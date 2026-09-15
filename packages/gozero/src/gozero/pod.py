"""Explicit host selection and safely quoted pdsh/SSH commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex


def supervise(commands, *, directory, environment, timeout_seconds, cancel, grace_seconds=15.):
    """Watch each rank launcher; request remote cancellation on first failure.

    Local process groups are owned Popen sessions. The cancellation callback
    must be bounded and arrange remote attempt-scoped cleanup. A partition can
    still prevent delivery; callers retain each rank's independent deadline.
    """
    from contextlib import ExitStack
    import os
    import signal
    import subprocess
    import time
    if (not commands or timeout_seconds<=0 or grace_seconds<=0
            or len({rank for rank,_ in commands})!=len(commands)
            or any(type(rank) is not int or rank<0 or not argv for rank,argv in commands)):
        raise ValueError('Require unique rank commands and positive deadlines')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+timeout_seconds;processes={};reason=None;receipt=None
    with ExitStack() as stack:
        try:
            for rank,argv in commands:
                stdout=stack.enter_context((directory/f'rank-{rank}.stdout.log').open('x'))
                stderr=stack.enter_context((directory/f'rank-{rank}.stderr.log').open('x'))
                processes[rank]=subprocess.Popen(argv,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            while True:
                codes={rank:p.poll() for rank,p in processes.items()}
                failed=next((rank for rank,code in codes.items() if code is not None and code!=0),None)
                if reason is None and (failed is not None or time.monotonic()>=deadline):
                    reason=f'rank {failed} exited with status {codes[failed]}' if failed is not None else 'controller deadline expired'
                    try:receipt=cancel(reason)
                    except Exception as error:receipt={'status':'failed','error':repr(error)}
                    deadline=time.monotonic()+grace_seconds
                if all(code is not None for code in codes.values()):break
                if reason is not None and time.monotonic()>=deadline:break
                time.sleep(0.05)
        except BaseException:
            if reason is None:
                reason='controller interrupted or rank launch failed'
                try:receipt=cancel(reason)
                except Exception as error:receipt={'status':'failed','error':repr(error)}
            raise
        finally:
            for process in processes.values():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for process in processes.values():
                try:process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
    return {'status':'passed' if reason is None and all(p.returncode==0 for p in processes.values()) else 'failed',
            'reason':reason,'returncodes':{str(rank):p.returncode for rank,p in processes.items()},
            'cancellation':receipt,'elapsed_seconds':time.monotonic()-started}

from .snapshots import read_json


SSH_OPTIONS = ('-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10')
SSH_TARGET = re.compile(r'[a-z_][a-z0-9_-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\Z')


@dataclass(frozen=True)
class Host:
    rank: int
    ssh: str

    @property
    def hostname(self) -> str:
        return self.ssh.split('@', 1)[1]


def load_hosts(path: Path) -> tuple[Host, ...]:
    manifest = read_json(path)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported host manifest')
    hosts = tuple(Host(**row) for row in manifest['hosts'])
    if not hosts or sorted(h.rank for h in hosts) != list(range(len(hosts))):
        raise ValueError('Host ranks must be unique and contiguous from zero')
    if len({h.ssh for h in hosts}) != len(hosts) or len({h.hostname for h in hosts}) != len(hosts):
        raise ValueError('Hostnames must be unique')
    if any(not SSH_TARGET.fullmatch(h.ssh) for h in hosts):
        raise ValueError('Invalid SSH target')
    if manifest.get('coordinator_rank') not in range(len(hosts)):
        raise ValueError('Invalid coordinator rank')
    return tuple(sorted(hosts, key=lambda h: h.rank))


def pdsh_command(hosts: tuple[Host, ...], argv: list[str], timeout_seconds: int) -> list[str]:
    if not hosts or not argv or timeout_seconds < 1:
        raise ValueError('Require hosts, a command, and a positive timeout')
    return ['pdsh', '-R', 'ssh', '-S', '-b', '-f', str(len(hosts)), '-t', '10', '-u', str(timeout_seconds),
            '-w', ','.join(host.ssh for host in hosts), shlex.join(argv)]


def pdsh_environment(environment: dict[str, str]) -> dict[str, str]:
    result = dict(environment)
    result.pop('PDSH_SSH_ARGS', None)
    result['PDSH_SSH_ARGS_APPEND'] = shlex.join(SSH_OPTIONS)
    return result
