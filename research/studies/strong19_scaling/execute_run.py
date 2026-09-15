"""Execute and audit one registered joint-model run using the qualified pod runner.

This operator does not choose subsequent experiments. Every invocation needs a
new immutable plan, explicit prerequisites, and checkpoint/memory reserves.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def inspect(path, digest):
    require(sha(path) == digest, 'Plan changed')
    p = read(path)
    require(p['kind'] == 'registered_joint19_run', 'Wrong execution scope')
    for name, expected in p['operators'].items():
        require(sha(ROOT / name) == expected, 'Operator changed: ' + name)
    require(p['operators'].get(str(Path(__file__).relative_to(ROOT))) == sha(Path(__file__)), 'Runner is not pinned')
    for name, expected in p['prerequisites'].items():
        artifact = ROOT / name
        require(sha(artifact) == expected and read(artifact)['status'] in ('passed', 'prepared'), 'Prerequisite differs: ' + name)
    snapshot = ROOT / '.gozero/snapshots' / p['snapshot']
    verify(snapshot)
    c = read(snapshot / 'resolved_config.json')
    require(sha(snapshot / 'resolved_config.json') == p['config_sha256']
            and c['training']['purpose'] == p['purpose']
            and c['steps'] == c['checkpoint_every'] == p['steps']
            and c['expected_processes'] == 4 and c['expected_devices'] == 16
            and c['platform'] == 'tpu' and c['checkpoint_temporary']
            and not c['evaluation']['run_test'], 'Configuration differs')
    require(not [q for q in (ROOT / 'runs').glob('pod-*/launch.json')
                 if not (q.parent / 'result.json').exists()], 'An accelerator attempt remains open')
    return p, snapshot


def resources(p):
    from gozero.pod import load_hosts, SSH_OPTIONS
    hosts = load_hosts(ROOT / '.gozero/snapshots' / p['snapshot'] / 'ops/hosts.json')
    code = "import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');d=os.statvfs('.');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,disk_free=d.f_bavail*d.f_frsize,memory_available=m['MemAvailable'])))"
    def one(host):
        command = 'cd ' + shlex.quote(str(ROOT)) + ' && taskset -c 0,1 python3 -c ' + shlex.quote(code)
        value = json.loads(subprocess.check_output(['ssh', *SSH_OPTIONS, host.ssh, command], text=True, timeout=30))
        minimum = p['shm_floor_bytes'] + p['producer_growth_reserve_bytes'] + p['additional_reserve_by_host'][str(host.rank)]
        if host.rank in (0, p['replica_peer']):
            minimum += p['checkpoint_reserve_bytes']
        require(value['shm_free'] > minimum and value['memory_available'] > p['memory_floor_bytes']
                and value['disk_free'] > p['disk_floor_bytes'], 'Insufficient headroom on rank ' + str(host.rank))
        return dict(rank=host.rank, required_shm_free=minimum, **value)
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(one, hosts))


def run(path, digest):
    p, snapshot = inspect(path, digest)
    folder = STUDY / p['output_directory']; folder.mkdir(exist_ok=False)
    started = time.time()
    def event(kind, **values):
        row = dict(kind=kind, time=time.time(), **values)
        with (folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n'); stream.flush()
        print(json.dumps(row), flush=True)
    def command(argv, label, timeout, accelerator=False):
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('JAX_PLATFORMS', None)
        if not accelerator:
            env['JAX_PLATFORMS'] = 'cpu'
        log = folder / (label + '.log'); event('command_started', label=label)
        with log.open('xb') as stream:
            subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        event('command_passed', label=label, log_sha256=sha(log))
        return log
    attempt = None
    try:
        preflight = resources(p)
        argv = [ROOT / '.venv/bin/python', '-B', snapshot / 'ops/pod_run.py', '--snapshot', snapshot,
                '--workspace-root', ROOT, '--timeout', str(p['timeout_seconds']), '--prepare-timeout', '600',
                '--controller-cpus', '6', '--controller-cpu-list', '2,3,4,5,6,7']
        publish(folder / 'launch-intent.json', dict(plan_sha256=digest, argv=list(map(str, argv)),
                preflight=preflight, created=time.time()))
        log = command(argv, 'controller', p['timeout_seconds'] + 900, accelerator=True)
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
        headers = [x for x in rows if x.get('kind') == 'pod_attempt']
        require(len(headers) == 1, 'Expected exactly one attempt')
        header = headers[0]; attempt = Path(header['attempt'])
        require(attempt.parent == ROOT / 'runs' and attempt.name == header['attempt_id']
                and header['snapshot_id'] == snapshot.name, 'Attempt differs')
        closed = read(attempt / 'result.json')
        require(closed['status'] == 'passed' and closed['snapshot_id'] == snapshot.name
                and closed['resume_attempt'] is None and closed['stop_after_turn'] is None, 'Attempt did not close')
        for rank in range(4):
            process = read(attempt / f'rank-{rank}/result.json')
            report = read(attempt / f'rank-{rank}/artifacts/result.json')
            require(process['status'] == 'passed' and process['source_integrity'] and process['returncode'] == 0
                    and not process['timed_out'] and not process['cancelled']
                    and report['status'] == 'passed' and report['turn'] == p['steps']
                    and report['snapshot_id'] == snapshot.name, 'A rank did not finish')
        publish(folder / 'attempt.json', dict(attempt=attempt.name, result_sha256=sha(attempt / 'result.json'), plan_sha256=digest))
        command([p['audit_python'], '-B', STUDY / 'audit_run.py', '--snapshot', snapshot,
                 '--artifacts', *[attempt / f'rank-{r}/artifacts' for r in range(4)],
                 '--purpose', p['purpose'], '--output', folder / 'audit.json'], 'audit', 900)
        audit = read(folder / 'audit.json')
        require(audit['status'] == 'passed' and audit['parameters'] == p['parameters'], 'Full model audit differs')
        if p.get('expected_positions') is not None:
            require(audit['positions'] == p['expected_positions'], 'Registered exposure replay differs')
        if p.get('validation_positions') is not None:
            require(all(r['raw_totals']['expert_count'] == p['validation_positions'] for r in audit['validation_history']),
                    'Validation does not cover the complete registered population')
        command([sys.executable, '-B', ROOT / 'research/studies/strong19_source_muon/replicate_full_size.py',
                 '--workspace-root', ROOT, '--attempt', attempt.name, '--peer', str(p['replica_peer']),
                 '--output', folder / 'replica.json'], 'replica', 600)
        outcome = dict(status='passed', attempt=attempt.name, audit_sha256=sha(folder / 'audit.json'),
                       replica_sha256=sha(folder / 'replica.json'))
    except BaseException as error:
        outcome = dict(status='failed', attempt=attempt.name if attempt else None, error=repr(error))
        publish(folder / 'result.json', dict(kind=p['kind'], plan_sha256=digest, started=started,
                                           finished=time.time(), **outcome))
        event('execution_failed', **outcome)
        raise
    publish(folder / 'result.json', dict(kind=p['kind'], plan_sha256=digest, started=started,
                                       finished=time.time(), **outcome))
    event('execution_passed', **outcome)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--plan-sha256', required=True)
    p.add_argument('--run', action='store_true'); args = p.parse_args()
    if not args.run:
        plan, snapshot = inspect(args.plan, args.plan_sha256)
        print(json.dumps(dict(status='prepared', snapshot=snapshot.name, accelerator_jobs_started=False)))
        return
    with (STUDY / '.registered-run.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
