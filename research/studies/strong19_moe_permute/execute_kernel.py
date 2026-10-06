"""Run one prepared kernel qualification after the registered learner closes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from gozero.snapshots import canonical_json, read_json, verify


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if sha(args.registration) != args.sha256:
        raise ValueError('Registration changed')
    plan = read_json(args.registration)
    if plan['kind'] != 'moe_routing_kernel_qualification_registration':
        raise ValueError('Unexpected experiment scope')
    for name, expected in {**plan['prerequisites'], **plan['operators']}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Registered source/evidence changed: ' + name)
    if plan['operators'].get(str(Path(__file__).relative_to(ROOT))) != sha(Path(__file__)):
        raise ValueError('Runner was not registered')
    snapshot = ROOT / '.gozero/snapshots' / plan['snapshot']
    verify(snapshot)
    if sha(snapshot / 'resolved_config.json') != plan['config_sha256']:
        raise ValueError('Configuration changed')
    if not args.run:
        print(json.dumps(dict(status='prepared', snapshot=snapshot.name, accelerator_jobs_started=False)))
        return
    prior = ROOT / 'runs' / plan['must_follow_successful_attempt'] / 'result.json'
    if not prior.is_file() or read_json(prior)['status'] != 'passed':
        raise ValueError('The registered learner has not successfully closed')
    if any(not (p.parent / 'result.json').exists() for p in (ROOT / 'runs').glob('pod-*/launch.json')):
        raise ValueError('Another accelerator attempt is still open')
    from gozero.pod import load_hosts, SSH_OPTIONS
    code = "import os,json;d=os.statvfs('.');s=os.statvfs('/dev/shm');print(json.dumps(dict(disk_free=d.f_bavail*d.f_frsize,shm_free=s.f_bavail*s.f_frsize)))"
    def inspect(host):
        command = 'cd ' + shlex.quote(str(ROOT)) + ' && taskset -c 0,1 python3 -c ' + shlex.quote(code)
        observed = json.loads(subprocess.check_output(['ssh', *SSH_OPTIONS, host.ssh, command], text=True, timeout=30))
        if observed['disk_free'] < 2 * (1 << 30) or observed['shm_free'] < 4 * (1 << 30):
            raise OSError('Insufficient headroom on host rank ' + str(host.rank))
        return dict(rank=host.rank, **observed)
    with ThreadPoolExecutor(4) as pool:
        resources = list(pool.map(inspect, load_hosts(snapshot / 'ops/hosts.json')))
    folder = STUDY / plan['output_directory']; folder.mkdir(exist_ok=False)
    started = time.time()
    argv = [ROOT / '.venv/bin/python', '-B', snapshot / 'ops/pod_run.py', '--snapshot', snapshot,
            '--workspace-root', ROOT, '--timeout', str(plan['timeout_seconds']), '--prepare-timeout', '600',
            '--controller-cpus', '6', '--controller-cpu-list', '2,3,4,5,6,7']
    publish(folder / 'launch-intent.json', dict(registration_sha256=args.sha256, resources=resources,
            argv=list(map(str, argv)), created=started, prior_result_sha256=sha(prior)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('JAX_PLATFORMS', None)
    attempt = None
    try:
        with (folder / 'controller.log').open('xb') as stream:
            subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=plan['timeout_seconds'] + 1200, check=True)
        rows = [json.loads(s) for s in (folder / 'controller.log').read_text().splitlines() if s.startswith('{')]
        headers = [r for r in rows if r.get('kind') == 'pod_attempt']
        if len(headers) != 1 or headers[0]['snapshot_id'] != snapshot.name:
            raise ValueError('Unexpected attempt identity')
        attempt = Path(headers[0]['attempt'])
        if attempt.parent != ROOT / 'runs' or attempt.name != headers[0]['attempt_id']:
            raise ValueError('Attempt path differs')
        audit = [ROOT / '.venv/bin/python', '-B', STUDY / 'audit_qualification.py',
                 '--attempt', attempt, '--snapshot', snapshot.name, '--output', folder / 'audit.json']
        with (folder / 'audit.log').open('xb') as stream:
            subprocess.run(list(map(str, audit)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
        outcome = dict(status='passed', attempt=attempt.name, audit_sha256=sha(folder / 'audit.json'))
    except BaseException as error:
        outcome = dict(status='failed', attempt=attempt.name if attempt else None, error=repr(error))
        publish(folder / 'result.json', dict(registration_sha256=args.sha256, started=started, finished=time.time(), **outcome))
        raise
    publish(folder / 'result.json', dict(registration_sha256=args.sha256, started=started, finished=time.time(), **outcome))
    print(json.dumps(outcome), flush=True)


if __name__ == '__main__':
    main()
