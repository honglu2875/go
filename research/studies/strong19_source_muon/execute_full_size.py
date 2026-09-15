"""Run only the frozen full-size source-optimizer recovery sequence.

Invocation requires the preceding 9x9 intervention to be completely reviewed.
There is no unattended wait/auto-launch option and no scientific-run selection.
The existing pod controller enforces accelerator ownership and bounded cleanup.
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


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def inspect(path, digest):
    require(sha(path) == digest, 'Execution plan changed')
    plan = read(path)
    require(plan['kind'] == 'joint_source_muon_full_size_execution' and plan['stages'] == ['full', 'prefix', 'resumed']
            and plan['horizon'] == 4 and plan['prefix_turn'] == 2, 'Execution scope differs')
    for name, expected in plan['operators'].items():
        require(sha(ROOT / name) == expected, 'Execution operator changed: ' + name)
    for name, expected in plan['prerequisites'].items():
        require(sha(ROOT / name) == expected, 'Execution evidence changed: ' + name)
        value = read(ROOT / name)
        require(value['status'] in ('passed', 'prepared'), 'Execution prerequisite did not pass')
    snapshot = ROOT / '.gozero/snapshots' / plan['snapshot']
    manifest = verify(snapshot)
    config = read(snapshot / 'resolved_config.json')
    require(config['steps'] == 4 and config['expected_processes'] == 4 and config['expected_devices'] == 16
            and config['training']['purpose'] == 'qualification' and config['platform'] == 'tpu'
            and config['checkpoint_every'] == 4 and config['eval_every'] == 2
            and config['checkpoint_temporary'] and not config['evaluation']['run_test'], 'Wrong execution fixture')
    prior = plan['preceding_intervention']
    require(sha(ROOT / prior['registration']) == prior['registration_sha256'], 'Preceding registration changed')
    outcome_path = ROOT / prior['outcome']
    if not outcome_path.exists():
        return plan, snapshot, dict(ready=False, reason='The registered 9x9 intervention has not closed')
    outcome = read(outcome_path)
    require(outcome['kind'] == 'strong9_scalar_continuation_outcome' and outcome['status'] == 'passed'
            and outcome['registration_sha256'] == prior['registration_sha256'], 'Preceding closure differs')
    stages = outcome['stages']
    require('seed1' in stages and (not stages['seed1']['screen_passed'] or 'seed2' in stages),
            'The registered confirmation seed is still required')
    registration = read(ROOT / prior['registration'])
    for stage, result in stages.items():
        contrast = ROOT / 'research/studies/strong9_followups' / registration['trial'] / (stage + '-contrast-001.json')
        require(sha(contrast) == result['contrast_sha256'] and read(contrast)['status'] == 'passed',
                'Preceding stage evidence changed')
    return plan, snapshot, dict(ready=True, preceding_outcome_sha256=sha(outcome_path))


def resources(plan, remaining):
    from gozero.pod import load_hosts, SSH_OPTIONS
    hosts = load_hosts(ROOT / '.gozero/snapshots' / plan['snapshot'] / 'ops/hosts.json')
    code = "import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');d=os.statvfs('.');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,disk_free=d.f_bavail*d.f_frsize,memory_available=m['MemAvailable'])))"
    def one(host):
        command = 'cd ' + shlex.quote(str(ROOT)) + ' && taskset -c 0,1 python3 -c ' + shlex.quote(code)
        value = json.loads(subprocess.check_output(['ssh', *SSH_OPTIONS, host.ssh, command], text=True, timeout=30))
        reserve = (remaining * plan['checkpoint_reserve_bytes'] if host.rank in (0, 2) else 0)
        # Fresh continuation stages the prefix's owner arrays on every peer.
        # Reserve this separately from the retained completed-stage replicas.
        if host.rank != 0:
            reserve += plan['checkpoint_reserve_bytes']
        minimum = plan['shm_floor_bytes'] + plan['producer_growth_reserve_bytes'] + reserve
        require(value['shm_free'] > minimum and value['memory_available'] > plan['memory_floor_bytes']
                and value['disk_free'] > plan['disk_floor_bytes'], 'Insufficient execution headroom on host index ' + str(host.rank))
        return dict(host_index=host.rank, required_shm_free=minimum, **value)
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(one, hosts))


def run(path, digest):
    plan, snapshot, closure = inspect(path, digest)
    require(closure['ready'], closure.get('reason', 'Preceding intervention remains open'))
    folder = STUDY / plan['output_directory']
    require(folder.parent == STUDY and not folder.exists(), 'Execution already started or output escapes study')
    folder.mkdir()
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
        log = folder / (label + '.log')
        event('command_started', label=label)
        with log.open('xb') as stream:
            subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        event('command_passed', label=label, log_sha256=sha(log))
        return log
    attempts = {}
    try:
        for index, stage in enumerate(plan['stages']):
            inspect(path, digest)
            open_attempts = [p.parent.name for p in (ROOT / 'runs').glob('pod-*/launch.json')
                             if not (p.parent / 'result.json').exists()]
            require(not open_attempts, 'An accelerator attempt is still open')
            preflight = resources(plan, len(plan['stages']) - index)
            argv = [ROOT / '.venv/bin/python', '-B', snapshot / 'ops/pod_run.py', '--snapshot', snapshot,
                    '--workspace-root', ROOT, '--timeout', str(plan['stage_timeout_seconds']),
                    '--prepare-timeout', '600', '--controller-cpus', '6',
                    '--controller-cpu-list', '2,3,4,5,6,7']
            if stage == 'prefix':
                argv += ['--stop-after-turn', '2']
            elif stage == 'resumed':
                argv += ['--resume-attempt', attempts['prefix'], '--resume-turn', '2']
            publish(folder / (stage + '-launch-intent.json'), dict(plan_sha256=digest, stage=stage,
                    closure=closure, argv=list(map(str, argv)), preflight=preflight, created=time.time()))
            log = command(argv, stage + '-controller', plan['stage_timeout_seconds'] + 900, accelerator=True)
            launched = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
            headers = [r for r in launched if r.get('kind') == 'pod_attempt']
            require(len(headers) == 1, 'Controller did not publish exactly one attempt')
            header = headers[0]; attempt = Path(header['attempt'])
            require(attempt.parent == ROOT / 'runs' and header['snapshot_id'] == snapshot.name
                    and attempt.name == header['attempt_id'], 'Wrong launched attempt')
            closed = read(attempt / 'result.json')
            require(closed['status'] == 'passed' and closed['snapshot_id'] == snapshot.name, 'Attempt did not close successfully')
            expected_turn = 2 if stage == 'prefix' else 4
            for host in range(4):
                rank = read(attempt / f'rank-{host}/result.json')
                report = read(attempt / f'rank-{host}/artifacts/result.json')
                require(rank['status'] == 'passed' and rank['source_integrity'] and rank['returncode'] == 0
                        and not rank['timed_out'] and report['status'] == 'passed'
                        and report['turn'] == expected_turn and report['snapshot_id'] == snapshot.name,
                        'A rank did not complete the intended stage')
            attempts[stage] = attempt.name
            publish(folder / (stage + '-attempt.json'), dict(attempt=attempt.name, stage=stage,
                    result_sha256=sha(attempt / 'result.json'), plan_sha256=digest))
            command([sys.executable, '-B', STUDY / 'replicate_full_size.py', '--workspace-root', ROOT,
                     '--attempt', attempt.name, '--peer', '2', '--output', folder / (stage + '-replica.json')],
                    stage + '-replica', 600)
        argv = [plan['audit_python'], '-B', STUDY / 'distributed_recovery.py', '--snapshot', snapshot,
                '--prefix-turn', '2', '--output', folder / 'recovery-audit.json']
        for stage in plan['stages']:
            argv += ['--' + stage + '-artifacts', *[ROOT / 'runs' / attempts[stage] / f'rank-{h}/artifacts' for h in range(4)]]
        command(argv, 'recovery-audit', 900)
        audit = read(folder / 'recovery-audit.json')
        require(audit['status'] == 'passed' and audit['world_size'] == 4 and audit['model_parameters'] == 233220870,
                'Full-size distributed recovery did not pass')
        result = dict(status='passed', attempts=attempts, audit_sha256=sha(folder / 'recovery-audit.json'))
    except BaseException as error:
        result = dict(status='failed', attempts=attempts, error=repr(error))
        publish(folder / 'result.json', dict(kind=plan['kind'], plan_sha256=digest, started=started,
                                          finished=time.time(), **result))
        event('execution_failed', **result)
        raise
    publish(folder / 'result.json', dict(kind=plan['kind'], plan_sha256=digest, started=started,
                                      finished=time.time(), **result))
    event('execution_passed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        plan, snapshot, closure = inspect(args.plan, args.plan_sha256)
        print(json.dumps(dict(status='prepared', snapshot=snapshot.name, **closure, accelerator_jobs_started=False)))
        return
    with (STUDY / '.full-size-execution.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
