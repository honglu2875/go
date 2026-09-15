"""Run one bounded SPMD rank from verified source, retaining process/results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def select_controller_cpus(available, count, explicit=None):
    available = sorted(available)
    if explicit is None:
        return available[:count]
    fields = explicit.split(',')
    if not fields or any(not x.isdecimal() for x in fields):
        raise ValueError('Explicit CPU list must contain comma-separated nonnegative integers')
    selected = list(map(int, fields))
    if len(selected) != count or len(set(selected)) != count or not set(selected) <= set(available):
        raise ValueError('Explicit CPU list must contain the declared number of distinct available CPUs')
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--controller-cpus', type=int, default=16)
    parser.add_argument('--controller-cpu-list', help='Explicit comma-separated CPUs; must match --controller-cpus')
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--resume-attempt', type=Path)
    parser.add_argument('--resume-turn', type=int)
    parser.add_argument('--stop-after-turn', type=int)
    args = parser.parse_args()
    if args.timeout <= 0 or args.controller_cpus < 1:
        parser.error('Timeout and controller CPU count must be positive')
    if (args.resume_attempt is None) != (args.resume_turn is None):
        parser.error('Resume attempt and turn must be paired')
    sys.path.insert(0, str(args.snapshot / 'packages/gozero/src'))
    from gozero.pod import load_hosts
    from gozero.snapshots import canonical_json, read_json, verify
    manifest = verify(args.snapshot)
    hosts = load_hosts(args.snapshot / 'ops/hosts.json')
    controller_rank = read_json(args.snapshot / 'ops/hosts.json')['coordinator_rank']
    host = next(h for h in hosts if h.hostname == socket.gethostname().split('.')[0])
    rank_dir = args.attempt / ('rank-%d' % host.rank)
    rank_dir.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((args.environment / 'gozero-runtime.json').read_text())
    if runtime['uv_lock_sha256'] != hashlib.sha256((args.snapshot / 'uv.lock').read_bytes()).hexdigest():
        raise RuntimeError('Runtime lock does not match the source snapshot')
    available_cpus = sorted(os.sched_getaffinity(0))
    controller_cpus = select_controller_cpus(available_cpus, args.controller_cpus, args.controller_cpu_list)
    environment = dict(os.environ)
    # The library used by the trainer must come from this source bundle.
    environment.update(PYTHONPATH=str(args.snapshot / 'packages/gozero/src'), PYTHONDONTWRITEBYTECODE='1',
                       PYTHONNOUSERSITE='1', JAX_PLATFORMS='tpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       GOZERO_HOST_RANK=str(host.rank), GOZERO_WORLD_SIZE=str(len(hosts)),
                       GOZERO_CONTROLLER_HOSTNAME=hosts[controller_rank].hostname,
                       GOZERO_SNAPSHOT_ID=manifest['snapshot_id'], GOZERO_ATTEMPT_ID=args.attempt.name)
    native = None
    recipe = read_json(args.snapshot / manifest['recipe'] / 'recipe.json')
    if recipe.get('requires_native') and args.native_receipt is None:
        raise RuntimeError('Native receipt is required by this recipe')
    if args.native_receipt is not None:
        native = read_json(args.native_receipt)
        if native['snapshot_id'] != manifest['snapshot_id'] or native['filename'] != 'lib_gozero_native.so':
            raise RuntimeError('Native build differs from the frozen source')
        with (args.native_receipt.parent / native['filename']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != native['binary_sha256']:
                raise RuntimeError('Staged native binary hash mismatch')
        environment['GOZERO_NATIVE_RECEIPT'] = str(args.native_receipt)
    command = ['taskset', '-c', ','.join(map(str, controller_cpus)), str(args.environment / 'bin/python'), '-u',
               str(args.snapshot / manifest['recipe'] / 'train.py'), '--config', str(args.snapshot / 'resolved_config.json'),
               '--output', str(rank_dir / 'artifacts')]
    if args.resume_attempt is not None:
        checkpoint = args.resume_attempt / ('rank-%d' % host.rank) / 'artifacts/checkpoints' / ('turn-%09d' % args.resume_turn)
        command.extend(['--resume', str(checkpoint)])
    if args.stop_after_turn is not None:
        command.extend(['--stop-after-turn', str(args.stop_after_turn)])
    start = time.time()
    status = {'schema_version': 1, 'kind': 'rank_execution', 'snapshot_id': manifest['snapshot_id'],
              'hostname': host.hostname, 'rank': host.rank, 'world_size': len(hosts), 'command': command,
              'controller_cpus': controller_cpus, 'start_unix_time': start, 'runtime': runtime}
    if native is not None:
        status['native'] = native
    (rank_dir / 'start.json').write_bytes(canonical_json(status))
    def cancellation_requested():
        try:
            request = read_json(args.attempt / 'cancel.json')
        except (OSError, ValueError):
            return False
        return (isinstance(request, dict) and request.get('kind') == 'cancel_attempt'
                and request.get('schema_version') == 1 and request.get('snapshot_id') == manifest['snapshot_id']
                and request.get('attempt_id') == args.attempt.name)
    if cancellation_requested():
        result = {**status, 'end_unix_time': time.time(), 'returncode': None, 'timed_out': False,
                  'cancelled': True, 'requested_signal': None, 'source_integrity': True, 'status': 'failed'}
        (rank_dir / 'result.json').write_bytes(canonical_json(result))
        print(json.dumps({'kind':'rank_finished','rank':host.rank,'status':'failed','returncode':None,
                          'result':str(rank_dir/'result.json')}),flush=True)
        raise SystemExit(1)
    timed_out = False
    cancelled = False
    reported_failure = False
    requested_signal = None
    with (rank_dir / 'stdout.log').open('w') as stdout, (rank_dir / 'stderr.log').open('w') as stderr:
        child = subprocess.Popen(command, cwd=args.snapshot, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        (rank_dir / 'process.json').write_bytes(canonical_json({'pid': child.pid, 'pgid': child.pid,
            'start_ticks': Path('/proc/%d/stat' % child.pid).read_text().split(') ', 1)[1].split()[19]}))
        def stop(signum, frame):
            nonlocal requested_signal
            requested_signal = signum
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, stop)
        deadline = time.monotonic() + args.timeout
        terminating = False
        while child.poll() is None:
            if not cancelled:
                cancelled = cancellation_requested()
            cancelled |= requested_signal is not None
            if not reported_failure:
                # A trainer may publish its failure then block in distributed
                # shutdown while another rank is inside a collective. Surface
                # that failure promptly to the existing peer-cancel protocol.
                scientific = rank_dir / 'artifacts/result.json'
                try:
                    if scientific.stat().st_size <= 16 * 2**20:
                        outcome = read_json(scientific)
                        reported_failure = (isinstance(outcome, dict) and outcome.get('schema_version') == 1
                            and outcome.get('snapshot_id') == manifest['snapshot_id'] and outcome.get('status') == 'failed')
                except (OSError, ValueError):
                    pass  # The writer can still be publishing the bounded file.
            if not terminating and (cancelled or reported_failure or time.monotonic() >= deadline):
                timed_out = not cancelled and not reported_failure
                terminating = True
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                deadline = time.monotonic() + 10
            elif terminating and time.monotonic() >= deadline:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                break
            time.sleep(0.05)
        returncode = child.wait()
        # Completed or crashed leaders can leave descendants behind. Every
        # process in this private session belongs to this finished attempt.
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    integrity = True
    try:
        verify(args.snapshot)
    except Exception:
        integrity = False
    result = {**status, 'end_unix_time': time.time(), 'returncode': returncode,
              'timed_out': timed_out, 'cancelled': cancelled, 'requested_signal': requested_signal, 'source_integrity': integrity,
              'reported_scientific_failure': reported_failure,
              'status': 'passed' if returncode == 0 and integrity and not timed_out and not cancelled and not reported_failure else 'failed'}
    (rank_dir / 'result.json').write_bytes(canonical_json(result))
    print(json.dumps({'kind': 'rank_finished', 'rank': host.rank, 'status': result['status'],
                      'returncode': returncode, 'result': str(rank_dir / 'result.json')}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
