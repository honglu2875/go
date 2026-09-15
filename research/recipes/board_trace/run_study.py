#!/usr/bin/env python3
"""Execute the registered CPU trace conditions once, with bounded process groups."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    p.add_argument('--attempts-record', default='research/studies/board_trace/attempts.json')
    args = p.parse_args(); verify(SOURCE); root = args.workspace_root.resolve()
    study = root / 'research/studies/board_trace'; path = study / 'spec.json'
    if sha256(path) != args.expected_protocol_sha256:
        raise ValueError('Registration differs')
    protocol = read_json(path)
    if protocol['kind'] != 'state_validated_paired_causal_execution' or protocol['maximum_attempts_per_condition'] != 1:
        raise ValueError('Unexpected execution contract')
    final = (root / args.attempts_record).resolve()
    if not final.is_relative_to(study) or final.parent != study:
        raise ValueError('Attempt record must be a direct child of the study directory')
    progress = final.with_suffix('.progress.json')
    if final.exists() or progress.exists():
        raise FileExistsError('Registered conditions were already attempted')
    report = {'schema_version': 1, 'operator_snapshot': SOURCE.name,
              'protocol_sha256': args.expected_protocol_sha256, 'attempts': []}
    with (study / '.operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for condition in protocol['conditions']:
                source = artifact(root, '.gozero/snapshots/' + condition['snapshot']); verify(source)
                if sha256(path) != args.expected_protocol_sha256 or sha256(source / 'resolved_config.json') != condition['config_sha256']:
                    raise ValueError('Protocol or configuration changed')
                output = (root / condition['output']).resolve(); log = output.with_name(output.name + '.log')
                if not output.is_relative_to(root / 'runs') or not output.parent.is_dir():
                    raise ValueError('New output must stay within an existing runs directory')
                if output.exists() or log.exists():
                    raise FileExistsError('Condition output already exists; no retry')
                command = [sys.executable, '-B', str(source / 'research/recipes/board_trace/train.py'),
                           '--config', str(source / 'resolved_config.json'), '--native-receipt',
                           str(artifact(root, '.gozero/native/' + source.name + '/receipt.json')),
                           '--artifacts-root', str(root), '--output', str(output)]
                environment = {'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                attempt = {'name': condition['name'], 'command': command, 'environment': environment,
                           'started_unix': time.time(), 'status': 'running'}
                report['attempts'].append(attempt)
                progress.write_bytes(canonical_json(report))
                print(__import__('json').dumps({'kind': 'board_trace_start', 'name': condition['name']}), flush=True)
                with log.open('x') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             env={**os.environ, **environment}, start_new_session=True)
                    try:
                        code = child.wait(timeout=protocol['maximum_seconds_per_condition'])
                    except BaseException as error:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                        attempt['error'] = repr(error); code = child.returncode
                        if not isinstance(error, subprocess.TimeoutExpired):
                            attempt.update(status='failed', exit_code=code, finished_unix=time.time())
                            raise
                attempt.update(exit_code=code, finished_unix=time.time(), log_sha256=sha256(log))
                result_path = output / 'result.json'
                attempt['result_sha256'] = sha256(result_path) if result_path.exists() else None
                result = read_json(result_path) if result_path.exists() else {}
                attempt['status'] = 'passed' if code == 0 and result.get('status') == 'passed' and 'error' not in attempt else 'failed'
                verify(source)
                progress.write_bytes(canonical_json(report))
                print(__import__('json').dumps({'kind': 'board_trace_finished', 'name': condition['name'],
                    'status': attempt['status'], 'seconds': attempt['finished_unix'] - attempt['started_unix'],
                    'modes': {k: {'moves_per_game_per_dispatch': v['moves_per_game_per_dispatch'],
                        'sequential_prefix_exact': v.get('sequential_prefix_exact'),
                        'state_mismatch_stops': v.get('stop_state_mismatch', 0)} for k, v in result.get('modes', {}).items()}}), flush=True)
        except BaseException as error:
            report['error'] = repr(error)
            raise
        finally:
            verify(SOURCE)
            with final.open('xb') as stream:
                stream.write(canonical_json(report)); stream.flush(); os.fsync(stream.fileno())
            final.chmod(0o444)
    if any(a['status'] != 'passed' for a in report['attempts']):
        raise RuntimeError('An execution condition failed; all attempts retained without retry')


if __name__ == '__main__':
    main()
