#!/usr/bin/env python3
"""Run an ordered, hash-pinned sequence of bounded visual-model profiles."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--expected-schedule-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE)
    root = a.workspace_root.resolve(); schedule = read_json(a.schedule)
    if sha256(a.schedule) != a.expected_schedule_sha256 or schedule['kind'] != 'visual_supplied_path_profile_schedule':
        raise ValueError('Profile schedule identity differs')
    if not 1 <= len(schedule['cases']) <= 12 or schedule['seconds_per_attempt'] > 900:
        raise ValueError('Schedule exceeds bounded profile scope')
    recipe = schedule.get('recipe', 'research/recipes/visual_causal')
    if recipe not in ('research/recipes/visual_causal', 'research/recipes/visual_draft'):
        raise ValueError('Unsupported profiling recipe')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    report = {'schema_version': 1, 'kind': schedule['kind'], 'operator_snapshot': SOURCE.name,
              'schedule_sha256': a.expected_schedule_sha256, 'started_unix': time.time(),
              'status': 'running', 'attempts': []}
    try:
        for case in schedule['cases']:
            snapshot = root / '.gozero/snapshots' / case['snapshot_id']
            manifest = verify(snapshot)
            if (sha256(snapshot / 'resolved_config.json') != case['config_sha256']
                    or manifest['recipe'] != recipe):
                raise ValueError('Frozen case differs')
            before = {p.parent.name for p in (root / 'runs').glob('pod-*/launch.json')}
            command = [sys.executable, '-B', str(snapshot / 'ops/pod_run.py'), '--snapshot', str(snapshot),
                       '--workspace-root', str(root), '--timeout', str(schedule['seconds_per_attempt']),
                       '--prepare-timeout', '180', '--controller-cpus', '32']
            with (a.output / (case['id'] + '.log')).open('x') as log:
                done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=schedule['seconds_per_attempt'] + 700)
            added = {p.parent.name for p in (root / 'runs').glob('pod-*/launch.json')} - before
            if len(added) != 1:
                raise ValueError('Could not uniquely attribute profile attempt')
            attempt = added.pop(); result = root / 'runs' / attempt / 'result.json'
            observed = read_json(result)
            entry = {**case, 'attempt': attempt, 'result_sha256': sha256(result),
                     'status': observed['status'], 'returncode': done.returncode}
            report['attempts'].append(entry)
            with (a.output / 'progress.jsonl').open('ab') as log:
                log.write(canonical_json(entry)); log.flush()
            print(json.dumps(entry), flush=True)
            # A correctness or execution failure ends this schedule. Any repair
            # gets a new frozen source and explicit follow-up registration.
            if done.returncode or observed['status'] != 'passed':
                raise RuntimeError('Profile case failed: ' + case['id'])
        verify(SOURCE)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        (a.output / 'result.json').write_bytes(canonical_json(report))


if __name__ == '__main__':
    main()
