#!/usr/bin/env python3
"""Run the registered 584-game prefetch learning suite without retries."""
import argparse
import fcntl
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

REGISTRATION = 'd7d45556c9bfdb041d09d0d5f161eb080b3249ded8aac66d40a1f592e1ebe6f7'


def panel_limit(source, item):
    spec = read_json(source / item['path'])
    if item['kind'] == 'direct':
        pairs_per_worker = (len(spec['openings']) + len(spec['cpu_groups']) - 1) // len(spec['cpu_groups'])
        return 2 * pairs_per_worker * spec['game_timeout_seconds'] + 180
    children = [read_json(source / m['spec']) for m in spec['matches']]
    return max(2 * len(s['openings']) * s['game_timeout_seconds'] for s in children) + 180


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Prefetch registration changed')
    protocol = read_json(args.protocol)
    for name, expected in protocol['evaluation_code_sha256'].items():
        if sha256(SOURCE / name) != expected:
            raise ValueError('Registered evaluator differs: ' + name)
    for item in protocol['evaluation_panels_in_order']:
        if sha256(SOURCE / item['path']) != item['sha256']:
            raise ValueError('Registered panel differs')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'wide_prefetch_learning_suite', 'snapshot_id': SOURCE.name,
              'protocol_sha256': REGISTRATION, 'started_unix': time.time(), 'status': 'running', 'panels': []}
    try:
        with (args.output / '.operator.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for index, item in enumerate(protocol['evaluation_panels_in_order']):
                limit = panel_limit(SOURCE, item)
                spent = time.time() - report['started_unix']
                if spent + limit > protocol['budget']['maximum_new_eval_seconds']:
                    raise RuntimeError('Remaining registered budget cannot cover this bounded panel')
                identity = f'{index:02d}-{item["kind"]}'
                directory = args.output / identity
                print(json.dumps({'kind': 'prefetch_panel_start', 'id': identity, 'spent_seconds': spent}), flush=True)
                program = 'selfmatch.py' if item['kind'] == 'direct' else 'panel.py'
                command = [sys.executable, '-B', str(SOURCE / 'eval' / program), '--spec', str(SOURCE / item['path']),
                           '--artifacts-root', str(args.workspace_root.resolve()), '--output', str(directory)]
                with (args.output / (identity + '.stdout.log')).open('w') as out, (args.output / (identity + '.stderr.log')).open('w') as err:
                    child = subprocess.Popen(command, stdout=out, stderr=err)
                    try:
                        code = child.wait(timeout=limit - 90)
                    except BaseException:
                        child.terminate()
                        try:
                            child.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        raise
                result_path = directory / 'result.json'
                result = read_json(result_path)
                summaries = [result['summary']] if item['kind'] == 'direct' else list(result['summaries'].values())
                scheduled = sum(s['scheduled_games'] for s in summaries)
                failed = sum(s['failed_games'] for s in summaries)
                entry = {'id': identity, 'registration': item, 'returncode': code,
                         'result_sha256': sha256(result_path), 'summaries': summaries,
                         'elapsed_seconds': result['finished_unix'] - result['started_unix']}
                report['panels'].append(entry)
                (args.output / 'progress.json').write_bytes(canonical_json(report))
                print(json.dumps({'kind': 'prefetch_panel_finished', **entry}), flush=True)
                if (code not in (0, 1) or scheduled != item['games'] or failed or 'error' in result
                        or any(m['timed_out'] or not m['result_sha256'] for m in result.get('matches', []))):
                    raise RuntimeError('Evaluation integrity/process failure; records retained, no retry')
                verify(SOURCE)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        (args.output / 'result.json').write_bytes(canonical_json(report))


if __name__ == '__main__':
    main()
