#!/usr/bin/env python3
"""Execute a fixed portion of the registered checkpoint curve, without retries."""
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

REGISTRATION = 'd7dbd7461cdc76e2222609971b8895f17d6baa7cfcec82d2515449130a61b6d3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-index', type=int, required=True)
    parser.add_argument('--stop-index', type=int, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Curve registration changed')
    protocol = read_json(args.protocol)
    if not 0 <= args.start_index < args.stop_index <= len(protocol['checkpoints']):
        raise ValueError('Invalid registered checkpoint interval')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.operator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in protocol['checkpoints'][args.start_index:args.stop_index]:
            output = args.output / item['id']
            if output.exists():
                raise FileExistsError('No automatic retry or overwrite: ' + str(output))
            elapsed = 0.
            for previous in protocol['checkpoints']:
                path = args.output / previous['id'] / 'result.json'
                if path.exists():
                    result = read_json(path)
                    elapsed += result['finished_unix'] - result['started_unix']
            for file, expected in protocol['evaluation_code_sha256'].items():
                if sha256(SOURCE / file) != expected:
                    raise ValueError('Registered evaluation code differs: ' + file)
            if sha256(SOURCE / item['panel']) != item['panel_sha256']:
                raise ValueError('Registered panel differs')
            panel = read_json(SOURCE / item['panel'])
            specs = [read_json(SOURCE / m['spec']) for m in panel['matches']]
            # panel.py bounds each child by its game deadlines plus 60 seconds.
            # Termination unwinds the match's GTP contexts; reserve another
            # minute for bounded parent cleanup before starting a new panel.
            worst_case = max(s['game_timeout_seconds'] * 2 * len(s['openings']) + 60 for s in specs) + 60
            if elapsed + worst_case > protocol['budget']['maximum_total_panel_seconds']:
                raise RuntimeError('Remaining study budget cannot cover this bounded panel')
            if any(not (SOURCE / s['candidate']).is_file() for s in specs):
                raise ValueError('Checkpoint descriptor is not yet frozen; no replacement is permitted')
            print(json.dumps({'kind': 'curve_checkpoint_start', 'id': item['id'],
                              'source': SOURCE.name, 'spent_panel_seconds': elapsed,
                              'started_unix': time.time()}), flush=True)
            command = [sys.executable, '-B', str(SOURCE / 'eval/panel.py'), '--spec', str(SOURCE / item['panel']),
                       '--artifacts-root', str(args.workspace_root.resolve()), '--output', str(output.resolve())]
            child = subprocess.run(command)
            result = read_json(output / 'result.json')
            summary = result['summaries'][item['id']]
            print(json.dumps({'kind': 'curve_checkpoint_finished', 'id': item['id'],
                              'returncode': child.returncode, 'summary': summary,
                              'result_sha256': sha256(output / 'result.json')}), flush=True)
            # A capped game has no outcome but remains valid curve evidence.
            # Actual failed games or missing child records stop the suite.
            if (summary['scheduled_games'] != 64 or summary['failed_games']
                    or any(m['timed_out'] or not m['result_sha256'] for m in result['matches'])
                    or child.returncode not in (0, 1)):
                raise RuntimeError('Evaluation integrity/process failure; inspect retained records')
            verify(SOURCE)


if __name__ == '__main__':
    main()
