#!/usr/bin/env python3
"""Require equal native rollout work when only inference batch size changes."""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def compare(left, right):
    import numpy as np
    results = [read_json(p / 'result.json') for p in (left, right)]
    if any(r['status'] != 'passed' for r in results):
        raise ValueError('Both benchmark runs must pass')
    configs = [read_json(p / 'resolved_config.json') for p in (left, right)]
    batches = [c.pop('inference_batch') for c in configs]
    modes = [c.pop('dispatch_mode', 'serial') for c in configs]
    for c in configs:
        c.pop('verify_outputs', None)
    if configs[0] != configs[1] or not configs[0]['verify_trace']:
        raise ValueError('Only inference execution and diagnostic output verification may differ; trace verification is required')
    hashes = ('request_features_sha256', 'training_targets_sha256', 'game_records_sha256')
    for key in hashes:
        if results[0]['validation'][key] != results[1]['validation'][key]:
            raise ValueError('Rollout trace differs: ' + key)
    states = []
    root_values = []
    for directory, result in zip((left, right), results):
        path = directory / 'actors.json.gz'
        if sha256(path) != result['validation']['actors_gzip_sha256']:
            raise ValueError('Actor checkpoint hash differs')
        with gzip.open(path, 'rt') as stream:
            state = json.load(stream)
        values = []
        for actor in state['actors']:
            for row in actor['rows']:
                values.append(row.pop('root_value'))
        root_values.append(np.asarray(values, dtype=np.float64))
        states.append(state)
    if states[0] != states[1]:
        raise ValueError('Native actor states differ after excluding diagnostic root values')
    np.testing.assert_allclose(root_values[0], root_values[1], atol=1e-5, rtol=0)
    work = ('real_moves', 'neural_slots', 'active_neural_evaluations', 'completed_games', 'truncated_games', 'eligible_rows')
    for key in work:
        if results[0]['counters'][key] != results[1]['counters'][key]:
            raise ValueError('Useful work differs: ' + key)
    return {
        'status': 'passed', 'paths': [str(left), str(right)],
        'result_sha256': [sha256(p / 'result.json') for p in (left, right)],
        'snapshot_ids': [r['snapshot_id'] for r in results], 'inference_batches': batches, 'dispatch_modes': modes,
        'exact_trace_hashes': {key: results[0]['validation'][key] for key in hashes},
        'actor_state_exact_excluding_root_values': True,
        'maximum_root_value_error': float(np.max(np.abs(root_values[0] - root_values[1]), initial=0)),
        'work': {key: results[0]['counters'][key] for key in work},
        'elapsed_segment_seconds': [r['elapsed_segment_seconds'] for r in results],
        'inference_seconds': [r['counters']['inference_seconds'] for r in results],
        'neural_dispatches': [r['counters']['neural_dispatches'] for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hosts', type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    verify(ROOT)
    report = {'schema_version': 1, 'analysis_snapshot': ROOT.name, 'status': 'failed', 'hosts': [],
              'claims_mfu': False, 'claims_go_strength': False}
    try:
        for host in range(args.hosts):
            suffix = Path('.') if args.hosts == 1 else Path(f'rank-{host}/artifacts')
            report['hosts'].append(compare(args.left / suffix, args.right / suffix))
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
