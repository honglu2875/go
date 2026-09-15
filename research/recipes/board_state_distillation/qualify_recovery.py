#!/usr/bin/env python3
"""Compare every saved causal learner array and non-timing scientific field."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import read as checkpoint_read, sha256
from gozero.snapshots import canonical_json, read_json, verify


def scientific(state):
    return {**state, 'counters': {k: v for k, v in state['counters'].items() if not k.endswith('_seconds')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--hosts', type=int, choices=[1, 4], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify(SOURCE)
    report = {'schema_version': 1, 'kind': 'causal_learner_exact_recovery', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'baseline': str(args.baseline), 'resumed': str(args.resumed), 'ranks': []}
    try:
        for host in range(args.hosts):
            a, b = (d if args.hosts == 1 else d / f'rank-{host}/artifacts' for d in (args.baseline, args.resumed))
            ra, rb = (read_json(d / 'result.json') for d in (a, b))
            if (ra['status'] != 'passed' or rb['status'] != 'passed' or not ra['training_complete'] or not rb['training_complete']
                    or ra['turn'] != rb['turn'] or ra['snapshot_id'] != rb['snapshot_id']
                    or ra['jax_rank'] != rb['jax_rank'] or ra['world_size'] != args.hosts or rb['world_size'] != args.hosts
                    or ra['model_export_sha256'] != rb['model_export_sha256'] or ra['test'] != rb['test']
                    or ra['last_metrics'] != rb['last_metrics']):
                raise ValueError('Run identity, export or final metrics differ')
            verify(args.workspace_root / '.gozero/snapshots' / ra['snapshot_id'])
            states = []; arrays = []
            for path, result in ((a, ra), (b, rb)):
                checkpoint = path / 'checkpoints' / f'turn-{result["turn"]:09d}'
                group = read_json(checkpoint.with_suffix('.group.json'))
                state, saved, _ = checkpoint_read(checkpoint, expected_manifest_sha256=group['rank_manifests'][result['jax_rank']])
                if (group['snapshot_id'] != result['snapshot_id'] or group['turn'] != state['turn']
                        or sha256(checkpoint / 'arrays.npz') != group['replicated_arrays_sha256']
                        or sha256(path / 'model_export.npz') != result['model_export_sha256']):
                    raise ValueError('Checkpoint group or export integrity differs')
                states.append(state); arrays.append(saved)
            if scientific(states[0]) != scientific(states[1]) or set(arrays[0]) != set(arrays[1]):
                raise ValueError('Scientific state or tree differs after recovery')
            if any(not np.array_equal(arrays[0][k], arrays[1][k]) for k in arrays[0]):
                raise ValueError('Parameters or Adam moments differ after recovery')
            report['ranks'].append({'host_rank': host, 'jax_rank': ra['jax_rank'], 'arrays_exact': len(arrays[0]),
                                     'numpy_sampler_state_exact': True, 'final_metrics_exact': True,
                                     'baseline_result_sha256': sha256(a / 'result.json'), 'resumed_result_sha256': sha256(b / 'result.json'),
                                     'source_snapshot': ra['snapshot_id'], 'model_export_sha256': ra['model_export_sha256'],
                                     'turn': ra['turn'], 'resume_turn': rb['resumed_from']['turn']})
        if {r['jax_rank'] for r in report['ranks']} != set(range(args.hosts)):
            raise ValueError('Rank coverage differs')
        if args.hosts == 4:
            pods = [read_json(d / 'result.json') for d in (args.baseline, args.resumed)]
            if any(p['status'] != 'passed' for p in pods):
                raise ValueError('Pod controller did not pass')
            report['recorded_attempt_chip_hours'] = sum(p['reserved_chip_hours'] for p in pods)
        report['status'] = 'passed'; verify(SOURCE)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        args.output.chmod(0o444)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
