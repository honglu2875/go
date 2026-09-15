#!/usr/bin/env python3
"""Check that assembling the attention recipe preserved its CNN control."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    verify(ROOT)
    root = args.workspace_root.resolve()
    sources = ['a7bc602431e0e4c21f7adeefff646471ff0758aabfafdf48fcb1156383c327b6',
               'c458f549cae6862befd773fd6692531ff948081e1b4d8b15bf4d82be8008e385']
    directories = [root / 'runs/qualification/gumbel-a7bc6024/continuous',
                   root / 'runs/qualification/gumbel-attention-cnn-c458f549-r2']
    report = {'schema_version': 1, 'kind': 'gumbel_attention_cnn_regression',
              'analysis_snapshot': ROOT.name, 'status': 'failed', 'sources': sources,
              'scope': 'CPU 3x3 full trainer regression; excludes only declared model configuration extensions, CPU placement and pinned artifact identities. No TPU or architecture strength claim.'}
    try:
        configs = []; states = []; arrays = []; actors = []; results = []
        for source_id, directory in zip(sources, directories):
            source = root / '.gozero/snapshots' / source_id
            verify(source)
            c = read_json(source / 'resolved_config.json')
            r = read_json(directory / 'result.json')
            if r['status'] != 'passed' or r['snapshot_id'] != source_id or r['turn'] != 96:
                raise ValueError('Training identity or status differs')
            if checkpoints.sha256(directory / 'model_export.npz') != r['model_export_sha256']:
                raise ValueError('Export hash differs')
            state, saved, engine = checkpoints.read(directory / 'checkpoints/turn-000000096',
                expected_manifest_sha256=r['latest_checkpoint']['manifest_sha256'])
            configs.append(c); states.append(state); arrays.append(saved)
            actors.append(json.loads(engine)); results.append(r)
        for key, value in {'architecture': 'cnn', 'heads': 4, 'mlp_ratio': 4, 'max_board_size': 26}.items():
            if configs[1]['model'].pop(key) != value:
                raise ValueError('Unexpected CNN configuration extension')
        for index, expected in enumerate(([8, 9], [104, 105])):
            if configs[index]['actors'].pop('worker_cpus') != expected or actors[index]['config'].pop('worker_cpus') != expected:
                raise ValueError('Unexpected worker placement')
        if configs[0] != configs[1] or actors[0] != actors[1]:
            raise ValueError('Scientific config or full actors differ')
        if set(arrays[0]) != set(arrays[1]):
            raise ValueError('Saved array trees differ')
        for key in arrays[0]:
            np.testing.assert_array_equal(arrays[0][key], arrays[1][key], err_msg=key)
        for state, result in zip(states, results):
            for key in ('snapshot_id', 'config_sha256'):
                if state.pop(key) != result[key]:
                    raise ValueError('State provenance differs')
            if state.pop('native_sha256') != result['native']['binary_sha256']:
                raise ValueError('State native provenance differs')
            state['counters'] = {key: value for key, value in state['counters'].items() if not key.endswith('_seconds')}
        if states[0] != states[1] or results[0]['model_export_sha256'] != results[1]['model_export_sha256']:
            raise ValueError('Non-timing scientific state differs')
        games = sorted(p.name for p in (directories[0] / 'games').glob('*.json'))
        if games != sorted(p.name for p in (directories[1] / 'games').glob('*.json')):
            raise ValueError('Completed or truncated game set differs')
        for name in games:
            for suffix in ('.json', '.sgf'):
                paths = [(directory / 'games' / name).with_suffix(suffix) for directory in directories]
                if paths[0].read_bytes() != paths[1].read_bytes():
                    raise ValueError('Game record differs: ' + name)
        report.update(status='passed', all_arrays_exact=True, compared_arrays=len(arrays[0]),
                      actors_exact_except_declared_worker_placement=True, scientific_state_exact=True,
                      compared_games=len(games), model_export_sha256=results[0]['model_export_sha256'],
                      result_sha256=[checkpoints.sha256(directory / 'result.json') for directory in directories])
        verify(ROOT)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
