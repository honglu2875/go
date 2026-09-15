#!/usr/bin/env python3
"""Validate registered CPU prefetch controls, full-state equivalence and resume."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'ops'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from compare_resume import compare as compare_resume

REGISTRATION = '165c17de53b2de6bc19ebc99ead57eadceeeb4f9ae56c7b074dc26d437f32791'
SYSTEM_COUNTERS = {'neural_batches', 'neural_slots', 'active_neural_evaluations',
                   'inference_syncs', 'prefetch_neural_evaluations', 'prefetch_hits'}


def load(root, directory, expected_source, final_turn=96):
    import numpy as np
    source = root / '.gozero/snapshots' / expected_source
    verify(source)
    result = read_json(directory / 'result.json')
    if result['status'] != 'passed' or result['snapshot_id'] != expected_source or result['turn'] != final_turn:
        raise ValueError('Training source, status or turn differs')
    config = read_json(source / 'resolved_config.json')
    if checkpoints.sha256(directory / 'model_export.npz') != result['model_export_sha256']:
        raise ValueError('Model export differs')
    state, arrays, actors = checkpoints.read(directory / 'checkpoints' / f'turn-{final_turn:09d}',
        expected_manifest_sha256=result['latest_checkpoint']['manifest_sha256'])
    for key in ('snapshot_id', 'config_sha256'):
        if state[key] != result[key]:
            raise ValueError('Checkpoint provenance differs')
    if state['native_sha256'] != result['native']['binary_sha256']:
        raise ValueError('Checkpoint native provenance differs')
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite saved training state')
    return {'directory': directory, 'config': config, 'state': state, 'arrays': arrays,
            'actors': json.loads(actors), 'result': result,
            'result_sha256': checkpoints.sha256(directory / 'result.json')}


def compare(left, right, legacy=False):
    import numpy as np
    configs = [copy.deepcopy(x['config']) for x in (left, right)]
    for c in configs:
        c.pop('root_prefetch', None)
        c['actors'].pop('worker_cpus')
    if configs[0] != configs[1]:
        raise ValueError('Scientific configuration differs')
    if set(left['arrays']) != set(right['arrays']):
        raise ValueError('Array tree differs')
    for key, array in left['arrays'].items():
        np.testing.assert_array_equal(array, right['arrays'][key], err_msg=key)
    actors = [copy.deepcopy(x['actors']) for x in (left, right)]
    for actor in actors:
        actor.pop('round')
        actor['config'].pop('worker_cpus')
    if actors[0] != actors[1]:
        raise ValueError('Actors differ outside request rounds and worker placement')
    states = [copy.deepcopy(x['state']) for x in (left, right)]
    for state in states:
        for key in ('snapshot_id', 'config_sha256', 'native_sha256'):
            state.pop(key)
        state['counters'] = {k: v for k, v in state['counters'].items()
                             if not k.endswith('_seconds') and k not in SYSTEM_COUNTERS}
    if states[0] != states[1]:
        raise ValueError('Scientific training state differs')
    names = [sorted(p.name for p in (x['directory'] / 'games').glob('*.json')) for x in (left, right)]
    if names[0] != names[1]:
        raise ValueError('Game identities differ')
    for name in names[0]:
        for suffix in ('.json', '.sgf'):
            a, b = [(x['directory'] / 'games' / name).with_suffix(suffix) for x in (left, right)]
            if a.read_bytes() != b.read_bytes():
                raise ValueError('Game record differs: ' + name)
    if left['result']['model_export_sha256'] != right['result']['model_export_sha256']:
        raise ValueError('Model exports differ')
    if legacy:
        for key in ('neural_batches', 'neural_slots', 'active_neural_evaluations'):
            if left['result']['counters'][key] != right['result']['counters'][key]:
                raise ValueError('Off changed legacy work counts')
    return {'status': 'passed', 'compared_arrays': len(left['arrays']), 'compared_games': len(names[0]),
            'all_arrays_exact': True, 'scientific_actors_exact': True, 'scientific_state_exact': True,
            'model_export_sha256': right['result']['model_export_sha256'],
            'exceptions': ['Request round numbers', 'Declared worker placement',
                           'Pinned source/config/native identities', 'Timing and explicit systems counters']}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace-root', 'protocol', 'directory', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if checkpoints.sha256(args.protocol) != REGISTRATION:
        raise ValueError('Qualification registration differs')
    protocol = read_json(args.protocol)
    root = args.workspace_root.resolve()
    directory = args.directory.resolve()
    report = {'schema_version': 1, 'kind': 'root_prefetch_cpu_qualification_result',
              'protocol_sha256': REGISTRATION, 'analysis_snapshot': SOURCE.name,
              'status': 'failed', 'claims_throughput': False, 'claims_mfu': False,
              'claims_strength': False, 'claims_sample_efficiency': False}
    try:
        runs = {mode: load(root, directory / mode, source) for mode, source in protocol['sources'].items()}
        for mode, run in runs.items():
            if run['config']['root_prefetch'] != {'mode': mode, 'limit': 16}:
                raise ValueError('Registered execution intervention differs')
            if run['config']['actors']['worker_cpus'] != [104, 105]:
                raise ValueError('CPU placement differs')
        legacy = load(root, root / 'runs/qualification/gumbel-a7bc6024/continuous',
                      'a7bc602431e0e4c21f7adeefff646471ff0758aabfafdf48fcb1156383c327b6')
        if legacy['config']['actors']['worker_cpus'] != [8, 9]:
            raise ValueError('Legacy placement differs')
        report['legacy_regression'] = compare(legacy, runs['off'], legacy=True)
        report['queued_equivalence'] = compare(runs['off'], runs['queued'])
        report['queued_resume'] = compare_resume(directory / 'queued', directory / 'resumed', 48, 96)
        wide = runs['wide']
        from gozero.native import Actors
        native = root / '.gozero/native' / protocol['sources']['wide']
        receipt = read_json(native / 'receipt.json')
        with Actors(wide['config']['actors'], native / receipt['filename'], receipt['binary_sha256'],
                    checkpoint=json.dumps(wide['actors'])) as engine:
            if json.loads(engine.checkpoint()) != wide['actors']:
                raise ValueError('Wide native state cannot restore exactly')
        differences = {}
        for key, value in runs['off']['arrays'].items():
            other = wide['arrays'][key]
            if value.shape != other.shape:
                differences[key] = {'shape': [list(value.shape), list(other.shape)]}
            elif not np.array_equal(value, other):
                differences[key] = {'max_absolute': float(np.max(np.abs(value.astype(np.float64) - other.astype(np.float64))))}
        report['wide'] = {'execution_and_native_restore_passed': True,
                          'saved_arrays_differing_from_off': differences,
                          'all_saved_arrays_exact': not differences,
                          'model_export_sha256': wide['result']['model_export_sha256']}
        report['arms'] = {}
        for mode, run in runs.items():
            counters = run['result']['counters']
            if counters['prefetch_hits'] > counters['prefetch_neural_evaluations']:
                raise ValueError('Cache hits exceed actual prefetched evaluations')
            report['arms'][mode] = {'source': protocol['sources'][mode],
                'result_sha256': run['result_sha256'], 'counters': counters,
                'unused_prefetched_evaluations': counters['prefetch_neural_evaluations'] - counters['prefetch_hits'],
                'elapsed_segment_seconds': run['result']['elapsed_segment_seconds']}
        report['limitations'] = protocol['limitations']
        report['decision'] = 'CPU queued semantics and recovery qualified; TPU equivalence, useful batch sizing and approximate wide execution need separate qualification.'
        report['status'] = 'passed'
        verify(SOURCE)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
