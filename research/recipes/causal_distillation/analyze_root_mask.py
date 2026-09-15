#!/usr/bin/env python3
"""Verify and report the registered known-root legality execution probe."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify

REGISTRATION = '413cbda60f9cd50bfa74c84a7900a4124a6c25820fc11d33ae7b0afe30bc65c4'
MODES = ('sequential', 'behavior_independent', 'oracle_independent', 'oracle_shared')


def inspect(root, directory, source_id, protocol, masked):
    result_path = directory / 'result.json'
    result = read_json(result_path)
    source = root / '.gozero/snapshots' / source_id
    verify(source)
    if result['status'] != 'passed' or result['snapshot_id'] != source_id:
        raise ValueError('Run identity or qualification differs')
    if result['started_unix'] < protocol['registered_unix']:
        raise ValueError('Run predates registration')
    config = read_json(source / 'resolved_config.json')
    if read_json(directory / 'resolved_config.json') != config or sha256(source / 'resolved_config.json') != result['config_sha256']:
        raise ValueError('Configuration differs')
    for key in ('games', 'horizon', 'samples', 'packets', 'seed'):
        if config[key] != protocol[key]:
            raise ValueError('Registered budget differs: ' + key)
    if config['actors']['root_legal_mask'] is not masked:
        raise ValueError('Registered mask setting differs')
    parameter_hash = sha256(directory / 'untrained_parameters.npz')
    if parameter_hash != result['parameter_sha256']:
        raise ValueError('Parameter export identity differs')
    native = root / '.gozero/native' / source_id
    receipt = read_json(native / 'receipt.json')
    if receipt != result['native'] or sha256(native / receipt['filename']) != receipt['binary_sha256']:
        raise ValueError('Native binary identity differs')
    events = {}
    for mode in MODES:
        record = read_json(directory / (mode + '-events.json'))
        counts = result['modes'][mode]
        if sha256(directory / (mode + '-events.json')) != counts['event_sha256']:
            raise ValueError('Event archive identity differs')
        if sha256(directory / (mode + '.hlo.txt')) != result['compilation'][mode]['hlo_sha256']:
            raise ValueError('HLO identity differs')
        streams = record['events']
        if len(streams) != config['games'] or sum(map(len, streams)) != counts['real_moves']:
            raise ValueError('Event count differs')
        if len(record['packets']) != counts['dispatches']:
            raise ValueError('Dispatch count differs')
        expected_dispatches = counts['dispatches'] if mode == 'sequential' else protocol['packets']
        if counts['dispatches'] != expected_dispatches:
            raise ValueError('Packet budget differs')
        if counts['moves_per_game_per_dispatch'] != counts['real_moves'] / (config['games'] * counts['dispatches']):
            raise ValueError('Acceptance diagnostic differs')
        resolved = [x for packet in record['packets'] for x in packet['resolutions']]
        if sum(len(x['actions']) for x in resolved) != counts['real_moves']:
            raise ValueError('Resolved move count differs')
        if sum(x['legality_corrections'] for x in resolved) != counts['legality_corrections']:
            raise ValueError('Legality correction count differs')
        events[mode] = streams
    comparisons = {}
    for mode in MODES[1:]:
        if any(actual != reference[:len(actual)] for actual, reference in zip(events[mode], events['sequential'])):
            raise ValueError('Multi-ply actions differ from sequential policy')
        comparisons[mode] = sum(map(len, events[mode]))
    return {
        'path': str(directory), 'result_sha256': sha256(result_path),
        'snapshot_id': source_id, 'native_binary_sha256': receipt['binary_sha256'],
        'parameter_sha256': parameter_hash, 'parameter_count': result['parameter_count'],
        'model_implementation_sha256': sha256(source / 'research/recipes/dual_trace_probe/model.py'),
        'runner_implementation_sha256': sha256(source / 'research/recipes/dual_trace_probe/train.py'),
        'compared_events_by_mode': comparisons, 'modes': result['modes'],
    }, config, events['sequential']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Registration changed')
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'kind': 'known_root_legality_trace_analysis',
              'analysis_snapshot': ROOT.name, 'protocol_sha256': REGISTRATION,
              'status': 'failed', 'claims_training': False, 'claims_go_strength': False,
              'claims_throughput_improvement': False, 'claims_mfu': False, 'arms': {}}
    try:
        configs = []; sequences = []
        for arm, masked in (('control', False), ('candidate', True)):
            result, config, sequence = inspect(args.workspace_root.resolve(), getattr(args, arm).resolve(),
                                               protocol[arm + '_snapshot'], protocol, masked)
            report['arms'][arm] = result
            config['actors'].pop('root_legal_mask')
            configs.append(config); sequences.append(sequence)
        if configs[0] != configs[1]:
            raise ValueError('Configs differ outside root masking')
        a, b = (report['arms'][arm] for arm in ('control', 'candidate'))
        for key in ('parameter_sha256', 'parameter_count', 'native_binary_sha256',
                    'model_implementation_sha256', 'runner_implementation_sha256'):
            if a[key] != b[key]:
                raise ValueError('Arm identity differs: ' + key)
        compared = 0
        for left, right in zip(*sequences):
            length = min(len(left), len(right))
            if left[:length] != right[:length]:
                raise ValueError('Root masking changed the sequential action stream')
            compared += length
        report['cross_arm_sequential_prefix_events'] = compared
        report['diagnostics'] = {
            mode: {
                'control_moves_per_game_per_dispatch': a['modes'][mode]['moves_per_game_per_dispatch'],
                'candidate_moves_per_game_per_dispatch': b['modes'][mode]['moves_per_game_per_dispatch'],
                'resolved_moves_ratio_at_fixed_packets': b['modes'][mode]['real_moves'] / a['modes'][mode]['real_moves'],
                'control_legality_corrections': a['modes'][mode]['legality_corrections'],
                'candidate_legality_corrections': b['modes'][mode]['legality_corrections'],
            } for mode in MODES[1:]
        }
        report['limitations'] = protocol['limitations']
        report['decision'] = 'Root masking passes the registered event-preservation check and improves resolved moves in this fixed probe; retain for learned-model experiments.'
        report['status'] = 'passed'
        verify(ROOT)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(canonical_json({key: value for key, value in report.items() if key != 'arms'}).decode(), flush=True)


if __name__ == '__main__':
    main()
