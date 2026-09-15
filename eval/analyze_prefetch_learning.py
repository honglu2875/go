#!/usr/bin/env python3
"""Audit the preregistered wide-prefetch learning comparison and paired bounds."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import read as read_checkpoint, sha256
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import canonical_json, read_json, verify
from match import summarize
from run_prefetch_learning import REGISTRATION

KATA_BINARY = '1ae1ed2108caa025bba853634b7f1aad3a29baa0ba27b4a298759ca6b172d104'


def paired_statistics(games, pairs=256):
    if len(games) != 2 * pairs or {(g['pair'], g['candidate_color']) for g in games} != {
            (i, c) for i in range(pairs) for c in ('B', 'W')}:
        raise ValueError('Every scheduled color pair must be recorded exactly once')
    bounds = np.zeros((pairs, 2, 2), np.float64)
    for game in games:
        if game['status'] == 'completed' and game.get('candidate_points') in (0., .5, 1.):
            value = [game['candidate_points']] * 2
        elif game['status'] == 'truncated' and game.get('candidate_points') is None:
            value = [0., 1.]
        else:
            raise ValueError('Failed or assigned incomplete game invalidates paired evidence')
        bounds[game['pair'], int(game['candidate_color'] == 'W')] = value
    pair_bounds = bounds.mean(axis=1)
    lower, upper = pair_bounds.mean(axis=0)
    radius = math.sqrt(math.log(40) / (2 * pairs))
    completed = sum(g['status'] == 'completed' for g in games)
    interval = [max(0., float(lower) - radius), min(1., float(upper) + radius)]
    return {'scheduled_games': len(games), 'opening_pairs': pairs, 'completed_games': completed,
            'scheduled_score_bounds': [float(lower), float(upper)], 'paired_hoeffding_95_outer_interval': interval,
            'minimum_completion_criterion': completed / len(games) >= .95,
            'noninferiority_criterion': interval[0] > .4,
            'registered_criterion_met': completed / len(games) >= .95 and interval[0] > .4,
            'uncertainty_scope': 'Conditional on this one trained checkpoint pair. Caps are unassigned [0,1]; openings are the independent bounded units, not individual colors or training seeds.'}


def train_evidence(root, attempt, descriptor, expected_source):
    validated = validate_candidate(root, descriptor)
    if descriptor['training_snapshot'] != expected_source:
        raise ValueError('Training source differs')
    config = validated['config']
    directory = root / 'runs' / attempt
    pod = read_json(directory / 'result.json')
    if pod['status'] != 'passed' or pod['snapshot_id'] != expected_source:
        raise ValueError('Training attempt incomplete or different')
    with np.load(validated['weights'], allow_pickle=False) as saved:
        parameters = {k: saved[k] for k in saved.files}
    integers = ('real_moves', 'completed_games', 'truncated_games', 'eligible_rows', 'active_neural_evaluations',
                'neural_slots', 'neural_batches', 'inference_syncs', 'prefetch_neural_evaluations', 'prefetch_hits', 'prefetch_waste')
    counts = {k: 0 for k in integers}
    ranks, results, group_hashes = [], [], []
    for host in range(4):
        artifacts = directory / f'rank-{host}/artifacts'
        result = read_json(artifacts / 'result.json')
        checkpoint = artifacts / 'checkpoints' / f'turn-{config["selfplay_turns"]:09d}'
        group = read_json(checkpoint.with_suffix('.group.json'))
        group_hashes.append(sha256(checkpoint.with_suffix('.group.json')))
        manifest_hash = sha256(checkpoint / 'manifest.json')
        state, arrays, _ = read_checkpoint(checkpoint, expected_manifest_sha256=manifest_hash, array_prefix='p_')
        rank = state['jax_rank']
        if (result['status'] != 'passed' or result['snapshot_id'] != expected_source or state['snapshot_id'] != expected_source
                or state['turn'] != config['selfplay_turns'] or result['turn'] != state['turn']
                or state['world_size'] != 4 or rank not in range(4)
                or group['rank_manifests'][rank] != manifest_hash
                or group['snapshot_id'] != expected_source or group['turn'] != state['turn']
                or state['config_sha256'] != sha256(validated['snapshot'] / 'resolved_config.json')
                or group['config_sha256'] != state['config_sha256']
                or state['native_sha256'] != validated['native_binary_sha256']
                or group['updates'] != descriptor['network_version']
                or state['counters']['updates'] != descriptor['network_version']
                or result['model_export_sha256'] != descriptor['model_export_sha256']
                or sha256(artifacts / 'model_export.npz') != descriptor['model_export_sha256']):
            raise ValueError('Rank checkpoint or export identity differs')
        if set(arrays) != set(parameters) or any(not np.array_equal(arrays[k], v) for k, v in parameters.items()):
            raise ValueError('Replicated checkpoint parameters differ from evaluation model')
        for key in counts:
            def counter(counters):
                if key == 'inference_syncs' and key not in counters:
                    return counters['neural_batches']
                if key == 'prefetch_waste' and key not in counters:
                    return counters.get('prefetch_neural_evaluations', 0) - counters.get('prefetch_hits', 0)
                return counters.get(key, 0)
            value = counter(result['counters'])
            if type(value) is not int or value < 0 or counter(state['counters']) != value:
                raise ValueError('Training integer work accounting differs')
            counts[key] += value
        ranks.append(rank); results.append(result)
    if set(ranks) != set(range(4)) or len(set(group_hashes)) != 1:
        raise ValueError('Rank coverage or checkpoint group differs')
    return {'attempt': attempt, 'snapshot_id': expected_source, 'pod_result_sha256': sha256(directory / 'result.json'),
            'rank_result_sha256': [sha256(directory / f'rank-{h}/artifacts/result.json') for h in range(4)],
            'model_export_sha256': descriptor['model_export_sha256'], 'checkpoint_group_sha256': group_hashes[0],
            'counter_provenance': 'Legacy serial inference fetches inferred from one device_get per neural batch in its frozen trainer. Unused prefetch predictions are evaluated minus consumed. Other absent prefetch counters are zero for the serial control.',
            'global_counters': counts, 'learner_updates': descriptor['network_version'],
            'global_learner_exposures': descriptor['network_version'] * config['learner']['batch_size'] * 4,
            'recorded_attempt_chip_hours': pod['reserved_chip_hours'], 'attempt_elapsed_seconds': pod['elapsed_seconds'],
            'max_training_seconds': max(r['elapsed_segment_seconds'] for r in results),
            'max_inference_seconds': max(r['counters']['inference_seconds'] for r in results),
            'max_native_seconds': max(r['counters']['native_seconds'] for r in results),
            'max_learner_seconds': max(r['counters']['learner_seconds'] for r in results),
            'host_average_cpu_cores': [r['process_cpu_segment_seconds'] / r['elapsed_segment_seconds'] for r in results]}, config, pod


def audit_games(games, directory, spec, *, direct):
    expected = {(i, c) for i in range(len(spec['openings'])) for c in ('B', 'W')}
    if len(games) != len(expected) or {(g['pair'], g['candidate_color']) for g in games} != expected:
        raise ValueError('Scheduled game coverage differs')
    for game in games:
        folder = directory / f'pair-{game["pair"]:03d}-{game["candidate_color"]}'
        rules = game['referee_rules' if direct else 'katago_rules']
        if (read_json(folder / 'result.json') != game or game['opening'] != spec['openings'][game['pair']]
                or game['checked_positions'] != len(game['moves']) or game.get('integrity_failure') or 'error' in game
                or rules['ko'] != 'POSITIONAL' or rules['scoring'] != 'AREA' or rules['tax'] != 'NONE' or not rules['suicide']):
            raise ValueError('Game bytes, board checks or rules differ')
        if not direct and sha256(folder / 'katago.cfg') != game['katago_config_sha256']:
            raise ValueError('Per-game KataGo configuration differs')
        if game['status'] == 'completed':
            points = .5 if game['score'] == '0' else float(game['score'][0] == game['candidate_color'])
            scores = list(game['referee_scores'].values()) if direct else [game['katago_adjudicated_score']]
            if (not game['score_margin_agrees_with_katago'] or game['candidate_points'] != points
                    or any(s != game['score'] for s in scores)
                    or [m['vertex'].lower() for m in game['moves'][-2:]] != ['pass', 'pass']):
                raise ValueError('Completed game score differs')
        elif game['status'] != 'truncated' or game.get('candidate_points') is not None or len(game['moves']) != spec['max_game_moves']:
            raise ValueError('Failed game or invalid cap')
        totals = ({s: {k: 0 for k in game['search_totals'][s]} for s in ('candidate', 'opponent')} if direct
                  else {'candidate': {k: 0 for k in game['candidate_search_totals']}})
        for ply, move in enumerate(game['moves']):
            if move['color'] != ('B' if ply % 2 == 0 else 'W') or move['opening'] != (ply < len(game['opening'])):
                raise ValueError('Move color or opening boundary differs')
            if move['opening']:
                if move['vertex'] != game['opening'][ply]:
                    raise ValueError('Played opening differs')
                continue
            side = 'candidate' if move['color'] == game['candidate_color'] else 'opponent'
            if ('search' in move) != (direct or side == 'candidate'):
                raise ValueError('Actual search coverage differs')
            if 'search' in move:
                budget = spec[side + '_simulations'] if direct else spec['candidate_simulations_excluding_root']
                if move['search']['simulations'] != budget:
                    raise ValueError('Actual search budget differs')
                for key in totals[side]:
                    totals[side][key] += move['search'][key]
        if totals != (game['search_totals'] if direct else {'candidate': game['candidate_search_totals']}):
            raise ValueError('Actual neural/search work totals differ')


def audit_panel(root, protocol, item, directory, descriptors):
    result_path = directory / 'result.json'
    result = read_json(result_path)
    source = root / '.gozero/snapshots' / result['snapshot_id']
    verify(source)
    spec = read_json(source / item['path'])
    if (sha256(source / item['path']) != item['sha256'] or result['spec_sha256'] != item['sha256']
            or result['started_unix'] < protocol['registered_unix'] or result['finished_unix'] < result['started_unix']
            or (item['kind'] == 'direct' and read_json(directory / 'resolved_spec.json') != spec) or 'error' in result):
        raise ValueError('Panel identity or registration order differs')
    for file, expected in protocol['evaluation_code_sha256'].items():
        if sha256(source / file) != expected:
            raise ValueError('Evaluator changed after registration')
    all_games = []
    if item['kind'] == 'direct':
        for side in ('candidate', 'opponent'):
            descriptor = descriptors['candidate' if side == 'candidate' else 'control']
            native = root / '.gozero/native' / descriptor['training_snapshot'] / 'receipt.json'
            if (result['models'][side] != descriptor or read_json(source / spec[side]) != descriptor
                    or result[side + '_descriptor_sha256'] != sha256(source / spec[side])
                    or result['native_receipt_sha256'][side] != sha256(native)):
                raise ValueError('Direct model or engine identity differs')
        if (result['referee_binary_sha256'] != KATA_BINARY
                or result['referee_weights_sha256'] != protocol['strong_katago']['weights_sha256']
                or result['referee_config_sha256'] != sha256(source / spec['referee_config'])):
            raise ValueError('Referee identity differs')
        audit_games(result['games'], directory, spec, direct=True)
        if summarize(result['games']) != result['summary']:
            raise ValueError('Direct summary differs from raw games')
        all_games = result['games']
    else:
        declarations = spec['matches']
        if len(result['matches']) != len(declarations) or {m['id'] for m in result['matches']} != {m['id'] for m in declarations}:
            raise ValueError('KataGo child coverage differs')
        for declaration in declarations:
            entry = next(m for m in result['matches'] if m['id'] == declaration['id'])
            child_path = directory / entry['id'] / 'result.json'
            child = read_json(child_path)
            child_spec_path = source / declaration['spec']; child_spec = read_json(child_spec_path)
            native = root / '.gozero/native' / descriptors['candidate']['training_snapshot'] / 'receipt.json'
            if (entry['timed_out'] or entry['returncode'] not in (0, 1) or sha256(child_path) != entry['result_sha256']
                    or child['snapshot_id'] != source.name or child['spec_sha256'] != sha256(child_spec_path)
                    or read_json(directory / entry['id'] / 'resolved_spec.json') != child_spec
                    or child['candidate'] != descriptors['candidate']
                    or child['candidate_sha256'] != sha256(source / child_spec['candidate'])
                    or child['candidate_scoring_profile'] != 'pass_alive_area'
                    or child['native_receipt_sha256'] != sha256(native)
                    or child['base_katago_config_sha256'] != sha256(source / child_spec['katago_config'])
                    or child['katago_binary_sha256'] != KATA_BINARY or entry['games'] != child['games'] or 'error' in child):
                raise ValueError('KataGo child configuration or identity differs')
            weights_path = source / child_spec.get('katago_weights', 'eval/katago_9x9.json')
            weights = read_json(weights_path)
            anchor = protocol['early_katago' if item['kind'] == 'early' else 'strong_katago']
            if (weights['sha256'] != anchor['weights_sha256'] or child['katago_weights_sha256'] != weights['sha256']
                    or sha256(root / weights['path']) != weights['sha256']):
                raise ValueError('Official KataGo weights differ')
            if item['kind'] == 'early' and (sha256(weights_path) != anchor['descriptor_sha256']
                    or child['katago_weights_descriptor_sha256'] != anchor['descriptor_sha256']):
                raise ValueError('Early KataGo descriptor differs')
            audit_games(child['games'], directory / entry['id'], child_spec, direct=False)
            offset = len(all_games) // 2
            all_games.extend({**g, 'pair': g['pair'] + offset} for g in child['games'])
    if len(all_games) != item['games']:
        raise ValueError('Scheduled panel game count differs')
    return {'path': str(result_path.relative_to(root)), 'sha256': sha256(result_path), 'summary': summarize(all_games),
            'frozen_child_spec_sha256': ({m['spec']: sha256(source / m['spec']) for m in spec['matches']}
                                         if item['kind'] != 'direct' else {}),
            'checked_boards': sum(g['checked_positions'] for g in all_games),
            'checked_scores': sum(g['status'] == 'completed' for g in all_games),
            'elapsed_seconds': result['finished_unix'] - result['started_unix']}, all_games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--candidate-attempt', required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify(SOURCE)
    root = args.workspace_root.resolve(); protocol = read_json(args.protocol)
    args.suite = args.suite.resolve()
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Registration changed')
    report = {'schema_version': 1, 'kind': 'wide_prefetch_learning_analysis', 'analysis_snapshot': SOURCE.name,
              'protocol_sha256': REGISTRATION, 'status': 'failed', 'arms': {}, 'panels': [],
              'claims_repeated_systems_speedup': False, 'claims_mfu': False, 'production_promotion': False}
    try:
        suite = read_json(args.suite / 'result.json')
        source = root / '.gozero/snapshots' / suite['snapshot_id']; verify(source)
        build = read_json(source / 'eval/katago_build.json')
        if build['binary_sha256'] != KATA_BINARY or sha256(root / build['binary_path']) != KATA_BINARY:
            raise ValueError('Actual KataGo referee binary differs')
        attempts = [p for p in (root / 'runs').glob('pod-*/launch.json')
                    if read_json(p)['snapshot_id'] == protocol['candidate_snapshot']]
        if len(attempts) != protocol['budget']['maximum_new_training_attempts'] or attempts[0].parent.name != args.candidate_attempt:
            raise ValueError('Training attempt budget or selection differs')
        descriptors = {arm: read_json(source / f'eval/root_prefetch_learning_27/{arm}.json') for arm in ('control', 'candidate')}
        if sha256(source / 'eval/root_prefetch_learning_27/control.json') != protocol['control_descriptor_sha256']:
            raise ValueError('Reused control descriptor differs')
        configurations = []
        for arm, attempt in [('control', protocol['control_attempt']), ('candidate', args.candidate_attempt)]:
            evidence, config, pod = train_evidence(root, attempt, descriptors[arm], protocol[arm + '_snapshot'])
            report['arms'][arm] = evidence
            if (evidence['global_counters']['real_moves'] != protocol['budget']['global_real_moves']
                    or config['seed'] != 27 or config['actors']['seed'] != 27 or config['selfplay_turns'] != 32768):
                raise ValueError('Registered training budget differs')
            model_source = root / '.gozero/snapshots' / protocol[arm + '_snapshot']
            model_manifest = verify(model_source)
            if sha256(model_source / model_manifest['recipe'] / 'model.py') != protocol['model_implementation_sha256']:
                raise ValueError('Model implementation differs')
            if arm == 'candidate':
                if config.pop('root_prefetch') != protocol['intervention'] or pod['start_unix_time'] < protocol['registered_unix']:
                    raise ValueError('Intervention or registration order differs')
            configurations.append(config)
        if configurations[0] != configurations[1] or descriptors['control']['model_export_sha256'] != protocol['control_model_sha256']:
            raise ValueError('Control or non-prefetch configuration differs')
        if suite['status'] != 'passed' or suite['protocol_sha256'] != REGISTRATION or len(suite['panels']) != 10:
            raise ValueError('Registered suite did not finish')
        games = []
        for index, (item, entry) in enumerate(zip(protocol['evaluation_panels_in_order'], suite['panels'])):
            identity = f'{index:02d}-{item["kind"]}'
            if entry['id'] != identity or entry['registration'] != item or entry['returncode'] not in (0, 1):
                raise ValueError('Suite execution order differs')
            path = args.suite / identity
            if sha256(path / 'result.json') != entry['result_sha256']:
                raise ValueError('Suite child identity differs')
            evidence, raw = audit_panel(root, protocol, item, path, descriptors)
            report['panels'].append(evidence)
            if item['kind'] == 'direct':
                games.extend({**g, 'pair': index * 32 + g['pair']} for g in raw)
        report['primary'] = paired_statistics(games)
        report['direct_summary'] = summarize(games)
        report['by_color'] = {color: summarize([g for g in games if g['candidate_color'] == color]) for color in ('B', 'W')}
        report['new_eval_seconds'] = suite['finished_unix'] - suite['started_unix']
        if report['new_eval_seconds'] > protocol['budget']['maximum_new_eval_seconds']:
            raise ValueError('Registered evaluation budget exceeded')
        previous = root / protocol['strong_katago']['reused_control_result']
        if sha256(previous) != protocol['strong_katago']['reused_control_result_sha256']:
            raise ValueError('Reused strong anchor differs')
        report['reused_strong_control'] = {'path': str(previous.relative_to(root)), 'sha256': sha256(previous),
                                            'summaries': read_json(previous)['summaries'], 'new_eval_seconds': 0}
        control, candidate = (report['arms'][a] for a in ('control', 'candidate'))
        report['historical_timing_ratios'] = {key: control[key] / candidate[key] for key in ('max_training_seconds', 'max_inference_seconds')}
        report['learning_noninferiority_supported'] = report['primary']['registered_criterion_met']
        report['status'] = 'passed'; report['limitations'] = protocol['limitations']
        report['limitations'].append('Secondary KataGo panel paths and weights were registered, but their child configuration files were not individually hashed in the pre-training protocol. Their complete settings are bound to the immutable evaluation source and audited against raw execution. The eight direct primary specifications were individually hashed before training.')
        verify(SOURCE)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        args.output.chmod(0o444)
        print(json.dumps({k: v for k, v in report.items() if k in ('status', 'error', 'primary', 'learning_noninferiority_supported', 'new_eval_seconds')}), flush=True)


if __name__ == '__main__':
    main()
