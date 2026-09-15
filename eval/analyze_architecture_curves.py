#!/usr/bin/env python3
"""Validate registered checkpoint curves and paired, unresolved-outcome bounds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import read as read_checkpoint, sha256
from gozero.model_artifacts import artifact, validate_candidate
from gozero.snapshots import canonical_json, read_json, verify
from match import summarize

REGISTRATION = 'd7dbd7461cdc76e2222609971b8895f17d6baa7cfcec82d2515449130a61b6d3'
COUNT_KEYS = ('real_moves', 'completed_games', 'truncated_games', 'eligible_rows',
              'active_neural_evaluations', 'neural_batches', 'neural_slots')


def points_interval(game):
    if game['status'] == 'completed':
        points = game['candidate_points']
        if points not in (0., .5, 1.):
            raise ValueError('Invalid completed score')
        return points, points
    if game['status'] == 'truncated' and game.get('candidate_points') is None:
        return 0., 1.
    raise ValueError('Failed or assigned incomplete game invalidates study')


def curve_statistics(bounds, *, draws, seed):
    """Axes: seed, architecture (CNN, attention), budget, opening, color, bound."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if (bounds.shape != (2, 2, 4, 32, 2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds < 0) or np.any(bounds > 1)
            or np.any(bounds[..., 0] > bounds[..., 1])):
        raise ValueError('Registered complete paired tensor required')
    # Collapse colors, budgets and fixed training seeds inside each opening.
    pair_bounds = bounds.mean(axis=4)
    differences = np.stack((pair_bounds[:, 1, ..., 0] - pair_bounds[:, 0, ..., 1],
                            pair_bounds[:, 1, ..., 1] - pair_bounds[:, 0, ..., 0]), axis=-1)
    by_opening = differences.mean(axis=(0, 1))
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, 32, size=(draws, 32))
    samples = by_opening[indices].mean(axis=1)
    intervals = np.quantile(samples, [.025, .975], axis=0, method='linear')
    seed_bounds = differences.mean(axis=(1, 2))
    return {'mean_difference_bounds': by_opening.mean(axis=0).tolist(),
            'bootstrap_95_lower_bound_interval': intervals[:, 0].tolist(),
            'bootstrap_95_upper_bound_interval': intervals[:, 1].tolist(),
            'bootstrap_95_outer_interval': [float(intervals[0, 0]), float(intervals[1, 1])],
            'seed_mean_difference_bounds': {str(s): seed_bounds[i].tolist() for i, s in enumerate((27, 28))},
            'opening_difference_bounds': by_opening.tolist(),
            'bootstrap_draws': draws, 'bootstrap_seed': seed,
            'positive_effect_criterion': bool(intervals[0, 0] > 0 and np.all(seed_bounds[:, 0] > 0))}


def training_evidence(root, item, candidate, validated):
    if (candidate['schema_version'] != 2 or candidate['training_snapshot'] != item['training_snapshot']
            or candidate['checkpoint']['path'] != item['checkpoint_path']
            or validated['checkpoint_turn'] != item['turn']):
        raise ValueError('Candidate differs from registered checkpoint')
    config = validated['config']
    if config['seed'] != item['seed'] or config['actors']['size'] != 9:
        raise ValueError('Training seed or size differs')
    directory = artifact(root, 'runs/' + item['training_attempt'])
    attempt = read_json(directory / 'result.json')
    if attempt['status'] != 'passed' or attempt['snapshot_id'] != item['training_snapshot']:
        raise ValueError('Training attempt did not complete as registered')
    group_path = artifact(root, item['checkpoint_path']).with_suffix('.group.json')
    group = read_json(group_path)
    counts = {k: 0 for k in COUNT_KEYS}
    ranks = []
    with np.load(validated['weights'], allow_pickle=False) as saved:
        parameters = {key: saved[key] for key in saved.files}
    for host in range(4):
        checkpoint = directory / f'rank-{host}/artifacts/checkpoints/turn-{item["turn"]:09d}'
        if sha256(checkpoint.with_suffix('.group.json')) != sha256(group_path):
            raise ValueError('Rank checkpoint group differs')
        manifest_hash = sha256(checkpoint / 'manifest.json')
        state, arrays, _ = read_checkpoint(checkpoint, expected_manifest_sha256=manifest_hash, array_prefix='p_')
        rank = state['jax_rank']
        if (state['snapshot_id'] != item['training_snapshot'] or state['turn'] != item['turn']
                or state['world_size'] != 4 or rank not in range(4)
                or group['rank_manifests'][rank] != manifest_hash
                or state['config_sha256'] != sha256(validated['snapshot'] / 'resolved_config.json')
                or state['native_sha256'] != validated['native_binary_sha256']
                or state['counters']['updates'] != candidate['network_version']):
            raise ValueError('Rank checkpoint scientific identity differs')
        if set(arrays) != set(parameters) or any(not np.array_equal(arrays[k], v) for k, v in parameters.items()):
            raise ValueError('Checkpoint ranks disagree on model parameters')
        for key in counts:
            value = state['counters'][key]
            if type(value) is not int or value < 0:
                raise ValueError('Invalid checkpoint counter')
            counts[key] += value
        ranks.append({'host_rank': host, 'jax_rank': rank, 'manifest_sha256': manifest_hash})
    if {r['jax_rank'] for r in ranks} != set(range(4)) or counts['real_moves'] != item['global_real_moves']:
        raise ValueError('Training move budget or rank coverage differs')
    return {'checkpoint_ranks': ranks, 'global_counters': counts,
            'learner_updates': candidate['network_version'],
            'global_learner_exposures': candidate['network_version'] * config['learner']['batch_size'] * 4,
            'parameter_count': sum(a.size for a in parameters.values()),
            'whole_attempt': {'id': item['training_attempt'], 'result_sha256': sha256(directory / 'result.json'),
                              'reserved_chip_hours': attempt['reserved_chip_hours'],
                              'elapsed_seconds': attempt['elapsed_seconds']},
            'checkpoint_elapsed_wall_seconds': None}


def analyze_checkpoint(root, protocol, item, directory):
    panel_path = directory / 'result.json'
    panel = read_json(panel_path)
    source = artifact(root, Path('.gozero/snapshots') / panel['snapshot_id'])
    verify(source)
    if (panel['spec_sha256'] != item['panel_sha256']
            or sha256(source / item['panel']) != item['panel_sha256']
            or panel['started_unix'] < protocol['registered_unix']
            or panel['finished_unix'] < panel['started_unix'] or 'error' in panel):
        raise ValueError('Panel identity, registration order or execution differs')
    for file, expected in protocol['evaluation_code_sha256'].items():
        if sha256(source / file) != expected:
            raise ValueError('Registered evaluation code differs: ' + file)
    declarations = read_json(source / item['panel'])['matches']
    expected_ids = {f'{item["id"]}-chunk-{chunk}' for chunk in range(4)}
    if (len(declarations) != 4 or len(panel['matches']) != 4
            or {m['id'] for m in declarations} != expected_ids
            or {m['id'] for m in panel['matches']} != expected_ids):
        raise ValueError('Child identity or coverage differs')
    descriptor = source / f'eval/architecture_curves/{item["id"]}.json'
    candidate = read_json(descriptor)
    validated = validate_candidate(root, candidate)
    training = training_evidence(root, item, candidate, validated)
    native = root / '.gozero/native' / item['training_snapshot'] / 'receipt.json'
    if read_json(native)['binary_sha256'] != validated['native_binary_sha256']:
        raise ValueError('Evaluation native engine differs from training')
    games, children = [], []
    bounds = np.full((32, 2, 2), np.nan)
    budget, kata = protocol['budget'], protocol['katago']
    for chunk in range(4):
        identity = f'{item["id"]}-chunk-{chunk}'
        entry = next(m for m in panel['matches'] if m['id'] == identity)
        declaration = next(m for m in declarations if m['id'] == identity)
        if (entry['group'] != item['id'] or declaration['group'] != item['id']
                or entry['timed_out'] or entry['returncode'] not in (0, 1)):
            raise ValueError('Child group or process failure')
        child_path = directory / identity / 'result.json'
        if sha256(child_path) != entry['result_sha256']:
            raise ValueError('Child result identity differs')
        child = read_json(child_path)
        spec_path = artifact(source, declaration['spec'])
        spec = read_json(spec_path)
        openings = protocol['openings'][chunk * 8:(chunk + 1) * 8]
        required = {'size': 9, 'komi': 7.5, 'openings': openings,
                    'max_game_moves': budget['max_game_moves'], 'game_timeout_seconds': budget['game_timeout_seconds'],
                    'candidate_simulations_excluding_root': budget['candidate_simulations'],
                    'katago_max_visits': budget['katago_visits'], 'candidate_cpuct': 0.,
                    'candidate': str(descriptor.relative_to(source)), 'katago_weights': kata['descriptor'],
                    'seed_prefix': f'gozero-architecture-curves-{protocol["opening_seed"]}-chunk-{chunk}'}
        if (any(spec[key] != value for key, value in required.items())
                or read_json(directory / identity / 'resolved_spec.json') != spec
                or sha256(spec_path) != child['spec_sha256'] or child['snapshot_id'] != source.name
                or child['candidate'] != candidate or child['candidate_sha256'] != sha256(descriptor)
                or child['candidate_scoring_profile'] != 'pass_alive_area'
                or child['native_receipt_sha256'] != sha256(native)
                or sha256(source / spec['katago_config']) != child['base_katago_config_sha256']
                or 'error' in child):
            raise ValueError('Match budget, identity or configuration differs')
        weights_path = source / kata['descriptor']
        weights = read_json(weights_path)
        build = read_json(source / 'eval/katago_build.json')
        if (sha256(weights_path) != kata['descriptor_sha256']
                or weights['sha256'] != kata['weights_sha256']
                or sha256(artifact(root, weights['path'])) != kata['weights_sha256']
                or child['katago_weights_descriptor_sha256'] != kata['descriptor_sha256']
                or child['katago_weights_sha256'] != kata['weights_sha256']
                or child['katago_binary_sha256'] != kata['binary_sha256']
                or sha256(artifact(root, build['binary_path'])) != kata['binary_sha256']):
            raise ValueError('Official KataGo artifact differs')
        raw = child['games']
        expected = {(pair, color) for pair in range(8) for color in ('B', 'W')}
        if len(raw) != 16 or {(g['pair'], g['candidate_color']) for g in raw} != expected or entry['games'] != raw:
            raise ValueError('Paired game coverage or panel records differ')
        for game in raw:
            game_dir = directory / identity / f'pair-{game["pair"]:03d}-{game["candidate_color"]}'
            if read_json(game_dir / 'result.json') != game or sha256(game_dir / 'katago.cfg') != game['katago_config_sha256']:
                raise ValueError('Individual game record or KataGo config differs')
            rules = game['katago_rules']
            if (game['opening'] != openings[game['pair']] or game['checked_positions'] != len(game['moves'])
                    or game.get('integrity_failure') or 'error' in game
                    or rules['ko'] != 'POSITIONAL' or rules['scoring'] != 'AREA'
                    or rules['tax'] != 'NONE' or not rules['suicide']):
                raise ValueError('Opening, board verification or rules differ')
            points = points_interval(game)
            if game['status'] == 'completed':
                expected_points = .5 if game['score'] == '0' else float(game['score'][0] == game['candidate_color'])
                if (not game['score_margin_agrees_with_katago'] or game['score'] != game['katago_adjudicated_score']
                        or game['candidate_points'] != expected_points
                        or [m['vertex'].lower() for m in game['moves'][-2:]] != ['pass', 'pass']):
                    raise ValueError('Terminal score or outcome differs')
            elif len(game['moves']) != budget['max_game_moves']:
                raise ValueError('Unresolved game did not reach the registered move cap')
            totals = {k: 0 for k in game['candidate_search_totals']}
            for ply, move in enumerate(game['moves']):
                if move['color'] != ('B' if ply % 2 == 0 else 'W') or move['opening'] != (ply < len(game['opening'])):
                    raise ValueError('Move sequence or opening boundary differs')
                if move['opening'] and move['vertex'] != game['opening'][ply]:
                    raise ValueError('Played opening differs')
                ours = not move['opening'] and move['color'] == game['candidate_color']
                if ('search' in move) != ours:
                    raise ValueError('Candidate search coverage differs')
                if ours:
                    if move['search']['simulations'] != budget['candidate_simulations']:
                        raise ValueError('Actual candidate search budget differs')
                    for key in totals:
                        totals[key] += move['search'][key]
            if totals != game['candidate_search_totals']:
                raise ValueError('Candidate search work accounting differs')
            pair = chunk * 8 + game['pair']
            bounds[pair, 0 if game['candidate_color'] == 'B' else 1] = points
            games.append({**game, 'pair': pair})
        summary = summarize(raw)
        expected_status = 'passed' if summary['completed_games'] == 16 else 'failed'
        if (child['summary'] != summary or entry['summary'] != summary
                or child['status'] != expected_status or entry['status'] != expected_status
                or entry['returncode'] != int(expected_status != 'passed')):
            raise ValueError('Child aggregate or completion status differs')
        children.append({'id': identity, 'result_sha256': sha256(child_path), 'status': child['status']})
    summary = summarize(games)
    expected_status = 'passed' if summary['completed_games'] == 64 else 'failed'
    if (summary != panel['summaries'][item['id']] or summary['scheduled_games'] != 64
            or panel['status'] != expected_status):
        raise ValueError('Panel aggregate differs')
    evidence = {'id': item['id'], 'seed': item['seed'], 'architecture': item['architecture'],
                'turn': item['turn'], 'global_real_moves': item['global_real_moves'], 'training': training,
                'evaluation_snapshot': source.name, 'result_sha256': sha256(panel_path),
                'model_export_sha256': candidate['model_export_sha256'], 'children': children,
                'summary': summary, 'scheduled_score_bounds': bounds.mean(axis=(0, 1)).tolist(),
                'checked_boards': sum(g['checked_positions'] for g in games),
                'early_double_pass_games': sum(g['status'] == 'completed' and len(g['moves']) <= 25 for g in games),
                'colors': {c: {'wins': sum(g.get('candidate_points') == 1 for g in games if g['candidate_color'] == c),
                               'losses': sum(g.get('candidate_points') == 0 for g in games if g['candidate_color'] == c),
                               'draws': sum(g.get('candidate_points') == .5 for g in games if g['candidate_color'] == c),
                               'capped': sum(g['status'] == 'truncated' for g in games if g['candidate_color'] == c)}
                           for c in ('B', 'W')},
                'completion_coverage': summary['completed_games'] / 64,
                'started_unix': panel['started_unix'], 'finished_unix': panel['finished_unix'],
                'elapsed_seconds': panel['finished_unix'] - panel['started_unix']}
    return evidence, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'protocol', 'results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Registration changed')
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'kind': 'registered_architecture_curve_analysis', 'status': 'failed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': REGISTRATION, 'checkpoints': [],
              'production_promotion': False, 'claims_superiority_over_current_katago': False,
              'claims_population_sample_efficiency_improvement': False}
    try:
        root = args.workspace_root.resolve()
        if sha256(root / 'research/studies/katago_ladder/confirmation_result.json') != protocol['katago']['calibration_result_sha256']:
            raise ValueError('KataGo calibration identity differs')
        tensor = np.full((2, 2, 4, 32, 2, 2), np.nan)
        previous_finish = 0.
        for item in protocol['checkpoints']:
            evidence, bounds = analyze_checkpoint(root, protocol, item, args.results / item['id'])
            if evidence['started_unix'] < previous_finish:
                raise ValueError('Registered checkpoint execution order differs')
            previous_finish = evidence['finished_unix']
            tensor[item['seed'] - 27, int(item['architecture'] == 'attention'), item['turn'] // 8192 - 1] = bounds
            report['checkpoints'].append(evidence)
        report['total_eval_seconds'] = sum(c['elapsed_seconds'] for c in report['checkpoints'])
        if report['total_eval_seconds'] > protocol['budget']['maximum_total_panel_seconds']:
            raise ValueError('Registered evaluation time budget exceeded')
        report['primary'] = curve_statistics(tensor, draws=protocol['primary']['bootstrap_draws'], seed=protocol['primary']['bootstrap_seed'])
        report['coverage_criterion'] = all(c['completion_coverage'] >= protocol['budget']['minimum_completed_fraction_per_checkpoint'] for c in report['checkpoints'])
        report['registered_criterion_met'] = report['coverage_criterion'] and report['primary']['positive_effect_criterion']
        report['limitations'] = protocol['limitations']
        report['status'] = 'analyzed'
        verify(SOURCE)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        args.output.chmod(0o444)
        print(json.dumps({k: v for k, v in report.items() if k != 'checkpoints'}), flush=True)


if __name__ == '__main__':
    main()
