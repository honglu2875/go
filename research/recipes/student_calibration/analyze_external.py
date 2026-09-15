#!/usr/bin/env python3
"""Audit registered board-input matches and paired-opening score bounds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'eval'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify
from analyze_prefetch_learning import audit_games
from analyze_training import REGISTRATION, audit_arm, require
from match import summarize

TRAINING_AUDIT = 'aeedc9cc6dd68d5e4a579d07d659bb72c30dac968fd34cd5a339bab6dc24e8fe'


def ending_diagnostics(games):
    """Post hoc description; none of these values changes the registered gate."""
    opportunities = accepted = refused = pessimistic_refusals = 0
    tails = []
    for game in games:
        for previous, move in zip(game['moves'], game['moves'][1:]):
            if (move['color'] != game['candidate_color'] or move['opening']
                    or previous['vertex'].lower() != 'pass'):
                continue
            opportunities += 1
            if move['vertex'].lower() == 'pass':
                accepted += 1
            else:
                refused += 1
                pessimistic_refusals += move['search']['root_value'] <= -.95
        if game['status'] == 'truncated':
            tail = game['moves'][-32:]
            ours = [m for m in tail if m['color'] == game['candidate_color']]
            theirs = [m for m in tail if m['color'] != game['candidate_color']]
            tails.append({'opening': game['opening'], 'candidate_color': game['candidate_color'],
                          'tail_plies': len(tail), 'candidate_passes': sum(m['vertex'].lower() == 'pass' for m in ours),
                          'opponent_passes': sum(m['vertex'].lower() == 'pass' for m in theirs),
                          'candidate_root_value_range': [min(m['search']['root_value'] for m in ours),
                                                         max(m['search']['root_value'] for m in ours)]})
    return {'scope': 'Post hoc endgame diagnostics, including censored games; not an alternative strength metric or causal explanation.',
            'candidate_turns_after_opponent_pass': opportunities, 'responded_with_pass': accepted,
            'continued_with_play': refused, 'continued_with_play_at_root_value_at_most_minus_0_95': pessimistic_refusals,
            'capped_game_last_32_plies': tails}


def paired_comparison(control, candidate):
    """Resample opening units, keeping colors, arms and cap bounds together."""
    expected = {(i, color) for i in range(32) for color in ('B', 'W')}
    arms = []
    completions = []
    for games in (control, candidate):
        require(len(games) == 64 and {(g['pair'], g['candidate_color']) for g in games} == expected,
                'Historical opening/color coverage differs')
        bounds = np.zeros((32, 2, 2), np.float64)
        for game in games:
            if game['status'] == 'completed' and game.get('candidate_points') in (0., .5, 1.):
                value = [game['candidate_points']] * 2
            elif game['status'] == 'truncated' and game.get('candidate_points') is None:
                value = [0., 1.]
            else:
                raise ValueError('Process failure or assigned incomplete result cannot enter score bounds')
            bounds[game['pair'], int(game['candidate_color'] == 'W')] = value
        arms.append(bounds.mean(axis=1))
        completions.append(sum(g['status'] == 'completed' for g in games))
    for a, b in zip(sorted(control, key=lambda g: (g['pair'], g['candidate_color'])),
                    sorted(candidate, key=lambda g: (g['pair'], g['candidate_color']))):
        require(a['opening'] == b['opening'], 'Paired arms used different openings')
    # Difference of two intervals: [candidate.low-control.high, candidate.high-control.low].
    difference = np.stack((arms[1][:, 0] - arms[0][:, 1], arms[1][:, 1] - arms[0][:, 0]), axis=-1)
    indices = np.random.default_rng(19450324).integers(0, 32, size=(10000, 32))
    draws = difference[indices].mean(axis=1)
    interval = [float(np.quantile(draws[:, 0], .025)), float(np.quantile(draws[:, 1], .975))]
    completion_met = all(n / 64 >= .95 for n in completions)
    return {'opening_units': 32, 'completed_games': dict(zip(('empty', 'exact'), completions)),
            'arm_scheduled_score_bounds': {name: a.mean(axis=0).tolist() for name, a in zip(('empty', 'exact'), arms)},
            'exact_minus_empty_score_bounds': difference.mean(axis=0).tolist(),
            'paired_bootstrap_95_outer_interval': interval, 'bootstrap_draws': 10000, 'bootstrap_seed': 19450324,
            'minimum_completion_criterion_met': completion_met, 'positive_lower_endpoint': interval[0] > 0,
            'registered_external_criterion_met': completion_met and interval[0] > 0,
            'uncertainty_scope': 'Descriptive percentile interval conditional on one trained pair and these 32 openings. Colors and cap endpoints are resampled together. It does not estimate independent-training-seed uncertainty.'}


def loaded_identity(folder, game, mode, source, spec, descriptor, trained, native):
    require(game['candidate_version'] == 'gozero-board-causal-mcts-history-v1', 'Unexpected actual engine version')
    ready = []
    for line in (folder / 'candidate/stderr.log').read_text().splitlines():
        if line.startswith('{'):
            item = json.loads(line)
            if item.get('kind') == 'engine_ready':
                ready.append(item)
    require(len(ready) == 1, 'Engine readiness absent or duplicated')
    ready = ready[0]
    inference_sha = sha256(source / spec['causal_inference'])
    inference = read_json(source / spec['causal_inference'])
    require(ready['adapter'] == 'board_causal_full_prefill' and ready['board_mode'] == mode
            and ready['candidate_sha256'] == sha256(source / spec['candidate'])
            and ready['training_snapshot'] == descriptor['training_snapshot']
            and ready['weights_sha256'] == descriptor['model_export_sha256']
            and ready['model_code_sha256'] == trained['model_code_sha256']
            and ready['native_snapshot'] == native['snapshot_id'] and ready['native_sha256'] == native['binary_sha256']
            and ready['inference_config_sha256'] == inference_sha and ready['backend'] == 'cpu',
            'Actual loaded candidate identity differs')
    require(ready['simulations_excluding_root'] == spec['candidate_simulations_excluding_root'] == inference['simulations']
            and ready['maximum_game_moves'] == spec['max_game_moves'] == inference['max_game_moves']
            and ready['cpuct'] == spec['candidate_cpuct'] == inference['cpuct']
            and ready['maximum_game_moves'] + ready['simulations_excluding_root'] < ready['context_tokens'],
            'Actual inference search/context contract differs')


def audit_suite(root, protocol, mode, directory, training):
    suite = read_json(directory / 'result.json')
    source = artifact(root, '.gozero/snapshots/' + suite['snapshot_id'])
    verify(source); verify_inputs(source, protocol['evaluation_input_closure'])
    for name, expected in protocol['evaluation_code_sha256'].items():
        require(sha256(source / name) == expected, 'Registered evaluator code differs: ' + name)
    arm = protocol['arms'][mode]; descriptor = read_json(source / arm['descriptor']); trained = validate(root, descriptor)
    require(suite['kind'] == 'board_state_katago_suite' and suite['status'] == 'passed' and 'error' not in suite
            and suite['arm'] == mode and suite['protocol_sha256'] == REGISTRATION
            and suite['candidate_sha256'] == training['descriptor_sha256'] == sha256(source / arm['descriptor'])
            and suite['model_export_sha256'] == training['model_export_sha256'] == descriptor['model_export_sha256']
            and descriptor['training_snapshot'] == arm['snapshot_id'] and trained['config']['model']['board_mode'] == mode,
            'Suite status, registration or candidate differs')
    require(protocol['registered_unix'] < suite['started_unix'] <= suite['finished_unix']
            and suite['finished_unix'] - suite['started_unix'] <= protocol['maximum_eval_seconds_per_arm'],
            'Evaluation preceded registration or exceeded its budget')
    require([p['id'] for p in suite['panels']] == [p['id'] for p in arm['evaluation_panels']], 'Panel coverage/order differs')
    groups = {}; reports = {}; files = {}; previous_end = suite['started_unix']
    for declaration, recorded in zip(arm['evaluation_panels'], suite['panels']):
        panel_dir = directory / declaration['id']; panel = read_json(panel_dir / 'result.json')
        spec = read_json(source / declaration['path'])
        require(recorded['returncode'] in (0, 1) and 'error' not in panel
                and recorded['result_sha256'] == sha256(panel_dir / 'result.json')
                and panel['snapshot_id'] == source.name
                and panel['spec_sha256'] == recorded['spec_sha256'] == declaration['sha256'] == sha256(source / declaration['path'])
                and previous_end <= panel['started_unix'] <= panel['finished_unix'] <= suite['finished_unix'],
                'Panel execution identity or sequential timing differs')
        previous_end = panel['finished_unix']
        require(len(panel['matches']) == len(spec['matches'])
                and {m['id'] for m in panel['matches']} == {m['id'] for m in spec['matches']}, 'Child coverage differs')
        panel_groups = {}
        for child_decl in spec['matches']:
            entry = next(m for m in panel['matches'] if m['id'] == child_decl['id'])
            child_dir = panel_dir / child_decl['id']; child = read_json(child_dir / 'result.json')
            c = read_json(source / child_decl['spec'])
            require(entry['returncode'] in (0, 1) and not entry['timed_out'] and 'error' not in child
                    and entry['group'] == child_decl['group'] and sha256(child_dir / 'result.json') == entry['result_sha256']
                    and child['snapshot_id'] == source.name and child['spec_sha256'] == sha256(source / child_decl['spec'])
                    and read_json(child_dir / 'resolved_spec.json') == c
                    and child['candidate'] == descriptor and child['candidate_sha256'] == suite['candidate_sha256']
                    and read_json(source / c['candidate']) == descriptor and entry['games'] == child['games']
                    and panel['started_unix'] <= child['started_unix'] <= child['finished_unix'] <= panel['finished_unix'],
                    'Child execution, model, timing or game records differ')
            require(child['candidate_adapter'] == 'eval/causal_gtp.py' and child['candidate_scoring_profile'] == 'pass_alive_area'
                    and child['candidate_inference_config_sha256'] == sha256(source / c['causal_inference']),
                    'Candidate adapter, scoring or search configuration differs')
            receipt = artifact(root, '.gozero/native/' + c['inference_native_snapshot'] + '/receipt.json')
            native = read_json(receipt)
            require(child['native_receipt_sha256'] == sha256(receipt)
                    and native['snapshot_id'] == c['inference_native_snapshot'] == child['inference_native_snapshot']
                    and native['binary_sha256'] == c['inference_native_binary_sha256'] == sha256(receipt.parent / native['filename']),
                    'Actual native search/rules identity differs')
            weights = read_json(source / c['katago_weights']); kata = read_json(source / 'eval/katago_build.json')
            require(child['katago_binary_sha256'] == kata['binary_sha256'] == sha256(artifact(root, kata['binary_path']))
                    and child['katago_weights_sha256'] == weights['sha256'] == sha256(artifact(root, weights['path']))
                    and child['katago_weights_descriptor_sha256'] == sha256(source / c['katago_weights'])
                    and child['base_katago_config_sha256'] == sha256(source / c['katago_config']),
                    'Actual external engine, weights or base configuration differs')
            audit_games(child['games'], child_dir, c, direct=False)
            all_complete = all(g['status'] == 'completed' for g in child['games'])
            expected_status = 'passed' if all_complete else 'failed'
            require(child['status'] == entry['status'] == expected_status and entry['returncode'] == (0 if all_complete else 1)
                    and child['summary'] == entry['summary'] == summarize(child['games']), 'Child summary/status differs')
            for game in child['games']:
                folder = child_dir / f'pair-{game["pair"]:03d}-{game["candidate_color"]}'
                loaded_identity(folder, game, mode, source, c, descriptor, trained, native)
            group = panel_groups.setdefault(child_decl['group'], [])
            offset = len(group) // 2
            group.extend({**g, 'pair': offset + g['pair']} for g in child['games'])
        summaries = {name: summarize(games) for name, games in panel_groups.items()}
        require(summaries == panel['summaries'] == recorded['summaries']
                and sum(len(g) for g in panel_groups.values()) == declaration['games'], 'Panel summaries or game totals differ')
        expected_status = 'passed' if all(m['status'] == 'passed' for m in panel['matches']) else 'failed'
        require(panel['status'] == expected_status and recorded['returncode'] == (0 if expected_status == 'passed' else 1),
                'Panel status inconsistent with its children')
        require(not (groups.keys() & panel_groups.keys()), 'Group repeated across panels')
        groups.update(panel_groups)
        reports[declaration['id']] = {'result_sha256': sha256(panel_dir / 'result.json'), 'summaries': summaries,
            'elapsed_seconds': panel['finished_unix'] - panel['started_unix']}
    for path in directory.rglob('*'):
        if path.is_file() and path.name != '.operator.lock':
            files[str(path.relative_to(directory))] = sha256(path)
    games = [g for group in groups.values() for g in group]
    verify(source)
    return {'directory': str(directory.relative_to(root)), 'execution_snapshot': source.name,
            'suite_result_sha256': sha256(directory / 'result.json'), 'panels': reports,
            'scheduled_games': len(games), 'completed_games': sum(g['status'] == 'completed' for g in games),
            'capped_games': sum(g['status'] == 'truncated' for g in games), 'process_failures': 0,
            'checked_boards': sum(g['checked_positions'] for g in games),
            'checked_scores': sum(g['status'] == 'completed' for g in games),
            'search_totals': {k: sum(g['candidate_search_totals'][k] for g in games)
                              for k in ('simulations', 'neural_evaluations', 'terminal_evaluations')},
            'elapsed_seconds': suite['finished_unix'] - suite['started_unix'], 'raw_files_sha256': dict(sorted(files.items()))}, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--empty-suite', required=True); p.add_argument('--exact-suite', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(SOURCE); root = args.workspace_root.resolve()
    path = artifact(root, 'research/studies/board_state_distillation/pilot_spec.json')
    require(sha256(path) == REGISTRATION, 'Registration changed'); protocol = read_json(path)
    train_path = artifact(root, 'research/studies/board_state_distillation/training_result.json')
    require(sha256(train_path) == TRAINING_AUDIT, 'Training audit changed'); training = read_json(train_path)
    verify(artifact(root, '.gozero/snapshots/' + training['analysis_snapshot']))
    output = {'schema_version': 1, 'kind': 'board_state_distillation_pilot_analysis', 'status': 'failed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': REGISTRATION, 'training_audit_sha256': TRAINING_AUDIT,
              'arms': {}, 'production_promotion': False, 'claims_rl_sample_efficiency': False, 'claims_mfu': False,
              'claims_current_katago_superiority': False, 'limitations': protocol['limitations']}
    try:
        games = {}; attempts = []
        for mode, suite_name in (('empty', args.empty_suite), ('exact', args.exact_suite)):
            row, _ = audit_arm(root, protocol, mode)
            require(row == training['arms'][mode], 'Reaudited training evidence differs')
            launches = [p for p in (root / 'runs').glob('pod-*/launch.json')
                        if read_json(p)['snapshot_id'] == protocol['arms'][mode]['snapshot_id']]
            require(len(launches) == protocol['maximum_training_attempts_per_arm'] == 1
                    and launches[0].parent.name == row['attempt'], 'Training attempt budget or selection differs')
            attempts.append(read_json(launches[0].parent / 'result.json'))
            directory = artifact(root, suite_name)
            suites = []
            for candidate in (root / 'runs/eval').glob('*/result.json'):
                item = read_json(candidate)
                if item.get('kind') == 'board_state_katago_suite' and item.get('protocol_sha256') == REGISTRATION and item.get('arm') == mode:
                    suites.append(candidate)
            require(len(suites) == protocol['maximum_eval_attempts_per_arm'] == 1 and suites[0] == directory / 'result.json',
                    'Evaluation attempt budget or selection differs')
            output['arms'][mode], games[mode] = audit_suite(root, protocol, mode, directory, row)
            output['arms'][mode]['post_hoc_endgame_diagnostics'] = {
                group: ending_diagnostics(rows) for group, rows in games[mode].items()}
        require(protocol['training_order'] == ['empty', 'exact'] and attempts[0]['end_unix_time'] <= attempts[1]['start_unix_time'],
                'Registered sequential training order differs')
        output['external_comparison'] = paired_comparison(games['empty']['early'], games['exact']['early'])
        output['registered_prediction_criterion_met'] = training['registered_prediction_criterion_met']
        output['registered_combined_criterion_met'] = (training['registered_prediction_criterion_met']
            and output['external_comparison']['registered_external_criterion_met'])
        output['next_independent_seed_justified'] = output['registered_combined_criterion_met']
        output['recorded_pilot_attempt_chip_hours'] = training['recorded_pilot_attempt_chip_hours']
        output['status'] = 'passed'
    except Exception as e:
        output['error'] = repr(e)
        raise
    finally:
        verify(SOURCE); require(sha256(path) == REGISTRATION, 'Registration changed during analysis')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as f:
            f.write(canonical_json(output))
        args.output.chmod(0o444); print(sha256(args.output))


if __name__ == '__main__':
    main()
