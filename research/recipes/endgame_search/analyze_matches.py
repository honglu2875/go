#!/usr/bin/env python3
"""Audit the fixed-model rescaling ablation, preserving every scheduled cap."""
import argparse
import json
from pathlib import Path
import re
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
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from analyze_prefetch_learning import audit_games
from corpus import played_boards, require, responses
from learned_gtp import action
from match import summarize
from qualify_katago import kata_cells


def paired_comparison(on, off, openings, criterion):
    """Bootstrap opening units with colors, arms and cap endpoints kept together."""
    count = len(openings)
    expected = {(i, c) for i in range(count) for c in ('B', 'W')}
    bounds = {}
    completed = {}
    for name, games in (('on', on), ('off', off)):
        require(len(games) == 2 * count and {(g['pair'], g['candidate_color']) for g in games} == expected,
                'Paired opening/color coverage differs')
        rows = np.zeros((count, 2, 2), np.float64)
        for game in games:
            require(game['opening'] == openings[game['pair']], 'Registered opening differs')
            if game['status'] == 'completed' and game.get('candidate_points') in (0., .5, 1.):
                value = [game['candidate_points']] * 2
            elif game['status'] == 'truncated' and game.get('candidate_points') is None:
                value = [0., 1.]
            else:
                raise ValueError('Failed or assigned incomplete result cannot enter score bounds')
            rows[game['pair'], int(game['candidate_color'] == 'W')] = value
        bounds[name] = rows.mean(axis=1)
        completed[name] = sum(g['status'] == 'completed' for g in games)
    difference = np.stack((bounds['off'][:, 0] - bounds['on'][:, 1],
                           bounds['off'][:, 1] - bounds['on'][:, 0]), axis=-1)
    rng = np.random.default_rng(criterion['bootstrap_seed'])
    indices = rng.integers(0, count, size=(criterion['paired_bootstrap_replicates'], count))
    draws = difference[indices].mean(axis=1)
    alpha = (1. - criterion['confidence']) / 2
    interval = [float(np.quantile(draws[:, 0], alpha)), float(np.quantile(draws[:, 1], 1. - alpha))]
    completion_met = completed['off'] / (2 * count) >= criterion['minimum_off_completion_fraction']
    positive = interval[0] > criterion['minimum_lower_outer_interval_strictly_above']
    return {'opening_units': count, 'completed_games': completed,
            'arm_scheduled_score_bounds': {k: v.mean(axis=0).tolist() for k, v in bounds.items()},
            'off_minus_on_score_bounds': difference.mean(axis=0).tolist(),
            'paired_bootstrap_95_outer_interval': interval,
            'bootstrap_seed': criterion['bootstrap_seed'], 'bootstrap_draws': criterion['paired_bootstrap_replicates'],
            'off_completion_fraction': completed['off'] / (2 * count),
            'completion_difference_off_minus_on': (completed['off'] - completed['on']) / (2 * count),
            'minimum_off_completion_criterion_met': completion_met, 'positive_lower_endpoint': positive,
            'registered_criterion_met': completion_met and positive,
            'per_opening': [{'pair': i, 'opening': opening, 'on_score_bounds': bounds['on'][i].tolist(),
                             'off_score_bounds': bounds['off'][i].tolist(),
                             'off_minus_on_score_bounds': difference[i].tolist()} for i, opening in enumerate(openings)],
            'uncertainty_scope': 'Descriptive percentile outer interval conditional on one fixed trained model and these openings. Colors, arms and cap endpoints are resampled together. This is not independent-training-seed or Elo uncertainty.'}


def loaded_identity(folder, game, source, spec, descriptor, trained, native):
    ready = [json.loads(line) for line in (folder / 'candidate/stderr.log').read_text().splitlines()
             if line.startswith('{') and json.loads(line).get('kind') == 'engine_ready']
    require(len(ready) == 1, 'Engine readiness absent or duplicated')
    ready = ready[0]
    inference = read_json(source / spec['causal_inference'])
    require(game['candidate_version'] == 'gozero-board-causal-mcts-history-v1'
            and ready['adapter'] == 'board_causal_full_prefill' and ready['board_mode'] == 'exact'
            and ready['candidate_sha256'] == sha256(source / spec['candidate'])
            and ready['training_snapshot'] == descriptor['training_snapshot']
            and ready['weights_sha256'] == descriptor['model_export_sha256']
            and ready['model_code_sha256'] == trained['model_code_sha256']
            and ready['native_snapshot'] == native['snapshot_id'] and ready['native_sha256'] == native['binary_sha256']
            and ready['inference_config_sha256'] == sha256(source / spec['causal_inference'])
            and ready['backend'] == 'cpu', 'Actual loaded candidate identity differs')
    require(ready['simulations_excluding_root'] == spec['candidate_simulations_excluding_root'] == inference['simulations']
            and ready['maximum_game_moves'] == spec['max_game_moves'] == inference['max_game_moves']
            and ready['cpuct'] == spec['candidate_cpuct'] == inference['cpuct']
            and ready['maximum_game_moves'] + ready['simulations_excluding_root'] < ready['context_tokens'],
            'Actual inference budget or context contract differs')


def audit_transcript(path, game, side):
    pending = {}; seen = set(); moves = []; stats = []; starts = []; exits = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'start':
            starts.append(row['argv'])
        elif row['kind'] == 'exit':
            exits.append(row['returncode'])
        elif row['kind'] == 'transport_error':
            raise ValueError('Retained GTP transport failure')
        elif row['kind'] == 'command':
            require(row['id'] not in seen, 'Duplicated command ID'); seen.add(row['id'])
            pending[row['id']] = row['text']
        elif row['kind'] == 'response':
            require(row['id'] in pending and row['success'], 'Unmatched or failed GTP response')
            command = pending.pop(row['id']).split()
            if command[0] in ('play', 'genmove'):
                vertex = command[2] if command[0] == 'play' else row['text'].strip()
                moves.append((command[1].upper(), action(vertex, 9)))
            elif command[0] == 'gozero-search-stats':
                stats.append(json.loads(row['text']))
    require(not pending and exits == [0] and starts == [game[side + '_argv']],
            'Incomplete transcript, engine exit or actual process argv differs')
    require(moves == [(m['color'], action(m['vertex'], 9)) for m in game['moves']], 'Transcript moves differ')
    require(responses(path, 'version') == [game[side + '_version']], 'Transcript engine version differs')
    if side == 'candidate':
        require(stats == [m['search'] for m in game['moves'] if 'search' in m], 'Transcript search evidence differs')
    else:
        require([json.loads(r) for r in responses(path, 'kata-get-rules')] == [game['katago_rules']], 'Transcript rules differ')
    require(responses(path, 'final_score') == ([game['score']] if game['status'] == 'completed' else []),
            'Transcript final score or cap adjudication differs')


def replay_game(folder, game, native, spec):
    boards = played_boards(folder / 'candidate/gtp.jsonl', len(game['moves']), lambda b: b)
    other = played_boards(folder / 'katago/gtp.jsonl', len(game['moves']), lambda b: kata_cells(b, 9))
    require(boards == other, 'Candidate/KataGo board transcripts differ')
    for side in ('candidate', 'katago'):
        audit_transcript(folder / side / 'gtp.jsonl', game, side)
    position = native.Game(json.dumps({'size': 9, 'komi': 7.5, 'history': 4, 'scoring': 'pass_alive_area',
                                       'simulations': 0, 'cpuct': 0., 'max_search_edges': 1000000}))
    actions = []
    for ply, (board, move) in enumerate(zip(boards, game['moves'])):
        chosen = action(move['vertex'], 9); actions.append(chosen)
        require(chosen in position.legal(), 'Recorded move illegal in native replay')
        position.play(1 + ply % 2, chosen)
        expected = np.asarray(['.XO'.index(x) for x in board.replace('\n', '')], np.uint8)
        require(np.array_equal(position.state()[4], expected), 'Native replay differs from actual KataGo board')
        if 'search' in move:
            stats = move['search']
            require(all(type(stats[k]) is int and stats[k] >= 0 for k in ('simulations', 'neural_evaluations', 'terminal_evaluations'))
                    and stats['neural_evaluations'] + stats['terminal_evaluations'] == stats['simulations'] + 1,
                    'Neural/terminal work does not cover root plus simulations')
    _, _, outcomes = native.replay_observations(json.dumps({'size': 9, 'komi': 7.5, 'scoring': 'pass_alive_area'}),
                                               np.asarray(actions, np.int32), np.asarray([0, len(actions)], np.int64))
    outcome = json.loads(outcomes)[0]
    require(outcome['terminal'] == (game['status'] == 'completed'), 'Native terminal flag differs')
    if outcome['terminal']:
        score = game['score']; value = 0. if score == '0' else float(score[2:]) * (1 if score[0] == 'W' else -1)
        require(outcome['white_score'] == value, 'Native final score differs from real KataGo')
    seed = spec['seed_prefix'] + '-' + str(game['pair'])
    return seed


def diagnostics(games):
    opportunities = passes = pessimistic_refusals = 0
    search = []
    for game in games:
        for previous, move in zip(game['moves'], game['moves'][1:]):
            if move['opening'] or move['color'] != game['candidate_color']:
                continue
            search.append(move['search'])
            if previous['vertex'].lower() == 'pass':
                opportunities += 1; passes += move['vertex'].lower() == 'pass'
                pessimistic_refusals += move['vertex'].lower() != 'pass' and move['search']['root_value'] <= -.95
    return {'summary': summarize(games), 'candidate_turns_after_opponent_pass': opportunities,
            'responded_with_pass': passes, 'continued_with_play': opportunities - passes,
            'continued_with_play_at_root_value_at_most_minus_0_95': pessimistic_refusals,
            'search_totals': {k: sum(g['candidate_search_totals'][k] for g in games)
                              for k in ('simulations', 'neural_evaluations', 'terminal_evaluations')},
            'search_quantiles_0_25_50_75_100': {k: np.quantile([s[k] for s in search], [0., .25, .5, .75, 1.]).tolist()
                                             for k in ('neural_evaluations', 'terminal_evaluations', 'root_value')},
            'scope': 'Secondary endgame and search descriptions. More passes may concede losses; these counts do not change the strength criterion.'}


def audit_suite(root, protocol, protocol_sha):
    directory = artifact(root, protocol['output']); suite = read_json(directory / 'result.json')
    source = artifact(root, '.gozero/snapshots/' + protocol['snapshot']); verify(source)
    verify_inputs(source, protocol['evaluation_input_closure'])
    for name, expected in protocol['evaluation_code_sha256'].items():
        require(sha256(source / name) == expected, 'Registered evaluator code differs: ' + name)
    require(suite['kind'] == 'endgame_katago_ablation_attempt' and suite['status'] == 'passed' and 'error' not in suite
            and suite['snapshot'] == source.name and suite['protocol_sha256'] == protocol_sha,
            'Suite execution identity or completion differs')
    require(protocol['registered_unix'] < suite['started_unix'] <= suite['finished_unix']
            and suite['finished_unix'] - suite['started_unix'] <= protocol['maximum_seconds'],
            'Registration ordering or execution budget differs')
    attempts = [p for p in (root / 'runs/eval').glob('*/result.json')
                if read_json(p).get('kind') == 'endgame_katago_ablation_attempt'
                and read_json(p).get('protocol_sha256') == protocol_sha]
    require(attempts == [directory / 'result.json'] and protocol['maximum_attempts'] == 1, 'Attempt count differs')
    descriptor = read_json(source / protocol['candidate']); trained = validate(root, descriptor)
    require(sha256(source / protocol['candidate']) == protocol['candidate_sha256']
            and trained['config']['model']['board_mode'] == 'exact', 'Fixed candidate differs')
    openings_path = source / 'eval/endgame_search/openings.json'
    require(sha256(openings_path) == protocol['openings_sha256'], 'Opening book changed')
    openings = read_json(openings_path)
    searches = [read_json(source / ('eval/endgame_search/search_' + arm + '.json')) for arm in ('on', 'off')]
    require(searches[0]['gumbel'].pop('rescale_values') is True and searches[1]['gumbel'].pop('rescale_values') is False
            and searches[0] == searches[1], 'More than value rescaling differs between inference arms')
    require([r['id'] for r in suite['panels']] == [r['id'] for r in protocol['panels']], 'Panel coverage/order differs')
    groups = {}; reports = {}; previous_end = suite['started_unix']; contracts = {}
    for declaration, recorded in zip(protocol['panels'], suite['panels']):
        panel_dir = directory / declaration['id']; panel = read_json(panel_dir / 'result.json')
        spec = read_json(source / declaration['path'])
        require(recorded['result_sha256'] == sha256(panel_dir / 'result.json') and 'error' not in panel
                and panel['snapshot_id'] == source.name and panel['spec_sha256'] == declaration['sha256'] == sha256(source / declaration['path'])
                and previous_end <= panel['started_unix'] <= panel['finished_unix'] <= suite['finished_unix']
                and panel['finished_unix'] - panel['started_unix'] <= declaration['maximum_seconds'],
                'Panel identity, sequential order or budget differs')
        previous_end = panel['finished_unix']
        require(len(panel['matches']) == len(spec['matches'])
                and {m['id'] for m in panel['matches']} == {m['id'] for m in spec['matches']}, 'Child coverage differs')
        panel_groups = {}
        for child_decl in sorted(spec['matches'], key=lambda r: r['id']):
            entry = next(m for m in panel['matches'] if m['id'] == child_decl['id'])
            child_dir = panel_dir / child_decl['id']; child = read_json(child_dir / 'result.json')
            c = read_json(source / child_decl['spec'])
            require(entry['returncode'] in (0, 1) and not entry['timed_out'] and 'error' not in child
                    and entry['group'] == child_decl['group'] and sha256(child_dir / 'result.json') == entry['result_sha256']
                    and child['snapshot_id'] == source.name and child['spec_sha256'] == sha256(source / child_decl['spec'])
                    and read_json(child_dir / 'resolved_spec.json') == c and child['candidate'] == descriptor
                    and child['candidate_sha256'] == protocol['candidate_sha256'] and read_json(source / c['candidate']) == descriptor
                    and entry['games'] == child['games']
                    and panel['started_unix'] <= child['started_unix'] <= child['finished_unix'] <= panel['finished_unix'],
                    'Child identity, timing or game records differ')
            arm = child_decl['group'].split('-')[0]
            require(arm in ('on', 'off') and c['causal_inference'] == 'eval/endgame_search/search_' + arm + '.json'
                    and child['candidate_adapter'] == 'eval/causal_gtp.py' and child['candidate_scoring_profile'] == 'pass_alive_area'
                    and child['candidate_inference_config_sha256'] == sha256(source / c['causal_inference']),
                    'Candidate adapter, scoring or search arm differs')
            contract = {k: v for k, v in c.items() if k not in ('candidate_cpus', 'katago_cpus', 'causal_inference')}
            pair_key = child_decl['id'].replace('-on-', '-arm-').replace('-off-', '-arm-')
            contracts.setdefault(pair_key, []).append(contract)
            receipt = artifact(root, '.gozero/native/' + c['inference_native_snapshot'] + '/receipt.json'); native = read_json(receipt)
            require(child['native_receipt_sha256'] == sha256(receipt)
                    and native['snapshot_id'] == c['inference_native_snapshot'] == child['inference_native_snapshot']
                    and native['binary_sha256'] == c['inference_native_binary_sha256'] == sha256(receipt.parent / native['filename']),
                    'Actual native search/rules identity differs')
            native_module = load_library(receipt.parent / native['filename'], native['binary_sha256'])
            weights = read_json(source / c['katago_weights']); kata = read_json(source / 'eval/katago_build.json')
            require(child['katago_binary_sha256'] == kata['binary_sha256'] == sha256(artifact(root, kata['binary_path']))
                    and child['katago_weights_sha256'] == weights['sha256'] == sha256(artifact(root, weights['path']))
                    and child['katago_weights_descriptor_sha256'] == sha256(source / c['katago_weights'])
                    and child['base_katago_config_sha256'] == sha256(source / c['katago_config']),
                    'Actual external engine, weights or configuration differs')
            audit_games(child['games'], child_dir, c, direct=False)
            complete = all(g['status'] == 'completed' for g in child['games'])
            require(child['status'] == entry['status'] == ('passed' if complete else 'failed')
                    and entry['returncode'] == (0 if complete else 1)
                    and child['summary'] == entry['summary'] == summarize(child['games']), 'Child status or summary differs')
            for game in child['games']:
                folder = child_dir / f'pair-{game["pair"]:03d}-{game["candidate_color"]}'
                loaded_identity(folder, game, source, c, descriptor, trained, native)
                seed = replay_game(folder, game, native_module, c)
                expected_config = re.sub(r'^(searchRandSeed|nnRandSeed)\s*=.*$', lambda m: m[1] + ' = ' + seed,
                                         (source / c['katago_config']).read_text(), flags=re.M)
                require((folder / 'katago.cfg').read_text() == expected_config, 'Per-game KataGo seed/config differs')
            group = panel_groups.setdefault(child_decl['group'], []); offset = len(group) // 2
            group.extend({**g, 'pair': offset + g['pair']} for g in child['games'])
        summaries = {name: summarize(games) for name, games in panel_groups.items()}
        require(summaries == panel['summaries'] == recorded['summaries']
                and sum(len(g) for g in panel_groups.values()) == declaration['games'], 'Panel summaries or totals differ')
        complete = all(m['status'] == 'passed' for m in panel['matches'])
        require(panel['status'] == ('passed' if complete else 'failed') and recorded['returncode'] == (0 if complete else 1),
                'Panel status differs')
        require(not (groups.keys() & panel_groups.keys()), 'Group repeated across panels')
        groups.update(panel_groups)
        reports[declaration['id']] = {'result_sha256': sha256(panel_dir / 'result.json'), 'summaries': summaries,
                                     'elapsed_seconds': panel['finished_unix'] - panel['started_unix']}
    require(all(len(v) == 2 and v[0] == v[1] for v in contracts.values()), 'Paired arms differ beyond rescaling/CPU placement')
    require(set(groups) == {'on', 'off', 'on-v1', 'off-v1', 'on-v16', 'off-v16'}
            and len(openings['early']) == 16 and len(openings['strong']) == 2, 'Registered panel coverage differs')
    for name, games in groups.items():
        book = openings['strong' if '-v' in name else 'early']
        require(len(games) == 2 * len(book) and all(g['opening'] == book[g['pair']] for g in games), 'Opening book differs')
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != '.operator.lock'}
    games = [g for values in groups.values() for g in values]; verify(source)
    return {'directory': protocol['output'], 'execution_snapshot': source.name,
            'suite_result_sha256': sha256(directory / 'result.json'), 'panels': reports,
            'scheduled_games': len(games), 'completed_games': sum(g['status'] == 'completed' for g in games),
            'capped_games': sum(g['status'] == 'truncated' for g in games), 'process_failures': 0,
            'checked_boards': sum(g['checked_positions'] for g in games),
            'checked_scores': sum(g['status'] == 'completed' for g in games),
            'elapsed_seconds': suite['finished_unix'] - suite['started_unix'], 'raw_files_sha256': files}, groups, openings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(SOURCE); root = args.workspace_root.resolve()
    require(not args.output.exists(), 'Audit output already exists')
    require(sha256(args.protocol) == args.expected_protocol_sha256, 'Registration changed')
    protocol = read_json(args.protocol)
    require(protocol['kind'] == 'endgame_rescaling_katago_ablation', 'Wrong registration kind')
    diagnostic_path = artifact(root, 'research/studies/endgame_search/result.json')
    require(sha256(diagnostic_path) == protocol['diagnostic_audit_sha256'] and read_json(diagnostic_path)['status'] == 'passed',
            'Preceding diagnostic audit differs')
    report = {'schema_version': 1, 'kind': 'endgame_rescaling_katago_audit', 'status': 'failed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': args.expected_protocol_sha256,
              'diagnostic_audit_sha256': protocol['diagnostic_audit_sha256'],
              'production_promotion': False, 'claims_rl_sample_efficiency': False,
              'claims_mfu': False, 'claims_current_katago_superiority': False, 'limitations': protocol['limitations']}
    try:
        suite, groups, openings = audit_suite(root, protocol, args.expected_protocol_sha256)
        criterion = protocol['primary_criterion']
        require(criterion['panel'] == 'early' and criterion['confidence'] == .95, 'Unexpected primary panel or confidence')
        primary = paired_comparison(groups['on'], groups['off'], openings['early'], criterion)
        report.update(status='passed', suite=suite, primary=primary,
                      groups={k: diagnostics(v) for k, v in groups.items()},
                      registered_criterion_met=primary['registered_criterion_met'])
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        verify(SOURCE); require(sha256(args.protocol) == args.expected_protocol_sha256, 'Protocol changed during audit')
        print(json.dumps({k: v for k, v in report.items() if k not in ('suite', 'groups', 'primary')}))


if __name__ == '__main__':
    main()
