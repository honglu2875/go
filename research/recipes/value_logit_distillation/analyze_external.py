#!/usr/bin/env python3
"""Audit registered value-loss training and fresh paired real-KataGo games."""
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
from analyze_training import audit_arm, require
from match import summarize
from match_audit import loaded_identity, replay_game, diagnostics


def paired_comparison(control, candidate, openings, c):
    n = c['opening_units']; expected = {(i, color) for i in range(n) for color in ('B', 'W')}
    require(len(openings) == n, 'Opening book size differs')
    arms = {}; completed = {}
    for name, games in (('mse', control), ('bce2', candidate)):
        require(len(games) == 2 * n and {(g['pair'], g['candidate_color']) for g in games} == expected, 'Paired opening/color coverage differs')
        bounds = np.zeros((n, 2, 2), np.float64)
        for game in games:
            require(game['opening'] == openings[game['pair']], 'Registered paired opening differs')
            if game['status'] == 'completed' and game.get('candidate_points') in (0., .5, 1.):
                value = [game['candidate_points']] * 2
            elif game['status'] == 'truncated' and game.get('candidate_points') is None:
                value = [0., 1.]
            else:
                raise ValueError('Assigned cap or process failure cannot enter score bounds')
            bounds[game['pair'], int(game['candidate_color'] == 'W')] = value
        arms[name] = bounds.mean(axis=1)
        completed[name] = sum(g['status'] == 'completed' for g in games)
    difference = np.stack((arms['bce2'][:, 0] - arms['mse'][:, 1], arms['bce2'][:, 1] - arms['mse'][:, 0]), axis=-1)
    indices = np.random.default_rng(c['bootstrap_seed']).integers(0, n, size=(c['bootstrap_replicates'], n))
    draws = difference[indices].mean(axis=1); alpha = (1. - c['confidence']) / 2
    interval = [float(np.quantile(draws[:, 0], alpha)), float(np.quantile(draws[:, 1], 1. - alpha))]
    completion = all(count / (2 * n) >= c['minimum_each_arm_completion_fraction'] for count in completed.values())
    positive = interval[0] > c['minimum_bce2_minus_mse_lower_endpoint_strictly_above']
    return {'opening_units': n, 'completed_games': completed,
            'arm_scheduled_score_bounds': {name: b.mean(axis=0).tolist() for name, b in arms.items()},
            'bce2_minus_mse_score_bounds': difference.mean(axis=0).tolist(), 'paired_bootstrap_95_outer_interval': interval,
            'bootstrap_draws': c['bootstrap_replicates'], 'bootstrap_seed': c['bootstrap_seed'],
            'minimum_completion_criterion_met': completion, 'positive_lower_endpoint': positive,
            'registered_external_criterion_met': completion and positive,
            'per_opening': [{'pair': i, 'opening': opening, 'mse_score_bounds': arms['mse'][i].tolist(),
                             'bce2_score_bounds': arms['bce2'][i].tolist(), 'difference_bounds': difference[i].tolist()}
                            for i, opening in enumerate(openings)], 'uncertainty_scope': c['scope']}


def audit_suite(root, protocol, mode, directory, training, protocol_sha):
    suite = read_json(directory / 'result.json')
    source = artifact(root, '.gozero/snapshots/' + suite['snapshot_id'])
    verify(source); verify_inputs(source, protocol['evaluation_input_closure'])
    for name, expected in protocol['evaluation_code_sha256'].items():
        require(sha256(source / name) == expected, 'Registered evaluator code differs: ' + name)
    arm = protocol['arms'][mode]; descriptor = read_json(source / arm['descriptor']); trained = validate(root, descriptor)
    require(suite['kind'] == 'value_logit_katago_suite' and suite['status'] == 'passed' and 'error' not in suite
            and suite['arm'] == mode and suite['protocol_sha256'] == protocol_sha
            and suite['candidate_sha256'] == training['descriptor_sha256'] == sha256(source / arm['descriptor'])
            and suite['model_export_sha256'] == training['model_export_sha256'] == descriptor['model_export_sha256']
            and descriptor['training_snapshot'] == arm['snapshot_id'] and trained['config']['model']['board_mode'] == 'exact' and trained['config']['model']['value_objective'] == mode,
            'Suite status, registration or candidate differs')
    require(protocol['registered_unix'] < suite['started_unix'] <= suite['finished_unix']
            and suite['finished_unix'] - suite['started_unix'] <= protocol['maximum_eval_seconds_per_arm'],
            'Evaluation preceded registration or exceeded its budget')
    require([p['id'] for p in suite['panels']] == [p['id'] for p in arm['evaluation_panels']], 'Panel coverage/order differs')
    attempts = [p for p in (root / 'runs/eval').glob('*/result.json')
                if read_json(p).get('kind') == 'value_logit_katago_suite' and read_json(p).get('arm') == mode
                and read_json(p).get('protocol_sha256') == protocol_sha]
    require(attempts == [directory / 'result.json'] and protocol['maximum_eval_attempts_per_arm'] == 1, 'Evaluation attempt count differs')
    groups = {}; reports = {}; files = {}; previous_end = suite['started_unix']
    for declaration, recorded in zip(arm['evaluation_panels'], suite['panels']):
        panel_dir = directory / declaration['id']; panel = read_json(panel_dir / 'result.json')
        spec = read_json(source / declaration['path'])
        require(recorded['returncode'] in (0, 1) and 'error' not in panel
                and recorded['result_sha256'] == sha256(panel_dir / 'result.json')
                and panel['snapshot_id'] == source.name
                and panel['spec_sha256'] == recorded['spec_sha256'] == declaration['sha256'] == sha256(source / declaration['path'])
                and previous_end <= panel['started_unix'] <= panel['finished_unix'] <= suite['finished_unix']
                and panel['finished_unix'] - panel['started_unix'] <= declaration['maximum_seconds'],
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
                loaded_identity(folder, game, source, c, descriptor, trained, native)
                native_module = load_library(receipt.parent / native['filename'], native['binary_sha256'])
                seed = replay_game(folder, game, native_module, c)
                expected_config = re.sub(r'^(searchRandSeed|nnRandSeed)\s*=.*$', lambda m: m[1] + ' = ' + seed,
                                         (source / c['katago_config']).read_text(), flags=re.M)
                require((folder / 'katago.cfg').read_text() == expected_config, 'Actual external seed/config differs')
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
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--expected-training-audit-sha256', required=True)
    p.add_argument('--mse-suite', required=True); p.add_argument('--bce2-suite', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    require(not a.output.exists() and sha256(a.protocol) == a.expected_protocol_sha256, 'Audit output exists or protocol changed')
    protocol = read_json(a.protocol)
    require(protocol['kind'] == 'value_logit_distillation_pilot' and sha256(a.training_audit) == a.expected_training_audit_sha256,
            'Study kind or training audit differs')
    training = read_json(a.training_audit)
    require(training['status'] == 'passed' and training['protocol_sha256'] == a.expected_protocol_sha256, 'Training audit did not pass this protocol')
    report = {'schema_version': 1, 'kind': 'value_logit_distillation_pilot_analysis', 'status': 'failed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': a.expected_protocol_sha256,
              'training_audit_sha256': a.expected_training_audit_sha256, 'production_promotion': False,
              'claims_rl_sample_efficiency': False, 'claims_current_katago_superiority': False,
              'claims_mfu': False, 'limitations': protocol['limitations']}
    try:
        groups = {}; suites = {}; diagnostic = {}
        for mode, name in (('mse', a.mse_suite), ('bce2', a.bce2_suite)):
            evidence, _ = audit_arm(root, protocol, mode)
            require(evidence == training['arms'][mode], 'Reaudited training evidence differs')
            suites[mode], games = audit_suite(root, protocol, mode, artifact(root, name), evidence, a.expected_protocol_sha256)
            require(set(games) == {mode, mode + '-v1', mode + '-v16'} and suites[mode]['scheduled_games'] == 72,
                    'Historical/strong panel coverage differs')
            groups.update(games); diagnostic.update({k: diagnostics(v) for k, v in games.items()})
        book_path = SOURCE / 'eval/value_logit_distillation/openings.json'
        require(sha256(book_path) == protocol['openings_sha256'], 'Opening book changed')
        book = read_json(book_path)
        for mode in ('mse', 'bce2'):
            for visits in (1, 16):
                games = groups[f'{mode}-v{visits}']
                require(len(games) == 4 and {(g['pair'], g['candidate_color']) for g in games} == {(i, color) for i in range(2) for color in ('B', 'W')}
                        and all(g['opening'] == book['strong'][g['pair']] for g in games), 'Strong paired openings differ')
        primary = paired_comparison(groups['mse'], groups['bce2'], book['early'], protocol['external_comparison'])
        report.update(status='passed', arms=suites, groups=diagnostic, primary=primary,
                      registered_prediction_criterion_met=training['registered_prediction_criterion_met'],
                      registered_external_criterion_met=primary['registered_external_criterion_met'],
                      registered_combined_criterion_met=training['registered_prediction_criterion_met'] and primary['registered_external_criterion_met'],
                      scheduled_games=sum(s['scheduled_games'] for s in suites.values()),
                      completed_games=sum(s['completed_games'] for s in suites.values()),
                      capped_games=sum(s['capped_games'] for s in suites.values()),
                      process_failures=sum(s['process_failures'] for s in suites.values()),
                      checked_boards=sum(s['checked_boards'] for s in suites.values()),
                      checked_scores=sum(s['checked_scores'] for s in suites.values()),
                      recorded_training_attempt_chip_hours=training['recorded_pilot_attempt_chip_hours'])
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('xb') as stream:
            stream.write(canonical_json(report))
        a.output.chmod(0o444); verify(SOURCE)
        require(sha256(a.protocol) == a.expected_protocol_sha256 and sha256(a.training_audit) == a.expected_training_audit_sha256,
                'Protocol or training audit changed during analysis')
        print(json.dumps({k: v for k, v in report.items() if k not in ('arms', 'groups', 'primary')}))


if __name__ == '__main__':
    main()
