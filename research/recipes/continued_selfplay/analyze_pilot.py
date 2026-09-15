#!/usr/bin/env python3
"""Audit fixed-final online continuation games and the registered paired screen."""
import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(SOURCE / 'packages/gozero/src'), str(SOURCE / 'eval')]
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.model_artifacts import artifact, validate_candidate
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from analyze_prefetch_learning import audit_games
from match import summarize
from match_audit import loaded_identity, replay_game, replay_timeout_prefix, diagnostics
from paired_score import paired_comparison


def require(value, message):
    if not value: raise ValueError(message)


def audit_suite(root, spec, spec_sha, directory, arm, training):
    suite = read_json(directory / 'result.json'); source = artifact(root, '.gozero/snapshots/' + suite['snapshot']); verify(source)
    verify_inputs(source, spec['evaluation_input_closure'])
    for name, expected in spec['evaluation_code_sha256'].items():
        require(sha256(source / name) == expected, 'Registered evaluator changed: ' + name)
    require(suite['kind'] == 'online_annealing_katago_suite' and suite['status'] in ('passed', 'failed') and suite['arm'] == arm
            and suite['spec_sha256'] == spec_sha and spec['registered_unix'] < suite['started_unix'] <= suite['finished_unix']
            and suite['finished_unix'] - suite['started_unix'] <= spec['maximum_eval_seconds_each_arm'], 'Suite registration/status differs')
    attempts = [p for p in (root / 'runs/eval').glob('*/result.json') if read_json(p).get('kind') == 'online_annealing_katago_suite'
                and read_json(p).get('arm') == arm and read_json(p).get('spec_sha256') == spec_sha]
    require(attempts == [directory / 'result.json'], 'Evaluation attempt budget differs')
    descriptor = read_json(source / f'eval/online_annealing/{arm}.json'); identity = validate_candidate(root, descriptor)
    require(suite['candidate_sha256'] == sha256(source / f'eval/online_annealing/{arm}.json')
            and suite['model_export_sha256'] == descriptor['model_export_sha256'], 'Suite model differs')
    if arm == 'parent':
        require(descriptor['training_snapshot'] == spec['parent_snapshot'] and identity['checkpoint_turn'] == spec['parent_turn'], 'Parent identity differs')
    else:
        require(descriptor['training_snapshot'] == spec['arms'][arm]['snapshot'] and identity['checkpoint_turn'] == spec['final_turn']
                and descriptor['model_export_sha256'] == training['arms'][arm]['model_export_sha256'], 'Final model differs from training audit')
    native_path = identity['native_receipt'] or artifact(root, '.gozero/native/' + descriptor['training_snapshot'] + '/receipt.json')
    native = read_json(native_path); native_module = load_library(native_path.parent / native['filename'], native['binary_sha256'])
    require(len(suite['panels']) in (1, 2) and [p['path'] for p in suite['panels']] == spec['evaluation_panels'][arm][:len(suite['panels'])], 'Panel sequence differs')
    groups = {}; panels = {}; previous_end = suite['started_unix']
    for index, group in enumerate(('primary', 'strong')):
        panel_dir = directory / group; panel = read_json(panel_dir / 'result.json'); declared_path = spec['evaluation_panels'][arm][index]
        if index < len(suite['panels']): recorded = suite['panels'][index]
        else:
            require(group == 'strong' and suite['status'] == 'failed'
                    and suite.get('error') == "ValueError('Panel has a process or command failure')", 'Unexpected missing suite receipt')
            recorded = {'path': declared_path, 'spec_sha256': sha256(source / declared_path),
                        'result_sha256': sha256(panel_dir / 'result.json'), 'summaries': panel['summaries'], 'returncode': 1}
        declaration = read_json(source / recorded['path'])
        require(sha256(panel_dir / 'result.json') == recorded['result_sha256']
                and panel['snapshot_id'] == source.name and panel['spec_sha256'] == recorded['spec_sha256'] == sha256(source / recorded['path'])
                and previous_end <= panel['started_unix'] <= panel['finished_unix'] <= suite['finished_unix']
                and 'error' not in panel and len(panel['matches']) == len(declaration['matches']), 'Panel identity or timing differs')
        previous_end = panel['finished_unix']; games = []; offset = 0
        require({r['id'] for r in panel['matches']} == {d['id'] for d in declaration['matches']}, 'Child panel coverage differs')
        for child_decl in declaration['matches']:
            entry = next(row for row in panel['matches'] if row['id'] == child_decl['id'])
            child_dir = panel_dir / entry['id']; child = read_json(child_dir / 'result.json'); config = read_json(source / child_decl['spec'])
            require(sha256(child_dir / 'result.json') == entry['result_sha256'] and child['spec_sha256'] == sha256(source / child_decl['spec'])
                    and child['snapshot_id'] == source.name and child['candidate'] == descriptor
                    and child['candidate_sha256'] == suite['candidate_sha256'] and child['games'] == entry['games']
                    and read_json(child_dir / 'resolved_spec.json') == config and read_json(source / config['candidate']) == descriptor
                    and entry['group'] == arm + '-' + group and not entry['timed_out'] and 'error' not in child
                    and panel['started_unix'] <= child['started_unix'] <= child['finished_unix'] <= panel['finished_unix'], 'Child identity or games differ')
            require(child['candidate_adapter'] == 'eval/learned_gtp.py' and child['candidate_scoring_profile'] == 'pass_alive_area'
                    and child['inference_native_snapshot'] == native['snapshot_id'] and child['native_receipt_sha256'] == sha256(native_path)
                    and native['binary_sha256'] == identity['native_binary_sha256'], 'Actual native/adapter identity differs')
            weights = read_json(source / config['katago_weights']); kata = read_json(source / 'eval/katago_build.json')
            require(child['katago_binary_sha256'] == kata['binary_sha256'] == sha256(artifact(root, kata['binary_path']))
                    and child['katago_weights_sha256'] == weights['sha256'] == sha256(artifact(root, weights['path']))
                    and child['katago_weights_descriptor_sha256'] == sha256(source / config['katago_weights'])
                    and child['base_katago_config_sha256'] == sha256(source / config['katago_config']), 'External engine or weights differ')
            failures = [g for g in child['games'] if g['status'] == 'failed']
            if not failures:
                audit_games(child['games'], child_dir, config, direct=False)
            else:
                require(group == 'strong', 'Primary comparison has a process failure')
                require(len(child['games']) == 2 * len(config['openings'])
                        and {(g['pair'], g['candidate_color']) for g in child['games']} ==
                            {(i, color) for i in range(len(config['openings'])) for color in ('B', 'W')}, 'Failed diagnostic coverage differs')
            complete = all(g['status'] == 'completed' for g in child['games']); status = 'passed' if complete else 'failed'
            require(child['status'] == entry['status'] == status and entry['returncode'] == (0 if complete else 1)
                    and child['summary'] == entry['summary'] == summarize(child['games']), 'Child summary/status differs')
            for game in child['games']:
                folder = child_dir / f"pair-{game['pair']:03d}-{game['candidate_color']}"
                require(read_json(folder / 'result.json') == game and game['opening'] == config['openings'][game['pair']]
                        and game['katago_rules']['ko'] == 'POSITIONAL' and game['katago_rules']['scoring'] == 'AREA'
                        and game['katago_rules']['tax'] == 'NONE' and game['katago_rules']['suicide'], 'Raw diagnostic game or rules differ')
                loaded_identity(folder, game, source, config, descriptor, identity, native)
                if game['status'] == 'failed':
                    replay_timeout_prefix(folder, game, native_module); seed = config['seed_prefix'] + '-' + str(game['pair'])
                else:
                    seed = replay_game(folder, game, native_module, config)
                    if game['status'] == 'completed':
                        require(game['score_margin_agrees_with_katago'] and game['score'] == game['katago_adjudicated_score']
                                and game['candidate_points'] == (.5 if game['score'] == '0' else float(game['score'][0] == game['candidate_color'])), 'Completed diagnostic score differs')
                expected = re.sub(r'^(searchRandSeed|nnRandSeed)\s*=.*$', lambda m: m[1] + ' = ' + seed,
                                  (source / config['katago_config']).read_text(), flags=re.M)
                require((folder / 'katago.cfg').read_text() == expected, 'Actual external seed/config differs')
            games.extend({**g, 'pair': g['pair'] + offset} for g in child['games']); offset += len(config['openings'])
        summary = summarize(games)
        require(panel['summaries'] == recorded['summaries'] == {arm + '-' + group: summary}
                and len(games) == spec['scheduled_primary_games_each_arm' if group == 'primary' else 'scheduled_strong_diagnostic_games_each_arm'],
                'Panel totals differ')
        complete = all(g['status'] == 'completed' for g in games)
        require(panel['status'] == ('passed' if complete else 'failed') and recorded['returncode'] == (0 if complete else 1), 'Panel status differs')
        groups[group] = games; panels[group] = {'result_sha256': sha256(panel_dir / 'result.json'), 'summary': summary,
                                               'elapsed_seconds': panel['finished_unix'] - panel['started_unix'], 'diagnostics': diagnostics(games)}
    all_games = [g for games in groups.values() for g in games]
    failures = sum(g['status'] == 'failed' for g in all_games)
    require(suite['status'] == ('failed' if failures else 'passed'), 'Suite status does not reflect process failures')
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    return {'directory': str(directory.relative_to(root)), 'execution_snapshot': source.name,
            'result_sha256': sha256(directory / 'result.json'), 'panels': panels,
            'scheduled_games': len(all_games), 'completed_games': sum(g['status'] == 'completed' for g in all_games),
            'capped_games': sum(g['status'] == 'truncated' for g in all_games), 'process_failures': failures,
            'original_suite_status': suite['status'], 'complete_diagnostic_execution': failures == 0,
            'checked_boards': sum(g['checked_positions'] for g in all_games), 'model_export_sha256': descriptor['model_export_sha256'],
            'elapsed_seconds': suite['finished_unix'] - suite['started_unix'], 'raw_files_sha256': files}, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--training-audit', type=Path, required=True); p.add_argument('--expected-training-audit-sha256', required=True)
    p.add_argument('--parent-suite', type=Path, required=True); p.add_argument('--inherit-suite', type=Path, required=True)
    p.add_argument('--anneal-suite', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec); training = read_json(a.training_audit)
    report = {'schema_version': 1, 'kind': 'online_annealing_pilot_analysis', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'spec_sha256': sha256(a.spec), 'training_audit_sha256': sha256(a.training_audit), 'arms': {}, 'production_promotion': False}
    try:
        require(report['training_audit_sha256'] == a.expected_training_audit_sha256 and training['status'] == 'passed'
                and training['spec_sha256'] == report['spec_sha256'], 'Training audit differs')
        groups = {}
        for name in spec['evaluation_order']:
            report['arms'][name], groups[name] = audit_suite(root, spec, report['spec_sha256'], getattr(a, name + '_suite').resolve(), name, training)
        book = read_json(SOURCE / spec['opening_book']); require(sha256(SOURCE / spec['opening_book']) == spec['opening_book_sha256'], 'Opening book changed')
        report['paired_primary'] = paired_comparison(groups['inherit']['primary'], groups['anneal']['primary'], book['primary'], spec['criterion'])
        report['training_completion_criterion_met'] = all(row['training_completion_criterion_met'] for row in training['arms'].values())
        complete_execution = all(row['process_failures'] == 0 for row in report['arms'].values())
        report.update(status='passed', complete_study_execution=complete_execution,
                      registered_combined_criterion_met=report['paired_primary']['registered_external_criterion_met'] and report['training_completion_criterion_met'] and complete_execution,
                      scheduled_games=sum(r['scheduled_games'] for r in report['arms'].values()),
                      checked_boards=sum(r['checked_boards'] for r in report['arms'].values()),
                      completed_scores=sum(r['completed_games'] for r in report['arms'].values()),
                      capped_games=sum(r['capped_games'] for r in report['arms'].values()),
                      process_failures=sum(r['process_failures'] for r in report['arms'].values()),
                      recorded_training_attempt_chip_hours=sum(r['recorded_attempt_chip_hours'] for r in training['arms'].values()))
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        with a.output.open('xb') as stream: stream.write(canonical_json(report))
        print(canonical_json({k: v for k, v in report.items() if k != 'arms'}).decode(), flush=True)


if __name__ == '__main__': main()
