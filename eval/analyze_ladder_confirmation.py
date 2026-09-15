#!/usr/bin/env python3
"""Validate the registered 128-game official KataGo ladder confirmation."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact, validate_candidate
from gozero.snapshots import canonical_json, read_json, verify
from match import summarize

REGISTRATION = '713416d25c37912cac72ce2b96cba89eebeebe69f5ea174db182c19ac249af13'


def analyze_arm(root, protocol, arm, directory):
    source = root / '.gozero/snapshots' / protocol['evaluation_snapshot']
    verify(source)
    panel = read_json(directory / 'result.json')
    declared = protocol['arms'][arm]
    if (panel['snapshot_id'] != source.name
            or panel['spec_sha256'] != declared['panel_sha256']
            or sha256(source / declared['panel']) != declared['panel_sha256']
            or panel['started_unix'] < protocol['registered_unix']):
        raise ValueError('Panel identity or registration order differs')
    descriptor = source / declared['descriptor']
    if sha256(descriptor) != declared['descriptor_sha256']:
        raise ValueError('Candidate descriptor differs')
    candidate = read_json(descriptor)
    validate_candidate(root, candidate)
    matches = read_json(source / declared['panel'])['matches']
    expected_ids = {f'{arm}-level-{level}-chunk-{chunk}'
                    for level in (2, 3) for chunk in (0, 1)}
    if (len(matches) != 4 or len(panel['matches']) != 4
            or {m['id'] for m in matches} != expected_ids
            or {m['id'] for m in panel['matches']} != expected_ids):
        raise ValueError('Confirmation child identities differ')
    levels = []
    for level in protocol['levels']:
        index = level['level']
        games = []
        children = []
        for chunk in (0, 1):
            identity = f'{arm}-level-{index}-chunk-{chunk}'
            item = next(m for m in panel['matches'] if m['id'] == identity)
            declaration = next(m for m in matches if m['id'] == identity)
            if item['group'] != f'level-{index}' or declaration['group'] != item['group']:
                raise ValueError('Child grouping differs')
            child_path = directory / identity / 'result.json'
            if sha256(child_path) != item['result_sha256']:
                raise ValueError('Child result identity differs')
            child = read_json(child_path)
            spec_path = artifact(source, declaration['spec'])
            spec = read_json(spec_path)
            if (sha256(spec_path) != child['spec_sha256']
                    or child['snapshot_id'] != source.name):
                raise ValueError('Match source or spec identity differs')
            weights_descriptor = artifact(source, spec['katago_weights'])
            weights = read_json(weights_descriptor)
            if (sha256(weights_descriptor) != level['descriptor_sha256']
                    or weights['sha256'] != level['sha256']
                    or sha256(artifact(root, weights['path'])) != level['sha256']
                    or child['katago_weights_descriptor_sha256'] != level['descriptor_sha256']
                    or child['katago_weights_sha256'] != level['sha256']
                    or child['katago_binary_sha256'] != protocol['katago_binary_sha256']):
                raise ValueError('Official KataGo artifact differs')
            openings = protocol['openings'][chunk * 8:(chunk + 1) * 8]
            budget = protocol['budget']
            required = {'size': 9, 'komi': 7.5, 'openings': openings,
                        'max_game_moves': budget['max_game_moves'],
                        'game_timeout_seconds': budget['game_timeout_seconds'],
                        'candidate_simulations_excluding_root': budget['candidate_simulations'],
                        'katago_max_visits': budget['katago_visits'],
                        'candidate_cpuct': 1.5 if arm == 'control' else 0.,
                        'candidate': declared['descriptor']}
            if any(spec[key] != value for key, value in required.items()):
                raise ValueError('Registered match budget differs')
            if (child['candidate'] != candidate
                    or child['candidate_scoring_profile'] != 'pass_alive_area'
                    or sha256(source / spec['katago_config']) != child['base_katago_config_sha256']):
                raise ValueError('Candidate identity or configuration differs')
            raw_games = child['games']
            expected = {(p, color) for p in range(8) for color in ('B', 'W')}
            if len(raw_games) != 16 or {(g['pair'], g['candidate_color']) for g in raw_games} != expected:
                raise ValueError('Paired game identities differ')
            for game in raw_games:
                if (game['opening'] != openings[game['pair']]
                        or game['checked_positions'] != len(game['moves'])
                        or game.get('integrity_failure')):
                    raise ValueError('Opening or board verification differs')
                if game['status'] == 'completed':
                    expected_points = .5 if game['score'] == '0' else float(game['score'][0] == game['candidate_color'])
                    if not game['score_margin_agrees_with_katago'] or game['candidate_points'] != expected_points:
                        raise ValueError('Final score or points differ')
                elif game['status'] not in ('truncated', 'failed') or game.get('candidate_points') is not None:
                    raise ValueError('Incomplete game status or outcome differs')
                for move in game['moves']:
                    if 'search' in move and move['search']['simulations'] != budget['candidate_simulations']:
                        raise ValueError('Actual candidate search work differs')
                saved = read_json(directory / identity / f"pair-{game['pair']:03d}-{game['candidate_color']}" / 'result.json')
                if saved != game:
                    raise ValueError('Individual game record differs')
                # Child pair indices start at zero. Restore registered global pair IDs.
                games.append({**game, 'pair': chunk * 8 + game['pair']})
            children.append({'id': identity, 'result_sha256': sha256(child_path), 'status': child['status']})
        summary = summarize(games)
        if summary != panel['summaries'][f'level-{index}']:
            raise ValueError('Aggregate panel summary differs')
        complete = summary['completed_games'] == 32 and all(c['status'] == 'passed' for c in children)
        score = summary['paired_score']
        levels.append({'level': index, 'model': level['model'], 'sha256': level['sha256'],
                       'children': children, 'summary': summary,
                       'checked_boards': sum(g['checked_positions'] for g in games),
                       'within_registered_calibration_range': complete and .2 <= score <= .8})
    return {'path': str(directory), 'result_sha256': sha256(directory / 'result.json'),
            'model_export_sha256': candidate['model_export_sha256'], 'levels': levels,
            'started_unix': panel['started_unix'], 'finished_unix': panel['finished_unix'],
            'elapsed_seconds': panel['finished_unix'] - panel['started_unix']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'protocol', 'control', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Registration changed')
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'kind': 'official_early_katago_ladder_confirmation_analysis',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': REGISTRATION,
              'status': 'failed', 'claims_strength_improvement': False,
              'claims_sample_efficiency_improvement': False, 'production_promotion': False}
    try:
        root = args.workspace_root.resolve()
        if sha256(root / 'research/studies/katago_ladder/calibration_result.json') != protocol['parent_result_sha256']:
            raise ValueError('Parent calibration result differs')
        report['arms'] = {arm: analyze_arm(root, protocol, arm, getattr(args, arm).resolve())
                          for arm in ('control', 'candidate')}
        control, candidate = (report['arms'][arm] for arm in protocol['budget']['order'])
        if control['finished_unix'] > candidate['started_unix']:
            raise ValueError('Registered arm execution order differs')
        report['total_eval_seconds'] = sum(arm['elapsed_seconds'] for arm in report['arms'].values())
        if report['total_eval_seconds'] > protocol['budget']['maximum_total_eval_seconds']:
            raise ValueError('Registered evaluation time budget exceeded')
        report['eval_budget_respected'] = True
        report['retained_candidate_levels'] = [x['level'] for x in candidate['levels']
                                              if x['within_registered_calibration_range']]
        report['limitations'] = protocol['limitations']
        report['decision'] = 'Use every retained level in registered learning curves on new openings; keep the strong anchor. No algorithm or production promotion.'
        report['status'] = 'analyzed'
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
