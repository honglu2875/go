#!/usr/bin/env python3
"""Analyze the registered early KataGo checkpoint calibration without promotion."""
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

REGISTRATION = 'f57e29b2cf6881bc6200e1aea15023335740beb6368d5691627b582412c2381f'


def analyze_arm(root, protocol, arm, directory):
    source = root / '.gozero/snapshots' / protocol['evaluation_snapshot']
    verify(source)
    panel = read_json(directory / 'result.json')
    declared = protocol['arms'][arm]
    if (panel['snapshot_id'] != source.name or panel['spec_sha256'] != declared['panel_sha256']
            or sha256(source / declared['panel']) != declared['panel_sha256']):
        raise ValueError('Panel identity differs')
    if panel['started_unix'] < protocol['registered_unix']:
        raise ValueError('Panel predates registration')
    descriptor = source / declared['descriptor']
    if sha256(descriptor) != declared['descriptor_sha256']:
        raise ValueError('Candidate descriptor differs')
    candidate = read_json(descriptor)
    validate_candidate(root, candidate)
    matches = read_json(source / declared['panel'])['matches']
    if len(panel['matches']) != 4 or {item['id'] for item in matches} != {item['id'] for item in panel['matches']}:
        raise ValueError('Calibration levels differ')
    levels = []
    for index, level in enumerate(protocol['levels']):
        identity = arm + '-level-' + str(index)
        item = next(item for item in panel['matches'] if item['id'] == identity)
        child_path = directory / identity / 'result.json'
        if sha256(child_path) != item['result_sha256']:
            raise ValueError('Child result identity differs')
        child = read_json(child_path)
        spec_path = source / next(item['spec'] for item in matches if item['id'] == identity)
        spec = read_json(spec_path)
        if sha256(spec_path) != child['spec_sha256']:
            raise ValueError('Match spec identity differs')
        weights_descriptor = artifact(source, spec['katago_weights'])
        weights = read_json(weights_descriptor)
        if (sha256(weights_descriptor) != level['descriptor_sha256'] or weights['sha256'] != level['sha256']
                or sha256(artifact(root, weights['path'])) != level['sha256']
                or child['katago_weights_descriptor_sha256'] != level['descriptor_sha256']
                or child['katago_weights_sha256'] != level['sha256']
                or child['katago_binary_sha256'] != protocol['katago_binary_sha256']):
            raise ValueError('Official KataGo artifact differs')
        required = {'size': 9, 'komi': 7.5, 'openings': protocol['openings'],
                    'max_game_moves': 648, 'game_timeout_seconds': 180,
                    'candidate_simulations_excluding_root': 16, 'katago_max_visits': 1,
                    'candidate_cpuct': 1.5 if arm == 'control' else 0.}
        if any(spec[key] != value for key, value in required.items()):
            raise ValueError('Registered match budget differs')
        if child['candidate'] != candidate or child['candidate_scoring_profile'] != 'pass_alive_area':
            raise ValueError('Candidate identity or rules differ')
        if sha256(source / spec['katago_config']) != child['base_katago_config_sha256']:
            raise ValueError('KataGo configuration differs')
        games = child['games']
        if len(games) != 4 or {(g['pair'], g['candidate_color']) for g in games} != {(p, color) for p in range(2) for color in ('B', 'W')}:
            raise ValueError('Paired games differ')
        for game in games:
            if game['opening'] != protocol['openings'][game['pair']] or game['checked_positions'] != len(game['moves']):
                raise ValueError('Opening or board verification differs')
            if game['status'] == 'completed':
                expected = .5 if game['score'] == '0' else float(game['score'][0] == game['candidate_color'])
                if not game['score_margin_agrees_with_katago'] or game['candidate_points'] != expected:
                    raise ValueError('Final score or points differ')
            elif game.get('candidate_points') is not None:
                raise ValueError('Incomplete game received an outcome')
            for move in game['moves']:
                if 'search' in move and move['search']['simulations'] != 16:
                    raise ValueError('Actual candidate search work differs')
            saved = read_json(directory / identity / f"pair-{game['pair']:03d}-{game['candidate_color']}" / 'result.json')
            if saved != game:
                raise ValueError('Individual game record differs')
        summary = summarize(games)
        complete = summary['completed_games'] == 4 and child['status'] == 'passed'
        levels.append({'level': index, 'model': level['model'], 'sha256': level['sha256'],
                       'result_sha256': sha256(child_path), 'summary': summary,
                       'checked_boards': sum(g['checked_positions'] for g in games),
                       'informative_observed_score': complete and 0 < summary['paired_score'] < 1})
    return {'path': str(directory), 'result_sha256': sha256(directory / 'result.json'),
            'model_export_sha256': candidate['model_export_sha256'], 'levels': levels,
            'elapsed_seconds': panel['finished_unix'] - panel['started_unix']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if sha256(args.protocol) != REGISTRATION:
        raise ValueError('Registration changed')
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'kind': 'official_early_katago_ladder_analysis',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': REGISTRATION,
              'status': 'failed', 'claims_strength_improvement': False,
              'claims_sample_efficiency_improvement': False, 'production_promotion': False}
    try:
        report['arms'] = {arm: analyze_arm(args.workspace_root.resolve(), protocol, arm, getattr(args, arm).resolve())
                          for arm in ('control', 'candidate')}
        report['total_eval_seconds'] = sum(arm['elapsed_seconds'] for arm in report['arms'].values())
        report['eval_budget_respected'] = report['total_eval_seconds'] <= protocol['budget']['maximum_total_eval_seconds']
        report['candidate_informative_levels'] = [x['level'] for x in report['arms']['candidate']['levels'] if x['informative_observed_score']]
        report['limitations'] = protocol['limitations']
        report['decision'] = 'Carry every qualifying intermediate-score candidate level to a fresh larger calibration panel; no algorithm promotion.'
        if not report['candidate_informative_levels']:
            report['decision'] = 'No complete intermediate-score candidate level; report observed floors or ceilings without extrapolating Elo.'
        if not report['eval_budget_respected']:
            raise ValueError('Registered evaluation time budget exceeded')
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
