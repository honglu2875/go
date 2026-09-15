#!/usr/bin/env python3
"""Analyze a registered static score utility pilot with fresh matched controls."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def training(root, attempt, expected_source):
    import numpy as np
    directory = root / 'runs' / attempt
    pod = read_json(directory / 'result.json')
    if pod['status'] != 'passed' or pod['snapshot_id'] != expected_source:
        raise ValueError('Training attempt identity or status differs')
    source = root / '.gozero/snapshots' / expected_source
    manifest = verify(source)
    config = read_json(source / 'resolved_config.json')
    ranks = [read_json(directory / f'rank-{host}/artifacts/result.json') for host in range(4)]
    if any(r['status'] != 'passed' or r['snapshot_id'] != expected_source or r['turn'] != config['selfplay_turns'] for r in ranks):
        raise ValueError('Incomplete training')
    if len({r['model_export_sha256'] for r in ranks}) != 1 or len({r['counters']['updates'] for r in ranks}) != 1:
        raise ValueError('Replicated learner differs')
    supports = []; peaks = []; pass_mass = []; outcomes = []
    for host, result in enumerate(ranks):
        artifacts = directory / f'rank-{host}/artifacts'
        if sha256(artifacts / 'model_export.npz') != result['model_export_sha256']:
            raise ValueError('Export identity differs')
        checkpoint = artifacts / 'checkpoints' / f"turn-{result['turn']:09d}"
        if sha256(checkpoint / 'manifest.json') != result['latest_checkpoint']['manifest_sha256']:
            raise ValueError('Checkpoint manifest differs')
        saved = read_json(checkpoint / 'manifest.json')
        if sha256(checkpoint / 'arrays.npz') != saved['files']['arrays.npz']['sha256']:
            raise ValueError('Replay integrity differs')
        with np.load(checkpoint / 'arrays.npz', allow_pickle=False) as arrays:
            pi = arrays['replay_pi']; meta = arrays['replay_meta']
            if not np.all(meta[:, 3] == config['actors']['simulations']):
                raise ValueError('Replay search budget differs')
            supports.append(np.count_nonzero(pi, axis=1)); peaks.append(np.max(pi, axis=1))
            pass_mass.append(pi[:, -1]); outcomes.append(arrays['replay_z'])
    support = np.concatenate(supports); peak = np.concatenate(peaks)
    return {
        'attempt': attempt, 'snapshot_id': expected_source, 'pod_result_sha256': sha256(directory / 'result.json'),
        'rank_result_sha256': [sha256(directory / f'rank-{host}/artifacts/result.json') for host in range(4)],
        'model_export_sha256': ranks[0]['model_export_sha256'],
        'model_implementation_sha256': sha256(source / manifest['recipe'] / 'model.py'),
        'global': {k: sum(r['counters'][k] for r in ranks) for k in ('real_moves', 'completed_games', 'truncated_games', 'eligible_rows', 'active_neural_evaluations', 'neural_slots')},
        'global_updates': ranks[0]['counters']['updates'], 'max_training_seconds': max(r['elapsed_segment_seconds'] for r in ranks),
        'max_inference_seconds': max(r['counters']['inference_seconds'] for r in ranks),
        'max_native_seconds': max(r['counters']['native_seconds'] for r in ranks),
        'host_average_cpu_cores': [r['process_cpu_segment_seconds'] / r['elapsed_segment_seconds'] for r in ranks],
        'recorded_attempt_chip_hours': pod['reserved_chip_hours'], 'last_metrics': ranks[0]['last_metrics'],
        'final_replay_diagnostics': {'rows': len(support), 'mean_visited_root_actions': float(support.mean()),
            'mean_largest_visit_policy_mass': float(peak.mean()), 'mean_pass_policy_mass': float(np.concatenate(pass_mass).mean()),
            'mean_side_to_move_outcome': float(np.concatenate(outcomes).mean()),
            'scope': 'Final replay windows from different self-play distributions; not held-out accuracy or strength.'},
    }, config


def evaluation(root, path, model_hash):
    panel = read_json(path)
    source = root / '.gozero/snapshots' / panel['snapshot_id']; verify(source)
    games = []; profiles = []
    if len(panel['matches']) != 4:
        raise ValueError('Registered four-pair candidate screen differs')
    for entry in panel['matches']:
        child_path = path.parent / entry['id'] / 'result.json'
        if sha256(child_path) != entry['result_sha256']:
            raise ValueError('Child match identity differs')
        child = read_json(child_path)
        if child['candidate']['model_export_sha256'] != model_hash or len(child['games']) != 2:
            raise ValueError('Evaluation candidate or color pair differs')
        spec_path = source / 'eval/score_utility' / (entry['id'] + '.json')
        if sha256(spec_path) != child['spec_sha256']:
            raise ValueError('Frozen evaluation spec differs')
        spec = read_json(spec_path)
        if (spec['candidate_simulations_excluding_root'] != 16 or spec['candidate_cpuct'] != 1.5
                or spec['size'] != 9 or spec['komi'] != 7.5 or spec['max_game_moves'] != 324
                or spec['game_timeout_seconds'] != 360 or child['candidate_scoring_profile'] != 'pass_alive_area'):
            raise ValueError('Registered evaluation budget or rules differ')
        if child['katago_weights_sha256'] != 'a1298ce1adc1dad7bd868ca962b2384cc8388ed373a00e6bae1114fa6f9e2d61':
            raise ValueError('KataGo checkpoint differs')
        if child['katago_binary_sha256'] != '1ae1ed2108caa025bba853634b7f1aad3a29baa0ba27b4a298759ca6b172d104':
            raise ValueError('KataGo binary differs')
        if sha256(source / spec['katago_config']) != child['base_katago_config_sha256']:
            raise ValueError('KataGo configuration differs')
        profiles.append((spec['katago_max_visits'], json.dumps(spec['openings'])))
        for game in child['games']:
            if game['checked_positions'] != len(game['moves']):
                raise ValueError('Board checks incomplete')
            if game['status'] == 'completed' and not game['score_margin_agrees_with_katago']:
                raise ValueError('Completed adjudication differs')
        games.extend(child['games'])
    if sorted(profiles) != sorted((visits, json.dumps([opening])) for visits in (1,16) for opening in ([],['C3','G7'])):
        raise ValueError('Registered visit levels or openings differ')
    return {'path': str(path), 'panel_sha256': sha256(path), 'panel_status': panel['status'],
            'summaries': panel['summaries'], 'scheduled': len(games),
            'completed': sum(g['status'] == 'completed' for g in games),
            'truncated': sum(g['status'] == 'truncated' for g in games),
            'failed': sum(g['status'] == 'failed' for g in games),
            'wins': sum(g.get('candidate_points') == 1 for g in games),
            'checked_boards': sum(g['checked_positions'] for g in games)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    for arm in ('control', 'candidate'):
        p.add_argument('--' + arm + '-attempt', required=True)
        p.add_argument('--' + arm + '-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(ROOT); root = args.workspace_root.resolve()
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'analysis_snapshot': ROOT.name, 'kind': 'static_score_utility_pilot_analysis',
              'status': 'failed', 'protocol_sha256': sha256(args.protocol), 'arms': {}, 'evaluation': {},
              'claims_strength_improvement': False, 'claims_sample_efficiency_improvement': False, 'claims_mfu': False}
    try:
        configs = []
        for arm in ('control', 'candidate'):
            report['arms'][arm], config = training(root, getattr(args, arm + '_attempt'), protocol[arm + '_snapshot'])
            configs.append(config)
            report['evaluation'][arm] = evaluation(root, getattr(args, arm + '_panel'), report['arms'][arm]['model_export_sha256'])
        for c, factor in zip(configs, (0., 0.3)):
            if c['model'].pop('score_utility_factor') != factor or c['actors']['score_utility'].pop('factor') != factor:
                raise ValueError('Utility factor differs from registration')
        if configs[0] != configs[1]:
            raise ValueError('Training intervention differs beyond score utility')
        a, b = (report['arms'][arm] for arm in ('control', 'candidate'))
        if a['model_implementation_sha256'] != b['model_implementation_sha256']:
            raise ValueError('Model or optimizer implementation differs')
        report['eligible_rows_ratio'] = b['global']['eligible_rows'] / a['global']['eligible_rows']
        report['truncated_games_ratio'] = None if not a['global']['truncated_games'] else b['global']['truncated_games'] / a['global']['truncated_games']
        report['charged_training_attempt_chip_hours'] = a['recorded_attempt_chip_hours'] + b['recorded_attempt_chip_hours']
        no_wins = all(e['wins'] == 0 for e in report['evaluation'].values())
        incomplete = any(e['completed'] != e['scheduled'] for e in report['evaluation'].values())
        report['decision'] = ('No promotion: both arms remain at the observed zero-win floor against KataGo.' if no_wins
                              else 'Exploratory result only: larger complete panels and independent training seeds are required before promotion.')
        if incomplete: report['decision'] += ' Incomplete games invalidate the planned paired comparison.'
        report['limitations'] = protocol['limitations'] + [
            'Both arms train the score head; this isolates its use by search, not adding score supervision itself.',
            'Control evaluation overlapped candidate training on separate physical CPU cores; training timings are diagnostic and not an isolated systems comparison.',
            'Losses and replay diagnostics concern different evolving training distributions.',
            'Incomplete games have no assigned result.',
            'Attempt chip-hours exclude reserved idle and engineering time; see the allocation ledger.']
        report['status'] = 'analyzed'; verify(ROOT)
    except Exception as error:
        report['error'] = repr(error); raise
    finally:
        with args.output.open('xb') as stream: stream.write(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('arms', 'evaluation')}), flush=True)


if __name__ == '__main__': main()
