#!/usr/bin/env python3
"""Analyze a registered FPU follow-up, including its reused control and failed games."""
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


def evaluation(path, expected_model, expected_source):
    panel = read_json(path)
    verify(expected_source)
    selected = []
    for entry in panel['matches']:
        if not entry['group'].startswith('candidate-'):
            continue
        child_path = path.parent / entry['id'] / 'result.json'
        if sha256(child_path) != entry['result_sha256']:
            raise ValueError('Child match identity differs')
        child = read_json(child_path)
        if child['candidate']['model_export_sha256'] != expected_model:
            raise ValueError('Evaluation weights differ')
        for game in child['games']:
            if game['checked_positions'] != len(game['moves']):
                raise ValueError('Board checks incomplete')
            if game['status'] == 'completed' and not game['score_margin_agrees_with_katago']:
                raise ValueError('Adjudication differs')
        selected.append({'id': entry['id'], 'group': entry['group'], 'status': child['status'],
                         'result_sha256': entry['result_sha256'], 'games': child['games']})
    if len(selected) != 4 or any(len(e['games']) != 2 for e in selected):
        raise ValueError('Registered eight-game candidate screen differs')
    games = [g for e in selected for g in e['games']]
    return {'panel_sha256': sha256(path), 'panel_status': panel['status'],
            'summaries': {k: v for k, v in panel['summaries'].items() if k.startswith('candidate-')},
            'scheduled': len(games), 'completed': sum(g['status'] == 'completed' for g in games),
            'truncated': sum(g['status'] == 'truncated' for g in games), 'failed': sum(g['status'] == 'failed' for g in games),
            'wins': sum(g.get('candidate_points') == 1 for g in games), 'checked_boards': sum(g['checked_positions'] for g in games)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--candidate-attempt', required=True)
    p.add_argument('--candidate-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); verify(ROOT); root = args.workspace_root.resolve()
    protocol = read_json(args.protocol)
    report = {'schema_version': 1, 'analysis_snapshot': ROOT.name, 'kind': 'low_visit_fpu_pilot_analysis',
              'status': 'failed', 'protocol_sha256': sha256(args.protocol), 'arms': {}, 'evaluation': {},
              'claims_strength_improvement': False, 'claims_sample_efficiency_improvement': False, 'claims_mfu': False}
    try:
        configs = []
        for arm, attempt in [('control', protocol['reused_control_attempt']), ('candidate', args.candidate_attempt)]:
            report['arms'][arm], config = training(root, attempt, protocol[arm + '_snapshot'])
            configs.append(config)
        configs[0]['actors'].pop('fpu_reduction', None)
        reduction = configs[1]['actors'].pop('fpu_reduction')
        if configs[0] != configs[1] or reduction != 0.2:
            raise ValueError('Training intervention differs from the FPU protocol')
        a, b = (report['arms'][x] for x in ('control', 'candidate'))
        if a['model_implementation_sha256'] != b['model_implementation_sha256']:
            raise ValueError('Model or optimizer implementation differs')
        control_panel = root / protocol['reused_control_panel']
        if sha256(control_panel) != protocol['control_panel_sha256']:
            raise ValueError('Reused control panel differs')
        for arm, path in [('control', control_panel), ('candidate', args.candidate_panel)]:
            panel_source = root / '.gozero/snapshots' / read_json(path)['snapshot_id']
            report['evaluation'][arm] = evaluation(path, report['arms'][arm]['model_export_sha256'], panel_source)
        report['eligible_rows_ratio'] = b['global']['eligible_rows'] / a['global']['eligible_rows']
        report['truncated_games_ratio'] = b['global']['truncated_games'] / a['global']['truncated_games']
        report['charged_training_attempt_chip_hours'] = a['recorded_attempt_chip_hours'] + b['recorded_attempt_chip_hours']
        no_wins = all(e['wins'] == 0 for e in report['evaluation'].values())
        report['decision'] = ('Do not promote: both arms remain at the observed zero-win floor against KataGo, with incomplete games.'
                              if no_wins else 'Exploratory result only: confirm with independent seeds and a larger complete panel before promotion.')
        report['limitations'] = ['One seed; control reused after its negative pilot motivated the follow-up.',
            'Incomplete games have no assigned result and invalidate the planned paired strength comparison.',
            'Actual learner update counts differ with replay warmup; configured real-move and search budgets match.',
            'Lower training loss or more terminal rows on a changed self-play distribution is not improved sample efficiency.',
            'The fixed FPU reduction is one component; neither arm is a full KataGo training reproduction.',
            'Recorded attempt windows exclude idle reservation time; the allocation ledger is separate.']
        report['status'] = 'analyzed'; verify(ROOT)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('arms', 'evaluation')}), flush=True)


if __name__ == '__main__':
    main()
