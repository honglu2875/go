"""Audited fixed-registration reports for precision and real KataGo endpoints."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def precision(root, registration, attempts, remember):
    if len(attempts) != 1:
        raise ValueError('Expected one precision attempt')
    directory = root / 'runs' / attempts[0]; closed = remember(directory / 'result.json')
    if closed['snapshot_id'] != registration['snapshot_id'] or closed['status'] != 'passed':
        raise ValueError('Precision execution identity differs')
    reports = [remember(directory / f'rank-{h}/artifacts/result.json') for h in range(4)]
    if any(r['status'] != 'passed' or r['snapshot_id'] != closed['snapshot_id'] for r in reports):
        raise ValueError('A precision rank did not pass')
    if any(r['gradient_comparison'] != reports[0]['gradient_comparison'] or r['losses'] != reports[0]['losses'] for r in reports):
        raise ValueError('Replicated gradients or losses differ')
    critical = {mode: [max(r['timing'][i][mode] for r in reports) for i in range(5)] for mode in ('highest', 'default')}
    ratio = statistics.median(critical['highest']) / statistics.median(critical['default'])
    gradient = reports[0]['gradient_comparison']; loss = reports[0]['losses']; screen = registration['primary_screen']
    gates = {'speed': ratio >= screen['minimum_median_critical_path_speedup'],
             'gradient_cosine': gradient['cosine'] >= screen['minimum_gradient_cosine'],
             'gradient_relative_l2': gradient['relative_l2'] <= screen['maximum_gradient_relative_l2'],
             'forward_loss': abs(loss['highest'] - loss['default']) <= screen['maximum_forward_loss_absolute_difference']}
    return {'critical_path_seconds': critical, 'median_critical_path_speedup': ratio,
            'gradient_comparison': gradient, 'losses': loss, 'gates': gates, 'screen_passed': all(gates.values()),
            'decision': 'Retain original highest precision; reduced precision did not meet the registered speed threshold.' if not gates['speed'] else 'Eligible only for subsequent learning qualification.',
            'attempt_chip_hours': closed['reserved_chip_hours'],
            'limitations': 'One real batch at initialization, no optimizer trajectory comparison; compiler cost estimates are not measured MFU.'}


def katago(root, registration, attempts, remember):
    if len(attempts) != len(registration['arms']):
        raise ValueError('Every registered model needs an attempt')
    arms = {}; matched_openings = {}; cost = 0.
    for arm, attempt in zip(registration['arms'], attempts):
        snapshot = root / '.gozero/snapshots' / arm['snapshot_id']; verify(snapshot)
        config = read_json(snapshot / 'resolved_config.json')
        if sha256(snapshot / 'resolved_config.json') != arm['config_sha256']:
            raise ValueError('Panel configuration changed')
        directory = root / 'runs' / attempt; closed = remember(directory / 'result.json')
        if closed['snapshot_id'] != snapshot.name:
            raise ValueError('Panel attempt source changed')
        cost += closed['reserved_chip_hours']; anchors = defaultdict(list); timing = []
        for host in range(4):
            report = remember(directory / f'rank-{host}/artifacts/result.json')
            if report['candidate_sha256'] != arm['candidate_sha256']:
                raise ValueError('Match owner loaded a different candidate')
            timing.append({k: report.get(k) for k in ['match_seconds', 'cache_statistics', 'compilation']})
            for index, spec_name in enumerate(config['matches_by_host'][str(host)]):
                spec = read_json(snapshot / spec_name); path = directory / f'rank-{host}/artifacts/match-{index:03d}'
                result = remember(path / 'result.json')
                if (result['spec_sha256'] != sha256(snapshot / spec_name)
                        or result['candidate_sha256'] != arm['candidate_sha256']
                        or result['snapshot_id'] != snapshot.name or len(result['games']) != 2):
                    raise ValueError('Match identity or paired-color coverage changed')
                key = (result['katago_weights_sha256'], spec['seed_prefix'])
                previous = matched_openings.setdefault(key, spec['openings'])
                if previous != spec['openings']:
                    raise ValueError('Paired arms used different openings')
                pair = []
                for game in result['games']:
                    game_path = path / f"pair-000-{game['candidate_color']}"
                    saved = remember(game_path / 'result.json')
                    if game != saved or not (game_path / 'game.sgf').is_file():
                        raise ValueError('Canonical game and SGF coverage differ')
                    remember(game_path / 'game.sgf', parse=False)
                    if game['status'] == 'completed':
                        if (not game['winner_agrees_with_katago'] or not game['score_margin_agrees_with_katago']
                                or game['candidate_points'] not in (0., .5, 1.)):
                            raise ValueError('Independent terminal score audit failed')
                    pair.append({k: game.get(k) for k in ['status', 'candidate_points', 'candidate_color',
                        'checked_positions', 'score', 'elapsed_seconds', 'engine_seconds']})
                if sorted(g['candidate_color'] for g in pair) != ['B', 'W']:
                    raise ValueError('Paired colors are not complementary')
                anchors[result['katago_model']].append(pair)
        summary = {}
        for anchor, pairs in anchors.items():
            if len(pairs) != registration['pairs_per_anchor_per_arm']:
                raise ValueError('Anchor does not contain every registered opening')
            games = [g for pair in pairs for g in pair]; counts = Counter(g['status'] for g in games)
            points = sum(g['candidate_points'] for g in games if g['status'] == 'completed')
            complete_pairs = [sum(g['candidate_points'] for g in pair) / 2 for pair in pairs if all(g['status'] == 'completed' for g in pair)]
            interval = None; score = None
            if len(complete_pairs) == len(pairs):
                score = statistics.mean(complete_pairs); radius = math.sqrt(math.log(40) / (2 * len(pairs)))
                interval = [max(0., score - radius), min(1., score + radius)]
            unresolved = len(games) - counts['completed']
            summary[anchor] = {'scheduled_games': len(games), 'status_counts': dict(counts),
                'completed_wins': sum(g['status'] == 'completed' and g['candidate_points'] == 1. for g in games),
                'completed_draws': sum(g['status'] == 'completed' and g['candidate_points'] == .5 for g in games),
                'completed_losses': sum(g['status'] == 'completed' and g['candidate_points'] == 0. for g in games),
                'checked_positions': sum(g['checked_positions'] or 0 for g in games),
                'complete_pairs': len(complete_pairs), 'paired_score': score, 'paired_hoeffding_95_interval': interval,
                'scheduled_score_identification_bounds': [points / len(games), (points + unresolved) / len(games)],
                'bounds_scope': 'Assign each unresolved game any score in [0,1]; these are missing-outcome bounds, not confidence intervals.'}
        arms[arm['arm']] = {'attempt': attempt, 'execution_status': closed['status'], 'anchors': summary, 'host_timing': timing}
    return {'arms': arms, 'attempt_chip_hours': cost,
        'qualification_passed': all(v['execution_status'] == 'passed' for v in arms.values()),
        'limitations': 'One offline weak-teacher seed, fixed per-engine search budgets rather than equal compute. Caps remain unresolved and invalidate the complete-pair strength interval. No Elo, architecture superiority or faster RL claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--kind', choices=['precision', 'katago'], required=True); p.add_argument('--attempts', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); verify(SOURCE)
    if sha256(a.registration) != a.registration_sha256 or a.output.exists():
        raise ValueError('Registration identity changed or audit output exists')
    root = a.workspace_root.resolve(); evidence = {}
    def remember(path, parse=True):
        evidence[str(path.relative_to(root))] = sha256(path)
        return read_json(path) if parse else None
    result = (precision if a.kind == 'precision' else katago)(root, read_json(a.registration), a.attempts, remember)
    result = {'schema_version': 1, 'kind': 'visual_' + a.kind + '_audit', 'audit_status': 'passed',
              'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256, 'evidence': evidence, **result}
    a.output.write_bytes(canonical_json(result)); print(canonical_json({k: v for k, v in result.items() if k != 'evidence'}).decode().strip())


if __name__ == '__main__':
    main()
