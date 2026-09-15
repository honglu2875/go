"""Paired-opening bootstrap bounds that keep move-limit outcomes unresolved."""
import numpy as np


def paired_comparison(control, candidate, openings, criterion):
    n = criterion['opening_units']; expected = {(i, color) for i in range(n) for color in ('B', 'W')}
    if len(openings) != n: raise ValueError('Opening book size differs')
    arms = {}; completed = {}
    for name, games in (('inherit', control), ('anneal', candidate)):
        if len(games) != 2 * n or {(g['pair'], g['candidate_color']) for g in games} != expected:
            raise ValueError('Paired opening/color coverage differs')
        bounds = np.zeros((n, 2, 2), np.float64)
        for game in games:
            if game['opening'] != openings[game['pair']]: raise ValueError('Registered opening differs')
            if game['status'] == 'completed' and game.get('candidate_points') in (0., .5, 1.):
                value = [game['candidate_points']] * 2
            elif game['status'] == 'truncated' and game.get('candidate_points') is None:
                value = [0., 1.]
            else: raise ValueError('Assigned cap or process failure cannot enter score bounds')
            bounds[game['pair'], int(game['candidate_color'] == 'W')] = value
        arms[name] = bounds.mean(axis=1); completed[name] = sum(g['status'] == 'completed' for g in games)
    difference = np.stack((arms['anneal'][:, 0] - arms['inherit'][:, 1], arms['anneal'][:, 1] - arms['inherit'][:, 0]), axis=-1)
    indices = np.random.default_rng(criterion['bootstrap_seed']).integers(0, n, size=(criterion['bootstrap_replicates'], n))
    draws = difference[indices].mean(axis=1); alpha = (1. - criterion['confidence']) / 2
    interval = [float(np.quantile(draws[:, 0], alpha)), float(np.quantile(draws[:, 1], 1. - alpha))]
    completion = all(count / (2 * n) >= criterion['minimum_each_arm_completion_fraction'] for count in completed.values())
    positive = interval[0] > criterion['minimum_lower_endpoint_strictly_above']
    return {'opening_units': n, 'completed_games': completed,
            'arm_scheduled_score_bounds': {name: b.mean(axis=0).tolist() for name, b in arms.items()},
            'anneal_minus_inherit_score_bounds': difference.mean(axis=0).tolist(), 'paired_bootstrap_95_outer_interval': interval,
            'bootstrap_draws': criterion['bootstrap_replicates'], 'bootstrap_seed': criterion['bootstrap_seed'],
            'minimum_completion_criterion_met': completion, 'positive_lower_endpoint': positive,
            'registered_external_criterion_met': completion and positive,
            'per_opening': [{'pair': i, 'opening': opening, 'inherit_score_bounds': arms['inherit'][i].tolist(),
                             'anneal_score_bounds': arms['anneal'][i].tolist(), 'difference_bounds': difference[i].tolist()}
                            for i, opening in enumerate(openings)], 'uncertainty_scope': criterion['scope']}
