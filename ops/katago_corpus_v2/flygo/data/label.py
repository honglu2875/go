"""Label semantics: raw teacher targets, searched targets, and behavior stay separate."""
from __future__ import annotations

import numpy as np

from ..gtp import vertex_to_action


def parse_label(response: dict, legal, to_play: int, *, size=9) -> dict:
    if to_play not in (1, 2):
        raise ValueError('Invalid player')
    mask = np.zeros(size * size + 1, dtype=bool)
    mask[legal] = True
    policy = np.asarray(response['policy'], dtype=np.float64)
    if policy.shape != mask.shape or not np.all(np.isfinite(policy)):
        raise ValueError('Invalid raw policy')
    if not np.array_equal(policy >= 0, mask):
        raise ValueError('KataGo and native Go disagree on legal moves')
    policy = np.where(mask, policy, 0)
    if policy.sum() <= 0:
        raise ValueError('Empty raw policy')
    policy /= policy.sum()
    counts = np.zeros_like(policy)
    orders = np.full(len(policy), len(policy), dtype=np.int32)
    for move in response['moveInfos']:
        action = vertex_to_action(move['move'], size)
        if not mask[action] or 'isSymmetryOf' in move:
            raise ValueError('Unexpected illegal or symmetry-pruned root edge')
        count = move['edgeVisits']
        if count < 0 or not np.isfinite(count):
            raise ValueError('Invalid root edge count')
        counts[action] += count
        orders[action] = move['order']
    root = response['rootInfo']
    sign = 1 if to_play == 2 else -1
    raw_value = sign * (2 * root['rawWinrate'] - 1)
    search_value = sign * (2 * root['winrate'] - 1)
    if not np.isfinite([raw_value, search_value]).all() or max(abs(raw_value), abs(search_value)) > 1.00001:
        raise ValueError('Invalid teacher values')
    if root['currentPlayer'] != ('B' if to_play == 1 else 'W'):
        raise ValueError('Teacher response player does not match position')
    return dict(raw_policy=policy.astype(np.float32), raw_value=np.float32(raw_value),
                search_policy=(counts / max(1, counts.sum())).astype(np.float32),
                search_value=np.float32(search_value), search_policy_valid=bool(counts.sum() > 0),
                best_action=int(np.argmin(orders)), root_edge_visits=counts.astype(np.int32),
                raw_score=np.float32(sign * root.get('rawScoreSelfplay', 0)),
                raw_score_valid='rawScoreSelfplay' in root)
