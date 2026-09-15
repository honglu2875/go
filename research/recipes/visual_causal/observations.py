"""Data-only boundary for native feature planes and sequence validation."""
import numpy as np


def from_native(features):
    """Convert current-player history planes to absolute black/white + globals.

    Input ends in [H,W,2*history+4]. The four metadata/legality channels keep
    their native semantics. Older stacked boards are deliberately discarded:
    this model receives the actual sequence of observations separately.
    """
    x = np.asarray(features)
    if x.ndim < 3 or x.shape[-3] != x.shape[-2] or x.shape[-1] < 6 or x.shape[-1] % 2:
        raise ValueError('Invalid native board feature shape')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite native observation')
    black_to_play = x[..., -4]
    if not np.isin(black_to_play, [0, 1]).all():
        raise ValueError('Invalid side-to-play plane')
    black = np.where(black_to_play > .5, x[..., 0], x[..., 1])
    white = np.where(black_to_play > .5, x[..., 1], x[..., 0])
    return np.concatenate([black[..., None], white[..., None], x[..., -4:]], -1).astype(np.float32)


def validate_inputs(observations, actions, counts, c):
    o, a, n = map(np.asarray, (observations, actions, counts))
    if o.ndim != 5 or o.shape[2] != o.shape[3] or o.shape[4] != 6:
        raise ValueError('Expected six-channel square observation sequences')
    b, t, size, _, _ = o.shape
    if not 1 <= size <= c['max_board_size'] or not 1 <= t <= c['max_positions']:
        raise ValueError('Board or sequence exceeds model contract')
    if a.shape != (b, t) or n.shape != (b,) or a.dtype.kind not in 'iu' or n.dtype.kind not in 'iu':
        raise ValueError('Expected integral action IDs and observation counts')
    if np.any(n < 1) or np.any(n > t) or np.any(a > size * size) or np.any(a < 0):
        raise ValueError('Invalid lengths or action IDs (pass = board area)')
    if not np.isfinite(o).all() or not np.isin(o[..., [0, 1, 2, 5]], [0, 1]).all():
        raise ValueError('Invalid stones, side or legality planes')
    if np.any(o[..., 0] + o[..., 1] > 1):
        raise ValueError('Two stones occupy one point')
    if np.any((o[..., 4] < 0) | (o[..., 4] > 1)):
        raise ValueError('Pass-count plane out of range')
    for index in [2, 3, 4]:
        if not np.all(o[..., index] == o[:, :, :1, :1, index]):
            raise ValueError('Global plane is not spatially constant')
    return o.astype(np.float32), a.astype(np.int32), n.astype(np.int32)
