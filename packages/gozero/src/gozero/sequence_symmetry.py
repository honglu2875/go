"""One Go-board symmetry per complete causal history and all its spatial targets."""
import numpy as np


def transform_grid(array, symmetry, axes):
    if type(symmetry) is not int or not 0 <= symmetry < 8:
        raise ValueError('Expected one of eight square-board symmetries')
    transformed = np.rot90(array, symmetry % 4, axes=axes)
    return np.flip(transformed, axis=axes[1]) if symmetry // 4 else transformed


def action_map(size, symmetry):
    old_at_new = transform_grid(np.arange(size * size).reshape(size, size), symmetry, (0, 1)).reshape(-1)
    return np.concatenate([np.argsort(old_at_new), [size * size]]).astype(np.int32)


def augment(batch, symmetries):
    observations = batch['observations']; b, t, size, width, channels = observations.shape
    codes = np.asarray(symmetries)
    if (size != width or channels != 6 or codes.shape != (b,) or codes.dtype.kind not in 'iu'
            or np.any(codes < 0) or np.any(codes > 7) or batch['actions'].shape != (b, t)
            or np.any(batch['actions'] < 0) or np.any(batch['actions'] > size * size)):
        raise ValueError('Invalid whole-history symmetry dimensions')
    out = dict(batch)
    for field in ('observations', 'actions', 'expert_policies', 'ownership'):
        out[field] = np.empty_like(batch[field])
    for row, code in enumerate(codes):
        code = int(code); mapping = action_map(size, code)
        out['observations'][row] = transform_grid(observations[row], code, (1, 2))
        out['actions'][row] = mapping[batch['actions'][row]]
        out['expert_policies'][row, :, :-1] = transform_grid(
            batch['expert_policies'][row, :, :-1].reshape(t, size, size), code, (1, 2)).reshape(t, size * size)
        out['expert_policies'][row, :, -1] = batch['expert_policies'][row, :, -1]
        out['ownership'][row] = transform_grid(batch['ownership'][row], code, (1, 2))
    return out
