"""Deterministic synthetic tensor fixtures, explicitly not Go trajectories."""
import numpy as np


def inputs(seed, batch, positions, size):
    random = np.random.default_rng(seed)
    stones = random.integers(0, 3, (batch, positions, size, size))
    observations = np.zeros((*stones.shape, 6), np.float32)
    observations[..., 0] = stones == 1
    observations[..., 1] = stones == 2
    for t in range(positions):
        observations[:, t, :, :, 2] = 1 - t % 2
        observations[:, t, :, :, 3] = (1 if t % 2 else -1) * 7.5 / (size * size)
    observations[..., 5] = stones == 0
    actions = np.full((batch, positions), size * size, np.int32)
    counts = np.full(batch, positions, np.int32)
    return observations, actions, counts


def loss_batch(observations, actions, counts):
    b, t, size, _, _ = observations.shape
    policies = np.zeros((b, t, size * size + 1), np.float32)
    policies[..., -1] = 1.
    mask = np.ones((b, t), np.float32)
    return {'observations': observations, 'actions': actions, 'counts': counts,
            'expert_policies': policies, 'outcomes': np.zeros((b, t), np.float32),
            'scores': np.zeros((b, t), np.float32),
            'ownership': np.zeros((b, t, size, size), np.float32),
            'expert_mask': mask, 'behavior_mask': mask, 'value_mask': mask,
            'score_mask': mask, 'ownership_mask': mask,
            'loss_weights': np.array([1., 1., 1., 1., 1.5], np.float32)}
