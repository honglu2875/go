"""Exact visual histories from one GIL-free Rust replay call per packed batch."""
import hashlib
import json
import numpy as np


def observation_sha256(observation):
    return hashlib.sha256(np.asarray(observation, dtype='<f4').tobytes(order='C')).hexdigest()


class Replay:
    def __init__(self, native, rules, max_positions):
        if getattr(native, 'OBSERVATION_REPLAY_ABI_VERSION', None) != 1:
            raise ValueError('Exact visual inference needs observation replay ABI 1')
        if set(rules) != {'size', 'komi', 'scoring'} or type(max_positions) is not int or max_positions < 1:
            raise ValueError('Invalid visual replay contract')
        self.native, self.rules, self.max_positions = native, dict(rules), max_positions
        self.config = json.dumps(rules, sort_keys=True, allow_nan=False)
        self.size = rules['size']; self.area = self.size ** 2

    def validate(self, history):
        if (not isinstance(history, (list, tuple)) or len(history) >= self.max_positions
                or any(type(a) is not int or not 0 <= a <= self.area for a in history)):
            raise ValueError('Invalid history or complete context would overflow')
        return tuple(history)

    def __call__(self, histories):
        return self._replay(histories, None)

    def suffix(self, histories, starts):
        """Validate the full native tape while returning only changed board rows."""
        if getattr(self.native, 'OBSERVATION_SUFFIX_REPLAY_ABI_VERSION', None) != 1:
            raise ValueError('Suffix replay needs its explicitly qualified native capability')
        if (not isinstance(starts, (list, tuple)) or len(starts) != len(histories)
                or any(type(s) is not int or not 0 <= s <= len(h) for s, h in zip(starts, histories))):
            raise ValueError('Invalid suffix offsets')
        return self._replay(histories, starts)

    def _replay(self, histories, starts):
        histories = [self.validate(h) for h in histories]
        if not histories:
            return []
        # The extra pass supplies the pending leaf's pre-action observation.
        # A terminal leaf is rejected by Rust; search must score it directly.
        lengths = np.asarray([len(h) + 1 for h in histories], np.int64)
        offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
        actions = np.concatenate([np.asarray((*h, self.area), np.int32) for h in histories])
        if starts is None:
            stones, legal, _ = self.native.replay_observations(self.config, actions, offsets)
            starts = [0] * len(histories)
        else:
            stones, legal, _ = self.native.replay_suffix_observations(self.config, actions, offsets, np.asarray(starts, np.int64))
        offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths - np.asarray(starts))])
        stones = stones.reshape(-1, self.size, self.size)
        legal = legal.reshape(-1, self.area + 1)
        rows = []
        for h, start, begin, end in zip(histories, starts, offsets[:-1], offsets[1:]):
            n = len(h) + 1 - start; black = np.arange(start, start + n) % 2 == 0
            o = np.empty((n, self.size, self.size, 6), np.float32)
            o[..., 0] = stones[begin:end] == 1; o[..., 1] = stones[begin:end] == 2
            o[..., 2] = black[:, None, None]
            o[..., 3] = (np.where(black, -self.rules['komi'], self.rules['komi']) / self.area)[:, None, None]
            o[..., 4] = np.asarray([False, *(a == self.area for a in h)], np.float32)[start:, None, None] / 2.
            o[..., 5] = legal[begin:end, :-1].reshape(n, self.size, self.size)
            rows.append(o)
        return rows
