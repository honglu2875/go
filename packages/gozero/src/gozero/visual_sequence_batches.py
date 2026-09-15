"""Whole exact observation/action histories from pinned native replay overlays."""
from pathlib import Path
import numpy as np

from .checkpoints import sha256
from .sequence_batches import Dataset as HistoryDataset
from .snapshots import read_json


class Dataset(HistoryDataset):
    def __init__(self, directory, expected_sha256, *, rank=0, world=1):
        directory = Path(directory).resolve(strict=True)
        if sha256(directory / 'manifest.json') != expected_sha256:
            raise ValueError('Visual observation dataset identity differs')
        overlay = read_json(directory / 'manifest.json')
        if overlay['kind'] != 'visual_causal_teacher_dataset':
            raise ValueError('Expected exact visual observation overlay')
        base = overlay['parent_dataset']
        super().__init__(base['path'], base['manifest_sha256'], rank=rank, world=world)
        parent = self.manifest
        self.parent_manifest, self.manifest, self.directory = parent, overlay, directory
        original = [s for s in parent['shards'] if s['id'] % world == rank]
        selected = [s for s in overlay['shards'] if s['id'] % world == rank]
        if [s['id'] for s in original] != [s['id'] for s in selected]:
            raise ValueError('Visual shard coverage differs')
        self.komi = overlay['rules']['komi']
        for raw, visual, arrays in zip(original, selected, self.shards):
            path = directory / visual['arrays']
            if (Path(visual['arrays']).name != visual['arrays'] or sha256(path) != visual['sha256']
                    or visual['parent_arrays_sha256'] != raw['sha256']):
                raise ValueError('Visual shard bytes or parent differ')
            with np.load(path, allow_pickle=False) as saved:
                expected = {role + '_' + key for role in ('expert', 'behavior') for key in ('stones', 'legal', 'passes')}
                if set(saved.files) != expected:
                    raise ValueError('Unexpected visual replay fields')
                for role in ('expert', 'behavior'):
                    n = len(arrays[role + '_actions'])
                    stones, legal, passes = (saved[role + '_' + key] for key in ('stones', 'legal', 'passes'))
                    if (stones.dtype != np.uint8 or stones.shape != (n, self.size ** 2)
                            or not np.isin(stones, [0, 1, 2]).all() or legal.dtype != np.bool_
                            or legal.shape != (n, self.size ** 2 + 1) or not legal[:, -1].all()
                            or passes.dtype != np.uint8 or passes.shape != (n,) or np.any(passes > 1)):
                        raise ValueError('Invalid exact pre-action observation fields')
                    if not legal[np.arange(n), arrays[role + '_actions']].all():
                        raise ValueError('Observed move is illegal in its own input')
                    if role == 'expert' and not np.array_equal(legal, arrays['expert_legal']):
                        raise ValueError('Expert MCTS legality changed')
                    arrays.update({role + '_stones': stones, role + '_legal': legal, role + '_passes': passes})

    def batch(self, entries, *, positions=None, loss_weights=(1., 1., 1., 0., 0.)):
        """Keep every prior board and action; padding never creates target rows."""
        if len(loss_weights) != 5 or not np.isfinite(loss_weights).all() or min(loss_weights) < 0:
            raise ValueError('Invalid explicit loss weights')
        counts = np.asarray([0 if e is None else int(np.diff(self.shards[e[1]][e[0] + '_offsets'][e[2]:e[2]+2])[0]) for e in entries], np.int32)
        time = int(counts.max(initial=1)) if positions is None else positions
        if not 1 <= time or np.any(counts > time):
            raise ValueError('Batch bucket would truncate a complete history')
        b, size = len(entries), self.size
        shape = (b, time)
        result = {'observations': np.zeros((*shape, size, size, 6), np.float32),
                  'actions': np.zeros(shape, np.int32), 'counts': counts,
                  'expert_policies': np.zeros((*shape, size * size + 1), np.float32),
                  'outcomes': np.zeros(shape, np.float32), 'scores': np.zeros(shape, np.float32),
                  'ownership': np.zeros((*shape, size, size), np.float32),
                  'loss_weights': np.asarray(loss_weights, np.float32)}
        for role in ('expert', 'behavior', 'value', 'score', 'ownership'):
            result[role + '_mask'] = np.zeros(shape, np.float32)
        for row, entry in enumerate(entries):
            if entry is None:
                continue
            role, shard, episode = entry; arrays = self.shards[shard]
            begin, end = map(int, arrays[role + '_offsets'][episode:episode+2]); n = end - begin
            stones = arrays[role + '_stones'][begin:end].reshape(n, size, size)
            o = result['observations'][row, :n]
            o[..., 0] = stones == 1; o[..., 1] = stones == 2
            black = np.arange(n) % 2 == 0
            o[..., 2] = black[:, None, None]
            # Native signed komi is positive for White, negative for Black.
            o[..., 3] = np.where(black, -self.komi, self.komi)[:, None, None] / (size * size)
            o[..., 4] = arrays[role + '_passes'][begin:end, None, None] / 2.
            o[..., 5] = arrays[role + '_legal'][begin:end, :-1].reshape(n, size, size)
            result['actions'][row, :n] = arrays[role + '_actions'][begin:end]
            result[role + '_mask'][row, :n] = 1
            if role == 'expert':
                result['expert_policies'][row, :n] = arrays['expert_policies'][begin:end]
                result['outcomes'][row, :n] = arrays['expert_values'][begin:end]
                result['value_mask'][row, :n] = 1
        return result

    def bucket_entries(self, buckets, split=0):
        if tuple(sorted(set(buckets))) != tuple(buckets) or not buckets or buckets[-1] < self.time:
            raise ValueError('Buckets must cover every complete history')
        result = {(role, bucket): [] for role in ('expert', 'behavior') for bucket in buckets}
        for role in ('expert', 'behavior'):
            for shard, game in self.indices[role, split]:
                n = int(np.diff(self.shards[shard][role + '_offsets'][game:game+2])[0])
                bucket = next(b for b in buckets if b >= n)
                result[role, bucket].append((role, shard, game))
        return result
