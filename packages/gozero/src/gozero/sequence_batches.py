"""Deterministic whole-history batches with separate target populations."""
from pathlib import Path

import numpy as np

from .checkpoints import sha256
from .snapshots import read_json


class Dataset:
    def __init__(self, directory, expected_sha256, *, rank=0, world=1):
        self.directory = Path(directory).resolve(strict=True)
        if sha256(self.directory / 'manifest.json') != expected_sha256:
            raise ValueError('Causal dataset identity differs')
        self.manifest = read_json(self.directory / 'manifest.json')
        if (self.manifest['kind'] != 'causal_teacher_dataset' or not 0 <= rank < world
                or len(self.manifest['shards']) % world):
            raise ValueError('Dataset topology differs')
        spec = self.manifest['spec']
        self.size, self.time = spec['size'], spec['max_game_moves']
        self.shards = []
        self.indices = {(role, split): [] for role in ('expert', 'behavior') for split in range(3)}
        for item in self.manifest['shards']:
            if item['id'] % world != rank:
                continue
            for field, digest in [('arrays', 'sha256'), ('evidence', 'evidence_sha256')]:
                name = item[field]
                if Path(name).name != name or sha256(self.directory / name) != item[digest]:
                    raise ValueError('Dataset shard integrity differs')
            with np.load(self.directory / item['arrays'], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            shard = len(self.shards); self.shards.append(arrays)
            for role in ('expert', 'behavior'):
                offsets, splits = (arrays[role + suffix] for suffix in ('_offsets', '_splits'))
                if (len(offsets) != len(splits) + 1 or offsets[0] != 0 or offsets[-1] != len(arrays[role + '_actions'])
                        or np.any(np.diff(offsets) < 1) or np.any(np.diff(offsets) > self.time)
                        or not np.isin(splits, [0, 1, 2]).all()):
                    raise ValueError('Packed episode offsets or splits differ')
                eligible = arrays.get(role + '_training_eligible', np.ones(len(splits), np.bool_))
                if eligible.dtype != np.bool_ or eligible.shape != splits.shape:
                    raise ValueError('Training eligibility must be one boolean per episode')
                for split in range(3):
                    selected = splits == split
                    if split == 0:
                        selected = selected & eligible
                    self.indices[role, split].extend((shard, int(i)) for i in np.flatnonzero(selected))
        if any(not v for v in self.indices.values()):
            raise ValueError('Every local population needs train, validation and test episodes')

    def batch(self, entries):
        """Entries are (role, shard, episode) or None padding; targets never enter inputs."""
        count, time, actions = len(entries), self.time, self.size**2 + 1
        batch = {'tokens': np.full((count, time), actions + 1, np.int32), 'lengths': np.zeros(count, np.int32),
                 'policies': np.zeros((count, time, actions), np.float32), 'observed': np.zeros((count, time), np.int32),
                 'values': np.zeros((count, time), np.float32), 'legal': np.ones((count, time, actions), bool)}
        for name in ('expert_mask', 'behavior_mask', 'value_mask'):
            batch[name] = np.zeros((count, time), np.float32)
        batch['tokens'][:, 0] = actions  # BOS; PAD is actions+1.
        for row, entry in enumerate(entries):
            if entry is None:
                continue
            role, shard, game = entry
            data = self.shards[shard]; start, end = data[role + '_offsets'][game:game+2]
            moves = data[role + '_actions'][start:end]; n = len(moves)
            batch['tokens'][row, 1:n] = moves[:-1]
            batch['lengths'][row] = n - 1
            batch['observed'][row, :n] = moves
            batch[role + '_mask'][row, :n] = 1
            if role == 'expert':
                batch['policies'][row, :n] = data['expert_policies'][start:end]
                batch['values'][row, :n] = data['expert_values'][start:end]
                batch['legal'][row, :n] = data['expert_legal'][start:end]
                batch['value_mask'][row, :n] = 1
        return batch

    def sample(self, random, games_per_role):
        entries = []
        for role in ('expert', 'behavior'):
            indices = self.indices[role, 0]
            entries.extend((role, *indices[int(i)]) for i in random.integers(len(indices), size=games_per_role))
        return self.batch(entries)

    def evaluation_entries(self, split):
        if split not in (1, 2):
            raise ValueError('Evaluation requires a held-out split')
        return [(role, *index) for role in ('expert', 'behavior') for index in self.indices[role, split]]
