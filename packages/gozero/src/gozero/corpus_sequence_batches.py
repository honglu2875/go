"""Whole-history batches over lossless, memory-mapped strong-teacher corpora."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256
from .corpus_format import KIND, GAME_DTYPE, array_specs, unpack_spatial, unpack_legal
from .katago_sequence_batches import augment


class Dataset:
    def __init__(self, directory, expected_sha256, *, allow_partial=False):
        self.directory = Path(directory).resolve(strict=True)
        if sha256(self.directory/'manifest.json') != expected_sha256:
            raise ValueError('Corpus manifest changed')
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        m = self.manifest
        if (m['kind'] != KIND or m['schema_version'] != 1 or m['targets_changed']
                or m['feature_version'] != 7 or m['spatial_channels'] != 22 or m['global_channels'] != 19
                or m['test_arrays_included'] or (not m['complete_train_validation'] and not allow_partial)):
            raise ValueError('Unexpected or incomplete teacher corpus')
        self.size, self.time = m['size'], m['max_game_moves']
        if self.size not in (9, 19) or not 1 <= self.time <= 4*self.size*self.size:
            raise ValueError('Invalid board/context size')
        self.shards = []; self.indices = {('expert',split):[] for split in range(3)}
        ids, families = set(), {}
        counts, positions = Counter(), Counter()
        for number, item in enumerate(m['shards']):
            if item['id'] != number or not item['all_boards_equal'] or not item['all_legal_masks_equal'] or item['targets_changed']:
                raise ValueError('Unaudited feature shard')
            spec = array_specs(self.size,item['positions'],item['games'])
            if set(item['files']) != set(spec):
                raise ValueError('Incomplete packed arrays')
            arrays = {}
            for name, (shape, dtype) in spec.items():
                record = item['files'][name]; relative = Path(record['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Shard path escapes corpus')
                path = self.directory/relative
                if path.is_symlink() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                    raise ValueError('Shard bytes changed')
                array = np.load(path,mmap_mode='r',allow_pickle=False)
                if array.shape != shape or array.dtype != dtype or array.dtype.hasobject:
                    raise ValueError('Packed array schema differs')
                arrays[name] = array
            offsets = arrays['expert_offsets']; games = arrays['games']
            lengths = np.diff(offsets)
            if (offsets[0] != 0 or offsets[-1] != item['positions'] or np.any(lengths < 1)
                    or np.any(lengths > self.time) or not np.array_equal(lengths,games['length'])
                    or not np.isin(games['split'],[0,1]).all() or not np.isin(games['expert_color'],[1,2]).all()
                    or np.any(games['opponent'] > 7)):
                raise ValueError('Invalid complete-game metadata')
            for episode, game in enumerate(games):
                identity, family, split = bytes(game['game_id']), bytes(game['opening_family']), int(game['split'])
                if identity in ids or len(identity) != 64 or len(family) != 64:
                    raise ValueError('Invalid or duplicated record identity')
                ids.add(identity)
                if families.setdefault(family,split) != split:
                    raise ValueError('Opening family crosses splits')
                self.indices['expert',split].append((number,episode))
                counts[split] += 1; positions[split] += int(game['length'])
            self.shards.append(arrays)
        for name, split in (('train',0),('validation',1)):
            if m['populations'][name] != dict(games=counts[split],positions=positions[split]):
                raise ValueError('Manifest population differs')

    def game_info(self, entry):
        role, shard, episode = entry
        if role != 'expert':
            raise ValueError('Only fixed-teacher targets are defined')
        return self.shards[shard]['games'][episode]

    def bucket_entries(self, buckets, split=0):
        if list(buckets) != sorted(set(buckets)) or not buckets or buckets[-1] < self.time:
            raise ValueError('Buckets must cover complete histories')
        result = {('expert',b):[] for b in buckets}
        for shard, episode in self.indices['expert',split]:
            length = int(self.shards[shard]['games'][episode]['length'])
            bucket = next(b for b in buckets if b >= length)
            result['expert',bucket].append(('expert',shard,episode))
        return result

    def evaluation_family_weights(self, entries):
        """Weights giving each represented opening family equal total position mass."""
        totals = Counter()
        for entry in entries:
            info = self.game_info(entry)
            totals[bytes(info['opening_family'])] += int(info['length'])
        return {tuple(entry): 1./totals[bytes(self.game_info(entry)['opening_family'])] for entry in entries}

    def batch(self, entries, *, positions, family_weights=None, loss_weights=None):
        if loss_weights is not None:
            raise ValueError('Training uses one fixed teacher population')
        b,t,s = len(entries),positions,self.size
        result = dict(spatial=np.zeros((b,t,s,s,22),np.float32),global_features=np.zeros((b,t,19),np.float32),
            actions=np.zeros((b,t),np.int32),counts=np.zeros(b,np.int32),policies=np.zeros((b,t,s*s+1),np.float32),
            legal=np.ones((b,t,s*s+1),bool),values=np.zeros((b,t),np.float32),
            opponent=np.full(b,-1,np.int32),family_weights=np.zeros(b,np.float32))
        for row,entry in enumerate(entries):
            if entry is None:
                continue
            role,shard,episode = entry
            if role != 'expert':
                raise ValueError('No behavior-policy target is registered')
            arrays = self.shards[shard]; start,end = map(int,arrays['expert_offsets'][episode:episode+2]);n=end-start
            if n>t:
                raise ValueError('Would truncate a complete game history')
            result['counts'][row]=n
            result['spatial'][row,:n]=unpack_spatial(arrays['spatial'][start:end],s)
            result['legal'][row,:n]=unpack_legal(arrays['legal'][start:end],s)
            for name in ('global_features','actions','policies','values'):
                result[name][row,:n]=arrays[name][start:end]
            result['opponent'][row]=int(arrays['games'][episode]['opponent'])
            result['family_weights'][row]=1. if family_weights is None else family_weights[tuple(entry)]
        return result
