"""A shuffled, exact multiset of the existing CNN's position/D4 exposures."""
import hashlib
import math
import numpy as np
from gozero.snapshots import canonical_json
from gozero.sequence_symmetry import transform_grid, action_map


def digest(array):
    a = np.asarray(array)
    h = hashlib.sha256(canonical_json([list(a.shape), str(a.dtype)]))
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def build(data, reference, shuffle_seed, expected_positions):
    world = reference['expected_processes']; seed = reference['seed']
    buckets = reference['dataset']['buckets']; entries = data.bucket_entries(buckets)
    games = [np.random.Generator(np.random.PCG64(seed + 1 + 104729 * rank)) for rank in range(world)]
    symmetries = [np.random.Generator(np.random.PCG64(seed + 400003 + 104729 * rank)) for rank in range(world)]
    bucket_rng = np.random.Generator(np.random.PCG64(seed + 9143))
    records = np.empty((expected_positions, 3), np.int32)
    draws = [[] for _ in range(world)]; milestones = []; offset = 0
    for turn in range(reference['steps']):
        warm = reference['dataset']['warmup_buckets']
        bucket = warm[turn] if turn < len(warm) else int(bucket_rng.choice(buckets, p=reference['dataset']['bucket_probabilities']))
        for rank in range(world):
            chosen = [entries['expert', bucket][int(i)] for i in games[rank].integers(
                len(entries['expert', bucket]), size=reference['learner']['games_per_host'])]
            codes = symmetries[rank].integers(0, 8, len(chosen)).tolist()
            draws[rank].append({'turn': turn + 1, 'bucket': bucket,
                'local_entries_sha256': hashlib.sha256(canonical_json(chosen)).hexdigest(),
                'local_symmetries': codes})
            for (_, shard, episode), code in zip(chosen, codes):
                begin, end = map(int, data.shards[shard]['expert_offsets'][episode:episode + 2]); n = end - begin
                if offset + n > expected_positions: raise ValueError('Exposure reference exceeded its registered count')
                records[offset:offset+n, 0] = shard
                records[offset:offset+n, 1] = np.arange(begin, end, dtype=np.int32)
                records[offset:offset+n, 2] = code
                offset += n
        if (turn + 1) % reference['eval_every'] == 0 or turn + 1 == reference['steps']:
            milestones.append({'reference_turn': turn + 1, 'positions': offset})
    if offset != expected_positions: raise ValueError('Exposure reference count differs')
    original_digest = digest(records)
    permutation = np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(offset)
    shuffled = records[permutation]
    metadata = {'positions': offset, 'original_records_sha256': original_digest,
                'permutation_sha256': digest(permutation), 'shuffled_records_sha256': digest(shuffled),
                'shuffle_seed': shuffle_seed, 'milestones': milestones,
                'reference_draws_sha256': hashlib.sha256(canonical_json(draws)).hexdigest()}
    return shuffled, metadata, draws


def batch(data, records, start_update, updates, batch_size, world, rank):
    if batch_size % world or not 0 <= rank < world: raise ValueError('Uneven position batch')
    local = batch_size // world; size = data.size
    ids = np.arange(start_update * batch_size, (start_update + updates) * batch_size, dtype=np.int64)
    ids = ids.reshape(updates, world, local)[:, rank, :].reshape(-1)
    valid = ids < len(records); selected = records[np.minimum(ids, len(records) - 1)]
    n = len(ids)
    result = {'spatial': np.zeros((n, 1, size, size, 22), np.float32),
              'global_features': np.zeros((n, 1, 19), np.float32),
              'actions': np.zeros((n, 1), np.int32), 'counts': valid.astype(np.int32),
              'policies': np.zeros((n, 1, size*size+1), np.float32),
              'legal': np.ones((n, 1, size*size+1), bool)}
    for shard in np.unique(selected[valid, 0]):
        rows = np.flatnonzero(valid & (selected[:, 0] == shard)); indices = selected[rows, 1]
        arrays = data.shards[int(shard)]
        for name, key in [('spatial', 'spatial'), ('global_features', 'global_features'),
                          ('actions', 'expert_actions'), ('policies', 'expert_policies'), ('legal', 'expert_legal')]:
            result[name][rows, 0] = arrays[key][indices]
    for code in range(1, 8):
        rows = np.flatnonzero(valid & (selected[:, 2] == code))
        if not len(rows): continue
        result['spatial'][rows, 0] = transform_grid(result['spatial'][rows, 0], code, (1, 2))
        result['actions'][rows, 0] = action_map(size, code)[result['actions'][rows, 0]]
        for key in ('policies', 'legal'):
            result[key][rows, 0, :-1] = transform_grid(result[key][rows, 0, :-1].reshape(-1, size, size), code, (1, 2)).reshape(-1, size*size)
    return {k: v.reshape(updates, local, *v.shape[1:]) for k, v in result.items()}


def evaluation_steps(metadata, batch_size):
    return [{**m, 'sgd_step': math.ceil(m['positions'] / batch_size)} for m in metadata['milestones']]
