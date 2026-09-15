"""Join immutable terminal MCTS replay with its exact causal game histories.

Expert targets come from saved search policies. Behavior targets are observed
moves sampled independently from the archive, including capped games. No
outcome is assigned to a capped game and no future action is an input token.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import checkpoints
from .game_archives import open_records
from .snapshots import read_json


def split_game(snapshot_id, game_id):
    """Whole-game 80/10/10 split shared by expert and behavior examples."""
    digest = hashlib.sha256(f'{snapshot_id}:{game_id:016x}'.encode()).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 10000
    return 0 if bucket < 8000 else 1 if bucket < 9000 else 2


def validate_game(game, *, game_id, size, komi, scoring):
    if (game['game_id'] != game_id or game['size'] != size or game['komi'] != komi
            or game['scoring'] != scoring or type(game['truncated']) is not bool
            or not game['actions'] or len(game['actions']) != len(game['networks'])):
        raise ValueError('Game identity or history differs')
    if any(type(a) is not int or not 0 <= a <= size * size for a in game['actions']):
        raise ValueError('Invalid observed action')
    if any(type(n) is not int or not 0 <= n < 2**64 for n in game['networks']):
        raise ValueError('Invalid game network version')
    score = game['white_score']
    if game['truncated']:
        if score is not None:
            raise ValueError('Capped game has an assigned outcome')
    elif (type(score) not in (float, int) or not math.isfinite(score)
          or not -size * size + komi <= score <= size * size + komi
          or game['actions'][-2:] != [size * size, size * size]):
        raise ValueError('Completed game lacks a valid double-pass outcome')


def validate_teacher(game, features, policies, outcomes, metadata, *, size, simulations, allow_suffix=False):
    """Validate a whole-game block; an evicted first prefix can be excluded."""
    rows, actions = len(metadata), size * size + 1
    if (not rows or metadata.shape != (rows, 6) or policies.shape != (rows, actions)
            or outcomes.shape != (rows,) or features.ndim != 4 or features.shape[-1] < 1
            or features.shape[:3] != (rows, size, size)
            or metadata.dtype != np.uint64 or policies.dtype != np.float32
            or outcomes.dtype != np.float32 or features.dtype != np.float32
            or game['truncated'] or rows > len(game['actions'])):
        raise ValueError('Expert block shape, type or terminal status differs')
    full = rows == len(game['actions'])
    if not full and not allow_suffix:
        raise ValueError('Partial expert game is not the overwritten ring prefix')
    start = len(game['actions']) - rows
    if (not np.all(metadata[:, 0] == game['game_id'])
            or metadata[:, 1].tolist() != game['networks'][start:]
            or metadata[:, 2].tolist() != game['actions'][start:]
            or not np.all(metadata[:, 3] == simulations)):
        raise ValueError('Expert rows do not align with exact observed game plies')
    if not all(np.isfinite(a).all() for a in (features, policies, outcomes)) or np.any(policies < 0):
        raise ValueError('Invalid expert observation or policy value')
    spatial_legal = features[..., -1].reshape(rows, -1)
    if not np.isin(spatial_legal, [0., 1.]).all():
        raise ValueError('Expert legality plane is not binary')
    legal = np.concatenate((spatial_legal > .5, np.ones((rows, 1), bool)), axis=1)
    if (not np.all(legal[np.arange(rows), metadata[:, 2].astype(np.int64)])
            or not np.allclose(policies.sum(axis=1), 1., rtol=0, atol=2e-5)
            or np.any(policies[~legal] > 1e-8)):
        raise ValueError('Expert policy normalization or legality differs')
    white_value = np.sign(game['white_score'])
    expected = np.where((np.arange(rows) + start) % 2 == 0, -white_value, white_value)
    if not np.array_equal(outcomes, expected):
        raise ValueError('Expert value labels differ from terminal side-to-move outcomes')
    return full, legal


def extract_shard(root, reference, *, behavior_games, behavior_seed):
    """Return packed arrays and provenance; caller publishes after verification."""
    root = Path(root).resolve()
    checkpoint = (root / reference['checkpoint']).resolve(strict=True)
    if not checkpoint.is_relative_to(root / 'runs'):
        raise ValueError('Checkpoint escapes run artifacts')
    if checkpoints.sha256(checkpoint.with_suffix('.group.json')) != reference['group_sha256']:
        raise ValueError('Checkpoint group identity differs')
    state, arrays, _ = checkpoints.read(checkpoint, expected_manifest_sha256=reference['manifest_sha256'])
    group = read_json(checkpoint.with_suffix('.group.json'))
    source = root / '.gozero/snapshots' / reference['training_snapshot']
    from .snapshots import verify
    verify(source)
    config = read_json(source / 'resolved_config.json')
    if (state['snapshot_id'] != reference['training_snapshot'] or group['snapshot_id'] != state['snapshot_id']
            or state['turn'] != group['turn'] or state['turn'] != config['selfplay_turns']
            or state['world_size'] != group['world_size'] or not 0 <= state['jax_rank'] < state['world_size']
            or group['rank_manifests'][state['jax_rank']] != reference['manifest_sha256']
            or group['updates'] != state['counters']['updates']
            or group['config_sha256'] != state['config_sha256']
            or checkpoints.sha256(source / 'resolved_config.json') != state['config_sha256']):
        raise ValueError('Final checkpoint scientific identity differs')
    count, cursor, capacity = state['replay_count'], state['replay_cursor'], config['learner']['replay_capacity']
    if not 0 < count <= capacity or not 0 <= cursor < capacity or count < capacity and cursor != count:
        raise ValueError('Invalid replay ring state')
    for name in ('replay_x', 'replay_pi', 'replay_z', 'replay_meta'):
        if len(arrays[name]) != count:
            raise ValueError('Replay ring array length differs')
    order = np.arange(count) if count < capacity else (cursor + np.arange(count)) % capacity
    metadata = arrays['replay_meta'][order]
    cuts = np.r_[0, 1 + np.flatnonzero(metadata[1:, 0] != metadata[:-1, 0]), count]
    ids = metadata[cuts[:-1], 0]
    if len(set(map(int, ids))) != len(ids) or np.any(metadata[:, 1] > state['counters']['updates']):
        raise ValueError('Duplicate game blocks or future model versions in replay')
    actor = config['actors']
    artifacts = checkpoint.parents[1]
    teacher_chunks, teacher_offsets, teacher_ids, teacher_splits = [], [0], [], []
    behavior_actions, behavior_offsets, behavior_ids, behavior_splits = [], [0], [], []
    behavior_capped, behavior_versions = [], []
    records, excluded = {}, []
    with open_records(artifacts, root / '.gozero/game-archives') as archive:
        if (archive.archive_id != reference['archive_id'] or archive.training_snapshot != state['snapshot_id']
                or archive.turn != state['turn']):
            raise ValueError('Archive and checkpoint lineage differ')
        def game_record(game_id):
            name = f'{game_id:016x}.json'
            game = json.loads(archive.read(name))
            validate_game(game, game_id=game_id, size=actor['size'], komi=actor['komi'], scoring=actor['scoring'])
            actor_id = game_id >> 32
            if (not state['jax_rank'] * actor['games'] <= actor_id < (state['jax_rank'] + 1) * actor['games']
                    or len(game['actions']) > actor['max_game_moves']
                    or max(game['networks']) > state['counters']['updates']):
                raise ValueError('Archived game actor, length or network version differs')
            records[name] = archive.member_sha256(name)
            return game
        for block, (begin, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            game_id = int(metadata[begin, 0])
            game = game_record(game_id)
            selected = order[begin:end]
            full, legal = validate_teacher(game, arrays['replay_x'][selected], arrays['replay_pi'][selected],
                                           arrays['replay_z'][selected], metadata[begin:end], size=actor['size'],
                                           simulations=actor['simulations'], allow_suffix=block == 0 and count == capacity)
            if not full:
                excluded.append({'game_id': game_id, 'available_suffix_rows': int(end - begin),
                                 'full_game_rows': len(game['actions']), 'reason': 'overwritten replay prefix'})
                continue
            teacher_chunks.append((np.asarray(game['actions'], np.int32), np.asarray(game['networks'], np.uint64),
                                   arrays['replay_pi'][selected], legal, arrays['replay_z'][selected]))
            teacher_offsets.append(teacher_offsets[-1] + len(selected))
            teacher_ids.append(game_id)
            teacher_splits.append(split_game(state['snapshot_id'], game_id))
        choices = sorted(n for n in archive.names if n.endswith('.json'))
        if type(behavior_games) is not int or not 1 <= behavior_games <= min(16384, len(choices)):
            raise ValueError('Behavior sample size exceeds bounded archive membership')
        random = np.random.Generator(np.random.PCG64(behavior_seed))
        names = [choices[i] for i in sorted(random.choice(len(choices), behavior_games, replace=False))]
        for name in names:
            game_id = int(name[:16], 16)
            game = game_record(game_id)
            behavior_actions.extend(game['actions'])
            behavior_versions.extend(game['networks'])
            behavior_offsets.append(len(behavior_actions))
            behavior_ids.append(game_id)
            behavior_splits.append(split_game(state['snapshot_id'], game_id))
            behavior_capped.append(game['truncated'])
        archive_identity = {'archive_id': archive.archive_id, 'training_result_sha256': archive.training_result_sha256}
    if not teacher_chunks:
        raise ValueError('No complete expert game blocks remain')
    packed = {name: np.concatenate([chunk[i] for chunk in teacher_chunks], axis=0)
              for i, name in enumerate(('expert_actions', 'expert_networks', 'expert_policies', 'expert_legal', 'expert_values'))}
    packed.update(expert_offsets=np.asarray(teacher_offsets, np.int64), expert_game_ids=np.asarray(teacher_ids, np.uint64),
                  expert_splits=np.asarray(teacher_splits, np.uint8),
                  behavior_actions=np.asarray(behavior_actions, np.int32), behavior_networks=np.asarray(behavior_versions, np.uint64),
                  behavior_offsets=np.asarray(behavior_offsets, np.int64), behavior_game_ids=np.asarray(behavior_ids, np.uint64),
                  behavior_splits=np.asarray(behavior_splits, np.uint8), behavior_capped=np.asarray(behavior_capped, bool))
    evidence = {'reference': reference, **archive_identity, 'size': actor['size'], 'komi': actor['komi'],
                'scoring': actor['scoring'], 'simulations': actor['simulations'],
                'replay_rows': count, 'expert_games': len(teacher_ids), 'expert_rows': teacher_offsets[-1],
                'excluded_expert_suffixes': excluded, 'behavior_games': behavior_games,
                'behavior_rows': len(behavior_actions), 'behavior_capped_games': sum(behavior_capped),
                'behavior_seed': behavior_seed, 'game_json_sha256': records,
                'split_contract': 'SHA256(source_snapshot:16hex_game_id), first8 bytes big endian mod10000; train<8000, validation<9000, test otherwise.',
                'split_counts': {role: {str(split): int(np.count_nonzero(packed[role + '_splits'] == split)) for split in (0, 1, 2)}
                                 for role in ('expert', 'behavior')},
                'behavior_outcome_labels_created': False, 'katago_data_used': False}
    verify(source)
    return packed, evidence
