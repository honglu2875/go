"""Immutable, pickle-free game records. Publication is atomic and retry-safe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from ..go import Game, GameConfig


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def opening_family(actions, size=9, plies=8):
    # D4-canonical prefixes keep paired/symmetric openings in the same split.
    variants = []
    for flip in (False, True):
        for turns in range(4):
            transformed = []
            for action in actions[:plies]:
                if action == size * size:
                    transformed.append(action)
                    continue
                row, col = divmod(int(action), size)
                if flip:
                    col = size - 1 - col
                for _ in range(turns):
                    row, col = col, size - 1 - row
                transformed.append(row * size + col)
            variants.append(transformed)
    return identity(min(variants))


def publish_game(directory: Path, metadata: dict, rows: list[dict], actions: list[int]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (metadata['game_id'] + '.npz')
    if target.exists():
        with np.load(target, allow_pickle=False) as record:
            old = json.loads(record['metadata'].tobytes())
            if old['game_id'] != metadata['game_id'] or old['contract_id'] != metadata['contract_id']:
                raise ValueError('Existing game identity mismatch')
        return target
    if len(rows) != len(actions) or not rows:
        raise ValueError('A game needs one label row per played action')
    family = opening_family(actions, size=metadata['board_size'])
    split_number = int(family[:8], 16) % 100
    metadata = {**metadata, 'schema_version': 2, 'rows': len(rows), 'opening_family': family,
                'split': 'test' if split_number < 5 else 'validation' if split_number < 10 else 'train'}
    columns = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    columns['actions'] = np.asarray(actions, dtype=np.int32)
    columns['metadata'] = np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), np.uint8)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **columns)
        audit_game(temporary)
        # link(2) cannot replace an already published game, even across workers.
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    return target


def audit_game(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        # Load once: repeatedly indexing NpzFile re-decompresses entire columns.
        data = {key:archive[key] for key in archive.files}
        meta = json.loads(data['metadata'].tobytes())
        actions = data['actions']
        count = len(actions)
        if count != meta['rows'] or count == 0:
            raise ValueError('Bad row count')
        size = meta['board_size']
        if size not in (9, 19) or meta['schema_version'] != 2:
            raise ValueError('Unsupported board size or schema')
        game = Game(GameConfig(size=size, komi=meta['komi']))
        for ply, action in enumerate(actions):
            legal = np.zeros(size * size + 1, dtype=bool)
            legal[game.legal()] = True
            if game.state().terminal or not legal[action]:
                raise ValueError('Invalid game history')
            if not np.array_equal(data['legal'][ply], legal):
                raise ValueError('Stored legal mask differs from full-history replay')
            if not np.array_equal(data['stones'][ply], game.state().stones):
                raise ValueError('Stored pre-action board differs from replay')
            policy = data['raw_policy'][ply]
            if (not np.isfinite(policy).all() or (policy < 0).any()
                    or np.any(policy[~legal]) or abs(float(policy.sum()) - 1) > 2e-6):
                raise ValueError('Invalid raw teacher policy')
            if data['teacher_visits'][ply] != (meta['visits'] if 1 + ply % 2 == meta['expert_color'] else 1):
                raise ValueError('Wrong teacher labeling budget')
            if data['search_policy_valid'][ply]:
                searched = data['search_policy'][ply]
                if (not np.isfinite(searched).all() or (searched < 0).any()
                        or np.any(searched[~legal]) or abs(float(searched.sum()) - 1) > 2e-6):
                    raise ValueError('Invalid searched teacher policy')
            game.play(1 + ply % 2, int(action))
        if not np.isfinite(data['raw_value']).all() or np.any(np.abs(data['raw_value']) > 1.00001):
            raise ValueError('Invalid raw teacher value')
        state = game.state()
        if state.terminal != meta['terminal'] or state.white_score != meta['white_score']:
            raise ValueError('Incorrect behavior outcome/truncation')
        if opening_family(actions.tolist(), size=size) != meta['opening_family']:
            raise ValueError('Incorrect split family')
        bucket = int(meta['opening_family'][:8], 16) % 100
        expected_split = 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'
        if meta['split'] != expected_split:
            raise ValueError('Incorrect split assignment')
        return meta
