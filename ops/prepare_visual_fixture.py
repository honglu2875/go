#!/usr/bin/env python3
"""Small legal-game fixture for learning/recovery mechanics, never strength."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.native import load_library
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-receipt', type=Path, required=True)
    p.add_argument('--expected-native-receipt-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE)
    if sha256(a.native_receipt) != a.expected_native_receipt_sha256:
        raise ValueError('Fixture replay binary differs')
    receipt = read_json(a.native_receipt)
    native = load_library(a.native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
    root = a.output.resolve(); base = root / 'base'; visual = root / 'visual'
    root.mkdir(parents=True, exist_ok=False); base.mkdir(); visual.mkdir()
    rules = {'size': 3, 'komi': .5, 'scoring': 'pass_alive_area'}
    arrays, overlay = {}, {}
    for role in ('expert', 'behavior'):
        games = [[0, 1, 9, 9], [0, 1, 4, 5, 9, 9], [8, 6, 9, 9], [8, 6, 4, 3, 9, 9]] * 3
        actions = np.asarray([x for g in games for x in g], np.int32)
        offsets = np.cumsum([0] + list(map(len, games)), dtype=np.int64)
        stones, legal, outcomes = native.replay_observations(json.dumps(rules), actions, offsets)
        legal = legal.reshape(len(actions), 10); outcomes = json.loads(outcomes)
        passes = np.zeros(len(actions), np.uint8)
        arrays.update({role + '_actions': actions, role + '_offsets': offsets,
                       role + '_splits': np.repeat(np.arange(3, dtype=np.uint8), 4),
                       role + '_game_ids': np.arange(12, dtype=np.uint64)})
        values = np.zeros(len(actions), np.float32)
        for i, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if not outcomes[i]['terminal']:
                raise ValueError('Fixture game did not terminate')
            passes[begin+1:end] = actions[begin:end-1] == 9
            values[begin:end] = np.where(np.arange(end-begin) % 2 == 0, -np.sign(outcomes[i]['white_score']), np.sign(outcomes[i]['white_score']))
        if role == 'expert':
            # Explicit arbitrary soft targets on real legal inputs, not MCTS.
            policies = legal.astype(np.float32); policies /= policies.sum(-1, keepdims=True)
            policies *= .25; policies[np.arange(len(actions)), actions] += .75
            arrays.update(expert_policies=policies, expert_values=values, expert_legal=legal)
        else:
            arrays['behavior_capped'] = np.zeros(len(games), np.bool_)
        overlay.update({role + '_stones': stones.reshape(len(actions), 9), role + '_legal': legal, role + '_passes': passes})
    np.savez_compressed(base / 'shard-00.npz', **arrays)
    (base / 'shard-00.json').write_bytes(canonical_json({'kind': 'qualification_fixture', 'claims_learning_quality': False}))
    base_manifest = {'schema_version': 1, 'kind': 'causal_teacher_dataset',
        'spec': {**rules, 'max_game_moves': 8, 'teacher_cost': {'kind': 'artificial qualification targets; zero teacher training'}},
        'shards': [{'id': 0, 'arrays': 'shard-00.npz', 'sha256': sha256(base / 'shard-00.npz'),
                    'evidence': 'shard-00.json', 'evidence_sha256': sha256(base / 'shard-00.json')}]}
    (base / 'manifest.json').write_bytes(canonical_json(base_manifest))
    np.savez_compressed(visual / 'shard-00.npz', **overlay)
    manifest = {'schema_version': 1, 'kind': 'visual_causal_teacher_dataset', 'operator_snapshot': SOURCE.name,
        'qualification_fixture': True, 'rules': rules, 'native': receipt,
        'parent_dataset': {'path': str(base), 'manifest_sha256': sha256(base / 'manifest.json')},
        'shards': [{'id': 0, 'arrays': 'shard-00.npz', 'sha256': sha256(visual / 'shard-00.npz'),
                    'parent_arrays_sha256': sha256(base / 'shard-00.npz')}]}
    (visual / 'manifest.json').write_bytes(canonical_json(manifest))
    for path in root.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({'dataset': str(visual), 'manifest_sha256': sha256(visual / 'manifest.json'), 'operator_snapshot': SOURCE.name}))


if __name__ == '__main__':
    main()
