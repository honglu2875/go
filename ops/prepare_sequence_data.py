#!/usr/bin/env python3
"""Publish a bounded, source-verified causal dataset from completed teacher runs."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256, _sync_directory
from gozero.sequence_data import extract_shard
from gozero.snapshots import canonical_json, read_json, verify


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--expected-spec-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    spec = read_json(args.spec)
    if sha256(args.spec) != args.expected_spec_sha256 or spec['kind'] != 'causal_teacher_dataset':
        raise ValueError('Dataset registration identity differs')
    if not 1 <= len(spec['shards']) <= 16 or len({s['checkpoint'] for s in spec['shards']}) != len(spec['shards']):
        raise ValueError('Invalid bounded source selection')
    output = args.output.resolve()
    if not output.is_relative_to(args.workspace_root.resolve()) or output.is_relative_to(SOURCE):
        raise ValueError('Dataset publication must be outside source and inside workspace')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
    started = time.time()
    manifest = {'schema_version': 1, 'kind': 'causal_teacher_dataset', 'operator_snapshot': SOURCE.name,
                'spec_sha256': args.expected_spec_sha256, 'spec': spec, 'shards': []}
    seen = set()
    for index, item in enumerate(spec['shards']):
        arrays, evidence = extract_shard(args.workspace_root, item, behavior_games=spec['behavior_games_per_shard'],
                                         behavior_seed=spec['behavior_seed'] + index)
        for role in ('expert', 'behavior'):
            identities = {(item['training_snapshot'], role, int(g)) for g in arrays[role + '_game_ids']}
            if seen & identities:
                raise ValueError('A game occurs in more than one host shard')
            seen |= identities
        path = temporary / f'shard-{index:02d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        record = temporary / f'shard-{index:02d}.json'
        record.write_bytes(canonical_json(evidence))
        manifest['shards'].append({'id': index, 'arrays': path.name, 'sha256': sha256(path),
                                   'bytes': path.stat().st_size, 'evidence': record.name, 'evidence_sha256': sha256(record),
                                   'expert_rows': evidence['expert_rows'], 'behavior_rows': evidence['behavior_rows'],
                                   'expert_games': evidence['expert_games'], 'behavior_games': evidence['behavior_games'],
                                   'behavior_capped_games': evidence['behavior_capped_games'], 'split_counts': evidence['split_counts']})
        print(json.dumps({'kind': 'sequence_shard_verified', **manifest['shards'][-1]}), flush=True)
    verify(SOURCE)
    (temporary / 'manifest.json').write_bytes(canonical_json(manifest))
    receipt = {'schema_version': 1, 'status': 'passed', 'dataset_manifest_sha256': sha256(temporary / 'manifest.json'),
               'started_unix': started, 'finished_unix': time.time(), 'operator_snapshot': SOURCE.name}
    (temporary / 'receipt.json').write_bytes(canonical_json(receipt))
    for path in temporary.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    _sync_directory(temporary)
    if output.exists():
        raise FileExistsError(output)
    temporary.rename(output); _sync_directory(output.parent)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
