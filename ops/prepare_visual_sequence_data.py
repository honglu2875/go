#!/usr/bin/env python3
"""Replay pinned whole games into exact compact visual input histories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace-root', 'spec', 'native-receipt', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--expected-spec-sha256', required=True)
    a = p.parse_args(); verify(SOURCE)
    root = a.workspace_root.resolve(); spec = read_json(a.spec)
    if sha256(a.spec) != a.expected_spec_sha256 or spec['kind'] != 'visual_causal_teacher_dataset':
        raise ValueError('Visual dataset registration differs')
    if type(spec['workers']) is not int or not 1 <= spec['workers'] <= 8:
        raise ValueError('Bounded preprocessing worker count required')
    parent = artifact(root, spec['parent_dataset']['path']); manifest = read_json(parent / 'manifest.json')
    if sha256(parent / 'manifest.json') != spec['parent_dataset']['manifest_sha256'] or manifest['kind'] != 'causal_teacher_dataset':
        raise ValueError('Pinned teacher dataset differs')
    receipt = read_json(a.native_receipt)
    # Reuse the previously qualified pure replay ABI without rebuilding it.
    if sha256(a.native_receipt) != spec['native_receipt_sha256']:
        raise ValueError('Replay binary receipt differs')
    verify(artifact(root, '.gozero/snapshots/' + receipt['snapshot_id']))
    native = load_library(a.native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
    if getattr(native, 'OBSERVATION_REPLAY_ABI_VERSION', None) != 1:
        raise ValueError('Replay ABI unavailable')
    rules = {k: manifest['spec'][k] for k in ('size', 'komi', 'scoring')}
    output = a.output.resolve()
    if not output.is_relative_to(root) or output.is_relative_to(SOURCE) or output.exists():
        raise ValueError('Output must be a fresh workspace artifact directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
    report = {'schema_version': 1, 'kind': spec['kind'], 'operator_snapshot': SOURCE.name,
              'status': 'running', 'started_unix': time.time(), 'spec_sha256': a.expected_spec_sha256}
    def process(item):
        path = parent / item['arrays']
        if Path(item['arrays']).name != item['arrays'] or sha256(path) != item['sha256']:
            raise ValueError('Teacher shard bytes differ')
        result, counts = {}, {}
        with np.load(path, allow_pickle=False) as saved:
            for role in ('expert', 'behavior'):
                actions, offsets = saved[role + '_actions'], saved[role + '_offsets']
                stones, legal, endings = native.replay_observations(json.dumps(rules), np.ascontiguousarray(actions), np.ascontiguousarray(offsets))
                legal = legal.reshape(len(actions), rules['size'] ** 2 + 1)
                passes = np.zeros(len(actions), np.uint8)
                endings = json.loads(endings)
                for game, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                    begin, end = int(begin), int(end)
                    passes[begin+1:end] = actions[begin:end-1] == rules['size'] ** 2
                    ending = endings[game]
                    if role == 'expert':
                        if not ending['terminal'] or ending['white_score'] is None:
                            raise ValueError('Expert outcome missing')
                        expected = np.where(np.arange(end - begin) % 2 == 0, -np.sign(ending['white_score']), np.sign(ending['white_score'])).astype(np.float32)
                        if not np.array_equal(expected, saved['expert_values'][begin:end]):
                            raise ValueError('Replayed terminal outcome differs')
                    elif ending['terminal'] == bool(saved['behavior_capped'][game]):
                        raise ValueError('Behavior completion state differs')
                if role == 'expert' and not np.array_equal(legal, saved['expert_legal']):
                    raise ValueError('Replayed MCTS legality differs')
                result.update({role + '_stones': stones.reshape(len(actions), rules['size'] ** 2),
                               role + '_legal': legal, role + '_passes': passes})
                counts[role + '_rows'] = len(actions); counts[role + '_games'] = len(offsets) - 1
        name = 'shard-%02d.npz' % item['id']; path = temporary / name
        with path.open('xb') as f:
            np.savez_compressed(f, **result); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
        record = {'id': item['id'], 'arrays': name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                  'parent_arrays_sha256': item['sha256'], **counts}
        print(json.dumps({'kind': 'visual_shard_verified', **record}), flush=True)
        return record
    try:
        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
            shards = list(pool.map(process, manifest['shards']))
        verify(SOURCE)
        complete = {**report, 'kind': spec['kind'], 'schema_version': 1, 'native': receipt,
                    'parent_dataset': spec['parent_dataset'], 'rules': rules, 'shards': shards, 'spec': spec,
                    'targets_changed': False, 'input_contract': 'Exact pre-action stones, legality and previous-pass count. Absolute colors. Entire history from empty board. Current/future target actions are not embedded before their predictions.'}
        complete.pop('status'); complete.pop('started_unix')
        (temporary / 'manifest.json').write_bytes(canonical_json(complete))
        report.update(status='passed', finished_unix=time.time(), dataset_manifest_sha256=sha256(temporary / 'manifest.json'))
        (temporary / 'receipt.json').write_bytes(canonical_json(report))
        for path in temporary.iterdir():
            with path.open('rb') as f:
                os.fsync(f.fileno())
            path.chmod(0o444)
        _sync_directory(temporary); temporary.rename(output); _sync_directory(output.parent)
    except BaseException as error:
        report.update(status='failed', error=repr(error), finished_unix=time.time())
        (temporary / 'failure.json').write_bytes(canonical_json(report))
        raise
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
