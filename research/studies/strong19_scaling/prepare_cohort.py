"""Freeze and losslessly pack a metadata-selected scientific 19x19 cohort.

Process small shards so native feature replay and serialization have bounded
temporary memory. Publish the manifest only after every planned game passes.
The existing V7 worker and packed-loader implementation remain unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'research/studies/strong19_scaling/qualify_features.py'
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json

TEACHER = '73f6454eba62d2f6d099af8ce66d8c3fde6225e223c55817da0627590e98b0ae'
WORKER_RECEIPT = '9e7d8d4757c4214e25ea34a8ff3cbe57e1817abd7daf535117e0850eb0a6bd07'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def select(args):
    require(sha(args.inventory) == args.inventory_sha256, 'Inventory changed')
    inventory = json.loads(args.inventory.read_text())
    require(inventory['kind'] == 'completed_19_inventory' and not inventory['test_policy_value_targets_decoded'],
            'Unexpected inventory scope')
    all_records = inventory['records']
    require(len({r['game_id'] for r in all_records}) == len(all_records), 'Duplicate game ID')
    families = {}
    for row in all_records:
        require(families.setdefault(row['opening_family'], row['split']) == row['split'], 'Family crosses splits')
    records = sorted((r for r in all_records if r['terminal'] and r['split'] in ('train', 'validation')),
                     key=lambda r: (r['split'], r['game_id']))
    require({r['teacher_sha256'] for r in records} == {TEACHER}, 'Teacher differs')
    require(all(r['board_size'] == 19 and r['komi'] == 7.5 for r in records), 'Board rules differ')
    require(max(r['rows'] for r in records) <= 768, 'Scientific buckets do not cover the frozen cohort')
    populations = {}
    for split in ('train', 'validation'):
        rows = [r for r in records if r['split'] == split]
        require({r['opponent_index'] for r in rows} == set(range(8)), 'Missing opponent stratum')
        populations[split] = dict(games=len(rows), positions=sum(r['rows'] for r in rows),
            families=len({r['opening_family'] for r in rows}),
            opponents=dict(sorted(Counter(r['opponent_index'] for r in rows).items())),
            buckets=dict(sorted(Counter(512 if r['rows'] <= 512 else 768 for r in rows).items())))
    expected = 128 * populations['train']['positions'] / populations['train']['games']
    plan = dict(kind='real19_scientific_cohort_plan', created=time.time(),
        inventory=str(args.inventory.relative_to(ROOT)), inventory_sha256=args.inventory_sha256,
        operator_sha256=sha(Path(__file__)), feature_reference_sha256=sha(REFERENCE),
        library_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       [ROOT / 'packages/gozero/src/gozero' / n for n in
                        ('corpus_format.py', 'corpus_sequence_batches.py', 'sequence_symmetry.py', 'snapshots.py')]},
        records=records, populations=populations, buckets=[512, 768], games_per_shard=32,
        destination='/dev/shm/gozero-datasets/strong19-scientific-001',
        expected_positions_per_128_games=expected, rounded_reference_batch_positions=round(expected),
        test_targets_read=False, qualification_only=False,
        selection='Every terminal train/validation game in the pinned inventory, sorted by split and game ID. Existing family splits preserved; no target-dependent selection, truncation, resplitting or test target access.',
        scope='Fixed first scientific19 pilot cohort. Later generated games are excluded from this view. Hyperparameters and learner launches require a separate experiment registration.')
    publish(args.plan, plan)
    print(json.dumps(dict(status='selected', plan_sha256=sha(args.plan), populations=populations,
                          expected_positions_per_128_games=expected)), flush=True)


REMOTE = '''import io,json,sys,tarfile
from pathlib import Path
rows=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for row in rows:
  path=Path(row['path'])
  if path.is_symlink() or not path.is_relative_to('/dev/shm/go-corpus19/corpora') or path.name!=row['game_id']+'.npz':
   raise ValueError('Unexpected source path')
  raw=path.read_bytes(); info=tarfile.TarInfo(path.name);info.size=len(raw);out.addfile(info,io.BytesIO(raw))
'''


def fetch(rows, raw_inputs=None):
    def one(host):
        selected = [r for r in rows if r['host'] == host]
        if not selected:
            return {}
        command = ('taskset -c 0,1 /home/go-user/.local/share/uv/python/'
                   'cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -c ' + shlex.quote(REMOTE))
        value = subprocess.run(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            'go-user@worker-' + str(host), command], input=json.dumps(selected).encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
        result = {}
        with tarfile.open(fileobj=io.BytesIO(value.stdout), mode='r:') as archive:
            for member in archive:
                require(member.isfile() and member.name not in result, 'Invalid source archive member')
                result[member.name] = archive.extractfile(member).read()
        require(set(result) == {r['game_id'] + '.npz' for r in selected}, 'Source archive coverage differs')
        return result
    if raw_inputs is None:
        with ThreadPoolExecutor(4) as pool:
            raw = {name: data for part in pool.map(one, range(4)) for name, data in part.items()}
    else:
        raw = {}
        for row in rows:
            path = raw_inputs / (row['game_id'] + '.npz')
            require(not path.is_symlink(), 'Staged raw input is a symlink')
            raw[path.name] = path.read_bytes()
    for row in rows:
        data = raw[row['game_id'] + '.npz']
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Source bytes changed')
    return raw


def prepare(args):
    import numpy as np
    from gozero.corpus_format import KIND, SPLITS, array_specs, unpack_legal, unpack_spatial
    from gozero.corpus_sequence_batches import Dataset
    from gozero.sequence_symmetry import action_map
    require(sha(args.plan) == args.plan_sha256, 'Cohort plan changed')
    plan = json.loads(args.plan.read_text())
    require(plan['operator_sha256'] == sha(Path(__file__)) and plan['feature_reference_sha256'] == sha(REFERENCE),
            'Cohort operator changed')
    require(plan['kind'] == 'real19_scientific_cohort_plan' and not plan['test_targets_read'], 'Wrong plan kind')
    require(sha(ROOT / plan['inventory']) == plan['inventory_sha256'], 'Inventory changed')
    for path, digest in plan['library_sha256'].items():
        require(sha(ROOT / path) == digest, 'Packed library changed')
    target = Path(plan['destination'])
    require(target.parent == Path('/dev/shm/gozero-datasets'), 'Unexpected destination')
    require(not target.exists() and not args.output.exists(), 'Cohort preparation already started')
    planned_bytes = sum(r['rows'] for r in plan['records']) * 2571 + len(plan['records']) * 220
    fs = os.statvfs('/dev/shm')
    require(fs.f_bavail * fs.f_frsize > args.reserved_bytes + planned_bytes + (2 << 30), 'Insufficient reserved RAM headroom')
    receipt = ROOT / '.gozero/build/katago-v7-12e944b6/receipt.json'
    require(sha(receipt) == WORKER_RECEIPT, 'Feature worker receipt changed')
    binary = receipt.parent / 'feature_worker'
    require(sha(binary) == json.loads(receipt.read_text())['binary_sha256'], 'Feature worker changed')
    spec = importlib.util.spec_from_file_location('cohort_feature_reference', REFERENCE)
    feature = importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    started = time.monotonic(); target.mkdir(); shards = []; trajectories = {}; checked = 0
    for offset in range(0, len(plan['records']), plan['games_per_shard']):
        rows = plan['records'][offset:offset + plan['games_per_shard']]
        raw = fetch(rows, args.raw_inputs); originals = []
        for row in rows:
            require(row['terminal'] and row['split'] in ('train', 'validation'), 'Test/nonterminal record in view')
            with np.load(io.BytesIO(raw[row['game_id'] + '.npz']), allow_pickle=False) as data:
                metadata = json.loads(data['metadata'].tobytes())
                for key in ('game_id', 'rows', 'opening_family', 'split', 'teacher_sha256', 'terminal', 'board_size', 'komi'):
                    require(metadata[key] == row[key], 'Source metadata differs')
                originals.append({name: data[name] for name in ('actions', 'stones', 'legal', 'raw_policy', 'raw_value')})
        del raw
        spatial, glob, audit = feature.features(binary,
            [dict(size=19, komi=7.5, actions=d['actions'].tolist()) for d in originals], target)
        legal = np.concatenate([d['legal'] for d in originals]); count = len(spatial)
        boards = np.concatenate([d['stones'].reshape(-1, 361) for d in originals])
        require(count == sum(r['rows'] for r in rows) and not np.any(spatial > 1)
                and np.all(spatial[..., 0] == 1) and np.isfinite(glob).all(), 'Feature values/count differ')
        require(np.array_equal(audit[:, :361], boards) and np.array_equal(audit[:, 361:].astype(bool), legal),
                'Native board/legal replay differs')
        shard_id = len(shards); folder = target / f'shard-{shard_id:05d}'; folder.mkdir()
        arrays = {name: np.lib.format.open_memmap(folder / (name + '.npy'), mode='w+', shape=shape, dtype=dtype)
                  for name, (shape, dtype) in array_specs(19, count, len(rows)).items()}
        arrays['spatial'][:] = np.packbits(spatial.reshape(count, -1), axis=1, bitorder='little')
        arrays['legal'][:] = np.packbits(legal, axis=1, bitorder='little'); arrays['global_features'][:] = glob
        arrays['expert_offsets'][:] = np.concatenate(([0], np.cumsum([r['rows'] for r in rows], dtype=np.int64)))
        for name, original in (('actions', 'actions'), ('policies', 'raw_policy'), ('values', 'raw_value')):
            arrays[name][:] = np.concatenate([d[original] for d in originals])
        for i, (row, data) in enumerate(zip(rows, originals)):
            trajectory = hashlib.sha256(min(action_map(19, s)[data['actions']].astype('<i4').tobytes()
                                            for s in range(8))).hexdigest()
            require(trajectories.setdefault(trajectory, row['split']) == row['split'], 'Complete trajectory crosses train/validation')
            arrays['games'][i] = (row['game_id'], row['opening_family'], trajectory, SPLITS[row['split']],
                                  row['opponent_index'], row['expert_color'], row['rows'])
        files = {}
        for name, array in arrays.items():
            array.flush(); path = folder / (name + '.npy'); path.chmod(0o444)
            loaded = np.load(path, mmap_mode='r', allow_pickle=False)
            require(np.array_equal(loaded, array), 'Serialized array read-back differs')
            original_name = {'actions': 'actions', 'policies': 'raw_policy', 'values': 'raw_value'}.get(name)
            if original_name:
                require(np.array_equal(loaded, np.concatenate([d[original_name] for d in originals])),
                        'Serialized targets differ from source games')
            files[name] = dict(path=str(path.relative_to(target)), bytes=path.stat().st_size, sha256=sha(path),
                               shape=list(array.shape), dtype=str(array.dtype))
        require(np.array_equal(unpack_spatial(arrays['spatial'], 19), spatial)
                and np.array_equal(unpack_legal(arrays['legal'], 19), legal), 'Bit packing changes features')
        for name in ('policies', 'values'):
            require(np.isfinite(arrays[name]).all(), 'Nonfinite targets')
        require(np.all(np.abs(arrays['values']) <= 1) and np.all(arrays['policies'] >= 0)
                and np.allclose(arrays['policies'].sum(axis=1), 1, atol=3e-5), 'Invalid target range')
        require(np.all(arrays['policies'][~legal] == 0), 'Teacher assigned illegal action mass')
        shards.append(dict(id=shard_id, games=len(rows), positions=count, files=files,
                           all_boards_equal=True, all_legal_masks_equal=True, targets_changed=False))
        checked += count
        print(json.dumps(dict(kind='cohort_shard_passed', shard=shard_id, games=offset + len(rows), positions=checked)), flush=True)
        del arrays, loaded, spatial, glob, audit, originals, legal, boards
    manifest = dict(schema_version=1, kind=KIND, size=19, komi=7.5,
        max_game_moves=max(r['rows'] for r in plan['records']), shards=shards,
        populations={s: {k: v for k, v in p.items() if k in ('games', 'positions')} for s, p in plan['populations'].items()},
        feature_version=7, spatial_channels=22, global_channels=19, targets_changed=False,
        test_arrays_included=False, complete_train_validation=True,
        complete_train_validation_scope='Every terminal training/validation game in the pinned inventory; later collection excluded.',
        qualification_only=False, source_release='scientific19:' + args.plan_sha256,
        inventory_sha256=plan['inventory_sha256'], selection_plan_sha256=args.plan_sha256,
        source_game_sha256={r['game_id']: r['sha256'] for r in plan['records']}, target_teacher_sha256=TEACHER,
        feature_worker_binary_sha256=sha(binary), operator_sha256=sha(Path(__file__)))
    publish(target / 'manifest.json', manifest)
    data = Dataset(target, sha(target / 'manifest.json'))
    buckets = {s: {str(b): len(entries) for (_, b), entries in data.bucket_entries(plan['buckets'], split=SPLITS[s]).items()}
               for s in ('train', 'validation')}
    result = dict(kind='real19_scientific_cohort', status='passed', created=time.time(),
        plan_sha256=args.plan_sha256, manifest_sha256=sha(target / 'manifest.json'), target=str(target),
        games=len(plan['records']), positions=checked, populations=plan['populations'], buckets=buckets,
        all_native_boards_legal_and_packed_features_equal=True, all_source_targets_preserved=True,
        test_targets_read=False, original_splits_preserved=True, train_validation_trajectory_overlap=False,
        packed_array_bytes=sum(f['bytes'] for s in shards for f in s['files'].values()),
        seconds=time.monotonic() - started, operator_sha256=sha(Path(__file__)))
    publish(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'games', 'positions', 'manifest_sha256', 'packed_array_bytes', 'seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'prepare'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--inventory-sha256')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--raw-inputs', type=Path)
    parser.add_argument('--reserved-bytes', type=int, default=64 << 30)
    args = parser.parse_args()
    require(args.reserved_bytes >= 64 << 30, 'Reserve cannot fall below the shared-memory floor')
    if args.action == 'select':
        require(args.inventory is not None and args.inventory_sha256, 'Pinned inventory required')
        args.inventory = args.inventory.resolve(); select(args)
    else:
        require(args.plan_sha256 and args.output is not None, 'Pinned plan and output required'); prepare(args)


if __name__ == '__main__':
    main()
