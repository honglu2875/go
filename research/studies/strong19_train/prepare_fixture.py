"""Freeze a small real19 execution fixture without changing collection splits."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
INVENTORY = ROOT / 'research/studies/strong19_scaling/inventory-002.json'
INVENTORY_SHA = 'c7188947a3fb2443af54f5a001aa83e7b0f507263d21411b5b2d081b8981399c'
TARGET = Path('/dev/shm/gozero-datasets/joint19-execution-001')
REFERENCE = ROOT / 'research/studies/strong19_scaling/qualify_features.py'
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def select(args):
    if sha(INVENTORY) != INVENTORY_SHA:
        raise ValueError('Inventory changed')
    inventory = json.loads(INVENTORY.read_text())
    chosen = {}
    for split in ('train', 'validation'):
        pool = [r for r in inventory['records'] if r['terminal'] and r['split'] == split]
        for opponent in range(8):
            candidates = [r for r in pool if r['opponent_index'] == opponent]
            if not candidates:
                raise ValueError('Missing qualification opponent stratum')
            r = min(candidates, key=lambda r: r['game_id']); chosen[r['game_id']] = r
        for r in (min(pool, key=lambda r: (r['rows'], r['game_id'])),
                  min(pool, key=lambda r: (-r['rows'], r['game_id']))):
            chosen[r['game_id']] = r
    records = sorted(chosen.values(), key=lambda r: (r['split'], r['game_id']))
    if {r['teacher_sha256'] for r in records} != {'73f6454eba62d2f6d099af8ce66d8c3fde6225e223c55817da0627590e98b0ae'}:
        raise ValueError('Teacher identity differs')
    for split in ('train', 'validation'):
        if {512 if r['rows'] <= 512 else 1536 for r in records if r['split'] == split} != {512, 1536}:
            raise ValueError('Both shape buckets require complete games')
    plan = dict(kind='real19_execution_fixture_plan', created=time.time(),
        inventory_sha256=INVENTORY_SHA, operator_sha256=sha(Path(__file__)),
        feature_reference_sha256=sha(REFERENCE), destination=str(TARGET), records=records,
        buckets=[512, 1536], test_targets_read=False,
        selection='Within each existing train/validation split, first game ID per opponent plus shortest and longest terminal games, deduplicated. Selection uses metadata only; no relabeling, truncation, target-dependent choice or test records.',
        scope='Complete selected execution fixture only, not the full collection or an architecture learning dataset. Long real games are padded to1536 solely to exercise the maximum intended runtime shape.')
    publish(args.plan, plan)
    print(json.dumps(dict(status='selected', games=len(records), positions=sum(r['rows'] for r in records),
                         plan_sha256=sha(args.plan))), flush=True)


def prepare(args):
    import numpy as np
    from gozero.corpus_format import KIND, SPLITS, array_specs
    from gozero.corpus_sequence_batches import Dataset
    from gozero.sequence_symmetry import action_map
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Fixture plan changed')
    plan = json.loads(args.plan.read_text())
    if plan['operator_sha256'] != sha(Path(__file__)) or plan['feature_reference_sha256'] != sha(REFERENCE):
        raise ValueError('Fixture source changed')
    if sha(INVENTORY) != plan['inventory_sha256'] or plan['destination'] != str(TARGET):
        raise ValueError('Fixture input/destination changed')
    if args.output.exists() or TARGET.exists():
        raise FileExistsError('Fixture destination/evidence already exists')
    fs = os.statvfs('/dev/shm')
    if fs.f_bavail * fs.f_frsize < 66 * (1 << 30):
        raise ValueError('Insufficient fixture RAM headroom')
    started = time.monotonic(); TARGET.mkdir(); raw_root = TARGET / 'raw-inputs'; raw_root.mkdir()
    spec = importlib.util.spec_from_file_location('qualified_feature_operator', REFERENCE)
    feature = importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)
    receipt = ROOT / '.gozero/build/katago-v7-12e944b6/receipt.json'
    if sha(receipt) != '9e7d8d4757c4214e25ea34a8ff3cbe57e1817abd7daf535117e0850eb0a6bd07':
        raise ValueError('Feature worker receipt changed')
    binary = receipt.parent / 'feature_worker'
    if sha(binary) != json.loads(receipt.read_text())['binary_sha256']:
        raise ValueError('Feature worker changed')
    records = plan['records']; originals = []
    for row in records:
        source = Path(row['path'])
        if (row['split'] not in ('train', 'validation') or not row['terminal'] or
                not source.is_relative_to('/dev/shm/go-corpus19/corpora') or source.name != row['game_id'] + '.npz'):
            raise ValueError('Unapproved fixture record')
        if row['host'] == 0:
            raw = source.read_bytes()
        else:
            raw = subprocess.check_output(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                'go-user@worker-' + str(row['host']), 'cat -- ' + shlex.quote(str(source))], timeout=30)
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Source game bytes changed')
        dest = raw_root / source.name
        with dest.open('xb') as stream:
            stream.write(raw)
        dest.chmod(0o444)
        with np.load(dest, allow_pickle=False) as data:
            metadata = json.loads(data['metadata'].tobytes())
            for key in ('game_id', 'rows', 'opening_family', 'split', 'teacher_sha256', 'terminal'):
                if metadata[key] != row[key]:
                    raise ValueError('Source metadata changed')
            originals.append({name: data[name] for name in ('actions', 'stones', 'legal', 'raw_policy', 'raw_value')})
    games = [dict(size=19, komi=7.5, actions=d['actions'].tolist()) for d in originals]
    spatial, glob, audit = feature.features(binary, games, TARGET)
    legal = np.concatenate([d['legal'] for d in originals])
    boards = np.concatenate([d['stones'].reshape(-1, 361) for d in originals])
    if not np.array_equal(audit[:, :361], boards) or not np.array_equal(audit[:, 361:].astype(bool), legal):
        raise ValueError('Native board/legal replay differs')
    count = len(spatial)
    if count != sum(row['rows'] for row in records) or np.any(spatial > 1) or not np.isfinite(glob).all():
        raise ValueError('Feature count/values differ')
    shard = TARGET / 'shard-00000'; shard.mkdir()
    arrays = {name: np.lib.format.open_memmap(shard / (name + '.npy'), mode='w+', shape=shape, dtype=dtype)
              for name, (shape, dtype) in array_specs(19, count, len(records)).items()}
    arrays['spatial'][:] = np.packbits(spatial.reshape(count, -1), axis=1, bitorder='little')
    arrays['legal'][:] = np.packbits(legal, axis=1, bitorder='little'); arrays['global_features'][:] = glob
    offsets = np.concatenate(([0], np.cumsum([r['rows'] for r in records], dtype=np.int64)))
    arrays['expert_offsets'][:] = offsets
    for name, original in (('actions', 'actions'), ('policies', 'raw_policy'), ('values', 'raw_value')):
        arrays[name][:] = np.concatenate([d[original] for d in originals])
    for i, (row, data) in enumerate(zip(records, originals)):
        trajectory = hashlib.sha256(min(action_map(19, symmetry)[data['actions']].astype('<i4').tobytes()
                                        for symmetry in range(8))).hexdigest()
        arrays['games'][i] = (row['game_id'], row['opening_family'], trajectory, SPLITS[row['split']],
                              row['opponent_index'], row['expert_color'], row['rows'])
    files = {}
    for name, array in arrays.items():
        array.flush(); path = shard / (name + '.npy'); path.chmod(0o444)
        files[name] = dict(path=str(path.relative_to(TARGET)), bytes=path.stat().st_size, sha256=sha(path),
                           shape=list(array.shape), dtype=str(array.dtype))
    populations = {split: dict(games=sum(r['split'] == split for r in records),
                               positions=sum(r['rows'] for r in records if r['split'] == split))
                   for split in ('train', 'validation')}
    manifest = dict(schema_version=1, kind=KIND, size=19, komi=7.5,
        max_game_moves=max(r['rows'] for r in records),
        shards=[dict(id=0, games=len(records), positions=count, files=files,
                     all_boards_equal=True, all_legal_masks_equal=True, targets_changed=False)],
        populations=populations, feature_version=7, spatial_channels=22, global_channels=19,
        targets_changed=False, test_arrays_included=False, complete_train_validation=True,
        complete_train_validation_scope='Every planned train/validation fixture record is present; not the full live collection.',
        qualification_only=True, source_release='execution-fixture:' + args.plan_sha256,
        inventory_sha256=plan['inventory_sha256'], selection_plan_sha256=args.plan_sha256,
        source_game_sha256={r['game_id']: r['sha256'] for r in records},
        target_teacher_sha256=records[0]['teacher_sha256'], feature_worker_binary_sha256=sha(binary),
        operator_sha256=sha(Path(__file__)))
    publish(TARGET / 'manifest.json', manifest)
    data = Dataset(TARGET, sha(TARGET / 'manifest.json'))
    for i, (row, original) in enumerate(zip(records, originals)):
        batch = data.batch([('expert', 0, i)], positions=row['rows'])
        start, end = map(int, offsets[i:i+2])
        if not np.array_equal(batch['spatial'][0], spatial[start:end]) or not np.array_equal(batch['global_features'][0], glob[start:end]):
            raise ValueError('Packed feature read-back differs')
        for name, source in (('actions', 'actions'), ('policies', 'raw_policy'), ('values', 'raw_value'), ('legal', 'legal')):
            if not np.array_equal(batch[name][0], original[source]):
                raise ValueError('Packed target read-back differs')
    buckets = {split: {str(b): len(entries) for (_, b), entries in data.bucket_entries(plan['buckets'], split=SPLITS[split]).items()}
               for split in ('train', 'validation')}
    result = dict(kind='real19_execution_fixture', status='passed', created=time.time(),
        plan_sha256=args.plan_sha256, target=str(TARGET), manifest_sha256=sha(TARGET / 'manifest.json'),
        games=len(records), positions=count, populations=populations, buckets=buckets,
        min_game_moves=min(r['rows'] for r in records), max_game_moves=data.time,
        all_boards_and_legal_masks_equal=True, all_loaded_inputs_and_targets_exact=True,
        test_targets_read=False, qualification_only=True, original_splits_preserved=True,
        packed_array_bytes=sum(r['bytes'] for r in files.values()), seconds=time.monotonic()-started,
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                       str(REFERENCE.relative_to(ROOT)): sha(REFERENCE)},
        scope='Small complete real19 train/validation execution fixture. No gameplay or training was run by this preparation operator. Not a final learning corpus, architecture comparison or strength result.')
    publish(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'games', 'positions', 'buckets', 'manifest_sha256', 'packed_array_bytes', 'seconds')}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('select', 'prepare'))
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--plan-sha256')
    p.add_argument('--output', type=Path); args = p.parse_args()
    if args.action == 'select':
        select(args)
    else:
        if args.plan_sha256 is None or args.output is None:
            raise ValueError('Pinned plan and new output required')
        prepare(args)


if __name__ == '__main__':
    main()
