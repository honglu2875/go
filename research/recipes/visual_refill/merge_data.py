"""Audited terminal self-play targets with unchanged historical held-out sets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.sequence_symmetry import action_map
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset


def array_file(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)


def json_file(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)


def collection(root, spec, native, rules):
    attempt = artifact(root, spec['generation_attempt'])
    launch = read_json(attempt / 'launch.json'); closed = read_json(attempt / 'result.json')
    if closed['status'] != 'passed' or launch['snapshot_id'] != spec['generation_snapshot']:
        raise ValueError('Generation must be a passed, pinned attempt')
    generation = artifact(root, '.gozero/snapshots/' + launch['snapshot_id']); verify(generation)
    config = read_json(generation / 'resolved_config.json')
    if (config['kind'] != 'visual_selfplay_generation' or config['rules'] != rules
            or config['candidate'] is None or config['fixture_pass_after'] is not None):
        raise ValueError('Only real, complete-history generation under the training rules is eligible')
    candidate = artifact(generation, config['candidate'])
    if checkpoints.sha256(candidate) != spec['candidate_sha256']:
        raise ValueError('Generator candidate differs')
    version = read_json(candidate)['network_version']
    evidence = {str((attempt / 'result.json').relative_to(root)): checkpoints.sha256(attempt / 'result.json')}
    mappings = [action_map(rules['size'], i) for i in range(8)]
    jobs = []
    for host in range(config['expected_processes']):
        base = attempt / f'rank-{host}/artifacts'; report = read_json(base / 'result.json')
        if (report['status'] != 'passed' or not report['generation_complete'] or report['turn'] != config['rounds']
                or report['candidate_sha256'] != spec['candidate_sha256']
                or report['native_receipt_sha256'] != spec['native_receipt_sha256']):
            raise ValueError('Incomplete rank generation or input identity mismatch')
        evidence[str((base / 'result.json').relative_to(root))] = checkpoints.sha256(base / 'result.json')
        for turn in range(1, config['rounds'] + 1):
            path = base / 'checkpoints' / f'turn-{turn:09d}'
            group = read_json(path.with_suffix('.group.json'))
            if (group['kind'] != 'visual_selfplay_checkpoint_group' or group['snapshot_id'] != generation.name
                    or group['turn'] != turn or group['config_sha256'] != hashlib.sha256(canonical_json(config)).hexdigest()
                    or group['host_jax_mapping'] != report['host_jax_mapping']):
                raise ValueError('Generation checkpoint group differs')
            jobs.append((host, turn, path, group['host_manifests'][str(host)]))

    def replay(job):
        host, turn, path, digest = job
        state, arrays, actors = checkpoints.read(path, expected_manifest_sha256=digest)
        if (state['host_rank'] != host or state['turn'] != turn or state['snapshot_id'] != generation.name
                or state['candidate_sha256'] != spec['candidate_sha256']): raise ValueError('Rank checkpoint identity differs')
        actors = json.loads(actors); actions, offsets = arrays['actions'], arrays['offsets']
        if (len(actors) != config['slots'] or len(offsets) != config['slots'] + 1
                or offsets[0] != 0 or offsets[-1] != len(actions) or np.any(np.diff(offsets) < 2)):
            raise ValueError('Generation episode coverage differs')
        stones, legal, endings = native.replay_observations(json.dumps(rules), actions, offsets)
        stones = stones.reshape(len(actions), rules['size'] ** 2)
        legal = legal.reshape(len(actions), rules['size'] ** 2 + 1); endings = json.loads(endings)
        if not np.array_equal(legal, arrays['legal']): raise ValueError('Generated root legality differs on replay')
        policy = arrays['policies']
        if (policy.shape != legal.shape or not np.isfinite(policy).all() or np.any(policy < 0)
                or np.any(policy[~legal] != 0) or not np.allclose(policy.sum(-1), 1., atol=2e-5, rtol=0)):
            raise ValueError('Generated search targets are malformed')
        result = []
        for slot, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            begin, end = int(begin), int(end); moves = actions[begin:end]; ending = endings[slot]; actor = actors[slot]
            expected_id = f'{generation.name}:host{host}:round{turn}:slot{slot}'
            if (actor['game_id'] != expected_id or actor['actions'] != moves.tolist()
                    or actor['moves'] != len(moves) or actor['terminal'] != ending['terminal']
                    or actor['white_score'] != ending['white_score']
                    or bool(arrays['terminal'][slot]) != ending['terminal']):
                raise ValueError('Replay ending or recorded game identity differs')
            identity = hashlib.sha256(expected_id.encode()).digest()
            canonical = min(tuple(mapping[moves].tolist()) for mapping in mappings)
            split_digest = hashlib.sha256(canonical_json([spec['split_seed'], list(canonical)])).digest()
            entry = {'game_id': expected_id, 'id': int.from_bytes(identity[:8], 'little'),
                     'shard': int.from_bytes(identity[8:16], 'little') % spec['shards'],
                     'terminal': ending['terminal'], 'moves': len(moves),
                     'train': int.from_bytes(split_digest[:8], 'little') % 5 != 0,
                     'trace_d4_sha256': hashlib.sha256(canonical_json(list(canonical))).hexdigest()}
            if ending['terminal']:
                expected = np.where(np.arange(len(moves)) % 2 == 0, -np.sign(ending['white_score']), np.sign(ending['white_score'])).astype(np.float32)
                if not np.array_equal(expected, arrays['outcomes'][begin:end]): raise ValueError('Terminal outcome target differs')
                passes = np.zeros(len(moves), np.uint8); passes[1:] = moves[:-1] == rules['size'] ** 2
                entry['arrays'] = {'actions': moves.copy(), 'policies': policy[begin:end].copy(),
                    'legal': legal[begin:end].copy(), 'values': expected, 'stones': stones[begin:end].copy(), 'passes': passes}
            elif np.any(arrays['outcomes'][begin:end] != 0):
                raise ValueError('Capped generation contains fabricated outcomes')
            result.append(entry)
        return result, {'host': host, 'round': turn, 'checkpoint': str(path.relative_to(root)), 'manifest_sha256': digest}

    with ThreadPoolExecutor(max_workers=spec['workers']) as pool: parts = list(pool.map(replay, jobs))
    games = [g for rows, _ in parts for g in rows]
    if len(games) != config['slots'] * config['rounds'] * config['expected_processes']:
        raise ValueError('Missing generated games')
    if len({g['id'] for g in games}) != len(games): raise ValueError('Generated game ID collision')
    terminal = sum(g['terminal'] for g in games)
    if terminal < spec['minimum_terminal_games']: raise ValueError('Insufficient terminal self-play data for the registered pilot')
    return games, version, evidence, [x for _, x in parts], closed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    if checkpoints.sha256(a.spec) != a.expected_sha256: raise ValueError('Dataset registration changed')
    spec = read_json(a.spec)
    fields = 'schema_version kind parent_dataset generation_attempt generation_snapshot candidate_sha256 native_receipt native_receipt_sha256 split_seed shards buckets minimum_fresh_per_bucket minimum_terminal_games workers protocol'
    if (set(spec) != set(fields.split()) or spec['kind'] != 'visual_online_dataset' or spec['schema_version'] != 1
            or not 1 <= spec['workers'] <= 8 or spec['minimum_fresh_per_bucket'] < 1): raise ValueError('Invalid online dataset spec')
    old = Dataset(spec['parent_dataset']['path'], spec['parent_dataset']['manifest_sha256'])
    if len(old.shards) != spec['shards']: raise ValueError('Parent shard coverage differs')
    receipt_path = artifact(root, spec['native_receipt'])
    if checkpoints.sha256(receipt_path) != spec['native_receipt_sha256']: raise ValueError('Native receipt changed')
    receipt = read_json(receipt_path); native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    games, version, evidence, batches, generation_result = collection(root, spec, native, old.manifest['rules'])
    fresh = [g for g in games if g['terminal']]
    def bucket(n): return next(b for b in spec['buckets'] if b >= n)
    counts = {b: sum(g['train'] and bucket(g['moves']) == b for g in fresh) for b in spec['buckets']}
    replacing = {b for b, n in counts.items() if n >= spec['minimum_fresh_per_bucket']}
    if not replacing: raise ValueError('No bucket qualifies for training population replacement')
    old_ids = {int(i) for s in old.shards for role in ('expert', 'behavior') for i in s[role + '_game_ids']}
    if any(g['id'] in old_ids for g in games): raise ValueError('Generated IDs collide with parent dataset')
    output = a.output.resolve()
    if output.exists() or not output.is_relative_to(root) or output.is_relative_to(SOURCE): raise ValueError('Output must be fresh workspace data')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
    base, visual = temporary / 'base', temporary / 'visual'; base.mkdir(); visual.mkdir()
    records, overlays, training_counts = [], [], {str(b): {'new': 0, 'old': 0} for b in spec['buckets']}
    for index, source in enumerate(old.shards):
        additions = [g for g in fresh if g['shard'] == index]
        parent_record = old.parent_manifest['shards'][index]
        with np.load(Path(old.manifest['parent_dataset']['path']) / parent_record['arrays'], allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        overlay = {role + '_' + k: source[role + '_' + k] for role in ('expert', 'behavior') for k in ('stones', 'legal', 'passes')}
        n = len(arrays['expert_actions']); original_games = len(arrays['expert_splits'])
        eligible = np.ones(original_games, np.bool_)
        for game, (begin, end) in enumerate(zip(arrays['expert_offsets'][:-1], arrays['expert_offsets'][1:])):
            b = bucket(int(end - begin))
            if arrays['expert_splits'][game] == 0:
                eligible[game] = b not in replacing
                training_counts[str(b)]['old'] += int(eligible[game])
        for key in ('actions', 'policies', 'legal', 'values'):
            name = 'expert_' + key
            arrays[name] = np.concatenate([arrays[name], *[g['arrays'][key] for g in additions]]) if additions else arrays[name]
        for key in ('stones', 'legal', 'passes'):
            name = 'expert_' + key
            overlay[name] = np.concatenate([overlay[name], *[g['arrays'][key] for g in additions]]) if additions else overlay[name]
        lengths = [g['moves'] for g in additions]
        arrays['expert_networks'] = np.concatenate([arrays['expert_networks'], np.full(sum(lengths), version, np.uint64)])
        arrays['expert_offsets'] = np.concatenate([arrays['expert_offsets'], n + np.cumsum(lengths, dtype=np.int64)])
        arrays['expert_game_ids'] = np.concatenate([arrays['expert_game_ids'], np.asarray([g['id'] for g in additions], np.uint64)])
        # New collection holdouts remain ineligible training rows; the original
        # validation/test sets and their indices are preserved exactly.
        arrays['expert_splits'] = np.concatenate([arrays['expert_splits'], np.zeros(len(additions), np.uint8)])
        arrays['expert_training_eligible'] = np.concatenate([eligible, np.asarray([g['train'] and bucket(g['moves']) in replacing for g in additions], np.bool_)])
        for g in additions:
            training_counts[str(bucket(g['moves']))]['new'] += int(g['train'] and bucket(g['moves']) in replacing)
        name = f'shard-{index:02d}.npz'; proof = f'shard-{index:02d}.json'
        array_file(base / name, arrays); array_file(visual / name, overlay)
        proof_data = {'schema_version': 1, 'parent_shard': parent_record,
                      'generated_games': [{k: v for k, v in g.items() if k != 'arrays'} for g in additions],
                      'all_original_fields_preserved_as_prefixes': True}
        json_file(base / proof, proof_data)
        common = {'id': index, 'arrays': name, 'expert_games': len(arrays['expert_splits']), 'expert_rows': len(arrays['expert_actions']),
                  'behavior_games': len(arrays['behavior_splits']), 'behavior_rows': len(arrays['behavior_actions'])}
        records.append({**common, 'sha256': checkpoints.sha256(base / name), 'bytes': (base / name).stat().st_size,
                        'evidence': proof, 'evidence_sha256': checkpoints.sha256(base / proof)})
        overlays.append({**common, 'sha256': checkpoints.sha256(visual / name), 'bytes': (visual / name).stat().st_size,
                         'parent_arrays_sha256': records[-1]['sha256']})
    cost = {'attempt': spec['generation_attempt'], 'result_sha256': checkpoints.sha256(artifact(root, spec['generation_attempt']) / 'result.json'),
            'purpose': 'Fresh terminal self-play targets', 'global_real_moves': sum(g['moves'] for g in games),
            'recorded_attempt_chip_hours': generation_result['reserved_chip_hours']}
    parent = {'schema_version': 1, 'kind': 'causal_teacher_dataset', 'operator_snapshot': SOURCE.name,
              'spec': {**old.parent_manifest['spec'], 'teacher_cost': [*old.parent_manifest['spec']['teacher_cost'], cost]},
              'online_dataset_spec': spec, 'online_dataset_spec_sha256': a.expected_sha256, 'shards': records}
    json_file(base / 'manifest.json', parent)
    manifest = {'schema_version': 1, 'kind': 'visual_causal_teacher_dataset', 'operator_snapshot': SOURCE.name,
                'parent_dataset': {'path': str(output / 'base'), 'manifest_sha256': checkpoints.sha256(base / 'manifest.json')},
                'rules': old.manifest['rules'], 'native': receipt, 'spec': spec, 'spec_sha256': a.expected_sha256,
                'shards': overlays, 'training_counts': training_counts, 'replaced_buckets': sorted(replacing),
                'scope': spec['protocol'], 'generation_evidence': evidence, 'generation_batches': batches}
    json_file(visual / 'manifest.json', manifest)
    summary = {'schema_version': 1, 'kind': 'visual_online_dataset_audit', 'status': 'passed', 'operator_snapshot': SOURCE.name,
               'finished_unix': time.time(), 'spec_sha256': a.expected_sha256, 'games': len(games),
               'terminal_games': len(fresh), 'capped_games': len(games) - len(fresh),
               'checked_boards': sum(g['moves'] for g in games), 'training_counts': training_counts,
               'collection_holdout_terminal_games': sum(not g['train'] for g in fresh),
               'visual_manifest_sha256': checkpoints.sha256(visual / 'manifest.json'),
               'raw_game_index': [{k: v for k, v in g.items() if k != 'arrays'} for g in games],
               'scope': 'Native replay audit of every generated board, legality, ending and outcome target; not independent KataGo rules validation.'}
    json_file(temporary / 'receipt.json', summary); verify(SOURCE)
    temporary.rename(output); checkpoints._sync_directory(output.parent)
    # Load the final absolute manifests and prove held-out sets and behavior
    # inputs/targets are unchanged before making this dataset trainable.
    merged = Dataset(output / 'visual', summary['visual_manifest_sha256'])
    for role in ('expert', 'behavior'):
        for split in (1, 2):
            if old.indices[role, split] != merged.indices[role, split]: raise ValueError('Held-out population changed')
        for original, actual in zip(old.shards, merged.shards):
            for name, arr in original.items():
                if name.startswith(role + '_'):
                    if not np.array_equal(arr, actual[name][:len(arr)]): raise ValueError('Original target/input bytes changed')
    json_file(output / 'qualified.json', {'status': 'passed', 'receipt_sha256': checkpoints.sha256(output / 'receipt.json'),
        'visual_manifest_sha256': summary['visual_manifest_sha256'], 'original_inputs_targets_and_heldout_indices_exact': True})
    print(json.dumps({k: v for k, v in summary.items() if k != 'raw_game_index'}), flush=True)


if __name__ == '__main__': main()
