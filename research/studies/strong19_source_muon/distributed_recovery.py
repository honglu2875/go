"""Audit real source-optimizer full/prefix/resume artifacts on every host.

Reads checkpoint payloads sequentially. Exact array identity is established by
canonical tensor SHA256, plus exact checked payload/metadata manifests; this
does not allocate three full-size optimizer states at once. Sampling is replayed
from corpus metadata, without invoking the model or consuming held-out targets.
"""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify

TIMINGS = {'cumulative_learning_seconds', 'cumulative_sampling_seconds'}
LOCAL = {'local_entries_sha256', 'local_symmetries'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trimmed(row, ignored):
    return {k: v for k, v in row.items() if k not in ignored}


def tensor_identity(arrays):
    import numpy as np
    digest = hashlib.sha256()
    elements = size = 0
    for key, value in sorted(arrays.items()):
        a = np.asarray(value)
        require(not a.dtype.hasobject and np.isfinite(a).all(), 'Nonfinite checkpoint tensor: ' + key)
        digest.update(canonical_json([key, list(a.shape), str(a.dtype)]))
        digest.update(memoryview(np.ascontiguousarray(a)).cast('B'))
        elements += a.size
        size += a.nbytes
    return dict(sha256=digest.hexdigest(), arrays=len(arrays), elements=elements, bytes=size)


class Audit:
    def __init__(self, snapshot):
        self.snapshot = snapshot.resolve()
        manifest = verify(self.snapshot)
        self.config = read_json(self.snapshot / 'resolved_config.json')
        self.config_sha = hashlib.sha256(canonical_json(self.config)).hexdigest()
        self.inputs = {}
        # Use the numerical recipe from the frozen experiment, never the live recipe.
        sys.path.insert(0, str(self.snapshot / manifest['recipe']))
        from gozero import corpus_sequence_batches
        for module in (checkpoints, corpus_sequence_batches):
            path = Path(module.__file__).resolve()
            frozen = self.snapshot / path.relative_to(ROOT)
            require(checkpoints.sha256(path) == checkpoints.sha256(frozen), 'Shared audit reader changed')
        import joint
        import optimizer_io
        import source_runtime
        self.codec, self.runtime = optimizer_io, source_runtime
        c = self.config
        require(c['training']['purpose'] == 'qualification' and c['kind'] == 'fixed_joint_learning',
                'This audit is for the explicit execution fixture')
        require(not c['evaluation']['run_test'], 'Execution audit cannot open the test population')
        self.schema = joint.parameter_schema(c['model'], c['value_model'])
        self.world = c['expected_processes']
        self.runtime_config = c['learner']['source_runtime']
        self.runtime.validate(self.runtime_config, c['steps'])
        self.dataset = corpus_sequence_batches.Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])

    def record(self, path):
        path = Path(path)
        digest = checkpoints.sha256(path)
        require(str(path) not in self.inputs or self.inputs[str(path)] == digest, 'Artifact changed during audit')
        self.inputs[str(path)] = digest
        return digest

    def stage(self, directories, turn, first_turn):
        import numpy as np
        require(len(directories) == self.world, 'All host artifact directories are required')
        reports, states, metrics, groups = {}, {}, {}, {}
        owner_identity = None
        for directory in directories:
            directory = directory.resolve()
            for name in ('result.json', 'resolved_config.json', 'metrics.jsonl'):
                self.record(directory / name)
            r = read_json(directory / 'result.json')
            c = read_json(directory / 'resolved_config.json')
            host = r['host_rank']
            require(host not in reports and host in range(self.world), 'Repeated or invalid host')
            require(c == self.config and r['config_sha256'] == self.config_sha
                    and r['snapshot_id'] == self.snapshot.name and r['world_size'] == self.world
                    and r['dataset_manifest_sha256'] == c['dataset']['manifest_sha256']
                    and r['kind'] == c['kind'] and r['training_purpose'] == 'qualification'
                    and r['optimizer_family'] == 'source_muon_aux_adam_lookahead'
                    and r['status'] == 'passed' and r['turn'] == turn
                    and r['training_complete'] == (turn == c['steps'])
                    and r['model_schema'] == self.schema and r['initial_turn'] == 0
                    and r['fork_lineage'] is None, 'Report source, scope or progress differs')
            mapping = r['host_jax_mapping']
            require(len(mapping) == self.world and sorted(x['host'] for x in mapping) == list(range(self.world))
                    and sorted(x['jax_rank'] for x in mapping) == list(range(self.world))
                    and r['jax_rank'] == next(x['jax_rank'] for x in mapping if x['host'] == host),
                    'Host/JAX mapping is not a bijection')
            cp = r['latest_checkpoint']; path = Path(cp['path'])
            group_path = path.with_suffix('.group.json')
            require(self.record(group_path) == cp['group_sha256'], 'Checkpoint group hash differs')
            group = read_json(group_path)
            require(group['kind'] == 'visual_replicated_checkpoint_group'
                    and group['snapshot_id'] == self.snapshot.name and group['turn'] == turn
                    and group['config_sha256'] == self.config_sha and group['host_jax_mapping'] == mapping
                    and set(group['host_manifests']) == {str(h) for h in range(self.world)}
                    and group['host_manifests'][str(host)] == cp['manifest_sha256']
                    and group['replicated_arrays_elements_sha256'] == cp['replicated_arrays_elements_sha256'],
                    'Incomplete or inconsistent checkpoint group')
            saved, arrays, actors = checkpoints.read(path, expected_manifest_sha256=cp['manifest_sha256'])
            for member in path.iterdir():
                self.record(member)
            require(actors == '{}' and saved['host_rank'] == host and saved['jax_rank'] == r['jax_rank']
                    and saved['owns_replicated_arrays'] == (host == 0)
                    and saved['snapshot_id'] == self.snapshot.name and saved['config_sha256'] == self.config_sha
                    and saved['dataset_manifest_sha256'] == c['dataset']['manifest_sha256']
                    and saved['model_schema'] == self.schema and saved['turn'] == turn
                    and saved['initial_turn'] == 0 and saved['fork_lineage'] is None
                    and saved['initial_parameter_elements_sha256'] == r['initial_parameter_elements_sha256'],
                    'Saved rank state identity differs')
            for key in ('counters', 'source_runtime', 'validation_history', 'training_probe_history'):
                require(saved[key] == r[key], 'Saved diagnostic/runtime differs: ' + key)
            require(saved['overfit_history'] == r['overfit_observations'], 'Saved overfit history differs')
            self.runtime.validate_state(saved['source_runtime'], self.runtime_config, c['steps'],
                                        positions=saved['counters']['expert_positions'])
            if host == 0:
                require(str(path) == group['owner_checkpoint_path'] == cp['owner_checkpoint_path'], 'Wrong array owner')
                owner_identity = tensor_identity(arrays)
                require(owner_identity['sha256'] == group['replicated_arrays_elements_sha256'], 'Actual tensor hash differs')
                schema = [{k: row[k] for k in ('path', 'shape', 'dtype')} for row in self.schema]
                params, opt = self.codec.restore(saved['optimizer_metadata'], arrays, schema=schema,
                    configuration_sha256=self.config_sha, source_sha256=self.snapshot.name,
                    lookahead_config=self.runtime_config['lookahead'])
                require(int(opt['step']) == turn == saved['optimizer_metadata']['step'], 'Optimizer counter differs')
                require(np.array_equal(opt['hyperparameters'], self.runtime.vector(saved['source_runtime']['settings'])),
                        'Saved host schedule and device scalars differ')
                slow = opt['lookahead']['slow']
                owner_identity.update(lookahead_counter=int(opt['lookahead']['counter']),
                    fast_equals_slow=bool(slow) and all(np.array_equal(params[k], slow[k]) for k in params))
                del params, opt
            else:
                require(not arrays, 'Nonowner saved replicated arrays')
            del arrays
            import json
            rows = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines() if x]
            require([x['turn'] for x in rows] == list(range(first_turn, turn + 1)), 'Missing or repeated update rows')
            require(all(x['accepted'] == 1 and x['eligible'] == 1 for x in rows), 'Rejected update')
            for row in rows:
                require(all(math.isfinite(v) for v in row.values() if type(v) in (float, int)), 'Nonfinite update metric')
            reports[host], states[host], metrics[host], groups[host] = r, saved, rows, group
        for host in range(self.world):
            require(groups[host] == groups[0], 'Group differs between hosts')
            for key in ('source_runtime', 'counters', 'optimizer_metadata', 'validation_history',
                        'training_probe_history', 'overfit_history', 'initial_parameter_elements_sha256'):
                require(states[host][key] == states[0][key], 'Replicated scientific state differs: ' + key)
            require([trimmed(x, TIMINGS | LOCAL) for x in metrics[host]] ==
                    [trimmed(x, TIMINGS | LOCAL) for x in metrics[0]], 'Global update metrics differ across hosts')
        return dict(reports=reports, states=states, metrics=metrics, group=groups[0], arrays=owner_identity)

    def replay(self, full, prefix, resumed, boundary):
        import numpy as np
        c = self.config; buckets = c['dataset']['buckets']; entries = self.dataset.bucket_entries(buckets)
        rngs = {}
        for host, report in full['reports'].items():
            rank = report['jax_rank']
            rngs[host] = [np.random.Generator(np.random.PCG64(c['seed'] + offset)) for offset in
                          (1 + 104729 * rank, 9143, 400003 + 104729 * rank)]
        state = self.runtime.initialize(self.runtime_config, c['steps'], full['states'][0]['source_runtime']['baseline'])
        positions = slots = counter = 0
        events = {key: [] for key in ('lookahead_synchronized', 'source_epoch_flush', 'source_norm_snapshot')}
        for i in range(c['steps']):
            state, before = self.runtime.before_step(state, self.runtime_config, c['steps'])
            if before['subepoch_entry']:
                counter = 0
            count = 0
            for host, (random, bucket_rng, augmentation) in rngs.items():
                row = full['metrics'][host][i]; warm = c['dataset'].get('warmup_buckets', [])
                if i < len(warm):
                    bucket = warm[i]
                elif 'bucket_probabilities' in c['dataset']:
                    bucket = int(bucket_rng.choice(buckets, p=c['dataset']['bucket_probabilities']))
                else:
                    bucket = buckets[int(bucket_rng.integers(len(buckets)))]
                pool = entries['expert', bucket]
                chosen = [pool[int(j)] for j in random.integers(len(pool), size=c['learner']['games_per_host'])]
                symmetries = (augmentation.integers(0, 8, len(chosen)).tolist()
                              if c['learner'].get('augmentation') == 'd4' else [0] * len(chosen))
                require(row['bucket'] == bucket and row['local_entries_sha256'] ==
                        hashlib.sha256(canonical_json(chosen)).hexdigest() and row['local_symmetries'] == symmetries,
                        'Independent rank game/D4 replay differs')
                count += sum(int(self.dataset.game_info(e)['length']) for e in chosen)
                slots += len(chosen) * bucket
            row = full['metrics'][0][i]; positions += count
            require(row['positions'] == count, 'Actual global loss-position count differs')
            applied = state['settings']
            for key, expected in dict(normal_learning_rate=applied['rates']['normal'],
                normal_weight_decay=applied['decays']['normal'],
                source_sum_gradient_clip_cap=applied['source_sum_gradient_clip_cap'],
                schedule_reference_batch_positions=self.runtime_config['reference_batch_positions']).items():
                require(row[key] == float(np.float32(expected)), 'Applied source scalar differs: ' + key)
            require(row['source_epoch_entry'] == before['epoch_entry']
                    and row['source_subepoch_entry'] == before['subepoch_entry'], 'Source entry event differs')
            pre_norms = ({k: row['source_pre_norm_' + k] for k in ('input', 'normal')}
                         if before['is_print_batch'] else None)
            if pre_norms is None:
                require(row['source_pre_norm_input'] == row['source_pre_norm_normal'] == 0, 'Unexpected print norms')
            state, after = self.runtime.after_step(state, self.runtime_config, c['steps'],
                                                   positions=count, pre_update_norms=pre_norms)
            k = self.runtime_config['lookahead']['k']
            counter += int(k is not None)
            sync = k is not None and counter == k
            if sync:
                counter = 0
            require(row['lookahead_synchronized'] == sync and row['lookahead_counter'] == counter
                    and row['source_norm_snapshot'] == after['norm_snapshot']
                    and row['source_epoch_flush'] == after['epoch_end']
                    and row['source_samples'] == state['samples']
                    and row['source_refresh_count'] == state['refresh_count'], 'Source clock/event replay differs')
            for key in events:
                if row[key]:
                    events[key].append(i + 1)
            stages = [prefix] if i + 1 == boundary else ([full, resumed] if i + 1 == c['steps'] else [])
            for stage in stages:
                require(stage['arrays']['lookahead_counter'] == counter, 'Saved Lookahead clock differs')
                for host, saved in stage['states'].items():
                    require(saved['source_runtime'] == state, 'Complete source runtime replay differs')
                    require(saved['counters'] == dict(updates=i + 1, expert_positions=positions,
                                                     padded_position_slots=slots), 'Saved exposure counters differ')
                    for key, rng in zip(('numpy_rng', 'bucket_rng', 'augmentation_rng'), rngs[host]):
                        require(saved[key] == rng.bit_generator.state, 'Saved sampler replay differs: ' + key)
        return dict(events=events, position_exposures=positions, padded_position_slots=slots,
                    rank_updates_replayed=self.world * c['steps'])

    def compare(self, full, prefix, resumed, boundary):
        require(full['states'] == resumed['states'], 'Complete host states differ after fresh continuation')
        require(full['group']['host_jax_mapping'] == prefix['group']['host_jax_mapping'], 'Prefix topology changed')
        require(full['group']['host_manifests'] == resumed['group']['host_manifests'], 'Checked payload/metadata bytes differ')
        require(full['arrays'] == resumed['arrays'], 'Actual checkpoint tensors differ after continuation')
        require(not prefix['arrays']['fast_equals_slow'], 'Prefix does not exercise unsynchronized fast/slow weights')
        require(prefix['arrays']['lookahead_counter'] == self.runtime_config['lookahead']['k'] - 1,
                'Prefix must stop immediately before Lookahead synchronization')
        require(full['arrays']['fast_equals_slow'], 'Final epoch flush did not leave equal fast/slow weights')
        for host in range(self.world):
            require([trimmed(x, TIMINGS) for x in full['metrics'][host]] ==
                    [trimmed(x, TIMINGS) for x in prefix['metrics'][host] + resumed['metrics'][host]],
                    'Update metrics/draws differ after continuation')
            require(resumed['reports'][host]['resume_group_sha256'] ==
                    prefix['reports'][host]['latest_checkpoint']['group_sha256'], 'Resumed from wrong checkpoint')
        epoch_ends = []; turn = 0
        for epoch in self.runtime_config['epochs']:
            turn += sum(epoch); epoch_ends.append(turn)
        evaluations = sorted({0, self.config['steps'], *epoch_ends,
                              *range(self.config['eval_every'], self.config['steps'] + 1, self.config['eval_every'])})
        report = full['reports'][0]
        for key in ('validation_history', 'training_probe_history'):
            history = report[key]
            require([x['turn'] for x in history] == evaluations, 'Missing regular diagnostic observation')
            require(len({x['episode_ids_sha256'] for x in history}) == 1, 'Diagnostic population changed')
            for row in history:
                require(row['raw_totals']['expert_count'] == row['raw_totals']['value_count'] > 0,
                        'Policy and value diagnostic populations differ')
                require(all(math.isfinite(v) for v in row['metrics'].values()), 'Nonfinite diagnostic metric')
        require(len(report['overfit_observations']) == 4 * (len(evaluations) - 1), 'Missing policy/value overfit observations')
        replay = self.replay(full, prefix, resumed, boundary)
        return dict(status='passed', snapshot=self.snapshot.name, config_sha256=self.config_sha,
            world_size=self.world, horizon=self.config['steps'], prefix_turn=boundary,
            model_parameters=sum(x['elements'] for x in self.schema), checkpoint=full['arrays'],
            complete_all_rank_state_exact=True, checkpoint_payload_manifests_exact=True,
            update_metrics_exact_except_timings=True, diagnostic_histories_exact=True,
            evaluation_turns=evaluations, replay=replay,
            measurements={name: {str(host): {key: stage['reports'][host].get(key) for key in
                ('compiled_updates', 'segment_timing', 'segment_elapsed_seconds',
                 'peak_process_rss_kib', 'device_memory_stats')} for host in range(self.world)}
                for name, stage in [('full', full), ('prefix', prefix), ('resumed', resumed)]},
            scope='Execution/recovery only. Canonical tensor hashes and checked payload manifests establish exact identity. '
                  'Runtime replay uses the independently qualified source runtime; it is not another source-math proof. '
                  'Compiler memory analysis is distinct from device memory statistics. No learning/strength conclusion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    for stage in ('full', 'prefix', 'resumed'):
        parser.add_argument('--' + stage + '-artifacts', required=True, nargs='+', type=Path)
    parser.add_argument('--prefix-turn', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    start = time.monotonic(); audit = Audit(args.snapshot)
    require(0 < args.prefix_turn < audit.config['steps'], 'Boundary outside frozen horizon')
    full = audit.stage(args.full_artifacts, audit.config['steps'], 1)
    prefix = audit.stage(args.prefix_artifacts, args.prefix_turn, 1)
    resumed = audit.stage(args.resumed_artifacts, audit.config['steps'], args.prefix_turn + 1)
    result = audit.compare(full, prefix, resumed, args.prefix_turn)
    result.update(kind='joint_source_muon_all_rank_recovery', created=time.time(),
                  elapsed_seconds=time.monotonic() - start, inputs=audit.inputs,
                  operator_sha256=checkpoints.sha256(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(canonical_json(result))
    print(canonical_json(dict(status=result['status'], world_size=result['world_size'],
        output=str(args.output), sha256=checkpoints.sha256(args.output), seconds=result['elapsed_seconds'])).decode(), end='')


if __name__ == '__main__':
    main()
