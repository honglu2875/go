"""Cloneable pure-JAX whole-history learning with shared replicated checkpoints."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset
from learning_config import validate


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path); checkpoints._sync_directory(path.parent)


def run(args, c, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import learner
    import metrics as measurement
    import model
    rank, world = jax.process_index(), jax.process_count()
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if (world != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Learning topology differs from frozen contract')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    def replica(tree):
        return jax.tree.map(lambda x: np.asarray(x.addressable_shards[0].data), tree)
    def gather_json(value):
        raw = canonical_json(value)
        if len(raw) > 32760:
            raise ValueError('Checkpoint collective metadata exceeds bound')
        buffer = np.zeros(32768, np.uint8); buffer[:8] = np.frombuffer(len(raw).to_bytes(8, 'little'), np.uint8)
        buffer[8:8+len(raw)] = np.frombuffer(raw, np.uint8)
        rows = np.asarray(mh.process_allgather(buffer)).reshape(world, 32768)
        return [json.loads(bytes(row[8:8+int.from_bytes(bytes(row[:8]), 'little')])) for row in rows]
    def digest_arrays(arrays):
        h = hashlib.sha256()
        for key in sorted(arrays):
            a = np.asarray(arrays[key]); h.update(canonical_json([key, list(a.shape), str(a.dtype)])); h.update(a.tobytes(order='C'))
        return h.hexdigest()
    def global_batch(batch):
        return {k: jax.device_put(v, replicated) if k == 'loss_weights' else
                jax.make_array_from_process_local_data(batched, v) for k, v in batch.items()}

    net, opt, exits = c['model'], c['learner'], c['exits']
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    buckets = c['dataset']['buckets']; by_bucket = data.bucket_entries(buckets)
    if data.size > net['max_board_size'] or data.time > net['max_positions']:
        raise ValueError('Dataset exceeds exact model context')
    # Bucket choice and uniform episodes within each role/bucket are explicit;
    # both arms get identical draws and exposures. Rare long games must not be
    # silently oversampled simply to exercise a compilation bucket.
    if any(not by_bucket[role, b] for role in ('expert', 'behavior') for b in buckets):
        raise ValueError('Every bucket needs both training populations')
    config_sha = hashlib.sha256(canonical_json(c)).hexdigest()
    mapping = gather_json({'host': host, 'jax_rank': rank})
    if sorted(x['host'] for x in mapping) != list(range(world)):
        raise ValueError('Host/JAX mapping is not a bijection')
    schema = model.parameter_schema(net)
    report.update(host_rank=host, jax_rank=rank, world_size=world, host_jax_mapping=mapping,
                  parameter_count=sum(x['elements'] for x in schema), model_schema=schema,
                  config_sha256=config_sha, jax_version=jax.__version__,
                  teacher_cost=data.parent_manifest['spec']['teacher_cost'],
                  dataset_manifest_sha256=c['dataset']['manifest_sha256'],
                  global_sequences_per_batch=world * 2 * opt['games_per_role'],
                  sequences_per_device=2 * opt['games_per_role'] // len(jax.local_devices()),
                  training_population={f'{role}-{b}': len(v) for (role, b), v in by_bucket.items()},
                  sampling={'bucket_probabilities': c['dataset'].get('bucket_probabilities', [1 / len(buckets)] * len(buckets)),
                            'initial_bucket_sequence': c['dataset'].get('warmup_buckets', []),
                            'within_bucket': 'Uniform complete episodes per role, with replacement.'})
    print(json.dumps({'kind': 'visual_learning_initialize', 'parameters': report['parameter_count'], 'host': host}), flush=True)
    params = jax.jit(lambda: model.initialize(c['seed'], net), out_shardings=replicated)()
    state = jax.jit(learner.initialize, out_shardings=replicated)(params)
    jax.block_until_ready((params, state))
    leaves, definition = jax.tree.flatten(params)
    initial = {f'p_{i:04d}': x for i, x in enumerate(jax.tree.leaves(replica(params)))}
    report['initial_parameter_elements_sha256'] = digest_arrays(initial)
    del initial
    random = np.random.Generator(np.random.PCG64(c['seed'] + 1 + 104729 * rank))
    bucket_random = np.random.Generator(np.random.PCG64(c['seed'] + 9143))
    turn = 0
    counters = {'updates': 0, 'expert_positions': 0, 'behavior_positions': 0, 'padded_position_slots': 0}
    timing = {k: 0. for k in ['sampling_seconds', 'learning_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'compilation_seconds']}
    if args.resume is not None:
        group = read_json(args.resume.with_suffix('.group.json'))
        if (group['kind'] != 'visual_replicated_checkpoint_group' or group['snapshot_id'] != SOURCE.name
                or group['config_sha256'] != config_sha or group['host_jax_mapping'] != mapping):
            raise ValueError('Resume source, configuration or topology differs')
        identities = gather_json(checkpoints.sha256(args.resume.with_suffix('.group.json')))
        if len(set(identities)) != 1:
            raise ValueError('Checkpoint group differs across ranks')
        saved, local_arrays, _ = checkpoints.read(args.resume, expected_manifest_sha256=group['host_manifests'][str(host)])
        if (saved['host_rank'] != host or saved['jax_rank'] != rank or saved['model_schema'] != schema
                or saved['snapshot_id'] != SOURCE.name or saved['dataset_manifest_sha256'] != c['dataset']['manifest_sha256']):
            raise ValueError('Rank scientific state differs')
        owner = Path(group['owner_checkpoint_path'])
        owner_state, arrays, _ = checkpoints.read(owner, expected_manifest_sha256=group['host_manifests']['0'])
        if owner_state['turn'] != saved['turn'] or digest_arrays(arrays) != group['replicated_arrays_elements_sha256']:
            raise ValueError('Shared model/optimizer checkpoint differs')
        required = {f'{kind}_{i:04d}' for kind in ('p', 'm', 'v') for i in range(len(leaves))}
        if set(arrays) != required or (host != 0 and local_arrays):
            raise ValueError('Shared checkpoint array coverage differs')
        def restore(kind):
            values = [arrays[f'{kind}_{i:04d}'] for i in range(len(leaves))]
            if any(x.shape != tuple(s['shape']) or str(x.dtype) != s['dtype'] or not np.isfinite(x).all() for x, s in zip(values, schema)):
                raise ValueError('Invalid restored parameter or moment array')
            return jax.device_put(definition.unflatten(values), replicated)
        params = restore('p'); state = {'first': restore('m'), 'second': restore('v'),
                                      'step': jax.device_put(np.asarray(saved['turn'], np.int32), replicated)}
        turn = saved['turn']; counters = saved['counters']
        random.bit_generator.state = saved['numpy_rng']; bucket_random.bit_generator.state = saved['bucket_rng']
        report['resume_group_sha256'] = identities[0]
        del arrays, local_arrays
    stop = c['steps'] if args.stop_after_turn is None else args.stop_after_turn
    if not turn < stop <= c['steps']:
        raise ValueError('Stop must advance the saved state within its frozen schedule')
    def schedule(iteration):
        progress = jnp.clip((iteration - opt['warmup_steps']) / max(1, c['steps'] - opt['warmup_steps']), 0., 1.)
        rate = opt['end_learning_rate'] + .5 * (opt['learning_rate'] - opt['end_learning_rate']) * (1 + jnp.cos(jnp.pi * progress))
        return rate * jnp.minimum(1., iteration / max(1, opt['warmup_steps']))
    def local_loss(p, batch):
        loss, result = model.losses(p, batch, net, axis_name='data',
            exit_depths=tuple(exits['depths']) if exits['loss_weight'] else (),
            exit_loss_weight=exits['loss_weight'], exit_temperature=exits['temperature'])
        live = jnp.arange(batch['actions'].shape[1])[None, :] < batch['counts'][:, None]
        return loss, {**result, **{role + '_positions': jax.lax.psum(jnp.sum(batch[role + '_mask'] * live), 'data')
                                  for role in ('expert', 'behavior')}}
    def specs(batch):
        return {k: P() if k == 'loss_weights' else P('data') for k in batch}
    compiled_steps, compiled_evaluations = {}, {}
    def get_step(batch):
        bucket = batch['actions'].shape[1]
        if bucket not in compiled_steps:
            start = time.perf_counter()
            objective = jax.shard_map(local_loss, mesh=mesh, in_specs=(P(), specs(batch)), out_specs=P(), check_vma=False)
            def update(p, s, b):
                (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(p, b)
                rate = schedule(s['step'] + 1)
                p, s, extra = learner.apply_gradient(p, s, gradient, loss, learning_rate=rate,
                    **{k: opt[k] for k in ['beta1', 'beta2', 'epsilon', 'weight_decay', 'max_grad_norm']})
                return p, s, {**metrics, **extra, 'learning_rate': rate}
            print(json.dumps({'kind': 'visual_compile_update', 'bucket': bucket, 'host': host}), flush=True)
            lowered = jax.jit(update, donate_argnums=(0, 1)).lower(params, state, batch)
            executable = lowered.compile(); compiled_steps[bucket] = executable
            memory = executable.memory_analysis()
            report.setdefault('compiled_updates', {})[str(bucket)] = {
                'compile_seconds': time.perf_counter() - start,
                'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
                'memory_bytes': {k: int(getattr(memory, k)) for k in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']}}
            timing['compilation_seconds'] += time.perf_counter() - start
        return compiled_steps[bucket]
    def evaluate(split):
        start = time.perf_counter(); sums = {}; all_ids = []
        selected_buckets = data.bucket_entries(buckets, split=split)
        batch_size = 2 * opt['games_per_role']
        for bucket in buckets:
            entries = [entry for role in ('expert', 'behavior') for entry in selected_buckets[role, bucket][:c['evaluation']['games_per_role_per_bucket']]]
            all_ids.extend(entries); local = entries[rank::world]
            for begin in range(0, math.ceil(len(entries) / world), batch_size):
                chosen = local[begin:begin+batch_size]
                batch = global_batch(data.batch(chosen + [None] * (batch_size - len(chosen)), positions=bucket, loss_weights=opt['loss_weights']))
                if bucket not in compiled_evaluations:
                    start_compile = time.perf_counter()
                    fn = jax.shard_map(lambda p, b: measurement.totals(p, b, net, tuple(exits['depths']), axis_name='data'),
                        mesh=mesh, in_specs=(P(), specs(batch)), out_specs=P(), check_vma=False)
                    print(json.dumps({'kind': 'visual_compile_evaluation', 'bucket': bucket, 'host': host}), flush=True)
                    compiled_evaluations[bucket] = jax.jit(fn).lower(params, batch).compile()
                    timing['compilation_seconds'] += time.perf_counter() - start_compile
                totals = {k: float(v) for k, v in replica(compiled_evaluations[bucket](params, batch)).items()}
                if not all(math.isfinite(x) for x in totals.values()):
                    raise FloatingPointError('Nonfinite held-out observation')
                for k, v in totals.items():
                    sums[k] = sums.get(k, 0.) + v
        result = {'turn': turn, 'split': split, 'episode_ids_sha256': hashlib.sha256(canonical_json(all_ids)).hexdigest(),
                  'raw_totals': sums, 'metrics': measurement.averages(sums)}
        timing['evaluation_seconds'] += time.perf_counter() - start
        print(json.dumps({'kind': 'visual_heldout', 'host': host, **result}), flush=True)
        return result
    def save():
        start = time.perf_counter()
        arrays = {f'{kind}_{i:04d}': x for kind, tree in [('p', params), ('m', state['first']), ('v', state['second'])]
                  for i, x in enumerate(jax.tree.leaves(replica(tree)))}
        digest = digest_arrays(arrays)
        if len(set(gather_json(digest))) != 1:
            raise ValueError('Replicated parameters or optimizer diverged')
        path = args.output / 'checkpoints' / f'turn-{turn:09d}'
        saved = {'schema_version': 1, 'kind': 'visual_replicated_rank_state', 'snapshot_id': SOURCE.name,
                 'config_sha256': config_sha, 'host_rank': host, 'jax_rank': rank, 'turn': turn,
                 'dataset_manifest_sha256': c['dataset']['manifest_sha256'], 'model_schema': schema,
                 'numpy_rng': random.bit_generator.state, 'bucket_rng': bucket_random.bit_generator.state,
                 'counters': counters, 'owns_replicated_arrays': host == 0}
        identity = checkpoints.write(path, state=saved, arrays=arrays if host == 0 else {}, actors='{}', compress=True)
        del arrays
        records = gather_json({'host': host, 'manifest_sha256': identity, 'path': str(path)})
        group = {'schema_version': 1, 'kind': 'visual_replicated_checkpoint_group', 'snapshot_id': SOURCE.name,
                 'config_sha256': config_sha, 'turn': turn, 'host_jax_mapping': mapping,
                 'host_manifests': {str(r['host']): r['manifest_sha256'] for r in records},
                 'owner_checkpoint_path': next(r['path'] for r in records if r['host'] == 0),
                 'replicated_arrays_elements_sha256': digest}
        publish(path.with_suffix('.group.json'), group)
        report['latest_checkpoint'] = {'path': str(path), 'manifest_sha256': identity,
            'group_sha256': checkpoints.sha256(path.with_suffix('.group.json')),
            'replicated_arrays_elements_sha256': digest, 'owner_checkpoint_path': group['owner_checkpoint_path']}
        timing['checkpoint_seconds'] += time.perf_counter() - start
    segment_start = time.perf_counter()
    history = []
    if turn == 0:
        report['initial_validation'] = evaluate(1)
    with (args.output / 'metrics.jsonl').open('x') as log:
        while turn < stop:
            start = time.perf_counter()
            warmup_buckets = c['dataset'].get('warmup_buckets', [])
            if turn < len(warmup_buckets):
                bucket = warmup_buckets[turn]
            elif 'bucket_probabilities' in c['dataset']:
                bucket = int(bucket_random.choice(buckets, p=c['dataset']['bucket_probabilities']))
            else:
                bucket = buckets[int(bucket_random.integers(len(buckets)))]
            entries = [by_bucket[role, bucket][int(i)] for role in ('expert', 'behavior')
                       for i in random.integers(len(by_bucket[role, bucket]), size=opt['games_per_role'])]
            raw = data.batch(entries, positions=bucket, loss_weights=opt['loss_weights'])
            batch = global_batch(raw)
            timing['sampling_seconds'] += time.perf_counter() - start
            step = get_step(batch); start = time.perf_counter()
            params, state, metrics = step(params, state, batch)
            observed = {k: float(v) for k, v in replica(metrics).items()}
            timing['learning_seconds'] += time.perf_counter() - start
            if not all(math.isfinite(v) for v in observed.values()) or not observed['accepted']:
                raise FloatingPointError('Rejected or nonfinite visual learner update')
            turn += 1; counters['updates'] = turn
            for role in ('expert', 'behavior'):
                counters[role + '_positions'] += int(observed[role + '_positions'])
            counters['padded_position_slots'] += world * len(entries) * bucket
            record = {'turn': turn, 'bucket': bucket, 'local_entries_sha256': hashlib.sha256(canonical_json(entries)).hexdigest(), **observed}
            log.write(json.dumps(record, sort_keys=True) + '\n'); log.flush()
            if turn % c['log_every'] == 0:
                print(json.dumps({'kind': 'visual_update', 'host': host, **record}), flush=True)
            if turn % c['eval_every'] == 0 or turn == c['steps']:
                history.append(evaluate(1))
            if turn % c['checkpoint_every'] == 0 or turn == stop:
                save()
    if turn == c['steps'] and c['evaluation']['run_test']:
        report['test'] = evaluate(2)
    # The owner checkpoint already contains exact parameters; avoid a redundant
    # near-gigabyte export. A separately pinned inference descriptor can select
    # its p_ arrays through the standard non-pickle checkpoint reader.
    report.update(status='passed', turn=turn, training_complete=turn == c['steps'],
                  counters=counters, segment_timing=timing, validation_history=history,
                  last_metrics=observed, segment_elapsed_seconds=time.perf_counter() - segment_start,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  device_memory_stats=[d.memory_stats() for d in jax.local_devices()])
    verify(SOURCE); mh.sync_global_devices('visual-learning-complete')


def entry(args):
    c = validate(read_json(args.config)); verify(SOURCE)
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')):
        raise ValueError('Training configuration is not frozen')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / 'resolved_config.json', c)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
              'claims_go_strength': False, 'claims_rl_sample_efficiency': False,
              'input_kind': 'Pinned exact complete weak-teacher Go histories; offline learning.'}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        publish(args.output / 'result.json', report)
        if distributed:
            jax.distributed.shutdown()
