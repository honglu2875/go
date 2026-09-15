"""Frozen position-batch SGD, with multiple true updates per compiled dispatch."""
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
from gozero.katago_sequence_batches import Dataset
from config import validate


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    checkpoints._sync_directory(path.parent)


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = arrays[key]
        h.update(canonical_json([key, list(a.shape), str(a.dtype)])); h.update(a.tobytes(order='C'))
    return h.hexdigest()


def run(args, c, report):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import sgd_optimizer as optimizer
    import policy_model as model
    import position_stream
    import checkpoint_storage

    rank, world = jax.process_index(), jax.process_count()
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if world != c['expected_processes'] or len(jax.devices()) != c['expected_devices'] or any(d.platform != c['platform'] for d in jax.devices()):
        raise ValueError('Wrong learning topology')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    replicated = NamedSharding(mesh, P())
    def replica(tree):
        return jax.tree.map(lambda x: np.asarray(x.addressable_shards[0].data), tree)
    def global_batch(batch, *, scan=False):
        sharding = NamedSharding(mesh, P(None, 'data') if scan else P('data'))
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding, x), batch)
    def gather(value):
        raw = canonical_json(value)
        if len(raw) > 32760: raise ValueError('Oversized collective metadata')
        buf = np.zeros(32768, np.uint8)
        buf[:8] = np.frombuffer(len(raw).to_bytes(8, 'little'), np.uint8)
        buf[8:8+len(raw)] = np.frombuffer(raw, np.uint8)
        rows = np.asarray(mh.process_allgather(buf)).reshape(world, 32768)
        return [json.loads(bytes(row[8:8+int.from_bytes(bytes(row[:8]), 'little')])) for row in rows]

    net, opt = c['model'], c['learner']
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    reference = read_json(Path(__file__).with_name('reference_config.json'))
    reference['steps'] = c['sampling']['reference_steps']
    reference['eval_every'] = c['evaluation']['every_reference_turns']
    records, exposure, _ = position_stream.build(data, reference, c['sampling']['shuffle_seed'], c['sampling']['position_exposures'])
    identities = gather(exposure)
    if any(x != identities[0] for x in identities): raise ValueError('Hosts generated different position tapes')
    mapping = gather({'host': host, 'jax_rank': rank})
    if sorted(x['host'] for x in mapping) != list(range(world)): raise ValueError('Invalid host mapping')
    schema = model.parameter_schema(net)
    reservation = args.output / '.checkpoint-reservation.bin'
    if host == 0:
        required = checkpoint_storage.reservation_size(schema)
        disk = os.statvfs(args.output)
        if disk.f_bavail * disk.f_frsize < required + 150_000_000:
            raise ValueError('Insufficient persistent space for a complete SGD checkpoint and logs')
        checkpoint_storage.reserve(reservation, required)
        report['checkpoint_reserved_bytes'] = required
    report.update(host_rank=host, jax_rank=rank, world_size=world, host_jax_mapping=mapping,
                  parameter_count=sum(x['elements'] for x in schema), model_schema=schema,
                  config_sha256=hashlib.sha256(canonical_json(c)).hexdigest(), exposure=exposure,
                  dataset_manifest_sha256=c['dataset']['manifest_sha256'], jax_version=jax.__version__,
                  optimizer='Nesterov SGD', batch_positions=opt['batch_size'],
                  updates_per_dispatch=opt['updates_per_dispatch'], training_positions=len(records))
    params = jax.jit(lambda: model.initialize(c['seed'], net), out_shardings=replicated)()
    state = jax.jit(optimizer.initialize, out_shardings=replicated)(params)
    jax.block_until_ready((params, state))
    report['initial_parameter_elements_sha256'] = array_digest({f'p_{i:04d}': x for i, x in enumerate(jax.tree.leaves(replica(params)))})
    _, definition = jax.tree.flatten(params)
    step = 0; initial_step = 0
    if args.resume is not None:
        group = read_json(args.resume.with_suffix('.group.json'))
        if group['snapshot_id'] != SOURCE.name or group['config_sha256'] != report['config_sha256'] or group['host_jax_mapping'] != mapping:
            raise ValueError('Resume configuration/topology changed')
        saved, _, _ = checkpoints.read(args.resume, expected_manifest_sha256=group['host_manifests'][str(host)])
        owner = Path(group['owner_checkpoint_path'])
        owner_state, arrays, _ = checkpoints.read(owner, expected_manifest_sha256=group['host_manifests']['0'])
        if array_digest(arrays) != group['replicated_arrays_elements_sha256'] or saved['exposure'] != exposure or owner_state['step'] != saved['step']:
            raise ValueError('Resume arrays or position stream changed')
        expected = {f'{k}_{i:04d}' for k in ('p', 'm') for i in range(len(schema))}
        if set(arrays) != expected: raise ValueError('Unexpected checkpoint arrays')
        def restore(prefix):
            values = [arrays[f'{prefix}_{i:04d}'] for i in range(len(schema))]
            if any(v.shape != tuple(s['shape']) or str(v.dtype) != s['dtype'] or not np.isfinite(v).all() for v, s in zip(values, schema)):
                raise ValueError('Invalid restored values')
            return jax.device_put(definition.unflatten(values), replicated)
        params = restore('p')
        state = {'momentum': restore('m'), 'step': jax.device_put(np.int32(saved['step']), replicated),
                 'samples': jax.device_put(np.int32(saved['samples']), replicated)}
        step = saved['step']; initial_step = step
        report['initial_parameter_elements_sha256'] = saved['initial_parameter_elements_sha256']
        del arrays
    steps = math.ceil(len(records) / opt['batch_size'])
    stop = steps if args.stop_after_turn is None else args.stop_after_turn
    if not step < stop <= steps: raise ValueError('Invalid stop boundary')
    timings = {k: 0. for k in ('compilation_seconds', 'sampling_seconds', 'learning_seconds', 'evaluation_seconds', 'checkpoint_seconds')}
    evaluations = {}; milestones = position_stream.evaluation_steps(exposure, opt['batch_size'])
    report['evaluation_milestones'] = milestones
    eval_steps = {m['sgd_step'] for m in milestones}

    def evaluate():
        started = time.perf_counter(); sums = {}; ids = []
        selected = data.bucket_entries(c['dataset']['buckets'], split=1)
        n = reference['learner']['games_per_host']
        for bucket in c['dataset']['buckets']:
            entries = selected['expert', bucket][:c['evaluation']['games_per_bucket']]
            ids.extend(entries); local = entries[rank::world]
            for begin in range(0, math.ceil(len(entries)/world), n):
                chosen = local[begin:begin+n]
                batch = global_batch(data.batch(chosen + [None] * (n - len(chosen)), positions=bucket))
                if bucket not in evaluations:
                    start = time.perf_counter()
                    fn = jax.shard_map(lambda p, b: model.total_metrics(model.logits(p, b, net), b, 'data', stratify=True),
                        mesh=mesh, in_specs=(P(), jax.tree.map(lambda _: P('data'), batch)), out_specs=P(), check_vma=False)
                    evaluations[bucket] = jax.jit(fn).lower(params, batch).compile()
                    timings['compilation_seconds'] += time.perf_counter() - start
                totals = {k: float(v) for k, v in replica(evaluations[bucket](params, batch)).items()}
                if any(not math.isfinite(v) for v in totals.values()): raise FloatingPointError('Nonfinite validation')
                for k, v in totals.items(): sums[k] = sums.get(k, 0.) + v
        result = {'step': step, 'position_exposures': min(step * opt['batch_size'], len(records)), 'split': 1,
                  'episode_ids_sha256': hashlib.sha256(canonical_json(ids)).hexdigest(), 'raw_totals': sums,
                  'metrics': {k: float(v) for k, v in model.averages(sums).items()},
                  'learning_seconds': timings['learning_seconds']}
        timings['evaluation_seconds'] += time.perf_counter() - started
        print(json.dumps({'kind': 'sgd_validation', 'host': host, **result}), flush=True)
        return result

    report['initial_validation'] = evaluate()
    sample_batch = global_batch(position_stream.batch(data, records, step, opt['updates_per_dispatch'], opt['batch_size'], world, rank), scan=True)
    single_specs = {k: P('data') for k in sample_batch}
    objective = jax.shard_map(lambda p, b: model.losses(p, b, net, axis_name='data'),
        mesh=mesh, in_specs=(P(), single_specs), out_specs=P(), check_vma=False)
    start = time.perf_counter()
    fn = optimizer.scan_updates(objective, opt)
    lowered = jax.jit(fn, donate_argnums=(0, 1)).lower(params, state, sample_batch)
    compiled = lowered.compile(); mem = compiled.memory_analysis()
    report['compiled_update'] = {'compile_seconds': time.perf_counter()-start,
        'compiler_cost_estimate': compiled.cost_analysis(),
        'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
        'memory_bytes': {k: int(getattr(mem, k)) for k in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}}
    timings['compilation_seconds'] += time.perf_counter()-start
    print(json.dumps({'kind': 'sgd_compiled', 'host': host, **report['compiled_update']}), flush=True)
    history = []; metrics_digest = hashlib.sha256(); dispatch = 0
    with (args.output / 'metrics.jsonl').open('x') as log:
        while step < stop:
            # End a dispatch at an evaluation boundary. Empty padding is skipped
            # in the same executable and does not create optimizer updates.
            boundary = min([stop, *[x for x in eval_steps if x > step]])
            active = min(opt['updates_per_dispatch'], boundary - step)
            start = time.perf_counter()
            raw = position_stream.batch(data, records, step, opt['updates_per_dispatch'], opt['batch_size'], world, rank)
            for key in raw:
                if key == 'legal': raw[key][active:] = True
                else: raw[key][active:] = 0
            batch = global_batch(raw, scan=True)
            timings['sampling_seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            params, state, observed, healthy = compiled(params, state, batch)
            observed = replica(observed); healthy = bool(replica(healthy))
            timings['learning_seconds'] += time.perf_counter()-start
            rows = [{k: float(v[i]) for k, v in observed.items()} for i in range(active)]
            invalid = not healthy or any(not r['accepted'] or any(not math.isfinite(v) for v in r.values()) for r in rows)
            payload = {'first_step': step + 1, 'active_updates': active,
                       'rows': [{k: v if math.isfinite(v) else str(v) for k, v in r.items()} for r in rows]}
            encoded = canonical_json(payload)
            stored = encoded if host == 0 else canonical_json({'first_step': step + 1, 'active_updates': active,
                                                               'metrics_sha256': hashlib.sha256(encoded).hexdigest()})
            log.write(stored.decode()); log.flush(); metrics_digest.update(encoded)
            if invalid:
                raise FloatingPointError('SGD update rejected; remaining updates in its dispatch were skipped')
            if any(int(r['update']) != step+i+1 for i, r in enumerate(rows)): raise ValueError('Unexpected update sequence')
            step += active; dispatch += 1
            if dispatch % 8 == 0:
                print(json.dumps({'kind': 'sgd_progress', 'host': host, 'step': step, 'total_steps': steps,
                    'samples': min(step*opt['batch_size'], len(records)), 'last': rows[-1], 'timing': timings}), flush=True)
            if step in eval_steps: history.append(evaluate())
    if len(set(gather(metrics_digest.hexdigest()))) != 1: raise ValueError('Replicated update metrics differ')
    start = time.perf_counter()
    arrays = {f'{kind}_{i:04d}': x for kind, tree in [('p', params), ('m', state['momentum'])]
              for i, x in enumerate(jax.tree.leaves(replica(tree)))}
    identity = array_digest(arrays)
    if len(set(gather(identity))) != 1: raise ValueError('Model or momentum differs across ranks')
    path = args.output / 'checkpoints' / f'turn-{step:09d}'
    saved = {'schema_version': 1, 'kind': 'position_sgd_rank_state', 'snapshot_id': SOURCE.name,
        'config_sha256': report['config_sha256'], 'host_rank': host, 'jax_rank': rank, 'model_schema': schema,
        'step': step, 'samples': min(step*opt['batch_size'], len(records)), 'exposure': exposure,
        'initial_parameter_elements_sha256': report['initial_parameter_elements_sha256']}
    if host == 0:
        manifest = checkpoint_storage.write(path, reservation=reservation, state=saved, arrays=arrays,
            scratch_root='/dev/shm' if c['platform'] == 'tpu' else '/tmp')
    else:
        manifest = checkpoints.write(path, state=saved, arrays={}, actors='{}', compress=True)
    del arrays
    saved_hosts = gather({'host': host, 'path': str(path), 'manifest_sha256': manifest})
    group = {'schema_version': 1, 'kind': 'position_sgd_checkpoint_group', 'snapshot_id': SOURCE.name,
        'config_sha256': report['config_sha256'], 'host_jax_mapping': mapping,
        'host_manifests': {str(r['host']): r['manifest_sha256'] for r in saved_hosts},
        'owner_checkpoint_path': next(r['path'] for r in saved_hosts if r['host'] == 0),
        'replicated_arrays_elements_sha256': identity}
    publish(path.with_suffix('.group.json'), group)
    timings['checkpoint_seconds'] += time.perf_counter()-start
    report.update(status='passed', step=step, initial_step=initial_step, training_complete=step == steps,
        total_steps=steps, segment_timing=timings, validation_history=history,
        training_metrics_sha256=metrics_digest.hexdigest(),
        latest_checkpoint={'path': str(path), 'manifest_sha256': manifest,
            'group_sha256': checkpoints.sha256(path.with_suffix('.group.json')), **group},
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    verify(SOURCE); mh.sync_global_devices('position-sgd-complete')


def entry(args):
    c = validate(read_json(args.config)); verify(SOURCE)
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')): raise ValueError('Configuration is not frozen')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / 'resolved_config.json', c)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
              'claims_go_strength': False, 'claims_rl_sample_efficiency': False}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu': jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time()); publish(args.output / 'result.json', report)
        if distributed: jax.distributed.shutdown()
