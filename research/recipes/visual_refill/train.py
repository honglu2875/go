#!/usr/bin/env python3
"""Bounded model qualification and supplied-path profiling, without Go training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from config import validate


def execute(c, output, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import fixtures
    import model
    import observations

    if (jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Runtime differs from frozen topology')
    mesh = Mesh(np.asarray(jax.local_devices()), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    report.update(jax_rank=jax.process_index(), world=jax.process_count(),
                  devices=[str(x) for x in jax.local_devices()], jax_version=jax.__version__,
                  logical_cpus=sorted(os.sched_getaffinity(0)))
    net, probe = c['model'], c['probe']
    cached_backend = probe.get('cached_backend', 'xla')
    report['cached_backend'] = cached_backend
    batch, positions, horizon, size = (probe[x] for x in ['sequences_per_host', 'positions', 'draft_plies', 'board_size'])
    report.update(sequences_per_host=batch, sequences_per_device=batch // len(jax.local_devices()),
                  sequences_across_pod=batch * jax.process_count(),
                  positions_per_sequence=positions, proposed_plies=horizon)
    schema = model.parameter_schema(net)
    (output / 'parameter_schema.json').write_bytes(canonical_json(schema))
    report['parameter_count'] = sum(row['elements'] for row in schema)
    report['parameter_schema_sha256'] = hashlib.sha256(canonical_json(schema)).hexdigest()
    report['parameters_float32_bytes'] = 4 * report['parameter_count']
    stages = report.setdefault('stages', {})

    def event(stage):
        print(json.dumps({'kind': 'visual_causal_stage', 'stage': stage, 'jax_rank': jax.process_index()}), flush=True)

    event('initialize')
    start = time.perf_counter()
    params = jax.jit(lambda: model.initialize(c['seed'], net), out_shardings=replicated)()
    jax.block_until_ready(params)
    stages['initialize_seconds'] = time.perf_counter() - start
    # Identical synthetic inputs on all hosts permit a separate rank check.
    obs, act, _ = fixtures.inputs(c['seed'] + 1, batch, positions + horizon, size)
    initial_counts = np.full(batch, positions, np.int32) - (np.arange(batch, dtype=np.int32) % 2 if positions > 1 else 0)
    final_counts = initial_counts + horizon
    observations.validate_inputs(obs, act, final_counts, net)
    rows = np.arange(batch)[:, None]
    future = obs[rows, initial_counts[:, None] + np.arange(horizon)[None, :]]
    proposed = act[rows, initial_counts[:, None] - 1 + np.arange(horizon)[None, :]]
    bucket = min(net['max_positions'], 1 << (positions + horizon - 1).bit_length())
    report['cache_positions_bucket'] = bucket
    stride = model.layout(size, net)['stride']
    report['tokens_per_position'] = stride
    report['block_query_token_rows_per_device'] = batch // len(jax.local_devices()) * horizon * stride
    def put(x):
        return jax.device_put(x, batched)
    inputs = (put(obs[:, :positions]), put(act[:, :positions]), put(initial_counts))
    future, proposed, added = put(future), put(proposed), put(np.full(batch, horizon, np.int32))
    full_inputs = (put(obs), put(act), put(final_counts))

    def compile_record(name, function, args):
        event('compile_' + name)
        started = time.perf_counter()
        lowered = jax.jit(function).lower(*args)
        ir = lowered.compiler_ir('hlo').as_hlo_text().encode()
        stages[name] = {'hlo_sha256': hashlib.sha256(ir).hexdigest(), 'hlo_bytes': len(ir)}
        executable = lowered.compile()
        stages[name]['compile_seconds'] = time.perf_counter() - started
        memory = executable.memory_analysis()
        if memory is not None:
            stages[name]['compiled_memory_bytes'] = {key: int(getattr(memory, key)) for key in
                ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']}
        cost = executable.cost_analysis()
        stages[name]['compiler_cost_estimate'] = cost
        started = time.perf_counter()
        result = executable(*args)
        jax.block_until_ready(result)
        stages[name]['warmup_seconds'] = time.perf_counter() - started
        return executable, result

    # TPU Mosaic kernels require explicit manual partitioning. Each host runs
    # its independent local-device mesh; no inference collectives cross hosts.
    cache_specs = {'keys': tuple(P('data') for _ in range(net['layers'])),
                   'values': tuple(P('data') for _ in range(net['layers'])),
                   'lengths': P('data'), 'valid': P('data'), 'network_version': P()}
    forward_specs = (P(), P('data'), P('data'), P('data'))
    prefill_function = jax.shard_map(
        lambda p, o, a, n: model.forward(p, o, a, n, net, with_cache=True, network_version=1, cache_positions=bucket),
        mesh=mesh, in_specs=forward_specs, out_specs=(P('data'), cache_specs), check_vma=False)
    full_function = jax.shard_map(lambda p, o, a, n: model.forward(p, o, a, n, net),
        mesh=mesh, in_specs=forward_specs, out_specs=P('data'), check_vma=False)
    prefill, (initial, cache) = compile_record('prefill', prefill_function, (params, *inputs))
    report['logical_cache_kv_bytes_per_host'] = sum(x.size * x.dtype.itemsize for key in ['keys', 'values'] for x in jax.tree.leaves(cache[key]))
    full, expected = compile_record('full_reference', full_function, (params, *full_inputs))
    block_fn = lambda p, ca, o, a, n: model.score_continuation(p, ca, o, a, n, net, network_version=1, cached_backend=cached_backend)
    cached_specs = (P(), cache_specs, P('data'), P('data'), P('data'))
    block_fn = jax.shard_map(block_fn, mesh=mesh, in_specs=cached_specs,
                            out_specs=(P('data'), cache_specs), check_vma=False)
    block, (block_out, final_cache) = compile_record('block_score', block_fn, (params, cache, future, proposed, added))

    def sequential_fn(p, ca, o, a, n):
        def step(current, index):
            next_obs = jax.lax.dynamic_slice_in_dim(o, index, 1, axis=1)
            next_actions = jax.lax.dynamic_slice_in_dim(a, index, 1, axis=1)
            out, current = model.score_continuation(p, current, next_obs, next_actions,
                (index < n).astype(jnp.int32), net, network_version=1, cached_backend=cached_backend)
            return current, jax.tree.map(lambda x: x[:, 0], out)
        current, out = jax.lax.scan(step, ca, jnp.arange(o.shape[1]))
        return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), out), current
    sequential_fn = jax.shard_map(sequential_fn, mesh=mesh, in_specs=cached_specs,
                                  out_specs=(P('data'), cache_specs), check_vma=False)
    sequential, (seq_out, seq_cache) = compile_record('sequential_score', sequential_fn, (params, cache, future, proposed, added))
    event('compare')
    reference, actual, serial = map(jax.device_get, (expected, block_out, seq_out))
    errors = {}
    for key in actual:
        target = reference[key][rows, initial_counts[:, None] + np.arange(horizon)[None, :]]
        if not np.isfinite(actual[key]).all() or not np.isfinite(serial[key]).all():
            raise FloatingPointError('Nonfinite output ' + key)
        errors[key] = {'block_vs_full_max_abs': float(np.max(np.abs(actual[key] - target))),
                       'block_vs_sequential_max_abs': float(np.max(np.abs(actual[key] - serial[key])))}
        np.testing.assert_allclose(actual[key], target, atol=probe['cache_tolerance'], rtol=probe['cache_tolerance'])
        np.testing.assert_allclose(actual[key], serial[key], atol=probe['cache_tolerance'], rtol=probe['cache_tolerance'])
    report['prediction_errors'] = errors
    # Compare caches on device to avoid copying large full-model state to Python.
    def cache_error(a, b):
        return jnp.max(jnp.stack([jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))
            for key in ['keys', 'values'] for x, y in zip(a[key], b[key])]))
    report['cache_max_abs_error'] = float(jax.jit(cache_error)(final_cache, seq_cache))
    np.testing.assert_array_equal(np.asarray(final_cache['lengths']), final_counts * stride - 1)
    if not np.asarray(final_cache['valid']).all() or report['cache_max_abs_error'] > probe['cache_tolerance']:
        raise ValueError('Cache qualification failed')
    # This digest checks actual prediction bytes across hosts, not model identity.
    prediction_hash = hashlib.sha256()
    for key in sorted(actual):
        prediction_hash.update(key.encode()); prediction_hash.update(actual[key].tobytes())
    hashes = np.asarray(mh.process_allgather(np.frombuffer(prediction_hash.digest(), np.uint8))).reshape(jax.process_count(), 32)
    report['prediction_sha256_per_rank'] = [bytes(x).hex() for x in hashes]
    if len(set(report['prediction_sha256_per_rank'])) != 1:
        raise ValueError('Independent local inference differs across hosts')

    event('timing')
    timings = {key: [] for key in ['block', 'sequential']}
    for repetition in range(probe['repetitions']):
        order = [('block', block), ('sequential', sequential)]
        if repetition % 2:
            order.reverse()
        for name, executable in order:
            mh.sync_global_devices(f'visual-causal-{repetition}-{name}')
            started = time.perf_counter()
            result = executable(params, cache, future, proposed, added)
            jax.block_until_ready(result)
            timings[name].append(time.perf_counter() - started)
    report['supplied_path_timing_seconds'] = timings
    report['timing_scope'] = 'Device-resident supplied paths; excludes drafting, Rust, queueing and acceptance. Not rollout latency.'
    if probe.get('profile_compute', False):
        # Profile after untraced timing. Only the controller host traces its
        # local chips. Explicit compute tracing is needed beyond XLA timelines.
        mh.sync_global_devices('visual-profile-start')
        if int(os.environ.get('GOZERO_HOST_RANK', '0')) == 0:
            options = jax.profiler.ProfileOptions()
            options.advanced_configuration = {'tpu_trace_mode': 'TRACE_COMPUTE_AND_SYNC',
                'tpu_perf_counters': True, 'tpu_num_chips_to_profile_per_task': 1,
                'tpu_enable_periodic_counter_sampling': True}
            report['profile_options'] = options.advanced_configuration
            jax.profiler.start_trace(str(output / 'compute_profile'), profiler_options=options)
            try:
                for name, executable in [('block', block), ('sequential', sequential)]:
                    with jax.profiler.TraceAnnotation('visual_supplied_path_' + name):
                        for _ in range(16):
                            result = executable(params, cache, future, proposed, added)
                            jax.block_until_ready(result)
            finally:
                jax.profiler.stop_trace()
        mh.sync_global_devices('visual-profile-complete')
    if probe['gradient_check']:
        n = min(4, positions)
        batch_data = fixtures.loss_batch(obs[:, :n], act[:, :n], np.full(batch, n, np.int32))
        batch_data = {key: jax.device_put(value, replicated if key == 'loss_weights' else batched) for key, value in batch_data.items()}
        batch_specs = {key: P() if key == 'loss_weights' else P('data') for key in batch_data}
        def local_loss(p, b):
            return model.losses(p, b, net, axis_name='data')
        global_loss = jax.shard_map(local_loss, mesh=mesh, in_specs=(P(), batch_specs), out_specs=P(), check_vma=False)
        def gradient_fn(p, b):
            (loss, metrics), gradient = jax.value_and_grad(global_loss, has_aux=True)(p, b)
            norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree.leaves(gradient)))
            return loss, norm, metrics
        _, gradient = compile_record('gradient', gradient_fn, (params, batch_data))
        loss, norm, metrics = jax.device_get(gradient)
        report['gradient'] = {'positions': n, 'loss': float(loss), 'norm': float(norm),
                              'metrics': {k: float(v) for k, v in metrics.items()}}
        if not np.isfinite(loss) or not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError('Invalid full-model gradient')
    report['device_memory_stats'] = [device.memory_stats() for device in jax.local_devices()]
    report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['status'] = 'passed'
    verify(SOURCE)
    mh.sync_global_devices('visual-causal-complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-turn', type=int)
    args = parser.parse_args()
    if read_json(args.config).get('kind') == 'visual_selfplay_generation':
        from generate import entry
        return entry(args)
    if read_json(args.config).get('kind') == 'visual_native_suffix_profile':
        from profile_replay import entry
        return entry(args)
    if read_json(args.config).get('kind') == 'visual_precision_profile':
        from profile_precision import entry
        return entry(args)
    if read_json(args.config).get('kind') == 'visual_causal_distillation':
        from train_learning import entry
        return entry(args)
    if read_json(args.config).get('kind') == 'visual_inference_service':
        from train_service import entry
        return entry(args)
    if args.resume is not None or args.stop_after_turn is not None:
        parser.error('Model-only qualification has no training state to resume')
    verify(SOURCE)
    c = validate(read_json(args.config))
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')):
        raise ValueError('Configuration is not frozen')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved_config.json').write_bytes(canonical_json(c))
    report = {'schema_version': 1, 'kind': 'visual_causal_model_qualification',
              'snapshot_id': SOURCE.name, 'status': 'running', 'trained': False,
              'input_kind': 'synthetic tensors, not legal Go trajectories',
              'claims_go_strength': False, 'claims_mfu': False, 'claims_rollout_speedup': False}
    started = time.perf_counter()
    distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90)
            distributed = True
        execute(c, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({'status': report['status'], 'output': str(args.output),
                          'parameter_count': report.get('parameter_count'), 'error': report.get('error')}), flush=True)
        if distributed:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
