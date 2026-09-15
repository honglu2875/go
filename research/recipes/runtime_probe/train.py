"""Plain-JAX distributed update, checkpoint, and compute qualification.

This is a numerical/runtime probe, not a Go learning experiment.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gozero.snapshots import canonical_json, read_json


def init_params(width, outputs):
    rng = np.random.default_rng(27)
    return {'w': rng.normal(0, 0.02, (width, outputs)).astype(np.float32),
            'b': np.zeros(outputs, np.float32)}


def model(params, x):
    return x @ params['w'] + params['b']


def train_step(params, velocity, x, y, learning_rate, momentum):
    def loss(p):
        return jnp.mean(jnp.square(model(p, x) - y))
    value, gradients = jax.value_and_grad(loss)(params)
    velocity = jax.tree.map(lambda v, g: momentum*v + g, velocity, gradients)
    params = jax.tree.map(lambda p, v: p - learning_rate*v, params, velocity)
    return params, velocity, value


def host_replica(tree):
    return jax.tree.map(lambda a: np.asarray(a.addressable_shards[0].data), tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # TPU metadata supplies topology/coordinator. JAX ranks can differ from
    # hostname order; all data partitioning uses the discovered JAX rank.
    jax.distributed.initialize(initialization_timeout=90)
    try:
        rank = jax.process_index()
        startup = {'jax_rank': rank, 'host_rank': int(os.environ['GOZERO_HOST_RANK']),
                   'process_count': jax.process_count(), 'device_count': jax.device_count(),
                   'devices': [str(d) for d in jax.devices()]}
        (args.output / 'startup.json').write_bytes(canonical_json(startup))
        print(json.dumps({'kind': 'startup', **startup}), flush=True)
        if (jax.process_count() != config['expected_processes']
                or jax.process_count() != int(os.environ['GOZERO_WORLD_SIZE'])
                or not 0 <= rank < jax.process_count()):
            raise RuntimeError('Discovered JAX process count differs from the launch contract')
        devices = jax.devices()
        if len(devices) != config['expected_devices'] or any(d.platform != 'tpu' for d in devices):
            raise RuntimeError('Expected the complete TPU mesh; CPU fallback is not qualification')
        print(json.dumps({'kind': 'devices_ready', 'rank': rank, 'devices': [str(d) for d in devices]}), flush=True)
        topology = [{'id': d.id, 'process_index': d.process_index, 'device_kind': d.device_kind,
                     'coords': list(d.coords), 'core_on_chip': d.core_on_chip,
                     'slice_index': getattr(d, 'slice_index', None)} for d in devices]
        mesh = Mesh(np.array(devices), ('data',))
        replicated = NamedSharding(mesh, P())
        batched = NamedSharding(mesh, P('data', None))
        b, d, o = config['global_batch'], config['input_width'], config['output_width']
        x_all = np.sin(np.arange(b*d, dtype=np.float32).reshape(b, d)*0.01)
        y_all = np.cos(np.arange(b*o, dtype=np.float32).reshape(b, o)*0.03)
        local_batch = b // jax.process_count()
        span = slice(rank*local_batch, (rank+1)*local_batch)
        x = jax.make_array_from_process_local_data(batched, x_all[span], global_shape=x_all.shape)
        y = jax.make_array_from_process_local_data(batched, y_all[span], global_shape=y_all.shape)
        initial = init_params(d, o)
        params = jax.device_put(initial, replicated)
        velocity = jax.tree.map(jnp.zeros_like, params)
        compile_start = time.monotonic()
        with jax.default_matmul_precision('highest'):
            update = jax.jit(lambda p, v, bx, by: train_step(p, v, bx, by, config['learning_rate'], config['momentum']),
                             in_shardings=(replicated, replicated, batched, batched),
                             out_shardings=(replicated, replicated, replicated), donate_argnums=(0, 1))
            executable = update.lower(params, velocity, x, y).compile()
        compile_seconds = time.monotonic()-compile_start
        for _ in range(config['steps']):
            params, velocity, loss = executable(params, velocity, x, y)
        jax.block_until_ready((params, velocity, loss))
        actual = host_replica(params)
        actual_velocity = host_replica(velocity)
        reference = {key: value.copy() for key, value in initial.items()}
        reference_velocity = {key: np.zeros_like(value) for key, value in initial.items()}
        for _ in range(config['steps']):
            diff = x_all @ reference['w'] + reference['b'] - y_all
            gradients = {'w': (2/(b*o))*(x_all.T @ diff), 'b': (2/o)*diff.mean(axis=0)}
            for key in reference:
                reference_velocity[key] = config['momentum']*reference_velocity[key] + gradients[key]
                reference[key] -= config['learning_rate']*reference_velocity[key]
        max_error = max(float(np.max(np.abs(actual[key]-reference[key]))) for key in reference)
        if max_error > 2e-5:
            raise RuntimeError('Distributed update differs from the global NumPy reference: %g' % max_error)
        checkpoint = args.output / 'checkpoint.npz'
        np.savez(checkpoint, w=actual['w'], b=actual['b'], velocity_w=actual_velocity['w'], velocity_b=actual_velocity['b'],
                 step=np.asarray(config['steps'], np.int64), seed=np.asarray(27, np.int64))
        with np.load(checkpoint, allow_pickle=False) as saved:
            restored = jax.device_put({'w': saved['w'], 'b': saved['b']}, replicated)
            restored_velocity = jax.device_put({'w': saved['velocity_w'], 'b': saved['velocity_b']}, replicated)
            assert int(saved['step']) == config['steps'] and int(saved['seed']) == 27
        p1, v1, _ = executable(params, velocity, x, y)
        p2, v2, _ = executable(restored, restored_velocity, x, y)
        jax.block_until_ready((p1, v1, p2, v2))
        for left, right in zip(jax.tree.leaves(host_replica((p1, v1))), jax.tree.leaves(host_replica((p2, v2)))):
            np.testing.assert_array_equal(left, right)
        n = config['matmul_size']
        local_shape = (jax.local_device_count(), n, n)
        shard3 = NamedSharding(mesh, P('data', None, None))
        left = jax.make_array_from_process_local_data(shard3, np.full(local_shape, 0.01, np.float32)).astype(jnp.bfloat16)
        right = jax.make_array_from_process_local_data(shard3, np.full(local_shape, 0.02, np.float32)).astype(jnp.bfloat16)
        mm = jax.jit(lambda a, b: a @ b, in_shardings=(shard3, shard3), out_shardings=shard3)
        result = mm(left, right)
        jax.block_until_ready(result)
        multihost_utils.sync_global_devices('matmul-start')
        benchmark_start = time.monotonic()
        for _ in range(config['matmul_repeats']):
            result = mm(left, right)
        jax.block_until_ready(result)
        multihost_utils.sync_global_devices('matmul-finished')
        elapsed = time.monotonic()-benchmark_start
        flops = config['matmul_repeats'] * len(devices) * 2*n**3
        result = {'schema_version': 1, 'kind': 'runtime_qualification', 'jax_rank': rank,
                  'host_rank': int(os.environ['GOZERO_HOST_RANK']),
                  'snapshot_id': os.environ['GOZERO_SNAPSHOT_ID'], 'topology': topology,
                  'packages': {name: importlib.metadata.version(name) for name in ('jax','jaxlib','libtpu','numpy')},
                  'global_batch': b, 'steps': config['steps'], 'numpy_update_max_abs_error': max_error,
                  'checkpoint_resume_exact': True, 'update_compile_seconds': compile_seconds,
                  'matmul_size': n, 'matmul_repeats': config['matmul_repeats'], 'matmul_seconds': elapsed,
                  'matmul_total_flops': flops, 'matmul_achieved_tflops': flops/elapsed/1e12,
                  'total_seconds': time.monotonic()-started,
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (args.output / 'result.json').write_bytes(canonical_json(result))
        multihost_utils.sync_global_devices('qualification-finished')
        print(json.dumps(result), flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
