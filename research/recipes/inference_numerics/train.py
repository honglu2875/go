#!/usr/bin/env python3
"""Fixed-input inference transport and numerical probe; no training occurs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def differences(reference, candidate, features):
    import numpy as np
    left, lv = (np.asarray(x, np.float64) for x in reference)
    right, rv = (np.asarray(x, np.float64) for x in candidate)
    if not all(np.isfinite(x).all() for x in (left, lv, right, rv)):
        raise ValueError('Nonfinite prediction')
    legal = np.concatenate((features[..., -1].reshape(len(features), -1) > .5,
                            np.ones((len(features), 1), bool)), axis=1)
    def log_prob(logits):
        masked = np.where(legal, logits, -1e9)
        shifted = masked - masked.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    lp, lq = log_prob(left), log_prob(right)
    p, q = np.exp(lp), np.exp(lq)
    def stats(value):
        return {'mean': float(value.mean()), 'p95': float(np.quantile(value, .95)), 'max': float(value.max())}
    return {'all_outputs_exact': all(np.array_equal(a, b) for a, b in zip(reference, candidate)),
            'absolute_logit_error': stats(np.abs(left - right)[legal]),
            'absolute_value_error': stats(np.abs(lv - rv)),
            'legal_policy_kl_serial_to_candidate': stats((p * (lp - lq)).sum(axis=1)),
            'legal_policy_total_variation': stats(np.abs(p - q).sum(axis=1) / 2),
            'legal_top_one_disagreements': int(np.count_nonzero(lp.argmax(axis=1) != lq.argmax(axis=1))),
            'positions': len(features)}


def probe(config, output, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import model

    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    rank, world, devices = jax.process_index(), jax.process_count(), jax.devices()
    if (world != config['expected_processes'] or len(devices) != config['expected_devices']
            or any(d.platform != config['platform'] for d in devices)):
        raise ValueError('Unexpected execution topology')
    report.update(host_rank=host, jax_rank=rank, world_size=world, jax_version=jax.__version__,
                  devices=[str(d) for d in devices])
    ref = config['input_checkpoints'][host]
    if ref['host_rank'] != host:
        raise ValueError('Checkpoint host mapping differs')
    workspace = ROOT.parents[2]
    path = (workspace / ref['path']).resolve(strict=True)
    if not path.is_relative_to(workspace / 'runs'):
        raise ValueError('Input checkpoint escapes run artifacts')
    if checkpoints.sha256(path.with_suffix('.group.json')) != ref['group_sha256']:
        raise ValueError('Input checkpoint group differs')
    state, arrays, _ = checkpoints.read(path, expected_manifest_sha256=ref['manifest_sha256'])
    group = read_json(path.with_suffix('.group.json'))
    if (state['snapshot_id'] != config['training_snapshot'] or state['turn'] != config['checkpoint_turn']
            or group['rank_manifests'][state['jax_rank']] != ref['manifest_sha256']
            or group['updates'] != state['counters']['updates']
            or checkpoints.sha256(Path(__file__).with_name('model.py')) != config['model_sha256']):
        raise ValueError('Input/model scientific identity differs')
    template = model.initialize(0, config['input_channels'], config['model'])
    leaves, definition = jax.tree.flatten(template)
    schema = [{'path': jax.tree_util.keystr(p), 'shape': list(v.shape), 'dtype': str(v.dtype)}
              for p, v in jax.tree_util.tree_flatten_with_path(template)[0]]
    if schema != state['model_schema']:
        raise ValueError('Model parameter schema differs')
    parameters = [arrays[f'p_{i:04d}'] for i in range(len(leaves))]
    small, branches = config['small_batch'], config['branches']
    count = small * branches
    if (not 1 <= branches <= 16 or not 1 <= count <= 2048 or count > state['replay_count']
            or not 1 <= config['warmup_iterations'] <= 10 or not 1 <= config['timed_iterations'] <= 100
            or not 1 <= len(config['orders']) <= 6
            or any(sorted(order) != ['queued', 'serial', 'wide'] for order in config['orders'])):
        raise ValueError('Unbounded or invalid probe configuration')
    indices = np.random.Generator(np.random.PCG64(config['input_seed'] + host)).choice(state['replay_count'], count, replace=False)
    features = np.ascontiguousarray(arrays['replay_x'][indices])
    expected_shape = (count, config['board_size'], config['board_size'], config['input_channels'])
    if features.shape != expected_shape or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('Invalid sampled replay observations')
    del arrays, template
    mesh = Mesh(np.array(devices), ('data',)).local_mesh
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    params = jax.device_put(definition.unflatten(parameters), replicated)
    parameter_hash = hashlib.sha256()
    for value in parameters:
        parameter_hash.update(value.tobytes())
    report['input'] = {**ref, 'training_snapshot': state['snapshot_id'], 'turn': state['turn'],
                       'network_version': state['counters']['updates'],
                       'parameters_sha256': parameter_hash.hexdigest(),
                       'features_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
                       'positions': count, 'sample_indices': indices.tolist()}
    grouped = features.reshape((small, branches, *features.shape[1:]))
    small_inputs = [np.ascontiguousarray(grouped[:, b]) for b in range(branches)]
    def inference(p, x):
        return model.apply(p, x, config['model'])
    before = time.perf_counter()
    executable = jax.jit(inference, in_shardings=(replicated, batched), out_shardings=(batched, batched))
    narrow = executable.lower(params, jax.device_put(small_inputs[0], batched)).compile()
    wide = executable.lower(params, jax.device_put(features, batched)).compile()
    report['compile_seconds'] = time.perf_counter() - before
    for name, compiled in (('small', narrow), ('wide', wide)):
        (output / (name + '.hlo.txt')).write_text(compiled.as_text())
    report['hlo_sha256'] = {name: checkpoints.sha256(output / (name + '.hlo.txt')) for name in ('small', 'wide')}
    def execute(mode):
        if mode == 'wide':
            return tuple(np.asarray(a) for a in jax.device_get(wide(params, jax.device_put(features, batched))))
        if mode == 'serial':
            predictions = [jax.device_get(narrow(params, jax.device_put(x, batched))) for x in small_inputs]
        else:
            predictions = jax.device_get([narrow(params, jax.device_put(x, batched)) for x in small_inputs])
        return tuple(np.stack([p[index] for p in predictions], axis=1).reshape((count, *predictions[0][index].shape[1:]))
                     for index in (0, 1))
    baseline = execute('serial')
    report['comparisons'] = {}
    payload = {'features': features, 'sample_indices': indices, 'serial_logits': baseline[0], 'serial_values': baseline[1]}
    for mode in ('queued', 'wide'):
        value = execute(mode)
        report['comparisons'][mode] = differences(baseline, value, features)
        payload[mode + '_logits'], payload[mode + '_values'] = value
    if not report['comparisons']['queued']['all_outputs_exact']:
        raise ValueError('Queuing the same executable changed predictions')
    np.savez_compressed(output / 'predictions.npz', **payload)
    report['predictions_sha256'] = checkpoints.sha256(output / 'predictions.npz')
    report['timings'] = []
    for repetition, order in enumerate(config['orders']):
        for mode in order:
            for _ in range(config['warmup_iterations']):
                execute(mode)
            if world > 1:
                mh.sync_global_devices(f'probe-{repetition}-{mode}')
            before, cpu_before = time.perf_counter(), time.process_time()
            for _ in range(config['timed_iterations']):
                value = execute(mode)
            elapsed, cpu = time.perf_counter() - before, time.process_time() - cpu_before
            # Check the timed result after its measurement, without charging a
            # second host copy or numerical analysis to only one timed mode.
            expected = (payload[mode + '_logits'], payload[mode + '_values'])
            if not all(np.array_equal(a, b) for a, b in zip(value, expected)):
                raise ValueError('Fixed-input execution changed within a timing repetition')
            entry = {'repetition': repetition, 'mode': mode, 'iterations': config['timed_iterations'],
                     'elapsed_seconds': elapsed, 'process_cpu_seconds': cpu,
                     'active_neural_evaluations': count * config['timed_iterations'],
                     'padded_slots': count * config['timed_iterations'],
                     'forward_dispatches': config['timed_iterations'] * (1 if mode == 'wide' else branches),
                     'host_fetches': config['timed_iterations'] * (branches if mode == 'serial' else 1)}
            report['timings'].append(entry)
            print(json.dumps({'kind': 'inference_timing', **entry}), flush=True)
    report['status'] = 'passed'
    verify(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT)
    config = read_json(args.config)
    if config != read_json(ROOT / 'resolved_config.json'):
        raise ValueError('Configuration is not frozen')
    os.environ['JAX_PLATFORMS'] = config['platform']
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'fixed_input_inference_probe', 'status': 'running',
              'snapshot_id': ROOT.name, 'no_training_performed': True,
              'claims_rollout_speedup': False, 'claims_go_strength': False, 'claims_mfu': False}
    distributed = False
    try:
        import jax
        if config['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90)
            distributed = True
        probe(config, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('input', 'timings')}), flush=True)
        if distributed:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
