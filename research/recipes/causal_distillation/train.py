#!/usr/bin/env python3
"""Cloneable, resumable pure-JAX causal expert/behavior distillation."""
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
from gozero import checkpoints
from gozero.sequence_batches import Dataset
from gozero.snapshots import canonical_json, read_json, verify
from config import validate


def publish(path, value):
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path); checkpoints._sync_directory(path.parent)


def train(args, c, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import model
    rank, world = jax.process_index(), jax.process_count()
    devices = jax.devices()
    if world != c['expected_processes'] or len(devices) != c['expected_devices'] or any(d.platform != c['platform'] for d in devices):
        raise ValueError('Runtime topology differs from frozen configuration')
    mesh = Mesh(np.array(devices), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))

    def replica(tree):
        return jax.tree.map(lambda a: np.asarray(a.addressable_shards[0].data), tree)

    def global_batch(batch):
        return jax.tree.map(lambda a: jax.make_array_from_process_local_data(batched, a), batch)

    def digests(value):
        return [bytes(row).hex() for row in np.asarray(mh.process_allgather(
            np.frombuffer(bytes.fromhex(value), np.uint8))).reshape(world, 32)]

    dataset_path = Path(c['dataset']['path'])
    if not dataset_path.is_absolute():
        raise ValueError('Dataset location must be explicit on every host')
    data = Dataset(dataset_path, c['dataset']['manifest_sha256'], rank=rank, world=world)
    if data.size != c['model']['size'] or data.time != 324:
        raise ValueError('Dataset context or board differs')
    config_sha = hashlib.sha256(canonical_json(c)).hexdigest()
    report.update(jax_rank=rank, world_size=world, jax_version=jax.__version__, devices=[str(d) for d in devices],
                  dataset_manifest_sha256=c['dataset']['manifest_sha256'], dataset_operator_snapshot=data.manifest['operator_snapshot'],
                  local_games={f'{role}-{split}': len(values) for (role, split), values in data.indices.items()},
                  teacher_cost=data.manifest['spec']['teacher_cost'])
    params = jax.device_put(jax.tree.map(np.asarray, model.initialize(c['seed'], c['model'])), replicated)
    first, definition = jax.tree.flatten(replica(params))
    schema = [{'path': jax.tree_util.keystr(path), 'shape': list(a.shape), 'dtype': str(a.dtype)}
              for path, a in jax.tree_util.tree_flatten_with_path(params)[0]]
    report['parameter_count'] = sum(a.size for a in first)
    m = jax.tree.map(jnp.zeros_like, params); v = jax.tree.map(jnp.zeros_like, params)
    random = np.random.Generator(np.random.PCG64(c['seed'] + 1 + 104729 * rank))
    turn = 0
    counters = {'updates': 0, 'expert_token_exposures': 0, 'behavior_token_exposures': 0,
                'sample_seconds': 0., 'learner_seconds': 0., 'evaluation_seconds': 0., 'checkpoint_seconds': 0.}
    if args.resume:
        group_path = args.resume.with_suffix('.group.json'); group = read_json(group_path)
        if (group['snapshot_id'] != SOURCE.name or group['config_sha256'] != config_sha or group['world_size'] != world
                or len(set(digests(checkpoints.sha256(group_path)))) != 1):
            raise ValueError('Resume checkpoint group differs')
        state, arrays, _ = checkpoints.read(args.resume, expected_manifest_sha256=group['rank_manifests'][rank])
        if (state['snapshot_id'] != SOURCE.name or state['config_sha256'] != config_sha or state['jax_rank'] != rank
                or state['world_size'] != world or state['turn'] != group['turn'] or state['model_schema'] != schema
                or state['dataset_manifest_sha256'] != c['dataset']['manifest_sha256']):
            raise ValueError('Resume scientific state differs')
        if set(arrays) != {f'{kind}_{i:04d}' for kind in ('p', 'm', 'v') for i in range(len(first))}:
            raise ValueError('Resume parameter/optimizer tree differs')
        def restore(kind):
            values = [arrays[f'{kind}_{i:04d}'] for i in range(len(first))]
            if any(a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() for a, b in zip(values, first)):
                raise ValueError('Resume arrays invalid')
            return jax.device_put(definition.unflatten(values), replicated)
        params, m, v = (restore(kind) for kind in ('p', 'm', 'v'))
        turn = state['turn']; random.bit_generator.state = state['numpy_rng']; counters = state['counters']
        report['resumed_from'] = {'turn': turn, 'group_sha256': checkpoints.sha256(group_path), 'path': str(args.resume)}
    stop = c['steps'] if args.stop_after_turn is None else args.stop_after_turn
    if not turn < stop <= c['steps']:
        raise ValueError('Invalid bounded training segment')
    learner, net = c['learner'], c['model']

    def update(p, m, v, batch, iteration):
        (loss, metrics), grad = jax.value_and_grad(lambda weights: model.losses(weights, batch, net), has_aux=True)(p)
        grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g * jnp.minimum(1., learner['grad_clip'] / jnp.maximum(grad_norm, 1e-12)), grad)
        m = jax.tree.map(lambda a, g: learner['beta1'] * a + (1 - learner['beta1']) * g, m, grad)
        v = jax.tree.map(lambda a, g: learner['beta2'] * a + (1 - learner['beta2']) * g * g, v, grad)
        warmup = jnp.minimum(1., iteration / learner['warmup_steps'])
        progress = jnp.clip((iteration - learner['warmup_steps']) / max(1, c['steps'] - learner['warmup_steps']), 0., 1.)
        rate = learner['learning_rate'] * warmup * (.1 + .9 * .5 * (1 + jnp.cos(jnp.pi * progress)))
        def apply(weight, first, second):
            first = first / (1 - learner['beta1'] ** iteration)
            second = second / (1 - learner['beta2'] ** iteration)
            decay = learner['weight_decay'] * weight if weight.ndim >= 2 else 0.
            return weight - rate * (first / (jnp.sqrt(second) + learner['epsilon']) + decay)
        p = jax.tree.map(apply, p, m, v)
        return p, m, v, {**metrics, 'loss': loss, 'grad_norm': grad_norm, 'learning_rate': rate}

    sample_state = random.bit_generator.state
    dummy = global_batch(data.sample(random, learner['games_per_role']))
    random.bit_generator.state = sample_state
    started_compile = time.perf_counter()
    lowered = jax.jit(update, in_shardings=(replicated, replicated, replicated, batched, replicated),
                      out_shardings=(replicated, replicated, replicated, replicated), donate_argnums=(0, 1, 2)).lower(
                          params, m, v, dummy, jax.device_put(np.asarray(1, np.float32), replicated))
    (args.output / 'update.hlo.txt').write_text(lowered.compiler_ir('hlo').as_hlo_text())
    step = lowered.compile()
    evaluate = jax.jit(lambda p, b: model.losses(p, b, net)[1], in_shardings=(replicated, batched), out_shardings=replicated)
    report['update_compile_seconds'] = time.perf_counter() - started_compile
    report['update_hlo_sha256'] = checkpoints.sha256(args.output / 'update.hlo.txt')
    report['compiler_cost_estimate'] = step.cost_analysis()
    (args.output / 'checkpoints').mkdir()

    def heldout(split):
        start = time.perf_counter(); entries = data.evaluation_entries(split)
        batch_size = 2 * learner['games_per_role']
        maximum = int(np.max(mh.process_allgather(np.asarray(len(entries), np.int32))))
        sums = {k: 0. for k in ('play_loss', 'behavior_loss', 'value_loss', 'target_entropy', 'play_top1', 'behavior_top1', 'illegal_probability')}
        counts = {k: 0 for k in ('expert_tokens', 'behavior_tokens', 'value_tokens')}
        for begin in range(0, maximum, batch_size):
            selected = entries[begin:begin+batch_size]
            batch = data.batch(selected + [None] * (batch_size - len(selected)))
            metrics = {k: float(a) for k, a in replica(evaluate(params, global_batch(batch))).items()}
            if not all(np.isfinite(a) for a in metrics.values()):
                raise FloatingPointError('Nonfinite held-out metric')
            for key in counts:
                counts[key] += int(metrics[key])
            for key in sums:
                population = 'behavior_tokens' if key.startswith('behavior') else 'value_tokens' if key == 'value_loss' else 'expert_tokens'
                sums[key] += metrics[key] * metrics[population]
        result = dict(counts)
        for key, value in sums.items():
            population = 'behavior_tokens' if key.startswith('behavior') else 'value_tokens' if key == 'value_loss' else 'expert_tokens'
            result[key] = value / counts[population]
        result['expert_kl'] = result['play_loss'] - result['target_entropy']
        counters['evaluation_seconds'] += time.perf_counter() - start
        print(json.dumps({'kind': 'causal_heldout', 'turn': turn, 'split': split, **result}), flush=True)
        return result

    def save():
        start = time.perf_counter()
        arrays = {f'{kind}_{i:04d}': a for kind, tree in [('p', params), ('m', m), ('v', v)]
                  for i, a in enumerate(jax.tree.leaves(replica(tree)))}
        state = {'schema_version': 1, 'snapshot_id': SOURCE.name, 'config_sha256': config_sha, 'jax_rank': rank,
                 'world_size': world, 'turn': turn, 'model_schema': schema, 'dataset_manifest_sha256': c['dataset']['manifest_sha256'],
                 'numpy_rng': random.bit_generator.state, 'counters': counters}
        path = args.output / 'checkpoints' / f'turn-{turn:09d}'
        identity = checkpoints.write(path, state=state, arrays=arrays, actors='{}', compress=True)
        replicated_hashes = digests(checkpoints.sha256(path / 'arrays.npz'))
        if len(set(replicated_hashes)) != 1:
            raise ValueError('Parameters and optimizer differ across ranks')
        group = {'schema_version': 1, 'snapshot_id': SOURCE.name, 'config_sha256': config_sha, 'turn': turn,
                 'world_size': world, 'rank_manifests': digests(identity), 'replicated_arrays_sha256': replicated_hashes[0]}
        publish(path.with_suffix('.group.json'), group)
        report['latest_checkpoint'] = {'path': str(path), 'manifest_sha256': identity, 'group_sha256': checkpoints.sha256(path.with_suffix('.group.json'))}
        counters['checkpoint_seconds'] += time.perf_counter() - start

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    if turn == 0:
        report['initial_validation'] = heldout(1)
    history = []
    with (args.output / 'metrics.jsonl').open('x') as log:
        while turn < stop:
            before = time.perf_counter()
            batch = data.sample(random, learner['games_per_role'])
            counters['sample_seconds'] += time.perf_counter() - before
            before = time.perf_counter()
            params, m, v, metrics = step(params, m, v, global_batch(batch), jax.device_put(np.asarray(turn + 1, np.float32), replicated))
            metrics = {k: float(a) for k, a in replica(metrics).items()}
            counters['learner_seconds'] += time.perf_counter() - before
            if not all(np.isfinite(a) for a in metrics.values()):
                raise FloatingPointError('Nonfinite causal learner')
            turn += 1; counters['updates'] = turn
            counters['expert_token_exposures'] += int(metrics['expert_tokens'])
            counters['behavior_token_exposures'] += int(metrics['behavior_tokens'])
            if turn % c['eval_every'] == 0 or turn == c['steps']:
                history.append({'turn': turn, 'validation': heldout(1)})
            if turn % c['checkpoint_every'] == 0 or turn == stop:
                save()
            entry = {'turn': turn, **metrics, 'learner_seconds': counters['learner_seconds']}
            log.write(json.dumps(entry, sort_keys=True) + '\n'); log.flush()
            if turn % c['log_every'] == 0:
                print(json.dumps({'kind': 'causal_update', **entry}), flush=True)
    if turn == c['steps']:
        report['test'] = heldout(2)
    final = jax.tree.leaves(replica(params))
    if any(not np.isfinite(a).all() for a in final):
        raise FloatingPointError('Nonfinite exported parameters')
    np.savez(args.output / 'model_export.npz', **{f'p_{i:04d}': a for i, a in enumerate(final)})
    (args.output / 'model_schema.json').write_bytes(canonical_json(schema))
    report.update(status='passed', turn=turn, training_complete=turn == c['steps'], counters=counters,
                  validation_history=history, config_sha256=config_sha, last_metrics=metrics,
                  elapsed_segment_seconds=time.perf_counter()-start_wall, process_cpu_segment_seconds=time.process_time()-start_cpu,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  model_export_sha256=checkpoints.sha256(args.output / 'model_export.npz'))
    if checkpoints.sha256(dataset_path / 'manifest.json') != c['dataset']['manifest_sha256']:
        raise ValueError('Dataset manifest changed during training')
    verify(SOURCE); mh.sync_global_devices('causal-training-complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path); parser.add_argument('--stop-after-turn', type=int)
    args = parser.parse_args(); verify(SOURCE); c = validate(read_json(args.config))
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')):
        raise ValueError('Configuration is not frozen')
    if os.environ.get('JAX_PLATFORMS', c['platform']) != c['platform']:
        raise ValueError('Requested platform differs')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved_config.json').write_bytes(canonical_json(c))
    report = {'schema_version': 1, 'kind': 'causal_expert_behavior_distillation', 'snapshot_id': SOURCE.name,
              'status': 'running', 'claims_rl_sample_efficiency': False, 'claims_go_strength': False,
              'behavior_updates_expert_trunk': False, 'observed_moves_used_as_expert_targets': False}
    distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        train(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        (args.output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('compiler_cost_estimate', 'devices', 'teacher_cost')}), flush=True)
        if distributed:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
