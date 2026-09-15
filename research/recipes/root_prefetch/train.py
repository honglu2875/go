#!/usr/bin/env python3
"""Cloneable synchronous AlphaZero-style self-play with complete local/group checkpoints.

Rust owns games and MCTS; this recipe owns the model, optimization and sampling.
CPU and TPU use the same training logic and explicit data-parallel JAX shardings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from config import validate


def sgf(game):
    size = game['size']
    text = '(;GM[1]FF[4]CA[UTF-8]SZ[%d]KM[%s]RU[Tromp-Taylor]AP[gozero:gumbel_search]' % (size, game['komi'])
    text += 'C[terminal scoring profile: '+game['scoring']+('; move-limit truncation; excluded from training' if game['truncated'] else '')+']'
    if game['white_score'] is not None:
        value = game['white_score']
        result = '0' if value == 0 else ('W' if value > 0 else 'B') + '+' + str(abs(value))
        text += 'RE[' + result + ']'
    for ply, action in enumerate(game['actions']):
        vertex = '' if action == size*size else chr(97+action%size)+chr(97+action//size)
        text += ';' + ('B' if ply%2 == 0 else 'W') + '[' + vertex + ']'
    return text + ')\n'


def publish_json(path, value):
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())
    if path.exists(): raise FileExistsError(path)
    temporary.rename(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def train(args, c, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gozero.native import Actors
    import model

    rank = jax.process_index(); world = jax.process_count()
    devices = jax.devices()
    if world != c['expected_processes'] or len(devices) != c['expected_devices'] or any(d.platform != c['platform'] for d in devices):
        raise RuntimeError('Discovered accelerator topology differs from the frozen configuration')
    mesh = Mesh(np.array(devices), ('data',))
    replicated = NamedSharding(mesh, P()); batched = NamedSharding(mesh, P('data'))
    inference_replicated = NamedSharding(mesh.local_mesh, P())
    inference_batched = NamedSharding(mesh.local_mesh, P('data'))
    report.update(jax_rank=rank, world_size=world, host_rank=int(os.environ.get('GOZERO_HOST_RANK', '0')),
                  devices=[str(d) for d in devices], jax_version=jax.__version__)
    print(json.dumps({'kind': 'startup', 'rank': rank, 'world': world, 'devices': report['devices']}), flush=True)

    def global_batch(value):
        return jax.make_array_from_process_local_data(batched, np.asarray(value))

    def local_batch(value):
        return np.asarray(mh.global_array_to_host_local_array(value, mesh, P('data')))

    def replica(tree):
        return jax.tree.map(lambda a: np.asarray(a.addressable_shards[0].data), tree)

    def gather_digest(value):
        array = np.frombuffer(bytes.fromhex(value), np.uint8)
        gathered = np.asarray(mh.process_allgather(array)).reshape(world, 32)
        return [bytes(row).hex() for row in gathered]

    actor = dict(c['actors']); actor['actor_offset'] = rank * actor['games']
    prefetch = c['root_prefetch']
    learner = c['learner']; net = c['model']
    size = actor['size']; channels = actor['history']*2+4; actions = size*size+1
    receipt = read_json(args.native_receipt)
    if receipt['snapshot_id'] != ROOT.name: raise ValueError('Native source snapshot differs')
    report['native'] = receipt
    output = args.output
    (output / 'games').mkdir(); (output / 'checkpoints').mkdir()
    (output / 'resolved_config.json').write_bytes(canonical_json(c))
    config_sha = hashlib.sha256(canonical_json(c)).hexdigest()

    # Initialization returns process-local JAX arrays. Materialize identical host
    # values before constructing global replicas; a local device array cannot be
    # device_put directly onto a sharding containing non-addressable devices.
    params = jax.device_put(jax.tree.map(np.asarray, model.initialize(c['seed'], channels, net)), replicated)
    velocity = jax.tree.map(jnp.zeros_like, params)
    initial, definition = jax.tree.flatten(replica(params))
    initial = [np.array(x, copy=True) for x in initial]
    schema = [{'path': jax.tree_util.keystr(path), 'shape': list(leaf.shape), 'dtype': str(leaf.dtype)}
              for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]]
    report['parameter_count'] = sum(x.size for x in initial)
    leaves_count = len(initial)
    capacity = learner['replay_capacity']
    replay_x = np.empty((capacity, size, size, channels), np.float32)
    replay_pi = np.empty((capacity, actions), np.float32)
    replay_owner = np.empty((capacity, size, size), np.float32)
    replay_z = np.empty(capacity, np.float32); replay_meta = np.empty((capacity, 6), np.uint64)
    count = cursor = turn = 0; ready = False; last_metrics = None
    random = np.random.default_rng(c['seed']+1+104729*rank)
    key = jax.random.wrap_key_data(jax.device_put(np.asarray(jax.random.key_data(jax.random.key(c['seed']+2))), replicated))
    counters = {k: 0 for k in ('real_moves','completed_games','truncated_games','eligible_rows','updates',
                               'neural_batches','neural_slots','active_neural_evaluations',
                               'inference_syncs','prefetch_neural_evaluations','prefetch_hits')}
    counters.update({k: 0.0 for k in ('native_seconds','inference_seconds','learner_seconds','checkpoint_seconds')})
    actor_state = None
    if args.resume:
        group = read_json(args.resume.parent / (args.resume.name + '.group.json'))
        if group['snapshot_id'] != ROOT.name or group['world_size'] != world or group['config_sha256'] != config_sha:
            raise ValueError('Checkpoint group source, topology or configuration differs')
        if len(set(gather_digest(checkpoints.sha256(args.resume.parent / (args.resume.name + '.group.json'))))) != 1:
            raise ValueError('Hosts selected different checkpoint groups')
        state, arrays, actor_state = checkpoints.read(args.resume, expected_manifest_sha256=group['rank_manifests'][rank])
        if (state['snapshot_id'] != ROOT.name or state['config_sha256'] != config_sha or state['jax_rank'] != rank
                or state['world_size'] != world or state['turn'] != group['turn'] or state['counters']['updates'] != group['updates']
                or state['model_schema'] != schema or state['native_sha256'] != receipt['binary_sha256']):
            raise ValueError('Checkpoint state does not match the run contract')
        expected = {'key', 'replay_x', 'replay_pi', 'replay_z', 'replay_meta', 'replay_owner'} | {f'{kind}_{i:04d}' for kind in ('p','v') for i in range(leaves_count)}
        if set(arrays) != expected: raise ValueError('Unexpected checkpoint array tree')
        for kind in ('p', 'v'):
            for i, template in enumerate(initial):
                a = arrays[f'{kind}_{i:04d}']
                if a.shape != template.shape or a.dtype != template.dtype or not np.isfinite(a).all():
                    raise ValueError('Invalid parameter or optimizer array')
        params = jax.device_put(definition.unflatten([arrays[f'p_{i:04d}'] for i in range(leaves_count)]), replicated)
        velocity = jax.device_put(definition.unflatten([arrays[f'v_{i:04d}'] for i in range(leaves_count)]), replicated)
        key = jax.random.wrap_key_data(jax.device_put(arrays['key'], replicated))
        count, cursor, turn, ready = state['replay_count'], state['replay_cursor'], state['turn'], state['learner_ready']
        if not 0 <= count <= capacity or not 0 <= cursor < capacity or (count < capacity and cursor != count):
            raise ValueError('Invalid replay ring metadata')
        for target, name in ((replay_x,'replay_x'), (replay_pi,'replay_pi'), (replay_z,'replay_z'), (replay_meta,'replay_meta'), (replay_owner,'replay_owner')):
            if arrays[name].shape != target[:count].shape or arrays[name].dtype != target.dtype:
                raise ValueError('Replay shape or dtype mismatch')
            target[:count] = arrays[name]
        counters = state['counters']; random.bit_generator.state = state['numpy_rng']; last_metrics = state['last_metrics']
        report['resumed_from'] = {'path': str(args.resume), 'group_sha256': checkpoints.sha256(args.resume.parent / (args.resume.name+'.group.json')), 'turn': turn}
    report['counters'] = counters
    stop_turn = c['selfplay_turns'] if args.stop_after_turn is None else args.stop_after_turn
    if not turn < stop_turn <= c['selfplay_turns']: raise ValueError('Invalid segment stop turn')

    def inference(p, features):
        return model.apply(p, features, net)

    def update(p, v, x, pi, z, owner, rng):
        next_key, draw = jax.random.split(rng)
        p, v, metrics = model.update(p, v, x, pi, z, owner, draw, net, learner)
        return p, v, next_key, metrics

    before = time.perf_counter()
    infer_params = mh.global_array_to_host_local_array(params, mesh, P())
    forward = jax.jit(inference, in_shardings=(inference_replicated, inference_batched), out_shardings=(inference_batched, inference_batched)).lower(
        infer_params, jax.device_put(np.zeros((actor['games'], size, size, channels), np.float32), inference_batched)).compile()
    wide_forward = None
    if prefetch['mode'] == 'wide':
        wide_forward = jax.jit(inference, in_shardings=(inference_replicated, inference_batched),
            out_shardings=(inference_batched, inference_batched)).lower(infer_params,
            jax.device_put(np.zeros((actor['games'] * prefetch['limit'], size, size, channels), np.float32),
                           inference_batched)).compile()
    del infer_params
    report['forward_compile_seconds'] = time.perf_counter()-before
    before = time.perf_counter()
    step = jax.jit(update, in_shardings=(replicated, replicated, batched, batched, batched, batched, replicated),
                   out_shardings=(replicated, replicated, replicated, replicated), donate_argnums=(0,1)).lower(
        params, velocity, global_batch(np.zeros((learner['batch_size'], size, size, channels), np.float32)),
        global_batch(np.zeros((learner['batch_size'], actions), np.float32)),
        global_batch(np.zeros(learner['batch_size'], np.float32)),
        global_batch(np.zeros((learner['batch_size'],size,size), np.float32)), key).compile()
    report['update_compile_seconds'] = time.perf_counter()-before
    # Compiler estimates are diagnostics, not a hardware utilization measurement.
    # Keep the raw result: backend counting conventions may differ.
    report['compiler_cost_estimates'] = {}
    for name, executable in (('local_forward',forward),('global_update',step)):
        try: report['compiler_cost_estimates'][name] = executable.cost_analysis()
        except (NotImplementedError,RuntimeError) as error:
            report['compiler_cost_estimates'][name] = {'unavailable':str(error)}
    ready_sum = jax.jit(lambda x: jnp.sum(x, dtype=jnp.int32), in_shardings=batched, out_shardings=replicated)
    start_time = time.perf_counter(); start_cpu = time.process_time()

    def save(engine, current_turn):
        before = time.perf_counter()
        p = jax.tree.leaves(replica(params)); v = jax.tree.leaves(replica(velocity))
        arrays = {**{f'p_{i:04d}': value for i, value in enumerate(p)}, **{f'v_{i:04d}': value for i, value in enumerate(v)},
                  'key': replica(jax.random.key_data(key)), 'replay_x': replay_x[:count], 'replay_pi': replay_pi[:count],
                  'replay_z': replay_z[:count], 'replay_meta': replay_meta[:count], 'replay_owner': replay_owner[:count]}
        state = {'schema_version': 1, 'snapshot_id': ROOT.name, 'config_sha256': config_sha,
                 'jax_rank': rank, 'world_size': world, 'turn': current_turn, 'learner_ready': ready,
                 'replay_count': count, 'replay_cursor': cursor, 'numpy_rng': random.bit_generator.state,
                 'counters': counters, 'last_metrics': last_metrics, 'model_schema': schema, 'native_sha256': receipt['binary_sha256']}
        path = output / 'checkpoints' / f'turn-{current_turn:09d}'
        manifest_sha = checkpoints.write(path, state=state, arrays=arrays, actors=engine.checkpoint(), compress=True)
        # No group commit is published unless all processes have durable rank state
        # and exactly equal replicated weights, momentum and augmentation RNG.
        h = hashlib.sha256()
        for value in [*p, *v, arrays['key']]: h.update(value.tobytes())
        replicated_sha = h.hexdigest()
        if len(set(gather_digest(replicated_sha))) != 1: raise RuntimeError('Replicated learner state differs across hosts')
        rank_manifests = gather_digest(manifest_sha)
        group = {'schema_version': 1, 'snapshot_id': ROOT.name, 'config_sha256': config_sha, 'world_size': world,
                 'turn': current_turn, 'updates': counters['updates'], 'replicated_state_sha256': replicated_sha,
                 'rank_manifests': rank_manifests}
        publish_json(path.parent / (path.name+'.group.json'), group)
        mh.sync_global_devices('checkpoint-'+str(current_turn))
        report['latest_checkpoint'] = {'path': str(path), 'manifest_sha256': manifest_sha,
                                       'group_sha256': checkpoints.sha256(path.parent/(path.name+'.group.json'))}
        counters['checkpoint_seconds'] += time.perf_counter()-before
        print(json.dumps({'kind': 'checkpoint', 'rank': rank, 'turn': current_turn, 'updates': counters['updates']}), flush=True)

    with Actors(actor, args.native_receipt.parent/receipt['filename'], receipt['binary_sha256'], checkpoint=actor_state) as engine:
        base_prefetch_evaluations = counters['prefetch_neural_evaluations']
        base_prefetch_hits = counters['prefetch_hits']
        with (output/'metrics.jsonl').open('x') as log:
            while turn < stop_turn:
                before = time.perf_counter(); batch = engine.start(counters['updates']); counters['native_seconds'] += time.perf_counter()-before
                # This executable has only local devices and no cross-host collective.
                infer_params = mh.global_array_to_host_local_array(params, mesh, P())
                first = True
                while batch.active_count:
                    before = time.perf_counter()
                    logits, values = jax.device_get(forward(infer_params, jax.device_put(batch.features, inference_batched)))
                    counters['neural_batches'] += 1; counters['neural_slots'] += actor['games']
                    counters['inference_syncs'] += 1
                    counters['active_neural_evaluations'] += batch.active_count
                    counters['inference_seconds'] += time.perf_counter()-before
                    before = time.perf_counter(); batch = engine.evaluate(batch, logits, values); counters['native_seconds'] += time.perf_counter()-before
                    if first and prefetch['mode'] != 'off' and batch.active_count:
                        before = time.perf_counter()
                        proposal = engine.prefetch(batch, prefetch['limit'])
                        counters['native_seconds'] += time.perf_counter()-before
                        before = time.perf_counter()
                        if proposal.active_count:
                            if prefetch['mode'] == 'queued':
                                features = proposal.features.reshape((actor['games'], prefetch['limit'], size, size, channels))
                                # Preserve the original executable and actor-to-slot mapping.
                                # Independent branch batches are queued before one host fetch.
                                pending = [forward(infer_params, jax.device_put(np.ascontiguousarray(features[:, branch]), inference_batched))
                                           for branch in range(prefetch['limit'])]
                                outputs = jax.device_get(pending)
                                logits = np.stack([p for p, _ in outputs], axis=1).reshape((-1, actions))
                                values = np.stack([v for _, v in outputs], axis=1).reshape(-1)
                                counters['neural_batches'] += prefetch['limit']
                            else:
                                logits, values = jax.device_get(wide_forward(infer_params, jax.device_put(proposal.features, inference_batched)))
                                counters['neural_batches'] += 1
                            counters['neural_slots'] += len(proposal.active)
                            counters['active_neural_evaluations'] += proposal.active_count
                            counters['inference_syncs'] += 1
                        else:
                            logits = np.zeros((len(proposal.active), actions), np.float32)
                            values = np.zeros(len(proposal.active), np.float32)
                        counters['inference_seconds'] += time.perf_counter()-before
                        before = time.perf_counter()
                        batch = engine.evaluate_prefetch(proposal, logits, values)
                        counters['native_seconds'] += time.perf_counter()-before
                    first = False
                del infer_params
                before = time.perf_counter(); rows = engine.commit(); counters['native_seconds'] += time.perf_counter()-before
                if prefetch['mode'] != 'off':
                    before = time.perf_counter()
                    evaluated, hits = engine.prefetch_counters()
                    counters['prefetch_neural_evaluations'] = base_prefetch_evaluations + evaluated
                    counters['prefetch_hits'] = base_prefetch_hits + hits
                    counters['native_seconds'] += time.perf_counter()-before
                counters['real_moves'] += actor['games']; counters['eligible_rows'] += len(rows.outcomes)
                for game in rows.games:
                    counters['truncated_games' if game['truncated'] else 'completed_games'] += 1
                    (output/'games'/f"{game['game_id']:016x}.sgf").write_text(sgf(game))
                    (output/'games'/f"{game['game_id']:016x}.json").write_bytes(canonical_json(game))
                n = min(capacity, len(rows.outcomes))
                if n:
                    indices = (cursor+np.arange(n))%capacity
                    replay_x[indices] = rows.features[-n:]; replay_pi[indices] = rows.policies[-n:]
                    replay_z[indices] = rows.outcomes[-n:]; replay_meta[indices] = rows.metadata[-n:]
                    replay_owner[indices] = rows.ownership[-n:]
                    cursor = (cursor+n)%capacity; count = min(capacity, count+n)
                if not ready:
                    flags = global_batch(np.full(jax.local_device_count(), count >= learner['warmup_rows'], np.int32))
                    ready = int(replica(ready_sum(flags))) == len(devices)
                if ready:
                    for _ in range(learner['updates_per_turn']):
                        indices = random.integers(0, count, size=learner['batch_size'])
                        before = time.perf_counter()
                        params, velocity, key, metrics = step(params, velocity, global_batch(replay_x[indices]), global_batch(replay_pi[indices]), global_batch(replay_z[indices]), global_batch(replay_owner[indices]), key)
                        metrics = replica(metrics); counters['learner_seconds'] += time.perf_counter()-before
                        if not all(np.isfinite(v) for v in metrics.values()): raise FloatingPointError('Nonfinite learner metrics')
                        counters['updates'] += 1; last_metrics = {k: float(v) for k, v in metrics.items()}
                turn += 1
                if turn % c['checkpoint_every'] == 0 or turn == stop_turn: save(engine, turn)
                entry = {'turn': turn, **counters, 'replay_rows': count, 'metrics': last_metrics}
                log.write(json.dumps(entry, sort_keys=True)+'\n'); log.flush()
                if turn % c['log_every'] == 0: print(json.dumps(entry, sort_keys=True), flush=True)
        if stop_turn == c['selfplay_turns'] and (counters['updates'] == 0 or counters['completed_games'] == 0):
            raise RuntimeError('No completed-game learning was demonstrated')
    final = jax.tree.leaves(replica(params))
    change = max(float(np.max(np.abs(a-b))) for a,b in zip(initial, final))
    if not np.isfinite(change): raise FloatingPointError('Nonfinite final parameters')
    np.savez(output/'model_export.npz', **{f'p_{i:04d}': value for i,value in enumerate(final)})
    (output/'model_schema.json').write_bytes(canonical_json(schema))
    report.update(status='passed', turn=turn, config_sha256=config_sha, maximum_parameter_change=change,
                  checkpoint_compression='zip-deflate-level1',
                  last_metrics=last_metrics, elapsed_segment_seconds=time.perf_counter()-start_time,
                  process_cpu_segment_seconds=time.process_time()-start_cpu,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  model_export_sha256=checkpoints.sha256(output/'model_export.npz'))
    verify(ROOT)
    mh.sync_global_devices('training-finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-receipt', type=Path, default=os.environ.get('GOZERO_NATIVE_RECEIPT'))
    parser.add_argument('--resume', type=Path); parser.add_argument('--stop-after-turn', type=int)
    args = parser.parse_args()
    verify(ROOT)
    c = validate(read_json(args.config))
    if canonical_json(c) != canonical_json(read_json(ROOT/'resolved_config.json')): raise ValueError('Configuration is not the frozen resolved config')
    if args.native_receipt is None: parser.error('--native-receipt or GOZERO_NATIVE_RECEIPT is required')
    args.native_receipt = Path(args.native_receipt).resolve()
    if os.environ.get('JAX_PLATFORMS', c['platform']) != c['platform']: raise ValueError('JAX platform environment differs from the recipe')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'gumbel_alphazero_training', 'status': 'running', 'snapshot_id': ROOT.name,
              'claims_go_strength': False, 'resumable_training_state': True, 'platform': c['platform']}
    distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        train(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        (args.output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k!='native'}, sort_keys=True), flush=True)
        if distributed: jax.distributed.shutdown()


if __name__ == '__main__': main()
