"""Profile local tensor/data meshes on unchanged trained decoder computations."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_artifacts import validate as validate_candidate
from gozero.visual_history import Replay
from gozero.visual_sequence_batches import Dataset


def run(c, output, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    import model
    import parallel
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if (jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Parallel inference topology differs')
    root = SOURCE.parents[2]; receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256']); size = data.size
    positions, capacity, batch = c['root_positions'], c['cache_positions'], c['sequences_per_host']
    if c['candidate'] is None:
        net = c['model']; params = model.initialize(c['seed'], net); version = 1
    else:
        descriptor_path = artifact(SOURCE, c['candidate']); descriptor = read_json(descriptor_path)
        checked = validate_candidate(root, descriptor); net = checked['config']['model']; version = descriptor['network_version']
        if checked['model_code_sha256'] != sha256(Path(__file__).with_name('model.py')) or checked['rules'] != data.manifest['rules']:
            raise ValueError('Decoder or complete-history rules differ from training')
        definition = jax.tree.structure(jax.eval_shape(lambda: model.initialize(0, net)))
        params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(checked['model_schema']))])
        report['candidate_sha256'] = sha256(descriptor_path); del checked
    longest = positions + max(c['horizons'])
    if not longest <= capacity <= net['max_positions']:
        raise ValueError('Complete sequence would exceed the registered cache')
    eligible = []
    for shard, episode in data.indices['expert', 1]:
        arrays = data.shards[shard]; begin, end = map(int, arrays['expert_offsets'][episode:episode + 2])
        if end - begin >= longest:
            eligible.append((shard, episode))
    if not eligible:
        raise ValueError('No held-out native trajectory reaches the registered length')
    rng = np.random.default_rng(c['seed'] + host)
    entries = [eligible[int(i)] for i in rng.choice(len(eligible), size=batch, replace=len(eligible) < batch)]
    tapes = []
    for shard, episode in entries:
        arrays = data.shards[shard]; start = int(arrays['expert_offsets'][episode])
        tapes.append(arrays['expert_actions'][start:start + longest - 1].tolist())
    obs = np.stack(Replay(native, data.manifest['rules'], capacity + 1)(tapes))
    actions = np.asarray([tape + [size * size] for tape in tapes], np.int32)
    counts = np.full(batch, positions, np.int32)
    report.update(host_rank=host, jax_rank=jax.process_index(), parameter_count=sum(s['elements'] for s in model.parameter_schema(net)),
        global_sequences=batch * jax.process_count(), root_positions=positions, exact_native_histories=tapes,
        selected_episode_ids=entries, exact_inputs_sha256=hashlib.sha256(obs.tobytes() + actions.tobytes()).hexdigest(),
        native_receipt_sha256=sha256(receipt_path), qualification=[], compilation={}, timings={})
    layouts = {}; caches = {}; executables = {}; inputs = {}; reference = None
    for tp in c['tensor_parallel_sizes']:
        started = time.perf_counter(); layout = parallel.Layout(params, net, tp); layouts[tp] = layout
        prediction, cache = layout.prefill(obs[:, :positions], actions[:, :positions], counts, capacity=capacity, version=version)
        jax.block_until_ready((prediction, cache)); caches[tp] = cache
        report['compilation'][f'tp{tp}-root'] = {'seconds': time.perf_counter() - started,
            'kv_local_shard_shape': list(cache['keys'][0].addressable_shards[0].data.shape),
            'mlp_expansion_local_shard_shape': list(layout.params['blocks'][0]['gate'].addressable_shards[0].data.shape)}
        if reference is None:
            full = jax.jit(lambda p, o, a, n: model.forward(p, o, a, n, layout.c),
                in_shardings=(layout.param_shardings, layout.batched, layout.batched, layout.batched), out_shardings=layout.batched)
            reference = jax.device_get(full(layout.params, layout.put(obs), layout.put(actions), layout.put(np.full(batch, longest, np.int32))))
        for horizon in c['horizons']:
            name = f'tp{tp}-h{horizon}'
            future = layout.put(obs[:, positions:positions + horizon]); proposed = layout.put(actions[:, positions - 1:positions + horizon - 1])
            added = layout.put(np.full(batch, horizon, np.int32)); prefix = layout.put(counts * model.layout(size, net)['stride'] - 1)
            static = (future, proposed, added, prefix); inputs[tp, horizon] = static
            start = time.perf_counter(); lowered = layout.continuation(version=version).lower(layout.params, caches[tp], *static)
            fn = lowered.compile(); executables[tp, horizon] = fn
            report['compilation'][name] = {'seconds': time.perf_counter() - start,
                'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
                'memory_bytes': {k: int(getattr(fn.memory_analysis(), k)) for k in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']}}
            result, caches[tp] = fn(layout.params, caches[tp], *static); actual = jax.device_get(result)
            if not np.asarray(caches[tp]['valid']).all():
                raise ValueError('Tensor-sharded cache rejected exact continuation')
            np.testing.assert_array_equal(np.asarray(caches[tp]['lengths']), (counts + horizon) * model.layout(size, net)['stride'] - 1)
            errors = {}; tvs = {}
            legal = np.concatenate([obs[:, positions:positions + horizon, ..., 5].reshape(batch, horizon, size * size) > .5,
                                    np.ones((batch, horizon, 1), bool)], -1)
            for key, value in actual.items():
                target = reference[key][:, positions:positions + horizon]
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite tensor-parallel head')
                errors[key] = float(np.max(np.abs(value - target)))
                np.testing.assert_allclose(value, target, atol=c['absolute_tolerance'], rtol=c['absolute_tolerance'])
                if key in ('expert_logits', 'behavior_logits'):
                    def probability(x):
                        x = np.where(legal, x.astype(np.float64), -1e30); y = np.exp(x - x.max(-1, keepdims=True))
                        return y / y.sum(-1, keepdims=True)
                    tvs[key] = float(np.max(np.abs(probability(value) - probability(target)).sum(-1) / 2))
            if max(tvs.values()) > c['maximum_policy_tv'] or errors['value'] > c['maximum_value_error']:
                raise ValueError('Tensor partition exceeds the registered policy/value numerical contract')
            report['qualification'].append({'name': name, 'maximum_absolute_head_errors': errors, 'maximum_legal_policy_tv': tvs})
            print(json.dumps({'kind': 'tensor_parallel_qualified', 'host': host, 'name': name, 'policy_tv': tvs}), flush=True)
    for horizon in c['horizons']:
        for repetition in range(c['repetitions']):
            order = c['tensor_parallel_sizes'] if repetition % 2 == 0 else list(reversed(c['tensor_parallel_sizes']))
            for tp in order:
                mh.sync_global_devices(f'tensor-profile-h{horizon}-r{repetition}-tp{tp}')
                started = time.perf_counter(); result, caches[tp] = executables[tp, horizon](layouts[tp].params, caches[tp], *inputs[tp, horizon])
                jax.block_until_ready((result, caches[tp])); jax.device_get(result)
                seconds = time.perf_counter() - started
                report['timings'].setdefault(f'tp{tp}-h{horizon}', []).append(seconds)
        print(json.dumps({'kind': 'tensor_parallel_timing', 'host': host, 'horizon': horizon,
                          'seconds': {str(tp): report['timings'][f'tp{tp}-h{horizon}'] for tp in c['tensor_parallel_sizes']}}), flush=True)
    report['status'] = 'passed'; verify(SOURCE); mh.sync_global_devices('tensor-profile-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices seed candidate model native_receipt dataset sequences_per_host root_positions cache_positions horizons tensor_parallel_sizes repetitions maximum_policy_tv maximum_value_error absolute_tolerance'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_tensor_parallel_profile'
            or c['platform'] not in ('cpu', 'tpu') or (c['candidate'] is None) == (c['model'] is None)
            or c['tensor_parallel_sizes'] != [1, 2, 4] or not 1 <= c['repetitions'] <= 7
            or not c['horizons'] or any(type(h) is not int or not 1 <= h <= 16 for h in c['horizons'])
            or not 0 < c['maximum_policy_tv'] <= .04 or not 0 < c['maximum_value_error'] <= .04
            or not 0 < c['absolute_tolerance'] <= .1
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))
            or args.resume is not None or args.stop_after_turn is not None):
        raise ValueError('Invalid frozen tensor-parallel profile')
    os.environ['JAX_PLATFORMS'] = c['platform']; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'claims_mfu': False, 'claims_rollout_speedup': False,
        'scope': 'Same parameters and exact held-out trajectories. XLA attention in every mesh; local tensor groups avoid inter-host inference collectives. Timing includes target inference, donated KV completion and head transfer, excludes prefill, board generation, queuing, drafting and acceptance.'}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(c, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        (args.output / 'result.json').write_bytes(canonical_json(report))
        if distributed:
            jax.distributed.shutdown()
