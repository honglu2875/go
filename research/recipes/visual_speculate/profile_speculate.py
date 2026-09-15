"""Bounded exact-dynamics draft/verification screening on complete held-out roots."""
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
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gozero import jax_go
    import model
    import speculate
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if (jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Speculative profile topology differs')
    root = SOURCE.parents[2]; receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    rules = data.manifest['rules']; size = rules['size']; batch = c['sequences_per_host']; positions = c['root_positions']
    capacity = c['cache_positions']; depth = c['draft_depth']
    if c['candidate'] is not None:
        descriptor_path = artifact(SOURCE, c['candidate']); descriptor = read_json(descriptor_path)
        checked = validate_candidate(root, descriptor)
        if checked['model_code_sha256'] != sha256(Path(__file__).with_name('model.py')) or checked['rules'] != rules:
            raise ValueError('Trained decoder or exact rules differ from probe')
        net = checked['config']['model']; schema = model.parameter_schema(net)
        definition = jax.tree.structure(jax.eval_shape(lambda: model.initialize(0, net)))
        params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(schema))])
        version = descriptor['network_version']; report['candidate_sha256'] = sha256(descriptor_path)
        del checked
    else:
        net = c['model']; params = model.initialize(c['seed'], net); version = 1
    if not positions + max(c['horizons']) < capacity <= net['max_positions'] or not 1 <= depth < net['layers']:
        raise ValueError('Profile context or exit depth differs')
    mesh = Mesh(np.asarray(jax.local_devices()), ('data',)); rep = NamedSharding(mesh, P()); bat = NamedSharding(mesh, P('data'))
    params = jax.device_put(params, rep)
    def put(x):
        return jax.device_put(x, bat)
    # Every root includes all earlier observations and actions. Prefix length is
    # a game position, never a truncated attention window or future label input.
    eligible = []
    for shard, episode in data.indices['expert', 1]:
        arrays = data.shards[shard]; a, b = map(int, arrays['expert_offsets'][episode:episode + 2])
        if b - a >= positions:
            eligible.append(('expert', shard, episode))
    rng = np.random.default_rng(c['seed'] + host)
    if not eligible:
        raise ValueError('No complete held-out roots reach registered prefix')
    selected = [eligible[int(i)] for i in rng.choice(len(eligible), size=batch, replace=len(eligible) < batch)]
    histories = []
    for _, shard, episode in selected:
        arrays = data.shards[shard]; begin = int(arrays['expert_offsets'][episode])
        histories.append(arrays['expert_actions'][begin:begin + positions - 1].tolist())
    replay = Replay(native, rules, capacity)
    obs = np.stack(replay(histories)); actions = np.asarray([t + [size * size] for t in histories], np.int32)
    counts = np.full(batch, positions, np.int32)
    report.update(host_rank=host, jax_rank=jax.process_index(), parameter_count=sum(s['elements'] for s in model.parameter_schema(net)),
        native_receipt_sha256=sha256(receipt_path), global_sequences=batch * jax.process_count(),
        root_positions=positions, exact_root_histories=histories, selected_episode_ids=selected,
        exact_root_input_sha256=hashlib.sha256(obs.tobytes() + actions.tobytes()).hexdigest())
    cache_specs = {'keys': (P('data'),) * net['layers'], 'values': (P('data'),) * net['layers'],
                   'lengths': P('data'), 'valid': P('data'), 'network_version': P()}
    prefill = jax.jit(jax.shard_map(lambda p, o, a, n: model.forward(p, o, a, n, net, with_cache=True,
        network_version=version, cache_positions=capacity, return_exits=(depth,)), mesh=mesh,
        in_specs=(P(), P('data'), P('data'), P('data')), out_specs=((P('data'), P('data')), cache_specs), check_vma=False))
    (deep, exits), cache = prefill(params, put(obs), put(actions), put(counts))
    deep_root = jax.tree.map(lambda x: x[:, positions - 1], deep)
    shallow_root = jax.tree.map(lambda x: x[:, positions - 1], exits[depth])
    initialize = jax.jit(jax.shard_map(jax.vmap(lambda a, n: jax_go.replay(a, n, size, rules['komi'], capacity)),
        mesh=mesh, in_specs=(P('data'), P('data')), out_specs=(P('data'), P('data')), check_vma=False))
    state, current = initialize(put(actions), put(counts - 1))
    np.testing.assert_array_equal(jax.device_get(current), obs[:, -1])
    if not np.asarray(state['valid']).all():
        raise ValueError('Compiled root rules rejected a native history')
    keys = put(jax.random.split(jax.random.key(c['seed'] + 1009 * host), batch))
    view = put(1 + np.arange(batch, dtype=np.int32) % 2)
    inputs = (params, cache, deep_root, shallow_root, state, keys, view)
    jax.block_until_ready(inputs)
    report['compilation'] = {}; report['conditions'] = []
    full_reference = jax.jit(jax.shard_map(lambda p, o, a, n: model.forward(p, o, a, n, net), mesh=mesh,
        in_specs=(P(), P('data'), P('data'), P('data')), out_specs=P('data'), check_vma=False))
    for horizon in c['horizons']:
        for role in c['draft_roles']:
            name = f'h{horizon}-{role}'
            def function(p, ca, r, q, s, k, v):
                rows, _, _ = speculate.packet(p, ca, r, q, s, k, v, net, horizon=horizon,
                    draft_depth=depth, size=size, komi=rules['komi'], network_version=version, draft_role=role)
                return rows
            fn = jax.shard_map(function, mesh=mesh,
                in_specs=(P(), cache_specs, P('data'), P('data'), P('data'), P('data'), P('data')),
                out_specs=P('data'), check_vma=False)
            started = time.perf_counter(); lowered = jax.jit(fn).lower(*inputs); compiled = lowered.compile()
            report['compilation'][name] = {'seconds': time.perf_counter() - started,
                'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest()}
            print(json.dumps({'kind': 'speculation_compiled', 'condition': name, 'host': host}), flush=True)
            rows = jax.device_get(compiled(*inputs))
            # Check every proposed pre-action state and each legal future
            # observation against Rust, including captures and terminal passes.
            checked_positions = 0; longest = positions + horizon
            all_obs = np.zeros((batch, longest, size, size, 6), np.float32)
            all_actions = np.zeros((batch, longest), np.int32); all_counts = np.zeros(batch, np.int32)
            for game, history in enumerate(histories):
                tape = list(history)
                all_obs[game, :positions] = obs[game]; all_actions[game, :positions - 1] = history
                n = positions
                for t in range(horizon):
                    if not rows['active'][game, t]:
                        break
                    exact = replay([tape])[0][-1]
                    np.testing.assert_array_equal(rows['legal'][game, t], np.r_[exact[..., 5].reshape(-1) > .5, True])
                    action = int(rows['actions'][game, t]); checked_positions += 1
                    if not rows['legal'][game, t, action]:
                        raise ValueError('Draft selected an illegal native action')
                    all_actions[game, positions + t - 1] = action; tape.append(action)
                    if rows['appended'][game, t]:
                        exact = replay([tape])[0][-1]
                        np.testing.assert_array_equal(rows['future_observations'][game, t], exact)
                        all_obs[game, positions + t] = exact; n += 1
                    elif tape[-2:] != [size * size, size * size]:
                        raise ValueError('Draft stopped without an exact terminal pass pair')
                all_counts[game] = n
            full = jax.device_get(full_reference(params, put(all_obs), put(all_actions), put(all_counts)))
            logits = full['expert_logits'][:, positions - 1:positions - 1 + horizon]
            logits = np.where(rows['legal'], logits.astype(np.float64), -1e30)
            weights = np.exp(logits - logits.max(-1, keepdims=True)); target = weights / weights.sum(-1, keepdims=True)
            tv = np.abs(rows['p'] - target).sum(-1) / 2
            maximum_tv = float(tv[rows['active']].max(initial=0.))
            if maximum_tv > c['maximum_policy_tv']:
                raise ValueError('Deep block probabilities exceed registered full-reference error')
            times = []; resolutions = []; host_audit_seconds = []
            for repetition in range(c['repetitions']):
                mh.sync_global_devices(f'speculate-{name}-{repetition}')
                start = time.perf_counter(); result = jax.device_get(compiled(*inputs)); times.append(time.perf_counter() - start)
                start = time.perf_counter()
                accepted = speculate.resolve(result, rng.random((batch, horizon)), rng.random((batch, horizon)))
                # One GIL-free packed Rust replay checks the actual accepted
                # prefix and replacement; terminal rows are handled explicitly.
                accepted_histories = []
                for history, decision in zip(histories, accepted):
                    tape = history + decision['actions']
                    terminal = len(tape) >= 2 and tape[-2:] == [size * size, size * size]
                    accepted_histories.append(tape[:-1] if terminal else tape)
                replay(accepted_histories)
                host_audit_seconds.append(time.perf_counter() - start); resolutions.append(accepted)
            np.savez_compressed(output / (name + '.npz'), **rows)
            report['conditions'].append({'name': name, 'horizon': horizon, 'draft_role': role,
                'exact_positions_checked': checked_positions, 'maximum_target_vs_full_policy_tv': maximum_tv,
                'packet_seconds': times, 'coupling_and_native_audit_seconds': host_audit_seconds,
                'resolutions': resolutions, 'packet_arrays_sha256': sha256(output / (name + '.npz')),
                'active_proposed_moves': int(rows['active'].sum()),
                'mean_same_path_overlap': float((1 - np.abs(rows['q'] - rows['p']).sum(-1) / 2)[rows['active']].mean())})
            print(json.dumps({'kind': 'speculation_finished', 'condition': name, 'host': host,
                'packet_seconds': times, 'maximum_policy_tv': maximum_tv}), flush=True)
    # Compare a single exact full-model step. This is an independent-root
    # amortization screen, not equal-work completed-game throughput.
    def one(p, ca, r, s, k, v):
        (_, next_prediction, _), rows = speculate.draft(p, ca, r, s, k, v, net, horizon=1, depth=net['layers'],
            size=size, komi=rules['komi'], network_version=version, role='expert')
        return rows, next_prediction
    fn = jax.jit(jax.shard_map(one, mesh=mesh,
        in_specs=(P(), cache_specs, P('data'), P('data'), P('data'), P('data')),
        out_specs=(P('data'), P('data')), check_vma=False))
    baseline_inputs = (params, cache, deep_root, state, keys, view)
    compiled = fn.lower(*baseline_inputs).compile(); jax.block_until_ready(compiled(*baseline_inputs))
    report['one_full_step_seconds'] = []
    for repetition in range(c['repetitions']):
        mh.sync_global_devices(f'speculate-baseline-{repetition}')
        start = time.perf_counter(); jax.device_get(compiled(*baseline_inputs)); report['one_full_step_seconds'].append(time.perf_counter() - start)
    report['status'] = 'passed'; verify(SOURCE); mh.sync_global_devices('speculate-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices seed candidate model native_receipt dataset sequences_per_host root_positions cache_positions draft_depth horizons draft_roles repetitions maximum_policy_tv'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_speculation_profile'
            or c['platform'] not in ('cpu', 'tpu') or (c['candidate'] is None) == (c['model'] is None)
            or not 1 <= c['repetitions'] <= 7 or not 1 <= len(c['horizons']) <= 4
            or any(type(h) is not int or not 1 <= h <= 16 for h in c['horizons'])
            or not c['draft_roles'] or set(c['draft_roles']) - {'expert', 'alternating'}
            or not 0 < c['maximum_policy_tv'] <= .05 or not 1 <= c['sequences_per_host'] <= 128
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))
            or args.resume is not None or args.stop_after_turn is not None):
        raise ValueError('Invalid frozen speculative profile')
    os.environ['JAX_PLATFORMS'] = c['platform']; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'claims_mcts_equivalence': False, 'claims_mfu': False, 'claims_completed_game_speedup': False,
        'scope': 'One fused early-exit draft scan with exact compiled Go transitions and deep block verification. Independent held-out roots and explicit host coupling/Rust audits; excludes root prefill and future cache-repair costs.'}
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
