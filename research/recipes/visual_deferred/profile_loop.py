"""Continuous exact-Go policy decoding with paid cache commit and repair."""
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
    import loop
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
    if not positions + c['moves_per_root'] + max(c['horizons']) < capacity <= net['max_positions'] or not 1 <= depth < net['layers']:
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
    def compile_fn(name, fn, args, in_specs, out_specs):
        started = time.perf_counter()
        lowered = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)).lower(*args)
        executable = lowered.compile()
        report['compilation'][name] = {'seconds': time.perf_counter() - started,
            'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest()}
        result = executable(*args); jax.block_until_ready(result)
        print(json.dumps({'kind': 'loop_compiled', 'host': host, 'condition': name}), flush=True)
        return executable, result
    remaining = put(np.full(batch, c['moves_per_root'], np.int32))
    common = (params, cache, deep_root, state, keys, remaining)
    serial, _ = compile_fn('serial', lambda p, ca, r, s, k, left: loop.serial(p, ca, r, s, k, left, net,
        size=size, komi=rules['komi'], version=version), common,
        (P(), cache_specs, P('data'), P('data'), P('data'), P('data')),
        (P('data'), cache_specs, P('data'), P('data')))
    functions = {}
    for horizon in c['horizons']:
        fn, result = compile_fn(f'packet-{horizon}', lambda p, ca, r, s, k, left, pending: loop.packet(p, ca, r, s, k, left, pending, net,
            horizon=horizon, depth=depth, size=size, komi=rules['komi'], version=version), (*common, put(np.full(batch, -1, np.int32))),
            (P(), cache_specs, P('data'), P('data'), P('data'), P('data'), P('data')),
            (P('data'), cache_specs, P('data')))
        rows, verified, predictions = result
        accepted = put(np.ones(batch, np.int32)); replacement = put(np.full(batch, -1, np.int32))
        repair, _ = compile_fn(f'settle-{horizon}', lambda p, before, r, s, rows, verified, predicted, a, replacement:
            loop.settle(p, before, r, s, rows, verified, predicted, a, replacement, net,
                size=size, komi=rules['komi'], version=version),
            (params, cache, deep_root, state, rows, verified, predictions, accepted, replacement),
            (P(), cache_specs, P('data'), P('data'), P('data'), cache_specs, P('data'), P('data'), P('data')),
            (cache_specs, P('data'), P('data'), P('data')))
        functions[horizon] = (fn, repair)
    drain, _ = compile_fn('drain', lambda p, ca, r, s, pending: loop.drain(p, ca, r, s, pending, net,
        size=size, komi=rules['komi'], version=version), (params, cache, deep_root, state, put(np.full(batch, -1, np.int32))),
        (P(), cache_specs, P('data'), P('data'), P('data')), (cache_specs, P('data')))
    full_reference = jax.jit(jax.shard_map(lambda p, o, a, n: model.forward(p, o, a, n, net), mesh=mesh,
        in_specs=(P(), P('data'), P('data'), P('data')), out_specs=P('data'), check_vma=False))
    reference_positions = positions + c['moves_per_root']
    full_obs = np.zeros((batch, reference_positions, size, size, 6), np.float32)
    full_actions = np.zeros((batch, reference_positions), np.int32)
    full_obs[:, :positions] = obs; full_actions[:, :positions] = actions
    jax.block_until_ready(full_reference(params, put(full_obs), put(full_actions), put(counts)))
    records = {h: [] for h in [0, *c['horizons']]}
    for repetition in range(c['repetitions']):
        order = [0, *c['horizons']] if repetition % 2 == 0 else [*reversed(c['horizons']), 0]
        for horizon in order:
            ca, prediction, current = cache, deep_root, state
            pending = put(np.full(batch, -1, np.int32)); deferred_replacements = 0
            keys = put(jax.random.split(jax.random.key(c['seed'] + 1009 * host + 100003 * repetition), batch))
            accept_rng = np.random.default_rng(c['seed'] + 700001 + 1009 * host + 100003 * repetition)
            tapes = [list(h) for h in histories]; left = np.full(batch, c['moves_per_root'], np.int32)
            iterations = dispatches = repairs = accepted_moves = rejected_packets = 0
            coupling_seconds = 0.; acceptance_records = []
            mh.sync_global_devices(f'loop-{horizon}-{repetition}')
            start = time.perf_counter()
            while np.any(left > 0):
                iterations += 1
                if iterations > c['moves_per_root']: raise ValueError('Loop failed to make bounded progress')
                if horizon == 0:
                    rows, ca, prediction, current = serial(params, ca, prediction, current, keys, put(left))
                    packet_rows, valid, passes = jax.device_get((rows, current['valid'], current['passes']))
                    dispatches += 1
                    for game in range(batch):
                        if packet_rows['active'][game]:
                            tapes[game].append(int(packet_rows['actions'][game])); left[game] -= 1
                else:
                    fn, repair = functions[horizon]
                    rows, verified, predictions = fn(params, ca, prediction, current, keys, put(left), pending)
                    packet_rows = jax.device_get({k: rows[k] for k in ['q', 'p', 'actions', 'active']})
                    now = time.perf_counter()
                    decisions = speculate.resolve(packet_rows, accept_rng.random((batch, horizon)), accept_rng.random((batch, horizon)))
                    coupling_seconds += time.perf_counter() - now
                    accepted = np.asarray([d['accepted_draft_moves'] for d in decisions], np.int32)
                    replacement = np.asarray([-1 if d['replacement'] is None else d['replacement'] for d in decisions], np.int32)
                    ca, prediction, current, pending = repair(params, ca, prediction, current, rows, verified, predictions, put(accepted), put(replacement))
                    valid, passes, pending_host = jax.device_get((current['valid'], current['passes'], pending))
                    dispatches += 2; deferred_replacements += int((pending_host >= 0).sum()); accepted_moves += int(accepted.sum())
                    rejected_packets += int((replacement >= 0).sum())
                    acceptance_records.append({'accepted': accepted.tolist(), 'replacement': replacement.tolist()})
                    for game, decision in enumerate(decisions):
                        tapes[game].extend(decision['actions']); left[game] -= len(decision['actions'])
                if not np.asarray(valid).all() or np.any(left < 0): raise ValueError('Invalid device move or excess work')
                left[np.asarray(passes) >= 2] = 0
            if horizon:
                repairs = int((np.asarray(pending) >= 0).sum())
                ca, prediction = drain(params, ca, prediction, current, pending); dispatches += 1
            jax.block_until_ready((ca, prediction, current))
            seconds = time.perf_counter() - start
            # Audits are symmetric and recorded separately from the complete
            # timed recurrence; no final KV or replacement work is omitted.
            audit_start = time.perf_counter(); native_count = 0
            final = jax.device_get(current); cache_metadata = jax.device_get({k: ca[k] for k in ['lengths', 'valid', 'network_version']})
            if not cache_metadata['valid'].all(): raise ValueError('Invalid retained cache')
            root_prediction = jax.device_get(prediction); final_counts = np.zeros(batch, np.int32); live = []
            for game, tape in enumerate(tapes):
                terminal = tape[-2:] == [size * size, size * size]
                exact = replay([tape[:-1] if terminal else tape])[0]
                if not np.array_equal(final['stones'][game], np.argmax(np.stack([1 - exact[-1, ..., 0] - exact[-1, ..., 1], exact[-1, ..., 0], exact[-1, ..., 1]], -1), -1).reshape(-1)):
                    raise ValueError('Committed device board differs from native complete-history replay')
                if int(final['ply'][game]) != len(tape) or (int(final['passes'][game]) >= 2) != terminal:
                    raise ValueError('Committed device ending differs')
                length = len(exact); final_counts[game] = length
                full_obs[game] = 0; full_obs[game, :length] = exact
                full_actions[game] = 0; full_actions[game, :length - 1] = tape[:length - 1]
                if int(cache_metadata['lengths'][game]) != length * model.layout(size, net)['stride'] - 1:
                    raise ValueError('Retained cache length differs from committed history')
                if not terminal: live.append(game)
                native_count += len(tape) - len(histories[game])
            reference = jax.device_get(full_reference(params, put(full_obs), put(full_actions), put(final_counts)))
            maximum_tv = 0.
            for game in live:
                legal = np.r_[full_obs[game, final_counts[game] - 1, ..., 5].reshape(-1) > .5, True]
                def prob(logits):
                    logits = np.where(legal, logits.astype(np.float64), -1e30); values = np.exp(logits - logits.max()); return values / values.sum()
                maximum_tv = max(maximum_tv, float(np.abs(prob(root_prediction['expert_logits'][game]) - prob(reference['expert_logits'][game, final_counts[game] - 1])).sum() / 2))
            if maximum_tv > c['maximum_policy_tv']: raise ValueError('Continuous retained-cache error exceeds registered full-reference tolerance')
            name = f'h{horizon}-r{repetition}'
            payload = {'histories': tapes, 'acceptance_records': acceptance_records, 'cache_lengths': cache_metadata['lengths'].tolist(),
                'final_passes': final['passes'].tolist(), 'final_stones': final['stones'].tolist(), 'remaining': left.tolist()}
            path = output / (name + '.json'); path.write_bytes(canonical_json(payload))
            audit_seconds = time.perf_counter() - audit_start
            record = {'repetition': repetition, 'horizon': horizon, 'seconds': seconds, 'audit_seconds': audit_seconds,
                'committed_moves': native_count, 'iterations': iterations, 'dispatches': dispatches,
                'native_positions_checked': native_count, 'live_final_predictions_checked': len(live), 'maximum_full_policy_tv': maximum_tv,
                'accepted_draft_moves': accepted_moves, 'replacement_moves': rejected_packets, 'neural_repairs': repairs,
                'deferred_replacements': deferred_replacements,
                'coupling_seconds': coupling_seconds, 'records_sha256': sha256(path), 'records_path': path.name}
            records[horizon].append(record)
            print(json.dumps({'kind': 'loop_condition', 'host': host, **record}), flush=True)
    report['conditions'] = [{'horizon': h, 'repetitions': records[h]} for h in [0, *c['horizons']]]
    report['status'] = 'passed'; verify(SOURCE); mh.sync_global_devices('visual-policy-loop-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices seed candidate model native_receipt dataset sequences_per_host root_positions cache_positions draft_depth horizons repetitions maximum_policy_tv moves_per_root'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_deferred_loop_profile'
            or c['platform'] not in ('cpu', 'tpu') or (c['candidate'] is None) == (c['model'] is None)
            or not 1 <= c['repetitions'] <= 5 or not 1 <= len(c['horizons']) <= 3
            or any(type(h) is not int or not 2 <= h <= 8 for h in c['horizons'])
            or not 1 <= c['moves_per_root'] <= 64 or not 0 < c['maximum_policy_tv'] <= .05
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))
            or args.resume is not None or args.stop_after_turn is not None): raise ValueError('Invalid continuous-loop profile')
    os.environ['JAX_PLATFORMS'] = c['platform']; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'claims_mcts_equivalence': False, 'claims_hardware_mfu': False, 'claims_strength_gain': False,
        'scope': 'Complete bounded expert-policy decoding with deferred deep replacement scoring. The next packet computes a shallow prediction after each pending replacement and verifies that position together with new proposals; its acceptance uses the actual deep target computed in that block. Final pending replacements are fully drained inside the timer. Includes all loop dispatches, host coupling and rollback; excludes initial prefill/compilation and reports symmetric final audits separately. No MCTS or equal-trace claim.'}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu': jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(c, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        (args.output / 'result.json').write_bytes(canonical_json(report))
        if distributed: jax.distributed.shutdown()
