"""Paired full/suffix native observation reconstruction during real MCTS."""
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
from gozero.visual_history import Replay, observation_sha256
from gozero.visual_sequence_batches import Dataset


def run(c, output, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    import inference
    import model
    import observations
    if (jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Replay profiling topology differs')
    root = SOURCE.parents[2]; host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    if getattr(native, 'OBSERVATION_SUFFIX_REPLAY_ABI_VERSION', None) != 1:
        raise ValueError('Qualified native suffix capability required')
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256']); size = data.size
    if c['candidate'] is None:
        net = c['model']; params = model.initialize(c['seed'], net); version = 1
    else:
        path = artifact(SOURCE, c['candidate']); candidate = read_json(path); checked = validate_candidate(root, candidate)
        net = checked['config']['model']; version = candidate['network_version']
        if checked['model_code_sha256'] != sha256(Path(__file__).with_name('model.py')) or checked['rules'] != data.manifest['rules']:
            raise ValueError('Training decoder or exact board rules differ')
        definition = jax.tree.structure(jax.eval_shape(lambda: model.initialize(0, net)))
        params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(checked['model_schema']))])
        report['candidate_sha256'] = sha256(path); del checked
    slots, capacity, positions = c['slots'], c['cache_positions'], c['root_positions']
    if not positions + c['plies'] + c['search']['simulations'] < capacity <= net['max_positions']:
        raise ValueError('Native search could exceed the complete decoder context')
    runners = {name: inference.Runner(params, net, native, data.manifest['rules'], slots=slots,
        cache_positions=capacity, network_version=version, max_block=c['max_block'], suffix_replay=name == 'suffix')
        for name in ('full', 'suffix')}
    for runner in runners.values(): runner.warmup()
    eligible = []
    for shard, episode in data.indices['expert', 1]:
        arrays = data.shards[shard]; begin, end = map(int, arrays['expert_offsets'][episode:episode + 2])
        if end - begin >= positions: eligible.append((shard, episode))
    if not eligible: raise ValueError('No held-out root reaches the registered history length')
    rng = np.random.default_rng(c['seed'] + host); record = []; episodes = []; observed_heads = 0; search_seconds = 0.
    work = hashlib.sha256()
    for repetition in range(c['repetitions']):
        selected = [eligible[int(i)] for i in rng.choice(len(eligible), slots, replace=len(eligible) < slots)]
        tapes = []
        for shard, episode in selected:
            arrays = data.shards[shard]; begin = int(arrays['expert_offsets'][episode])
            tapes.append(arrays['expert_actions'][begin:begin + positions - 1].tolist())
        exact = Replay(native, data.manifest['rules'], capacity)(tapes)
        requests = {i: (tape, observation_sha256(o[-1])) for i, (tape, o) in enumerate(zip(tapes, exact))}
        for runner in runners.values(): runner.score(requests)
        games = [native.Game(json.dumps({**data.manifest['rules'], **c['search'], 'history': 1})) for _ in tapes]
        for game, tape in zip(games, tapes):
            for i, action in enumerate(tape): game.play(1 + i % 2, action)
        before = {name: dict(r.stats) for name, r in runners.items()}
        mh.sync_global_devices(f'visual-replay-repetition-{repetition}')
        times = {name: 0. for name in runners}; roots = []; calls = 0; exact_leaves = 0
        for turn in range(c['plies']):
            started = time.perf_counter()
            pending = {slot: game.start(version) for slot, game in enumerate(games) if not game.state()[2]}
            search_seconds += time.perf_counter() - started
            while pending:
                started = time.perf_counter()
                requests = {}
                for slot, (request, features) in pending.items():
                    tape = games[slot].request_history(request).tolist()
                    board = observations.from_native(features.reshape(size, size, 6))
                    requests[slot] = (tape, observation_sha256(board))
                work.update(canonical_json({str(k): v for k, v in requests.items()}))
                search_seconds += time.perf_counter() - started
                prediction = {}
                order = ('full', 'suffix') if (calls + repetition) % 2 == 0 else ('suffix', 'full')
                for name in order:
                    started = time.perf_counter(); prediction[name] = runners[name].score(requests)
                    times[name] += time.perf_counter() - started
                for slot in requests:
                    for head, actual in prediction['full'][slot].items():
                        np.testing.assert_array_equal(actual, prediction['suffix'][slot][head]); observed_heads += 1
                calls += 1; exact_leaves += len(requests); next_pending = {}
                started = time.perf_counter()
                for slot, (request, _) in pending.items():
                    value = prediction['full'][slot]
                    request, features = games[slot].evaluate(request, np.ascontiguousarray(value['expert_logits'], np.float32), float(value['value']))
                    if request is not None: next_pending[slot] = (request, features)
                    else:
                        chosen, policy, value, simulations, neural, terminal = games[slot].finish()
                        color = games[slot].state()[1]; games[slot].play(color, chosen)
                        roots.append({'slot': slot, 'turn': turn, 'action': chosen, 'simulations': simulations,
                                      'neural': neural, 'terminal': terminal, 'policy': policy.tolist(), 'value': value})
                search_seconds += time.perf_counter() - started; pending = next_pending
        delta = {name: {k: r.stats[k] - before[name][k] for k in before[name]} for name, r in runners.items()}
        for key in ('requests', 'prefix_hits', 'dispatches', 'appended_positions', 'padded_position_slots', 'native_validated_history_moves'):
            if delta['full'][key] != delta['suffix'][key]: raise ValueError('Matched native search workload diverged')
        record.append({'repetition': repetition, 'scorer_seconds': times, 'stats': delta, 'batch_calls': calls,
                       'exact_leaf_predictions': exact_leaves, 'roots': roots})
        episodes.append(selected)
        print(json.dumps({'kind': 'native_suffix_profile', 'host': host, 'repetition': repetition,
            'seconds': times, 'requests': exact_leaves, 'encoded_positions': {n: s['native_encoded_positions'] for n, s in delta.items()}}), flush=True)
    report.update(status='passed', host_rank=host, jax_rank=jax.process_index(), parameter_count=sum(s['elements'] for s in model.parameter_schema(net)),
        repetitions=record, selected_heldout_episode_ids=episodes, work_sha256=work.hexdigest(), bitwise_head_comparisons=observed_heads,
        shared_native_search_seconds=search_seconds, compilation={name: r.compilation for name, r in runners.items()},
        native_receipt_sha256=sha256(receipt_path), device_memory_stats=[d.memory_stats() for d in jax.local_devices()])
    verify(SOURCE); mh.sync_global_devices('native-suffix-profile-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices seed candidate model native_receipt dataset slots cache_positions root_positions plies repetitions max_block search'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_native_suffix_profile'
            or c['platform'] not in ('cpu', 'tpu') or (c['candidate'] is None) == (c['model'] is None)
            or not 1 <= c['repetitions'] <= 5 or not 1 <= c['plies'] <= 32
            or args.resume is not None or args.stop_after_turn is not None
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))):
        raise ValueError('Invalid frozen native suffix profile')
    os.environ['JAX_PLATFORMS'] = c['platform']; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'claims_mfu': False, 'claims_go_strength_improvement': False,
        'scope': 'Actual native MCTS paths, paired bitwise-identical trained decoder heads and work. Time includes Rust replay, observation packing, TPU dispatch, KV completion and head transfer. It excludes shared search advancement, initial prefill/compilation and external process queues.'}
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
