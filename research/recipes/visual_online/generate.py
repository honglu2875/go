"""Frozen-policy native self-play in independently reproducible episode batches."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_artifacts import validate as candidate_validate
from gozero.visual_history import observation_sha256


def validate(c):
    fields = 'schema_version kind platform expected_processes expected_devices seed candidate model native_receipt rules slots rounds max_game_moves cache_positions max_block search sampling worker_cpus suffix_replay fixture_pass_after'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_selfplay_generation'
            or c['platform'] not in ('cpu', 'tpu') or (c['candidate'] is None) == (c['model'] is None)):
        raise ValueError('Invalid frozen self-play generation configuration')
    for key, low, high in [('slots', 1, 128), ('rounds', 1, 128), ('max_game_moves', 2, 1024),
                           ('cache_positions', 4, 2048), ('max_block', 1, 32), ('seed', 0, 2**32 - 1)]:
        if type(c[key]) is not int or not low <= c[key] <= high: raise ValueError('Invalid ' + key)
    if (c['max_game_moves'] + c['search']['simulations'] >= c['cache_positions']
            or type(c['suffix_replay']) is not bool or not c['worker_cpus']
            or len(set(c['worker_cpus'])) != len(c['worker_cpus']) or len(c['worker_cpus']) > 32):
        raise ValueError('Search context or worker placement is invalid')
    if set(c['sampling']) != {'temperature', 'moves'} or not 0 < c['sampling']['temperature'] <= 2 or not 0 <= c['sampling']['moves'] <= c['max_game_moves']:
        raise ValueError('Explicit bounded improved-policy sampling is required')
    if c['fixture_pass_after'] is not None and (c['candidate'] is not None or not 0 <= c['fixture_pass_after'] <= c['max_game_moves'] - 2):
        raise ValueError('Forced passes are restricted to untrained execution fixtures')
    if c['model'] is not None:
        from config import validate_model
        validate_model(c['model'])
    return c


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    checkpoints._sync_directory(path.parent)


def run(args, c, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    import inference
    import model
    import observations
    host = int(os.environ.get('GOZERO_HOST_RANK', '0')); rank = jax.process_index(); world = jax.process_count()
    if (world != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())): raise ValueError('Self-play topology differs')
    root = SOURCE.parents[2]; receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    candidate_hash = None
    if c['candidate'] is not None:
        path = artifact(SOURCE, c['candidate']); descriptor = read_json(path); checked = candidate_validate(root, descriptor)
        net = checked['config']['model']; version = descriptor['network_version']; candidate_hash = checkpoints.sha256(path)
        if checked['model_code_sha256'] != checkpoints.sha256(Path(__file__).with_name('model.py')) or checked['rules'] != c['rules']:
            raise ValueError('Frozen self-play decoder or rules differ from training')
        definition = jax.tree.structure(jax.eval_shape(lambda: model.initialize(0, net)))
        params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(checked['model_schema']))]); del checked
    else:
        net = c['model']; version = 1; params = model.initialize(c['seed'], net)
    if c['cache_positions'] > net['max_positions']: raise ValueError('Complete actor context exceeds model')
    runner = inference.Runner(params, net, native, c['rules'], slots=c['slots'], cache_positions=c['cache_positions'],
        network_version=version, max_block=c['max_block'], suffix_replay=c['suffix_replay'])
    runner.warmup(); size = c['rules']['size']; area = size * size
    config_sha = hashlib.sha256(canonical_json(c)).hexdigest()
    def gather(value):
        raw = canonical_json(value)
        if len(raw) > 32760: raise ValueError('Generation checkpoint metadata exceeds bound')
        buffer = np.zeros(32768, np.uint8); buffer[:8] = np.frombuffer(len(raw).to_bytes(8, 'little'), np.uint8)
        buffer[8:8 + len(raw)] = np.frombuffer(raw, np.uint8)
        values = np.asarray(mh.process_allgather(buffer)).reshape(world, 32768)
        return [json.loads(bytes(row[8:8 + int.from_bytes(bytes(row[:8]), 'little')])) for row in values]
    mapping = gather({'host': host, 'jax_rank': rank})
    turn = 0; batches = []; counters = {'games': 0, 'terminal_games': 0, 'capped_games': 0, 'moves': 0, 'terminal_positions': 0}
    if args.resume is not None:
        group = read_json(args.resume.with_suffix('.group.json'))
        if (group['kind'] != 'visual_selfplay_checkpoint_group' or group['snapshot_id'] != SOURCE.name
                or group['config_sha256'] != config_sha or group['host_jax_mapping'] != mapping): raise ValueError('Generation resume identity differs')
        state, _, _ = checkpoints.read(args.resume, expected_manifest_sha256=group['host_manifests'][str(host)])
        if (state['host_rank'] != host or state['jax_rank'] != rank or state['candidate_sha256'] != candidate_hash
                or state['native_receipt_sha256'] != checkpoints.sha256(receipt_path)): raise ValueError('Generation checkpoint lineage differs')
        turn = state['turn']; counters = state['counters']; batches = state['past_batches']
        batches.append({'path': str(args.resume), 'manifest_sha256': group['host_manifests'][str(host)], 'round': turn})
        for batch in batches:
            if checkpoints.sha256(Path(batch['path']) / 'manifest.json') != batch['manifest_sha256']: raise ValueError('Prior committed generation shard changed')
    stop = c['rounds'] if args.stop_after_turn is None else args.stop_after_turn
    if not turn < stop <= c['rounds']: raise ValueError('Generation stop must advance a committed episode batch')
    placement = []; placement_lock = threading.Lock(); next_worker = iter(c['worker_cpus'])
    def initialize_worker():
        with placement_lock: cpu = next(next_worker)
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}: raise RuntimeError('Native worker CPU pin did not hold')
        with placement_lock: placement.append({'tid': threading.get_native_id(), 'cpu': cpu})
    segment_started = time.perf_counter(); timings = {'native_search_seconds': 0., 'inference_seconds': 0., 'checkpoint_seconds': 0.}
    with ThreadPoolExecutor(max_workers=len(c['worker_cpus']), initializer=initialize_worker) as pool:
        while turn < stop:
            turn += 1; runner.reset()
            games = [native.Game(json.dumps({**c['rules'], **c['search'], 'history': 1})) for _ in range(c['slots'])]
            rngs = [np.random.default_rng(np.random.SeedSequence([c['seed'], host, turn, slot])) for slot in range(c['slots'])]
            rows = [{'actions': [], 'policies': [], 'legal': [], 'search_values': [], 'simulations': [], 'neural': [], 'terminal_evaluations': []} for _ in games]
            active = list(range(c['slots']))
            for ply in range(c['max_game_moves']):
                if not active: break
                started = time.perf_counter(); pending = dict(zip(active, pool.map(lambda slot: games[slot].start(version), active)))
                root_legal = {slot: np.concatenate([value[1].reshape(size, size, 6)[..., 5].reshape(-1) > .5, [True]]) for slot, value in pending.items()}
                timings['native_search_seconds'] += time.perf_counter() - started
                while pending:
                    def request_for(item):
                        slot, (request, features) = item
                        tape = games[slot].request_history(request).tolist()
                        board = observations.from_native(features.reshape(size, size, 6))
                        return slot, (tape, observation_sha256(board))
                    started = time.perf_counter(); requests = dict(pool.map(request_for, pending.items()))
                    timings['native_search_seconds'] += time.perf_counter() - started
                    started = time.perf_counter(); predictions = runner.score(requests)
                    timings['inference_seconds'] += time.perf_counter() - started
                    def evaluate(item):
                        slot, (request, _) = item; prediction = predictions[slot]
                        return slot, games[slot].evaluate(request, np.ascontiguousarray(prediction['expert_logits'], np.float32), float(prediction['value']))
                    started = time.perf_counter(); replies = list(pool.map(evaluate, pending.items())); pending = {}
                    for slot, (request, features) in replies:
                        if request is not None: pending[slot] = (request, features); continue
                        best, policy, value, simulations, neural, terminal = games[slot].finish()
                        policy = np.asarray(policy, np.float32); legal = root_legal[slot]
                        if (not np.isfinite(policy).all() or np.any(policy < 0) or np.any(policy[~legal] != 0)
                                or not np.isclose(policy.sum(), 1., atol=2e-5)): raise ValueError('Native search target is not a legal distribution')
                        if c['fixture_pass_after'] is not None and ply >= c['fixture_pass_after']: action = area
                        elif ply < c['sampling']['moves']:
                            probability = policy.astype(np.float64) ** (1 / c['sampling']['temperature']); probability /= probability.sum()
                            action = int(rngs[slot].choice(area + 1, p=probability))
                        else: action = int(best)
                        if not legal[action]: raise ValueError('Sampled search action is illegal')
                        game = games[slot]; game.play(1 + ply % 2, action)
                        row = rows[slot]; row['actions'].append(action); row['policies'].append(policy); row['legal'].append(legal)
                        row['search_values'].append(value); row['simulations'].append(simulations)
                        row['neural'].append(neural); row['terminal_evaluations'].append(terminal)
                    timings['native_search_seconds'] += time.perf_counter() - started
                active = [slot for slot in active if not games[slot].state()[2]]
            offsets = [0]; values = []; endings = []; summaries = []
            for slot, (game, row) in enumerate(zip(games, rows)):
                _, _, terminal, score, _ = game.state(); length = len(row['actions']); offsets.append(offsets[-1] + length)
                outcomes = np.where(np.arange(length) % 2 == 0, -np.sign(score), np.sign(score)).astype(np.float32) if terminal else np.zeros(length, np.float32)
                values.extend(outcomes); endings.append(terminal)
                game_id = f'{SOURCE.name}:host{host}:round{turn}:slot{slot}'
                summaries.append({'game_id': game_id, 'actions': row['actions'], 'terminal': bool(terminal),
                    'white_score': float(score) if terminal else None, 'moves': length,
                    'simulations': sum(row['simulations']), 'neural': sum(row['neural']), 'terminal_evaluations': sum(row['terminal_evaluations'])})
                counters['games'] += 1; counters['terminal_games'] += int(terminal); counters['capped_games'] += int(not terminal)
                counters['moves'] += length; counters['terminal_positions'] += length * int(terminal)
            arrays = {'actions': np.asarray([a for r in rows for a in r['actions']], np.int32), 'offsets': np.asarray(offsets, np.int64),
                'policies': np.asarray([p for r in rows for p in r['policies']], np.float32),
                'legal': np.asarray([p for r in rows for p in r['legal']], np.bool_), 'outcomes': np.asarray(values, np.float32),
                'terminal': np.asarray(endings, np.bool_), 'search_values': np.asarray([v for r in rows for v in r['search_values']], np.float32)}
            state = {'schema_version': 1, 'kind': 'visual_selfplay_rank_state', 'snapshot_id': SOURCE.name,
                'config_sha256': config_sha, 'host_rank': host, 'jax_rank': rank, 'turn': turn,
                'candidate_sha256': candidate_hash, 'native_receipt_sha256': checkpoints.sha256(receipt_path),
                'counters': dict(counters), 'past_batches': list(batches),
                'rng_contract': 'Independent PCG64 via SeedSequence[seed,host,round,slot]; every committed checkpoint is an episode-batch boundary.'}
            started = time.perf_counter(); path = args.output / 'checkpoints' / f'turn-{turn:09d}'
            identity = checkpoints.write(path, state=state, arrays=arrays, actors=canonical_json(summaries).decode(), compress=True)
            ranks = gather({'host': host, 'manifest_sha256': identity})
            group = {'schema_version': 1, 'kind': 'visual_selfplay_checkpoint_group', 'snapshot_id': SOURCE.name,
                'config_sha256': config_sha, 'turn': turn, 'host_jax_mapping': mapping,
                'host_manifests': {str(x['host']): x['manifest_sha256'] for x in ranks}}
            publish(path.with_suffix('.group.json'), group); timings['checkpoint_seconds'] += time.perf_counter() - started
            batches.append({'path': str(path), 'manifest_sha256': identity, 'round': turn})
            report['latest_checkpoint'] = {'path': str(path), 'manifest_sha256': identity, 'group_sha256': checkpoints.sha256(path.with_suffix('.group.json'))}
            print(json.dumps({'kind': 'visual_selfplay_batch', 'host': host, 'round': turn, 'counters': counters}), flush=True)
    report.update(status='passed', host_rank=host, jax_rank=rank, host_jax_mapping=mapping, turn=turn,
        generation_complete=turn == c['rounds'], candidate_sha256=candidate_hash, native_receipt_sha256=checkpoints.sha256(receipt_path),
        counters=counters, data_batches=batches, segment_seconds=time.perf_counter() - segment_started, timings=timings,
        worker_placement=placement, cache_stats=runner.stats, compilation=runner.compilation)
    verify(SOURCE); mh.sync_global_devices('visual-selfplay-generation-complete')


def entry(args):
    c = validate(read_json(args.config)); verify(SOURCE)
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')): raise ValueError('Generation configuration differs')
    os.environ['JAX_PLATFORMS'] = c['platform']; args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'claims_go_strength_improvement': False, 'claims_mfu': False,
        'scope': 'Frozen-model native self-play generation. Expert heads evaluate both players. Improved MCTS policy targets and sampled observed actions are distinct. Capped games have no outcome targets and are excluded from expert training. A partial uncommitted batch can be replayed from its deterministic independent seeds.'}
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
