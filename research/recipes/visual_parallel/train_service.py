"""Qualified local TPU cache owners and bounded native-search evaluation jobs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'eval'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_artifacts import validate as candidate_validate
from gozero.visual_history import observation_sha256
from gozero.visual_rpc import Server
from visual_gtp import contract, identity


def validate(c):
    keys = ('schema_version kind platform expected_processes expected_devices mode seed candidate '
            'model inference native_receipt slots max_block gather_seconds seconds exit_depth qualification matches_by_host')
    if (set(c) != set(keys.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_inference_service'
            or c['platform'] not in ('cpu', 'tpu') or c['mode'] not in ('qualification', 'matches')):
        raise ValueError('Unknown inference service contract')
    for key in ('expected_processes', 'expected_devices', 'slots', 'max_block', 'seconds'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Invalid ' + key)
    if (not 0 <= c['gather_seconds'] <= .1 or c['seconds'] > 3600
            or (c['candidate'] is None) == (c['model'] is None)
            or (c['mode'] == 'matches' and (c['candidate'] is None or c['qualification'] is not None))
            or (c['mode'] == 'qualification' and (c['qualification'] is None or c['matches_by_host'] is not None))):
        raise ValueError('Ambiguous service initialization or bounded workload')
    return c


def qualify(runner, native, search, q, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P
    import model
    import observations
    fields = {'plies', 'tolerance', 'full_bucket'}
    if (set(q) != fields or not 1 <= q['plies'] <= search['max_game_moves']
            or not 0 < q['tolerance'] <= .2 or not q['plies'] + search['simulations'] < q['full_bucket'] <= runner.capacity):
        raise ValueError('Invalid exact search qualification bounds')
    config = {k: search[k] for k in ('size', 'komi', 'scoring', 'simulations', 'cpuct', 'fpu_reduction', 'gumbel', 'max_search_edges')}
    games = [native.Game(json.dumps({**config, 'history': 1})) for _ in range(runner.slots)]
    exact_checks = 0; max_error = 0.; completed = 0; roots = []
    policy_comparisons = {role: {'greedy_differences': 0, 'total_variation_sum': 0., 'maximum_total_variation': 0.}
                          for role in ('expert', 'behavior')}
    max_value_error = 0.
    def full(p, obs, actions, counts):
        return model.forward(p, obs, actions, counts, runner.c, exit_depth=runner.depth)
    full_fn = jax.jit(jax.shard_map(full, mesh=runner.mesh,
        in_specs=(P(), P('data'), P('data'), P('data')), out_specs=P('data'), check_vma=False))
    for turn in range(q['plies']):
        pending = {}
        for slot, game in enumerate(games):
            if not game.state()[2]:
                request, features = game.start(runner.version)
                pending[slot] = (request, features)
        while pending:
            histories = {slot: games[slot].request_history(request).tolist() for slot, (request, _) in pending.items()}
            exact = {slot: observations.from_native(features.reshape(runner.size, runner.size, 6))
                     for slot, (_, features) in pending.items()}
            prediction = runner.score({slot: (histories[slot], observation_sha256(exact[slot])) for slot in pending})
            # Verify every pending MCTS leaf, including siblings and retained
            # prefixes, against a full fixed-bucket causal forward pass.
            replayed = dict(zip(histories, runner.replay(list(histories.values()))))
            obs = np.zeros((runner.slots, q['full_bucket'], runner.size, runner.size, 6), np.float32)
            act = np.zeros((runner.slots, q['full_bucket']), np.int32); counts = np.zeros(runner.slots, np.int32)
            for slot, tape in histories.items():
                n = len(tape) + 1; counts[slot] = n; obs[slot, :n] = replayed[slot]
                act[slot, :len(tape)] = tape
                np.testing.assert_array_equal(replayed[slot][-1], exact[slot])
            target = jax.device_get(full_fn(runner.params, *map(runner.put, (obs, act, counts))))
            next_pending = {}
            for slot, (request, _) in pending.items():
                for key in prediction[slot]:
                    x, y = prediction[slot][key], target[key][slot, counts[slot] - 1]
                    max_error = max(max_error, float(np.max(np.abs(x - y))))
                    np.testing.assert_allclose(x, y, atol=q['tolerance'], rtol=q['tolerance'])
                legal = np.concatenate([exact[slot][..., 5].reshape(-1) > .5, [True]])
                for role, comparison in policy_comparisons.items():
                    x, y = prediction[slot][role + '_logits'], target[role + '_logits'][slot, counts[slot] - 1]
                    def probability(logits):
                        logits = np.where(legal, logits.astype(np.float64), -1e30)
                        weights = np.exp(logits - logits.max()); return weights / weights.sum()
                    px, py = probability(x), probability(y); tv = float(np.abs(px - py).sum() / 2)
                    comparison['greedy_differences'] += int(np.argmax(px) != np.argmax(py))
                    comparison['total_variation_sum'] += tv
                    comparison['maximum_total_variation'] = max(comparison['maximum_total_variation'], tv)
                max_value_error = max(max_value_error, float(abs(prediction[slot]['value'] - target['value'][slot, counts[slot] - 1])))
                exact_checks += 1
                request, features = games[slot].evaluate(request, np.ascontiguousarray(prediction[slot]['expert_logits'], np.float32),
                                                       float(prediction[slot]['value']))
                if request is not None:
                    next_pending[slot] = (request, features)
                else:
                    chosen, policy, value, simulations, neural, terminal = games[slot].finish()
                    games[slot].play(1 + turn % 2, chosen)
                    roots.append({'slot': slot, 'turn': turn, 'action': chosen, 'simulations': simulations, 'neural': neural})
                    completed += int(games[slot].state()[2])
            pending = next_pending
    report['qualification'] = {'exact_leaf_predictions': exact_checks, 'maximum_absolute_error': max_error,
        'legal_policy_comparisons': policy_comparisons, 'maximum_value_error': max_value_error,
        'terminal_games': completed, 'root_actions': roots, 'all_heads_compared': True,
        'scope': 'Exact native MCTS leaves compared with whole-history transformer outputs, including cache rewinds.'}


def probe_gtp(runner, native, search, candidate_path, infer_path, native_path, output):
    """Run real GTP subprocesses through the queue and audit their boards."""
    from concurrent.futures import ThreadPoolExecutor
    from gozero.gtp import GTPClient
    from learned_gtp import action
    path = Path('/tmp') / ('gozero-' + hashlib.sha256(str(output).encode()).hexdigest()[:24] + '.sock')
    server = Server(path, identity(SOURCE, candidate_path, infer_path), runner, gather_seconds=.001)
    def game_probe(index):
        argv = [sys.executable, '-B', str(SOURCE / 'eval/visual_gtp.py'), '--candidate', str(candidate_path),
            '--inference-config', str(infer_path), '--native-receipt', str(native_path), '--socket', str(path)]
        config = {k: search[k] for k in ('size', 'komi', 'scoring', 'simulations', 'cpuct', 'fpu_reduction', 'gumbel', 'max_search_edges')}
        game = native.Game(json.dumps({**config, 'history': 1})); checked = 0; moves = []
        with GTPClient(argv, output / f'gtp-probe-{index}', environment={**os.environ, 'JAX_PLATFORMS': 'cpu',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}) as client:
            client.command('boardsize ' + str(search['size'])); client.command('komi ' + str(search['komi']))
            client.command('clear_board')
            for turn in range(min(4, search['max_game_moves'])):
                color = 'B' if turn % 2 == 0 else 'W'
                move = client.command('genmove ' + color, timeout=60).strip()
                game.play(1 + turn % 2, action(move, search['size'])); moves.append(move)
                expected = '\n'.join(''.join('.XO'[int(s)] for s in row) for row in game.state()[4].reshape(search['size'], search['size']))
                if client.command('showboard') != expected:
                    raise ValueError('RPC GTP board differs from independent native play')
                checked += 1
                stats = json.loads(client.command('gozero-search-stats'))
                if stats['simulations'] != search['simulations']:
                    raise ValueError('RPC GTP search budget changed')
                if game.state()[2]:
                    break
        return {'client': index, 'boards_checked': checked, 'moves': moves}
    try:
        with ThreadPoolExecutor(min(2, runner.slots)) as pool:
            futures = [pool.submit(game_probe, i) for i in range(min(2, runner.slots))]
            while not all(f.done() for f in futures):
                server.pump()
            results = [f.result() for f in futures]
        return {'clients': results, 'batches': server.records}
    finally:
        server.close()


def run(args, c, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    import inference
    import model
    if (jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Inference topology differs from registration')
    root = SOURCE.parents[2]; host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    infer_path = artifact(SOURCE, c['inference']); search = contract(read_json(infer_path))
    native_path = artifact(root, c['native_receipt'])
    if sha256(native_path) != search['native_receipt_sha256']:
        raise ValueError('Service native receipt differs')
    receipt = read_json(native_path); verify(artifact(root, '.gozero/snapshots/' + receipt['snapshot_id']))
    native = load_library(native_path.parent / receipt['filename'], receipt['binary_sha256'])
    candidate_path = None
    if c['candidate'] is not None:
        candidate_path = artifact(SOURCE, c['candidate']); descriptor = read_json(candidate_path)
        checked = candidate_validate(root, descriptor, allow_partial=c['mode'] == 'qualification')
        if checked['model_code_sha256'] != sha256(Path(__file__).with_name('model.py')):
            raise ValueError('Inference decoder implementation differs from trained source')
        net = checked['config']['model']; rules = checked['rules']; version = descriptor['network_version']
        if any(rules[k] != search[k] for k in rules):
            raise ValueError('Inference rules differ from complete training histories')
        abstract = jax.eval_shape(lambda: model.initialize(0, net)); definition = jax.tree.structure(abstract)
        if model.parameter_schema(net) != checked['model_schema']:
            raise ValueError('Inference parameter tree differs from checkpoint')
        params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(checked['model_schema']))])
        report['candidate_sha256'] = sha256(candidate_path); report['training_complete'] = descriptor['training_complete']
    else:
        net = c['model']; version = 17
        rules = {k: search[k] for k in ('size', 'komi', 'scoring')}
        params = jax.jit(lambda: model.initialize(c['seed'], net))()
        report['training_complete'] = False
    runner = inference.Runner(params, net, native, rules, slots=c['slots'], cache_positions=search['cache_positions'],
        network_version=version, exit_depth=c['exit_depth'], max_block=c['max_block'])
    report.update(host_rank=host, jax_rank=jax.process_index(), devices=[str(d) for d in jax.local_devices()],
                  parameter_count=sum(s['elements'] for s in model.parameter_schema(net)),
                  slots=c['slots'], slots_per_device=c['slots'] // len(jax.local_devices()),
                  inference_sha256=sha256(infer_path), native_receipt_sha256=sha256(native_path))
    if c['mode'] == 'qualification':
        qualify(runner, native, search, c['qualification'], report)
        if candidate_path is not None:
            report['gtp_probe'] = probe_gtp(runner, native, search, candidate_path, infer_path, native_path, args.output)
    else:
        path = Path('/tmp') / ('gozero-' + hashlib.sha256(str(args.output).encode()).hexdigest()[:24] + '.sock')
        server = Server(path, identity(SOURCE, candidate_path, infer_path), runner, gather_seconds=c['gather_seconds'])
        jobs = []; started = time.monotonic()
        try:
            specs = c['matches_by_host'][str(host)]
            if not specs or len(specs) > c['slots']:
                raise ValueError('Match concurrency exceeds fixed cache slots')
            for i, specification in enumerate(specs):
                spec_path = artifact(SOURCE, specification); spec = read_json(spec_path)
                if spec['candidate'] != c['candidate'] or spec['visual_inference'] != c['inference']:
                    raise ValueError('Match and inference owner identity differ')
                cpus = sorted(set(spec['candidate_cpus'] + spec['katago_cpus']))
                output = args.output / f'match-{i:03d}'
                log = (args.output / f'match-{i:03d}.log').open('xb')
                argv = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-B', str(SOURCE / 'eval/match.py'),
                        '--spec', str(spec_path), '--artifacts-root', str(root), '--output', str(output), '--visual-socket', str(path)]
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
                jobs.append((process, log, output, specification))
            while any(p.poll() is None for p, _, _, _ in jobs):
                if time.monotonic() - started > c['seconds']:
                    raise TimeoutError('Registered inference match window expired')
                server.pump()
            report['matches'] = []
            for process, _, output, spec in jobs:
                result = read_json(output / 'result.json')
                report['matches'].append({'spec': spec, 'exit_code': process.returncode,
                    'result_sha256': sha256(output / 'result.json'), 'summary': result['summary']})
            if any(process.returncode != 0 for process, _, _, _ in jobs):
                raise RuntimeError('One or more real KataGo match jobs failed; all game artifacts retained')
        finally:
            for process, log, _, _ in jobs:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                log.close()
            server.close()
            (args.output / 'inference_batches.json').write_bytes(canonical_json(server.records))
            report['match_seconds'] = time.monotonic() - started
            report['cache_stats'] = runner.stats; report['compilation'] = runner.compilation
    report['cache_stats'] = runner.stats; report['compilation'] = runner.compilation
    report['device_memory_stats'] = [d.memory_stats() for d in jax.local_devices()]
    report['status'] = 'passed'; verify(SOURCE)
    mh.sync_global_devices('visual-inference-complete')


def entry(args):
    c = validate(read_json(args.config)); verify(SOURCE)
    if (args.resume is not None or args.stop_after_turn is not None
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))):
        raise ValueError('Inference has no training continuation or mutable configuration')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'mode': c['mode'], 'snapshot_id': SOURCE.name,
              'status': 'running', 'claims_go_strength_improvement': False, 'claims_mfu': False}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        (args.output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({'status': report['status'], 'mode': c['mode'], 'error': report.get('error')}), flush=True)
        if distributed:
            jax.distributed.shutdown()
