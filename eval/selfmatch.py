#!/usr/bin/env python3
"""Paired learned-checkpoint matches with an independent pinned KataGo referee."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.scoring import direct_area, score_string
from gozero.snapshots import canonical_json, read_json, verify
from match import summarize
from qualify_katago import kata_cells, sgf_vertex

STOP = threading.Event()


def run(args):
    verify(SOURCE); root = args.artifacts_root.resolve()
    def frozen(path):
        path = (SOURCE / path).resolve()
        if not path.is_relative_to(SOURCE): raise ValueError('Specification must be frozen')
        return path
    spec = frozen(args.spec.resolve()); c = read_json(spec)
    if c['schema_version'] != 1 or c['size'] != 9 or c['komi'] != 7.5:
        raise ValueError('This panel uses the qualified 9x9, komi7.5 profile')
    if not 1 <= len(c['cpu_groups']) <= 8 or not 1 <= len(c['openings']) <= 32:
        raise ValueError('Bounded worker and opening counts required')
    if not 1 <= c['max_game_moves'] <= 1296 or not 1 <= c['game_timeout_seconds'] <= 600:
        raise ValueError('Bounded game limits required')
    used = set(); allowed = os.sched_getaffinity(0)
    for group in c['cpu_groups']:
        if set(group) != {'candidate', 'opponent', 'referee'}: raise ValueError('Three engine affinities required')
        for cpus in group.values():
            if not cpus or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed:
                raise ValueError('Invalid engine affinity')
            cores = {Path(f'/sys/devices/system/cpu/cpu{x}/topology/thread_siblings_list').read_text().strip() for x in cpus}
            if len(cores) != len(cpus) or used & cores: raise ValueError('Panel engines share physical cores')
            used |= cores
    descriptors = {}; descriptor_paths = {}; natives = {}
    for side in ('candidate', 'opponent'):
        path = frozen(c[side]); pinned = read_json(path)
        source = root / '.gozero/snapshots' / pinned['training_snapshot']; verify(source)
        trained = read_json(source / 'resolved_config.json')['actors']
        if trained['size'] != 9 or trained['komi'] != 7.5 or trained.get('scoring') != 'pass_alive_area':
            raise ValueError('Both checkpoints need the qualified scoring and board profile')
        if not 0 <= c[side + '_simulations'] <= 1024 or not 0 <= c[side + '_cpuct'] <= 10:
            raise ValueError('Invalid search budget')
        descriptors[side] = pinned; descriptor_paths[side] = path
        natives[side] = root / '.gozero/native' / pinned['training_snapshot'] / 'receipt.json'
    engine = read_json(SOURCE / 'eval/katago_build.json'); weights = read_json(SOURCE / 'eval/katago_9x9.json')
    binary = root / engine['binary_path']; model = root / weights['path']; referee_config = frozen(c['referee_config'])
    if sha256(binary) != engine['binary_sha256'] or sha256(model) != weights['sha256']:
        raise ValueError('KataGo referee artifact differs')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved_spec.json').write_bytes(canonical_json(c))
    report = {'schema_version': 1, 'kind': 'paired_checkpoint_matches_with_katago_referee', 'status': 'running',
              'snapshot_id': SOURCE.name, 'spec_sha256': sha256(spec), 'models': descriptors,
              'candidate_descriptor_sha256': sha256(descriptor_paths['candidate']),
              'opponent_descriptor_sha256': sha256(descriptor_paths['opponent']),
              'native_receipt_sha256': {side: sha256(p) for side, p in natives.items()},
              'referee_binary_sha256': engine['binary_sha256'], 'referee_weights_sha256': weights['sha256'],
              'referee_config_sha256': sha256(referee_config), 'started_unix': time.time(), 'games': [],
              'claims_strength_against_katago': False, 'claims_algorithm_improvement': False}
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    def worker(index):
        group = c['cpu_groups'][index]; games = []
        def prefix(side): return ['taskset', '-c', ','.join(map(str, group[side]))]
        commands = {}
        for side in ('candidate', 'opponent'):
            commands[side] = prefix(side) + [sys.executable, '-B', str(SOURCE / 'eval/learned_gtp.py'),
                '--candidate', str(descriptor_paths[side]), '--native-receipt', str(natives[side]), '--artifacts-root', str(root),
                '--simulations', str(c[side + '_simulations']), '--cpuct', str(c[side + '_cpuct'])]
        commands['referee'] = prefix('referee') + [str(binary), 'gtp', '-model', str(model), '-config', str(referee_config)]
        for pair in range(index, len(c['openings']), len(c['cpu_groups'])):
            for candidate_color in ('B', 'W'):
                if STOP.is_set(): return games
                opening = c['openings'][pair]
                directory = output / f'pair-{pair:03d}-{candidate_color}'; directory.mkdir()
                game = {'pair': pair, 'candidate_color': candidate_color, 'opening': opening, 'status': 'running',
                        'moves': [], 'checked_positions': 0, 'argv': commands,
                        'engine_seconds': {'candidate': 0., 'opponent': 0.},
                        'search_totals': {side: {key: 0 for key in ('simulations', 'neural_evaluations', 'terminal_evaluations')} for side in ('candidate', 'opponent')}}
                start = time.monotonic(); deadline = start + c['game_timeout_seconds']
                def command(client, text):
                    if STOP.is_set(): raise RuntimeError('Panel stop requested')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: raise TimeoutError('Game deadline expired')
                    return client.command(text, timeout=min(remaining, 60))
                try:
                    with ExitStack() as stack:
                        clients = {side: stack.enter_context(GTPClient(argv, directory / side, environment=env)) for side, argv in commands.items()}
                        for client in clients.values():
                            command(client, 'boardsize 9'); command(client, 'clear_board'); command(client, 'komi 7.5')
                        rules = json.loads(command(clients['referee'], 'kata-get-rules')); game['referee_rules'] = rules
                        if rules['ko'] != 'POSITIONAL' or rules['scoring'] != 'AREA' or not rules['suicide']:
                            raise ValueError('Referee rules differ')
                        passes = 0
                        for ply in range(c['max_game_moves']):
                            color = 'B' if ply % 2 == 0 else 'W'
                            if ply < len(opening):
                                move = opening[ply]
                                if move.lower() in ('pass', 'resign'): raise ValueError('Openings require stone placements')
                                for client in clients.values(): command(client, 'play ' + color + ' ' + move)
                                entry = {'color': color, 'vertex': move, 'opening': True}
                            else:
                                side = 'candidate' if color == candidate_color else 'opponent'
                                before = time.monotonic(); move = command(clients[side], 'genmove ' + color).strip()
                                elapsed = time.monotonic() - before
                                if move.lower() == 'resign': raise ValueError('Resignation is disabled')
                                for other, client in clients.items():
                                    if other != side: command(client, 'play ' + color + ' ' + move)
                                stats = json.loads(command(clients[side], 'gozero-search-stats'))
                                if stats['simulations'] != c[side + '_simulations']:
                                    raise ValueError('Completed search budget differs')
                                entry = {'color': color, 'vertex': move, 'opening': False, 'side': side, 'search': stats, 'engine_seconds': elapsed}
                                game['engine_seconds'][side] += elapsed
                                for key in game['search_totals'][side]: game['search_totals'][side][key] += stats[key]
                            game['moves'].append(entry)
                            board = command(clients['candidate'], 'showboard')
                            if board != command(clients['opponent'], 'showboard') or board != kata_cells(command(clients['referee'], 'showboard'), 9):
                                raise AssertionError('Learned engines and KataGo referee boards differ')
                            game['checked_positions'] += 1
                            passes = passes + 1 if move.lower() == 'pass' else 0
                            if passes == 2:
                                scores = {side: command(client, 'final_score') for side, client in clients.items()}
                                if len(set(scores.values())) != 1: raise AssertionError('Final scores differ')
                                score = scores['candidate']
                                game.update(status='completed', score=score, referee_scores=scores,
                                            independent_raw_score=score_string(direct_area(board, c['komi'])),
                                            score_margin_agrees_with_katago=True,
                                            candidate_points=0.5 if score == '0' else float(score[0] == candidate_color))
                                break
                        else: game['status'] = 'truncated'
                except Exception as error:
                    game.update(status='failed', error=repr(error))
                    # Board, rule, score or budget inconsistencies invalidate
                    # the engine comparison. Preserve the failing game and
                    # cancel peers rather than collecting further outcomes.
                    if isinstance(error, (AssertionError, ValueError)):
                        game['integrity_failure'] = True
                        STOP.set()
                finally:
                    game['elapsed_seconds'] = time.monotonic() - start
                    (directory / 'result.json').write_bytes(canonical_json(game))
                    result = 'RE[' + game['score'] + ']' if game['status'] == 'completed' else ''
                    names = {candidate_color: 'candidate', 'W' if candidate_color == 'B' else 'B': 'opponent'}
                    sgf = '(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RU[Tromp-Taylor]AP[gozero:selfmatch]C[pass-alive area; KataGo is referee only]' + result
                    sgf += 'PB[' + names['B'] + ']PW[' + names['W'] + ']'
                    sgf += ''.join(';' + m['color'] + '[' + sgf_vertex(m['vertex'], 9) + ']' for m in game['moves']) + ')\n'
                    (directory / 'game.sgf').write_text(sgf); games.append(game)
                    print(json.dumps({'kind': 'selfmatch', 'pair': pair, 'candidate_color': candidate_color, 'status': game['status'],
                                      'score': game.get('score'), 'plies': len(game['moves']), 'error': game.get('error')}), flush=True)
        return games
    try:
        with ThreadPoolExecutor(max_workers=len(c['cpu_groups'])) as executor:
            futures = [executor.submit(worker, i) for i in range(len(c['cpu_groups']))]
            for future in as_completed(futures):
                report['games'].extend(future.result())
                (output / 'progress.json').write_bytes(canonical_json(report))
        report['games'].sort(key=lambda g: (g['pair'], g['candidate_color']))
        report['summary'] = summarize(report['games'])
        report['status'] = 'passed' if len(report['games']) == 2 * len(c['openings']) and all(g['status'] == 'completed' for g in report['games']) else 'failed'
        verify(SOURCE)
    except BaseException as error:
        STOP.set(); report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time()
        (output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('games', 'models')}), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    def stop(signum, frame):
        STOP.set(); raise KeyboardInterrupt('selfmatch interrupted by signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True); parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
