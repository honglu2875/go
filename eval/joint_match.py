#!/usr/bin/env python3
"""Paired-color KataGo matches for the trained V7 policy/value RPC adapter."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.joint_artifacts import validate_descriptor
from gozero.snapshots import canonical_json, read_json, verify
from joint_gtp import contract
from match import summarize
from qualify_katago import kata_cells, sgf_vertex


def within(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Artifact is outside its declared root or missing')
    return path


def run(args):
    verify(SOURCE)
    spec = within(SOURCE, str(args.spec))
    c = read_json(spec)
    if c['kind'] != 'joint_v7_paired_katago_matches' or c['purpose'] not in ('qualification', 'benchmark'):
        raise ValueError('Invalid joint match purpose')
    root = args.artifacts_root.resolve()
    candidate = within(SOURCE, c['candidate'])
    descriptor = validate_descriptor(read_json(candidate))
    if c['purpose'] == 'benchmark' and descriptor['training_purpose'] != 'learning':
        raise ValueError('Execution-fixture weights cannot enter a scientific benchmark')
    search_path = within(SOURCE, c['search'])
    search = contract(read_json(search_path))
    size, komi = search['size'], search['komi']
    if (descriptor['input_contract']['size'] != size or descriptor['input_contract']['komi'] != komi
            or search['cache_positions'] > descriptor['max_positions']):
        raise ValueError('Candidate and match inputs differ')
    native = within(root, c['native_receipt'])
    if sha256(native) != search['native_receipt_sha256']:
        raise ValueError('Native receipt changed')
    engine = read_json(within(SOURCE, c['katago_build']))
    weights = read_json(within(SOURCE, c['katago_weights']))
    if weights['role'] != 'evaluation_only' or size not in weights['board_sizes']:
        raise ValueError('Wrong external model role or board size')
    binary, model = within(root, engine['binary_path']), within(root, weights['path'])
    for path, expected in ((binary, engine['binary_sha256']), (model, weights['sha256'])):
        if sha256(path) != expected:
            raise ValueError('KataGo artifact changed')
    allowed = set(os.sched_getaffinity(0))
    for key in ('candidate_cpus', 'katago_cpus'):
        if not c[key] or len(set(c[key])) != len(c[key]) or not set(c[key]) <= allowed:
            raise ValueError('Invalid engine CPU placement')
    if set(c['candidate_cpus']) & set(c['katago_cpus']):
        raise ValueError('Engine CPU placements overlap')
    base_path = within(SOURCE, c['katago_config'])
    base = base_path.read_text()
    visits = re.search(r'^maxVisits\s*=\s*(\d+)\s*$', base, re.M)
    if not visits or int(visits[1]) != c['katago_max_visits'] or not c['openings']:
        raise ValueError('Visit count or openings differ')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', c['seed_prefix']):
        raise ValueError('Invalid external search seed')
    if not 0 < c['game_timeout_seconds'] <= 3600:
        raise ValueError('Invalid per-game deadline')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind=c['kind'], purpose=c['purpose'], status='running',
                  snapshot=SOURCE.name, spec_sha256=sha256(spec), candidate_sha256=sha256(candidate),
                  parameter_manifest_sha256=descriptor['parameters']['manifest_sha256'],
                  search_sha256=sha256(search_path), native_receipt_sha256=sha256(native),
                  katago_binary_sha256=engine['binary_sha256'], katago_weights_sha256=weights['sha256'],
                  katago_model=weights['model'], katago_config_sha256=sha256(base_path),
                  candidate_simulations_excluding_root=search['simulations'],
                  katago_max_visits=c['katago_max_visits'], started_unix=time.time(), games=[])
    try:
        for pair, opening in enumerate(c['openings']):
            for candidate_color in ('B', 'W'):
                directory = args.output / f'pair-{pair:03d}-{candidate_color}'
                directory.mkdir()
                seed = c['seed_prefix'] + '-' + str(pair)
                cfg = re.sub(r'^(searchRandSeed|nnRandSeed)\s*=.*$',
                             lambda m: m[1] + ' = ' + seed, base, flags=re.M)
                config = directory / 'katago.cfg'
                config.write_text(cfg)
                ours = ['taskset', '-c', ','.join(map(str, c['candidate_cpus'])), sys.executable, '-B',
                        str(SOURCE / 'eval/joint_gtp.py'), '--candidate', str(candidate),
                        '--inference-config', str(search_path), '--native-receipt', str(native),
                        '--socket', str(args.socket)]
                theirs = ['taskset', '-c', ','.join(map(str, c['katago_cpus'])), str(binary), 'gtp',
                          '-model', str(model), '-config', str(config)]
                game = dict(pair=pair, candidate_color=candidate_color, opening=opening, status='running',
                            moves=[], checked_positions=0, engine_seconds=dict(candidate=0., katago=0.),
                            candidate_search_totals=dict(simulations=0, neural_evaluations=0, terminal_evaluations=0),
                            katago_config_sha256=sha256(config))
                start = time.monotonic()
                deadline = start + c['game_timeout_seconds']

                def command(client, text):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Joint match deadline expired')
                    return client.command(text, timeout=min(remaining, 120))

                try:
                    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                    with GTPClient(ours, directory / 'candidate', environment=env) as learned, GTPClient(theirs, directory / 'katago') as kata:
                        game['candidate_version'] = command(learned, 'version')
                        game['katago_version'] = command(kata, 'version')
                        for client in (learned, kata):
                            command(client, 'boardsize ' + str(size))
                            command(client, 'komi ' + str(komi))
                            command(client, 'clear_board')
                        rules = json.loads(command(kata, 'kata-get-rules'))
                        game['katago_rules'] = rules
                        if rules['ko'] != 'POSITIONAL' or rules['scoring'] != 'AREA' or not rules['suicide']:
                            raise ValueError('KataGo rules differ')

                        def check_board():
                            if command(learned, 'showboard') != kata_cells(command(kata, 'showboard'), size):
                                raise ValueError('Rust and KataGo boards differ')
                            game['checked_positions'] += 1

                        # This explicit protocol fixture is separate from game outcomes.
                        # It guarantees terminal scoring/reset coverage even if free play caps.
                        for color in ('B', 'W'):
                            for client in (learned, kata):
                                command(client, 'play ' + color + ' pass')
                            check_board()
                        left, right = command(learned, 'final_score'), command(kata, 'final_score')
                        if left != right:
                            raise ValueError('Empty-board two-pass score differs')
                        game['terminal_fixture'] = dict(score=left, katago_score=right, passes=2)
                        for client in (learned, kata):
                            command(client, 'clear_board')
                        check_board()
                        passes = 0
                        for ply in range(search['max_game_moves']):
                            color = 'B' if ply % 2 == 0 else 'W'
                            if ply < len(opening):
                                move = opening[ply]
                                if move.lower() in ('pass', 'resign'):
                                    raise ValueError('Openings require stone placements')
                                for client in (learned, kata):
                                    command(client, 'play ' + color + ' ' + move)
                                entry = dict(color=color, vertex=move, opening=True)
                            else:
                                our_turn = color == candidate_color
                                player, opponent = (learned, kata) if our_turn else (kata, learned)
                                before = time.monotonic()
                                move = command(player, 'genmove ' + color).strip()
                                elapsed = time.monotonic() - before
                                if move.lower() == 'resign':
                                    raise ValueError('Unexpected resignation')
                                command(opponent, 'play ' + color + ' ' + move)
                                game['engine_seconds']['candidate' if our_turn else 'katago'] += elapsed
                                entry = dict(color=color, vertex=move, opening=False, engine_seconds=elapsed)
                                if our_turn:
                                    stats = json.loads(command(learned, 'gozero-search-stats'))
                                    if stats['simulations'] != search['simulations']:
                                        raise ValueError('Candidate search work differs')
                                    entry['search'] = stats
                                    for key in game['candidate_search_totals']:
                                        game['candidate_search_totals'][key] += stats[key]
                            game['moves'].append(entry)
                            check_board()
                            passes = passes + 1 if move.lower() == 'pass' else 0
                            if passes == 2:
                                score, other = command(learned, 'final_score'), command(kata, 'final_score')
                                if score != other:
                                    raise ValueError('Native and KataGo pass-alive area scores differ')
                                game.update(status='completed', score=score, katago_score=other,
                                            candidate_points=.5 if score == '0' else float(score[0] == candidate_color))
                                break
                        else:
                            game['status'] = 'truncated'
                except Exception as error:
                    game.update(status='failed', error=repr(error))
                finally:
                    game['elapsed_seconds'] = time.monotonic() - start
                    (directory / 'result.json').write_bytes(canonical_json(game))
                    result = 'RE[' + game['score'] + ']' if game['status'] == 'completed' else ''
                    record = f'(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi}]RU[Tromp-Taylor]AP[gozero:joint-eval]'
                    record += result + 'PB[' + ('gozero' if candidate_color == 'B' else 'KataGo') + ']PW[' + ('gozero' if candidate_color == 'W' else 'KataGo') + ']'
                    record += ''.join(';' + m['color'] + '[' + sgf_vertex(m['vertex'], size) + ']' for m in game['moves']) + ')\n'
                    (directory / 'game.sgf').write_text(record)
                    report['games'].append(game)
                    print(json.dumps(dict(pair=pair, candidate_color=candidate_color, status=game['status'],
                                          plies=len(game['moves']), score=game.get('score'), error=game.get('error'))), flush=True)
        report['summary'] = summarize(report['games'])
        report['summary']['uncertainty_scope'] = ('Conservative bounded-score interval for independent completed opening pairs, conditional on those openings. '
                                                  'Incomplete pairs leave a strength panel incomplete; execution integrity is reported separately. No Elo estimate.')
        # Caps remain unresolved outcomes, not failures of transport and not draws.
        report['status'] = 'passed' if not report['summary']['failed_games'] else 'failed'
        report['complete_panel'] = report['summary']['completed_games'] == report['summary']['scheduled_games']
        report['scope'] = ('Pinned engine interoperability with paired colors; caps are unresolved. '
                           'Qualification-purpose small weights and the two-pass protocol fixture do not support strength claims.')
        verify(SOURCE)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        path = args.output / 'result.json'
        path.write_bytes(canonical_json(report))
        path.chmod(0o444)
        print(json.dumps(dict(status=report['status'], summary=report.get('summary'), sha256=sha256(path))), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt('Match interrupted by signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'artifacts-root', 'output', 'socket'):
        p.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
