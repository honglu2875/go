#!/usr/bin/env python3
"""Qualify a pinned KataGo network/backend with independent exact native boards."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.scoring import score_string
from gozero.snapshots import canonical_json, read_json, verify
from learned_gtp import action
from qualify_katago import kata_cells, sgf_vertex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True); a = p.parse_args(); verify(SOURCE)
    if not a.spec.resolve().is_relative_to(SOURCE):
        raise ValueError('Anchor qualification must be frozen')
    c = read_json(a.spec); root = a.workspace_root.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'visual_current_katago_qualification', 'status': 'running',
              'operator_snapshot': SOURCE.name, 'spec_sha256': sha256(a.spec), 'started_unix': time.time(),
              'scope': 'Official network self-play, independent native rule/board/score agreement; not candidate strength.', 'games': []}
    try:
        build = read_json(SOURCE / 'eval/katago_build.json'); weights = read_json(artifact(SOURCE, c['weights']))
        binary, network = artifact(root, build['binary_path']), artifact(root, weights['path'])
        if sha256(binary) != build['binary_sha256'] or sha256(network) != weights['sha256']:
            raise ValueError('KataGo artifact bytes differ')
        receipt_path = artifact(root, c['native_receipt'])
        if sha256(receipt_path) != c['native_receipt_sha256']:
            raise ValueError('Native reference identity differs')
        receipt = read_json(receipt_path); native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
        config = artifact(SOURCE, c['config']); rules = c['rules']; size = rules['size']
        report.update(weights_sha256=weights['sha256'], engine_sha256=build['binary_sha256'],
                      config_sha256=sha256(config), native_receipt_sha256=sha256(receipt_path))
        for index, opening in enumerate(c['openings']):
            game = native.Game(json.dumps({**rules, 'history': 1, 'simulations': 0, 'cpuct': 1., 'max_search_edges': size**2 + 1}))
            row = {'opening': opening, 'moves': [], 'status': 'running', 'checked_positions': 0, 'capture_moves': 0}
            report['games'].append(row); deadline = time.monotonic() + c['seconds_per_game']
            argv = [str(binary), 'gtp', '-model', str(network), '-config', str(config)]
            with GTPClient(argv, a.output / f'game-{index:02d}', environment={**os.environ, 'OMP_NUM_THREADS': '1'}) as client:
                def command(text):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('KataGo anchor qualification deadline expired')
                    return client.command(text, timeout=min(60., remaining))
                command('boardsize ' + str(size)); command('komi ' + str(rules['komi'])); command('clear_board')
                actual_rules = json.loads(command('kata-get-rules'))
                if actual_rules['ko'] != 'POSITIONAL' or actual_rules['scoring'] != 'AREA' or not actual_rules['suicide']:
                    raise ValueError('KataGo and native rules differ')
                for turn in range(c['max_game_moves']):
                    color = 'B' if turn % 2 == 0 else 'W'; before = time.perf_counter()
                    move = opening[turn] if turn < len(opening) else command('genmove ' + color).strip()
                    if turn < len(opening):
                        command('play ' + color + ' ' + move)
                    if move.lower() == 'resign':
                        raise ValueError('Unexpected resignation')
                    old_stones = int((game.state()[4] != 0).sum()); game.play(1 + turn % 2, action(move, size))
                    state = game.state(); expected = '\n'.join(''.join('.XO'[int(s)] for s in line) for line in state[4].reshape(size, size))
                    if kata_cells(command('showboard'), size) != expected:
                        raise ValueError('Exact board disagreement in current KataGo network self-play')
                    row['moves'].append({'color': color, 'vertex': move, 'seconds': time.perf_counter() - before})
                    row['checked_positions'] += 1
                    row['capture_moves'] += int(move.lower() != 'pass' and int((state[4] != 0).sum()) <= old_stones)
                    if state[2]:
                        score = score_string(state[3]); external = command('final_score')
                        if score != external:
                            raise ValueError('Exact terminal score differs')
                        row.update(status='completed', score=score); break
                else:
                    row['status'] = 'truncated'; raise ValueError('Current network qualification reached game cap')
                sgf = f'(;GM[1]FF[4]SZ[{size}]KM[{rules["komi"]}]RU[Tromp-Taylor]RE[{row["score"]}]'
                sgf += ''.join(';' + m['color'] + '[' + sgf_vertex(m['vertex'], size) + ']' for m in row['moves']) + ')\n'
                (a.output / f'game-{index:02d}.sgf').write_text(sgf)
            print(json.dumps({'kind': 'current_katago_game', 'index': index, 'status': row['status'], 'score': row.get('score'),
                              'boards': row['checked_positions']}), flush=True)
        report['status'] = 'passed'; verify(SOURCE)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (a.output / 'result.json').write_bytes(canonical_json(report))


if __name__ == '__main__':
    main()
