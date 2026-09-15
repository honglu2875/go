"""Independent board/score replay of a fixed sample of native generated games."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'eval'))
from gozero.checkpoints import sha256, read
from gozero.gtp import GTPClient
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.scoring import score_string
from gozero.snapshots import canonical_json, read_json, verify
from qualify_katago import kata_cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    attempt = artifact(root, 'runs/' + a.attempt); closed = read_json(attempt / 'result.json')
    if closed['status'] != 'passed': raise ValueError('Only completed generation can be audited')
    source = artifact(root, '.gozero/snapshots/' + closed['snapshot_id']); verify(source)
    c = read_json(source / 'resolved_config.json')
    if c['kind'] != 'visual_selfplay_generation' or c['fixture_pass_after'] is not None or c['candidate'] is None:
        raise ValueError('Expected real trained self-play')
    weights = read_json(SOURCE / 'eval/katago_early/level-0.json'); engine = read_json(SOURCE / 'eval/katago_build.json')
    if weights['role'] != 'evaluation_only': raise ValueError('Referee must remain evaluation-only')
    receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    binary = artifact(root, engine['binary_path']); model = artifact(root, weights['path'])
    if sha256(binary) != engine['binary_sha256'] or sha256(model) != weights['sha256']: raise ValueError('KataGo bytes differ')
    games = []; evidence = {str((attempt / 'result.json').relative_to(root)): sha256(attempt / 'result.json')}
    for host in range(c['expected_processes']):
        for turn in range(1, c['rounds'] + 1):
            path = attempt / f'rank-{host}/artifacts/checkpoints/turn-{turn:09d}'
            group = read_json(path.with_suffix('.group.json'))
            state, arrays, actors = read(path, expected_manifest_sha256=group['host_manifests'][str(host)])
            if state['snapshot_id'] != source.name or state['turn'] != turn or state['host_rank'] != host:
                raise ValueError('Generated batch identity differs')
            for slot, game in enumerate(json.loads(actors)):
                begin, end = map(int, arrays['offsets'][slot:slot + 2])
                if arrays['actions'][begin:end].tolist() != game['actions']: raise ValueError('Recorded actions differ')
                games.append(game)
            evidence[str((path / 'manifest.json').relative_to(root))] = sha256(path / 'manifest.json')
    def order(g): return hashlib.sha256(canonical_json([91312491, g['game_id']])).hexdigest()
    selected = sorted([g for g in games if g['terminal']], key=order)[:32]
    selected += sorted([g for g in games if not g['terminal']], key=order)[:8]
    if len(selected) < 32: raise ValueError('Insufficient complete games for the fixed independent replay sample')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'visual_generated_games_katago_replay', 'status': 'running',
        'operator_snapshot': SOURCE.name, 'generation_snapshot': source.name, 'attempt': a.attempt,
        'selection': 'SHA256([91312491,game_id]) order, first32 terminal and up to8 capped games; no outcome-based selection.',
        'selected_game_ids': [g['game_id'] for g in selected], 'native_receipt_sha256': sha256(receipt_path),
        'katago_binary_sha256': sha256(binary), 'katago_weights_sha256': sha256(model), 'evidence': evidence,
        'scope': 'KataGo executes recorded actions only and independently verifies boards and completed scores. It supplies no training moves, policies or outcome labels. Capped games remain without outcomes.', 'started_unix': time.time()}
    cfg = SOURCE / 'eval/visual_causal/katago_visits1_thread1.cfg'; report['config_sha256'] = sha256(cfg)
    def replay(item):
        index, recorded = item; directory = a.output / f'game-{index:03d}'
        game = native.Game(json.dumps({**c['rules'], 'simulations': 1, 'cpuct': 1., 'max_search_edges': 10000, 'history': 1}))
        argv = ['taskset', '-c', str(112 + index % 4), str(binary), 'gtp', '-model', str(model), '-config', str(cfg)]
        result = {'game_id': recorded['game_id'], 'terminal': recorded['terminal'], 'checked_boards': 0, 'status': 'running'}
        try:
            with GTPClient(argv, directory / 'katago') as kata:
                for command in ('boardsize 9', 'clear_board', 'komi 7.5'): kata.command(command, timeout=30)
                rules = json.loads(kata.command('kata-get-rules', timeout=30))
                if rules['ko'] != 'POSITIONAL' or rules['scoring'] != 'AREA' or not rules['suicide']: raise ValueError('KataGo rules differ')
                for ply, move in enumerate(recorded['actions']):
                    color = 1 + ply % 2; game.play(color, move)
                    vertex = 'pass' if move == 81 else 'ABCDEFGHJ'[move % 9] + str(move // 9 + 1)
                    kata.command('play ' + ('B' if color == 1 else 'W') + ' ' + vertex, timeout=30)
                    _, _, terminal, score, stones = game.state()
                    expected = '\n'.join(''.join('.XO'[int(x)] for x in row) for row in stones.reshape(9, 9)[::-1])
                    if kata_cells(kata.command('showboard', timeout=30), 9) != expected: raise ValueError('Independent board differs')
                    result['checked_boards'] += 1
                if bool(terminal) != recorded['terminal']: raise ValueError('Completion state differs')
                if terminal:
                    actual = kata.command('final_score', timeout=30)
                    if actual != score_string(score) or score != recorded['white_score']: raise ValueError('Independent terminal score differs')
                    result.update(white_score=score, katago_score=actual)
                result['status'] = 'passed'
        except BaseException as error:
            result.update(status='failed', error=repr(error)); raise
        finally:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'result.json').write_bytes(canonical_json(result))
        return result
    try:
        with ThreadPoolExecutor(max_workers=4) as pool: report['games'] = list(pool.map(replay, enumerate(selected)))
        report.update(status='passed', checked_boards=sum(g['checked_boards'] for g in report['games']),
                      checked_scores=sum(g['terminal'] for g in report['games']))
        verify(SOURCE)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (a.output / 'result.json').write_bytes(canonical_json(report))
    print(json.dumps({k: v for k, v in report.items() if k not in ('games', 'evidence', 'selected_game_ids')}))


if __name__ == '__main__': main()
