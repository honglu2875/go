#!/usr/bin/env python3
"""Compare parent/fork GTP search on every position of fixed external game traces."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec)
    require(spec['snapshot'] == SOURCE.name and spec['maximum_attempts'] == 1, 'Wrong frozen registration')
    for item in spec['inputs']:
        require(sha256(root / item['path']) == item['sha256'], 'Fixed input changed')
    output = root / spec['output']; output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'recorded_parent_fork_gtp_qualification', 'status': 'running',
              'snapshot': SOURCE.name, 'spec_sha256': sha256(a.spec), 'started_unix': time.time(), 'games': [],
              'scope': 'Fixed recorded opponent moves from all eight prior real-KataGo games. Exact candidate moves/search, boards and scores; no new competitive outcomes or strength claim.'}
    candidates = {name: read_json(SOURCE / path) for name, path in spec['candidates'].items()}
    identities = {name: validate_candidate(root, candidate) for name, candidate in candidates.items()}
    require(candidates['reference']['model_export_sha256'] == candidates['fork']['model_export_sha256'], 'Parameters differ')
    native = root / spec['native_receipt']; receipt = read_json(native)
    require(sha256(native) == spec['native_receipt_sha256'] and identities['fork']['native_receipt'] == native,
            'Fork native dependency differs')
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        for trace_name in ('reference', 'fork'):
            directory = root / spec['trace_directory'] / trace_name; match = read_json(directory / 'result.json')
            require(match['status'] == 'passed' and len(match['games']) == 4, 'Unexpected recorded games')
            for game in match['games']:
                label = f"{trace_name}-{game['pair']:03d}-{game['candidate_color']}"
                recorded = directory / f"pair-{game['pair']:03d}-{game['candidate_color']}"
                commands = {}; boards = []
                for line in (recorded / 'candidate/gtp.jsonl').read_text().splitlines():
                    row = json.loads(line)
                    if row['kind'] == 'command': commands[row['id']] = row['text']
                    if row['kind'] == 'response' and commands[row['id']] == 'showboard':
                        require(row['success'], 'Original board request failed'); boards.append(row['text'])
                require(len(boards) == len(game['moves']) + 1, 'Recorded board coverage differs')
                deadline = time.monotonic() + spec['timeout_seconds_per_game']
                with ExitStack() as stack:
                    clients = {}
                    for name in ('reference', 'fork'):
                        command = ['taskset', '-c', ','.join(map(str, spec['cpus'][name])), sys.executable, '-B',
                                   str(SOURCE / 'eval/learned_gtp.py'), '--candidate', str(SOURCE / spec['candidates'][name]),
                                   '--native-receipt', str(native), '--artifacts-root', str(root), '--simulations', '16', '--cpuct', '0']
                        clients[name] = stack.enter_context(GTPClient(command, output / label / name, environment=env))

                    def call(client, command):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0: raise TimeoutError('Recorded trace qualification timed out')
                        return client.command(command, timeout=min(60, remaining))

                    for client in clients.values():
                        call(client, 'boardsize 9'); call(client, 'clear_board'); call(client, 'komi 7.5')
                    searched = 0
                    for index, move in enumerate(game['moves']):
                        for client in clients.values():
                            if move['opening'] or move['color'] != game['candidate_color']:
                                call(client, 'play ' + move['color'] + ' ' + move['vertex'])
                            else:
                                require(call(client, 'genmove ' + move['color']) == move['vertex'], 'Candidate move differs on an identical history')
                                stats = json.loads(call(client, 'gozero-search-stats'))
                                require({k: v for k, v in stats.items() if k != 'seconds'} ==
                                        {k: v for k, v in move['search'].items() if k != 'seconds'}, 'Search result differs on an identical history')
                            require(call(client, 'showboard') == boards[index], 'Board differs from recorded candidate/KataGo position')
                        searched += int(not move['opening'] and move['color'] == game['candidate_color'])
                    for client in clients.values():
                        require(call(client, 'final_score') == game['score'] == game['katago_adjudicated_score'], 'Terminal score differs')
                # Verify the launched model/native identities in the raw stderr.
                for name, candidate in candidates.items():
                    ready = [json.loads(line) for line in (output / label / name / 'stderr.log').read_text().splitlines()
                             if line.startswith('{') and json.loads(line).get('kind') == 'engine_ready']
                    require(len(ready) == 1 and ready[0]['candidate_sha256'] == sha256(SOURCE / spec['candidates'][name])
                            and ready[0]['weights_sha256'] == candidate['model_export_sha256']
                            and ready[0]['training_snapshot'] == candidate['training_snapshot']
                            and ready[0]['native_sha256'] == receipt['binary_sha256'], 'Loaded engine identity differs')
                report['games'].append({'id': label, 'boards_exact_per_adapter': len(game['moves']),
                                        'searches_exact_per_adapter': searched, 'terminal_score_exact': game['score']})
                print(json.dumps(report['games'][-1]), flush=True)
        report.update(status='passed', recorded_games=8, board_comparisons=sum(2 * g['boards_exact_per_adapter'] for g in report['games']),
                      search_comparisons=sum(2 * g['searches_exact_per_adapter'] for g in report['games']), terminal_score_comparisons=16)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time()
        with (output / 'result.json').open('xb') as stream:
            stream.write(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
