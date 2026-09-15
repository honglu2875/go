"""Resident teacher/opponent pair with bounded concurrent games and exact histories."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import numpy as np

from ..go import Game, GameConfig
from ..runtime import pin
from ..storage import GIB, Limits, StorageBudget, StoragePressure, _process_identity
from .corpus import atomic_json, identity, publish_game
from .katago import AnalysisClient, query, model_path
from .label import parse_label


def play_game(teacher, opponent, contract, game_id, expert_color):
    size = contract['board_size']
    komi = contract['komi']
    game = Game(GameConfig(size=size, komi=komi))
    rng = np.random.default_rng(int(game_id[:16], 16))
    actions, rows = [], []
    for ply in range(contract['max_moves']):
        state = game.state()
        teacher_turn = state.to_play == expert_color
        teacher_visits = contract['visits'] if teacher_turn else 1
        teacher_future = teacher.submit(query(actions, visits=teacher_visits, size=size, komi=komi))
        behavior_future = teacher_future if teacher_turn else opponent.submit(query(actions, visits=contract['visits'], size=size, komi=komi))
        legal_actions = game.legal()
        label = parse_label(teacher_future.result(timeout=600), legal_actions, state.to_play, size=size)
        behavior = label if teacher_turn else parse_label(behavior_future.result(timeout=600), legal_actions, state.to_play, size=size)
        mask = np.zeros(size * size + 1, dtype=bool)
        mask[legal_actions] = True
        if ply < contract['opening_moves']:
            distribution = (0.75 * behavior['search_policy'] + 0.25 * behavior['raw_policy']
                            if behavior['search_policy_valid'] else behavior['raw_policy'])
            distribution = distribution.astype(np.float64)
            distribution /= distribution.sum()
            action = int(rng.choice(size * size + 1, p=distribution))
        else:
            action = behavior['best_action']
        rows.append(dict(stones=state.stones.copy(), legal=mask,
                         raw_policy=label['raw_policy'], raw_value=label['raw_value'],
                         search_policy=label['search_policy'], search_value=label['search_value'],
                         search_policy_valid=teacher_turn and label['search_policy_valid'],
                         root_edge_visits=label['root_edge_visits'],
                         raw_score=label['raw_score'], raw_score_valid=label['raw_score_valid'],
                         teacher_visits=np.int32(teacher_visits)))
        game.play(state.to_play, action)
        actions.append(action)
        if game.state().terminal:
            break
    state = game.state()
    return rows, actions, dict(terminal=state.terminal, white_score=state.white_score)


def run_worker(config_path: Path, worker_index: int, *, batches: int | None = None):
    config = json.loads(config_path.read_text())
    worker = config['workers'][worker_index]
    cpus = worker['cpus']
    pin(cpus)
    root = Path(config['storage_root'])
    run = root / 'runs' / config['run_id']
    output = run / f'worker-{worker_index:02d}'
    output.mkdir(parents=True, exist_ok=True)
    contract = config['contract']
    contract_id = identity(contract)
    budget = StorageBudget(root, Limits(**config['storage_limits']))
    concurrent_games = config['concurrent_games']
    opponent_index = worker['opponent_index']
    opponent_record = contract['opponents'][opponent_index]
    binary = root / 'artifacts' / contract['engine_sha256'] / 'katago'
    model = lambda record: model_path(root, record)
    completed, positions, batch = 0, 0, 0
    started = time.time()
    directory = root / 'corpora' / contract_id / f"host-{config['host_index']}" / f'worker-{worker_index:02d}'
    status_path = output / 'status.json'
    stop = run / 'stop'

    def status(state, **extra):
        atomic_json(status_path, dict(state=state, pid=os.getpid(), cpus=cpus, started=started,
                    updated=time.time(), completed_games=completed, positions=positions, batch=batch,
                    opponent=opponent_record['name'], contract_id=contract_id, **extra))

    with budget.reserve(files=128 * 1024**2, heap=12 * GIB, purpose=f'data worker {config["run_id"]}/{worker_index}'):
        with ExitStack() as stack:
            teacher = stack.enter_context(AnalysisClient(binary, model(contract['teacher']), output / 'teacher',
                    cpus, concurrency=concurrent_games, eigen_threads=4, max_pending=concurrent_games * 2, size=contract['board_size']))
            opponent = stack.enter_context(AnalysisClient(binary, model(opponent_record), output / 'opponent',
                    cpus, concurrency=concurrent_games, eigen_threads=2, max_pending=concurrent_games * 2, size=contract['board_size']))
            atomic_json(output / 'engine-processes.json', [dict(pid=c.process.pid,
                        identity=_process_identity(c.process.pid)) for c in (teacher, opponent)])
            identities = dict(teacher=teacher.request(dict(action='query_models')),
                              opponent=opponent.request(dict(action='query_models')))
            atomic_json(output / 'model-identities.json', identities)
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=concurrent_games))
            try:
                while not stop.exists() and (batches is None or batch < batches):
                    try:
                        budget.check(files=128 * 1024**2)
                    except StoragePressure as error:
                        status('paused_storage', reason=str(error))
                        time.sleep(20)
                        continue
                    jobs = {}
                    for slot in range(concurrent_games):
                        sequence = batch * concurrent_games + slot
                        game_id = identity([contract_id, config['host_index'], worker_index, sequence])
                        if (directory / (game_id + '.npz')).exists():
                            continue
                        expert_color = 1 + int(game_id[16:24], 16) % 2
                        jobs[pool.submit(play_game, teacher, opponent, contract, game_id, expert_color)] = (game_id, expert_color, sequence)
                    status('generating', pending=len(jobs))
                    for future in as_completed(jobs):
                        game_id, expert_color, sequence = jobs[future]
                        rows, actions, outcome = future.result()
                        metadata = dict(game_id=game_id, contract_id=contract_id,
                                        board_size=contract['board_size'], komi=contract['komi'],
                                        teacher_sha256=contract['teacher']['sha256'],
                                        opponent_sha256=opponent_record['sha256'], opponent_index=opponent_index,
                                        expert_color=expert_color, visits=contract['visits'],
                                        host_index=config['host_index'], worker_index=worker_index,
                                        sequence=sequence, **outcome)
                        publish_game(directory, metadata, rows, actions)
                        completed += 1
                        positions += len(rows)
                        status('generating', positions_per_second=positions / (time.time() - started))
                    batch += 1
                status('stopped')
            except BaseException as error:
                status('failed', error=repr(error))
                raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('worker', type=int)
    parser.add_argument('--batches', type=int)
    args = parser.parse_args()
    run_worker(args.config, args.worker, batches=args.batches)


if __name__ == '__main__':
    main()
