"""Resident teacher/opponent pair with bounded concurrent games and exact histories."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def play_game(teacher, opponent, contract, game_id, expert_color, progress=None):
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
        if progress is not None:progress(ply+1)
        if game.state().terminal:
            break
    state = game.state()
    return rows, actions, dict(terminal=state.terminal, white_score=state.white_score)


def run_worker(config_path: Path, worker_index: int, *, max_games: int | None = None):
    from . import stream
    import threading
    import fcntl
    config=json.loads(config_path.read_text());worker=config['workers'][worker_index]
    cpus=worker['cpus'];pin(cpus)
    root=Path(config['storage_root']);run=root/'runs'/config['run_id']
    output=run/f'worker-{worker_index:02d}';output.mkdir(parents=True,exist_ok=True)
    worker_lock=(output/'worker.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous_status=output/'status.json'
    if previous_status.exists():
        previous=previous_status.read_bytes()
        import hashlib
        archive=output/'session-history';archive.mkdir(exist_ok=True)
        retained=archive/(hashlib.sha256(previous).hexdigest()+'.json')
        if not retained.exists():retained.write_bytes(previous);retained.chmod(0o444)
    contract=config['contract'];contract_id=identity(contract)
    budget=StorageBudget(root,Limits(**config['storage_limits']))
    concurrent_games=config['concurrent_games'];opponent_index=worker['opponent_index']
    opponent_record=contract['opponents'][opponent_index]
    teacher_threads=worker['teacher_threads'];opponent_threads=worker['opponent_threads']
    if (type(teacher_threads) is not int or type(opponent_threads) is not int
            or min(teacher_threads,opponent_threads)<1 or teacher_threads+opponent_threads>len(cpus)):
        raise ValueError('Inference thread allocation exceeds the CPU mask')
    binary=root/'artifacts'/contract['engine_sha256']/'katago'
    model=lambda record:model_path(root,record)
    directory=root/'corpora'/contract_id/f"host-{config['host_index']}"/f'worker-{worker_index:02d}'
    directory.mkdir(parents=True,exist_ok=True)
    completed=positions=0
    for path in directory.glob('*.npz'):
        with np.load(path,allow_pickle=False) as archive:meta=json.loads(archive['metadata'].tobytes())
        if meta['contract_id']!=contract_id or path.stem!=meta['game_id']:
            raise ValueError('Existing publication has the wrong identity')
        completed+=1;positions+=meta['rows']
    sequence_path=output/'next-sequence.json'
    if sequence_path.exists():
        saved=json.loads(sequence_path.read_text())
        if saved['contract_id']!=contract_id:raise ValueError('Admission sequence belongs to another corpus')
        sequence=saved['next_sequence']
        if type(sequence) is not int or sequence<0:raise ValueError('Invalid admission counter')
    else:
        if completed:raise ValueError('Published games exist without an admission ledger')
        sequence=0
    started=time.time();stop=run/'stop';progress={};lock=threading.Lock()
    last_status=0.;pressure=None;pending_count=0
    def status(state,**extra):
        nonlocal last_status
        with lock:active={str(k):dict(v) for k,v in sorted(progress.items())}
        atomic_json(output/'status.json',dict(state=state,pid=os.getpid(),cpus=cpus,started=started,
            updated=time.time(),completed_games=completed,positions=positions,next_sequence=sequence,
            in_flight_games=pending_count,in_flight_positions=sum(v['plies'] for v in active.values()),
            in_flight=active,teacher_threads=teacher_threads,opponent_threads=opponent_threads,
            contract_id=contract_id,producer_snapshot=config['producer_snapshot'],**extra))
        last_status=time.monotonic()
    def on_tick(pending):
        nonlocal pending_count
        pending_count=len(pending)
        if time.monotonic()-last_status>=5:
            status('draining' if stop.exists() else 'paused_storage' if pressure else 'generating',storage_pressure=pressure)
    def can_admit():
        nonlocal pressure
        try:budget.check(files=128*1024**2)
        except StoragePressure as error:pressure=str(error);return False
        pressure=None;return True
    pool=ThreadPoolExecutor(max_workers=concurrent_games)
    try:
        with budget.reserve(files=128*1024**2,heap=12*GIB,purpose=f'data worker {config["run_id"]}/{worker_index}'):
            # Close engine clients before joining the executor on an error so
            # pending requests fail promptly instead of holding shutdown open.
            with ExitStack() as stack:
                teacher=stack.enter_context(AnalysisClient(binary,model(contract['teacher']),output/'teacher',
                    cpus,concurrency=concurrent_games,eigen_threads=teacher_threads,max_pending=concurrent_games*2,size=contract['board_size']))
                opponent=stack.enter_context(AnalysisClient(binary,model(opponent_record),output/'opponent',
                    cpus,concurrency=concurrent_games,eigen_threads=opponent_threads,max_pending=concurrent_games*2,size=contract['board_size']))
                atomic_json(output/'engine-processes.json',[dict(pid=c.process.pid,identity=_process_identity(c.process.pid)) for c in (teacher,opponent)])
                identities=dict(teacher=teacher.request(dict(action='query_models')),opponent=opponent.request(dict(action='query_models')))
                atomic_json(output/'model-identities.json',identities)
                def submit():
                    nonlocal sequence
                    current=sequence;sequence+=1
                    game_id=identity(dict(contract_id=contract_id,host=config['host_index'],worker=worker_index,sequence=current))
                    expert_color=1+int(game_id[:8],16)%2
                    atomic_json(sequence_path,dict(contract_id=contract_id,next_sequence=sequence))
                    meta=dict(sequence=current,game_id=game_id,expert_color=expert_color)
                    with lock:progress[current]=dict(game_id=game_id,plies=0,updated=time.time())
                    def advance(plies):
                        with lock:progress[current]=dict(game_id=game_id,plies=plies,updated=time.time())
                    return pool.submit(play_game,teacher,opponent,contract,game_id,expert_color,advance),meta
                def publish(meta,result):
                    nonlocal completed,positions
                    rows,actions,outcome=result
                    metadata=dict(contract_id=contract_id,game_id=meta['game_id'],board_size=contract['board_size'],
                        komi=contract['komi'],teacher_sha256=contract['teacher']['sha256'],opponent_sha256=opponent_record['sha256'],
                        opponent_index=opponent_index,expert_color=meta['expert_color'],visits=contract['visits'],
                        host_index=config['host_index'],worker_index=worker_index,sequence=meta['sequence'],
                        producer_snapshot=config['producer_snapshot'],**outcome)
                    publish_game(directory,metadata,rows,actions)
                    completed+=1;positions+=len(rows)
                    with lock:progress.pop(meta['sequence'])
                status('generating')
                result=stream.run(submit,publish,can_admit,stop.exists,concurrency=concurrent_games,
                                  limit=max_games,tick=on_tick,poll_seconds=1.)
                pending_count=0;status('stopped',stream_result=result)
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        worker_lock.close()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path);parser.add_argument('worker',type=int)
    parser.add_argument('--max-games',type=int);args=parser.parse_args()
    run_worker(args.config,args.worker,max_games=args.max_games)


if __name__=='__main__':main()
