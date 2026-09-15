"""19x19 producer qualification: real engines, sized histories, D4 splits, publication."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.environment/'site-packages'))
    import numpy as np
    from flygo.go import Game,GameConfig
    from flygo.data.katago import AnalysisClient,query,model_path
    from flygo.data.label import parse_label
    from flygo.data.corpus import publish_game,audit_game,opening_family,identity,atomic_json
    from flygo.data.generate import play_game
    contract=json.loads(args.contract.read_text())
    out=args.root/'qualification';out.mkdir(exist_ok=True)
    started=time.time()
    atomic_json(out/'status.json',dict(state='running',started=started))
    size=19;pass_action=361;binary=args.root/'artifacts'/contract['engine_sha256']/'katago'
    # Synthetic terminal record exercises indexing, full-history replay, storage and split.
    game=Game(GameConfig(size=size));rows=[];actions=[]
    for ply in range(2):
        state=game.state();legal=np.zeros(362,dtype=bool);legal[game.legal()]=True
        policy=legal.astype(np.float32);policy/=policy.sum()
        rows.append(dict(stones=state.stones.copy(),legal=legal,raw_policy=policy,raw_value=np.float32(0),
            search_policy=policy,search_value=np.float32(0),search_policy_valid=ply==0,
            root_edge_visits=np.zeros(362,dtype=np.int32),raw_score=np.float32(0),raw_score_valid=False,
            teacher_visits=np.int32(16 if ply==0 else 1)))
        game.play(state.to_play,pass_action);actions.append(pass_action)
    meta=dict(game_id=identity(['synthetic-qualification',contract]),contract_id=identity(contract),
        board_size=19,komi=7.5,expert_color=1,visits=16,terminal=True,white_score=game.state().white_score)
    published=publish_game(out/'synthetic',meta,rows,actions)
    assert audit_game(published)['rows']==2
    prefix=[0,18,38,57,95,113,190,360]
    assert opening_family(prefix,19)==opening_family([((a%19)*19+18-a//19) for a in prefix],19)
    assert query([360,361],visits=1,size=19)['moves']==[['B','T1'],['W','pass']]
    assert query([360,361],visits=1,size=19)['boardXSize']==19
    records=[contract['teacher'],*contract['opponents']]
    records=list({r['sha256']:r for r in records}.values())
    def check(item):
        i,record=item
        cpus=list(range((i%4)*8,(i%4+1)*8))
        with AnalysisClient(binary,model_path(args.root,record),out/record['sha256'],cpus,
                            concurrency=2,eigen_threads=4,max_pending=4,size=19) as client:
            model_info=client.request(dict(action='query_models'))
            latencies=[]
            for history in ([],prefix):
                position=Game(GameConfig(size=19))
                for ply,action in enumerate(history):position.play(1+ply%2,action)
                t=time.monotonic();response=client.request(query(history,visits=16,size=19),timeout=600)
                label=parse_label(response,position.legal(),position.state().to_play,size=19)
                assert label['raw_policy'].shape==(362,) and label['search_policy_valid']
                latencies.append(time.monotonic()-t)
            return dict(name=record['name'],sha256=record['sha256'],query_seconds=latencies,models=model_info,status='passed')
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks=list(pool.map(check,enumerate(records)))
    atomic_json(out/'status.json',dict(state='prefix_canary',models_passed=len(checks),started=started))
    # Both teacher colors, real search and raw labels; capped prefixes stay outside production.
    with AnalysisClient(binary,model_path(args.root,contract['teacher']),out/'canary-teacher',list(range(8)),
                        concurrency=2,eigen_threads=4,max_pending=4,size=19) as teacher:
        with AnalysisClient(binary,model_path(args.root,contract['opponents'][0]),out/'canary-opponent',list(range(8)),
                            concurrency=2,eigen_threads=2,max_pending=4,size=19) as opponent:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(play_game,teacher,opponent,{**contract,'max_moves':12},identity(['canary',color,contract]),color) for color in (1,2)]
                for color,future in zip((1,2),futures):
                    rows,actions,outcome=future.result()
                    record=dict(game_id=identity(['canary',color,contract]),contract_id=identity(contract),
                                board_size=19,komi=7.5,expert_color=color,visits=16,**outcome)
                    path=publish_game(out/'real-prefixes',record,rows,actions)
                    assert audit_game(path)['rows']==len(actions)
    result=dict(status='passed',board_size=19,synthetic_terminal_roundtrip=True,d4_family_invariant=True,
                models=checks,real_prefix_games=2,real_prefix_plies=24,production_games=0,
                elapsed_seconds=time.time()-started,completed=time.time())
    atomic_json(out/'result.json',result)
    atomic_json(out/'status.json',dict(state='complete',elapsed=time.time()-started))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
