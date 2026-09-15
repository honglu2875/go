#!/usr/bin/env python3
"""Paired-color real KataGo matches with pinned candidate/search artifacts and exact boards."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.model_artifacts import artifact,validate_candidate
from gozero.snapshots import canonical_json,read_json,verify
from gozero.scoring import direct_area,score_string
from qualify_katago import kata_cells,sgf_vertex


def summarize(games,alpha=0.05):
    valid=[g for g in games if g['status']=='completed']
    pairs={}
    for g in valid: pairs.setdefault(g['pair'],[]).append(g['candidate_points'])
    pair_points=[sum(v)/2 for v in pairs.values() if len(v)==2]
    mean=sum(pair_points)/len(pair_points) if pair_points else None
    radius=math.sqrt(math.log(2/alpha)/(2*len(pair_points))) if pair_points else None
    return {'scheduled_games':len(games),'completed_games':len(valid),'failed_games':sum(g['status']=='failed' for g in games),
            'truncated_games':sum(g['status']=='truncated' for g in games),'candidate_wins':sum(g['candidate_points']==1 for g in valid),
            'draws':sum(g['candidate_points']==0.5 for g in valid),'candidate_losses':sum(g['candidate_points']==0 for g in valid),
            'complete_pairs':len(pair_points),'paired_score':mean,
            'paired_hoeffding_95_interval':None if mean is None else [max(0,mean-radius),min(1,mean+radius)],
            'uncertainty_scope':'Conservative bounded-score interval assuming independent pairs, conditional on the scheduled openings. No Elo estimate. Incomplete pairs are listed and invalidate qualification.'}


def run(args):
    verify(SOURCE)
    spec_path=args.spec.resolve()
    if not spec_path.is_relative_to(SOURCE): raise ValueError('Match specification must be in the frozen source')
    c=read_json(spec_path);root=args.artifacts_root.resolve()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    candidate=SOURCE/c['candidate'];pinned=read_json(candidate)
    training=root/'.gozero/snapshots'/pinned['training_snapshot'];verify(training)
    training_config=read_json(training/'resolved_config.json')
    adapter='eval/learned_gtp.py';inference_args=[];inference_path=None
    native_snapshot=pinned['training_snapshot'];declared_native=None
    if pinned.get('kind') == 'visual_causal_checkpoint':
        from visual_gtp import contract
        adapter='eval/visual_gtp.py';inference_path=artifact(SOURCE,c['visual_inference'])
        inference=contract(read_json(inference_path))
        if (not pinned['training_complete'] or args.visual_socket is None
                or inference['simulations']!=c['candidate_simulations_excluding_root']
                or inference['cpuct']!=c['candidate_cpuct']
                or any(inference[k]!=c[k] for k in ('size','komi','max_game_moves'))):
            raise ValueError('Visual inference or completed candidate differs from match contract')
        native_snapshot=c['inference_native_snapshot'];verify(artifact(root,'.gozero/snapshots/'+native_snapshot))
        inference_args=['--inference-config',str(inference_path),'--socket',str(args.visual_socket)]
        scoring=inference['scoring']
    elif pinned.get('kind') in ('causal_history_policy','board_causal_history_policy','state_expert_policy'):
        from causal_gtp import inference_contract
        adapter='eval/state_expert_gtp.py' if pinned['kind']=='state_expert_policy' else 'eval/causal_gtp.py'
        inference_path=artifact(SOURCE,c['causal_inference'])
        inference=inference_contract(read_json(inference_path),training_config['model'])
        if (inference['simulations']!=c['candidate_simulations_excluding_root'] or inference['cpuct']!=c['candidate_cpuct']
                or any(inference[k]!=c[k] for k in ('size','komi','max_game_moves'))):
            raise ValueError('Causal inference and match contracts differ')
        native_snapshot=c['inference_native_snapshot'];verify(artifact(root,'.gozero/snapshots/'+native_snapshot))
        inference_args=['--inference-config',str(inference_path),'--inference-native-snapshot',native_snapshot]
        scoring=inference['scoring']
    else:
        if any(k in c for k in ('causal_inference','inference_native_snapshot','inference_native_binary_sha256')):
            raise ValueError('Causal inference overrides require a causal candidate')
        scoring=training_config['actors'].get('scoring','raw_area')
        if 'initialization' in training_config:
            declared_native=validate_candidate(root,pinned)['native_receipt']
            native_snapshot=read_json(declared_native)['snapshot_id']
    if scoring not in ('raw_area','pass_alive_area'): raise ValueError('Unsupported candidate scoring profile')
    native=declared_native or artifact(root,'.gozero/native/'+native_snapshot+'/receipt.json')
    if inference_path is not None and read_json(native)['binary_sha256']!=c['inference_native_binary_sha256']:
        raise ValueError('Causal inference native binary differs from registration')
    engine=read_json(SOURCE/'eval/katago_build.json')
    weights_descriptor=artifact(SOURCE,c.get('katago_weights','eval/katago_9x9.json'))
    weights=read_json(weights_descriptor)
    if weights['role']!='evaluation_only' or c['size'] not in weights['board_sizes']:
        raise ValueError('KataGo model does not support the registered evaluation role or board size')
    kata_binary=artifact(root,engine['binary_path']);kata_weights=artifact(root,weights['path'])
    for path,expected in ((kata_binary,engine['binary_sha256']),(kata_weights,weights['sha256'])):
        if sha256(path)!=expected: raise ValueError('External artifact hash mismatch: '+str(path))
    allowed=set(os.sched_getaffinity(0))
    for key in ('candidate_cpus','katago_cpus'):
        if not c[key] or len(set(c[key]))!=len(c[key]) or not set(c[key])<=allowed: raise ValueError('Invalid engine CPU affinity')
    if set(c['candidate_cpus'])&set(c['katago_cpus']): raise ValueError('Evaluation engine CPU sets must be disjoint')
    base=(SOURCE/c['katago_config']).read_text()
    visits=re.search(r'^maxVisits\s*=\s*(\d+)\s*$',base,re.M)
    if not visits or int(visits[1])!=c['katago_max_visits']: raise ValueError('KataGo visit contract differs')
    if c['size']!=9 or c['komi']!=7.5 or not c['openings']: raise ValueError('This benchmark is pinned to the 9x9 checkpoint profile')
    report={'schema_version':1,'kind':'paired_katago_matches','status':'running','claims_go_strength_improvement':False,
            'snapshot_id':SOURCE.name,'spec_sha256':sha256(spec_path),'candidate_sha256':sha256(candidate),
            'candidate_scoring_profile':scoring,
            'candidate_adapter':adapter,'inference_native_snapshot':native_snapshot,
            'candidate_inference_config_sha256':None if inference_path is None else sha256(inference_path),
            'candidate':pinned,'katago_binary_sha256':engine['binary_sha256'],'katago_weights_sha256':weights['sha256'],
            'katago_model':weights['model'],'katago_weights_descriptor_sha256':sha256(weights_descriptor),
            'base_katago_config_sha256':sha256(SOURCE/c['katago_config']),
            'native_receipt_sha256':sha256(native),'started_unix':time.time(),'games':[]}
    (output/'resolved_spec.json').write_bytes(canonical_json(c))
    try:
        for pair,opening in enumerate(c['openings']):
            for candidate_color in ('B','W'):
                directory=output/f'pair-{pair:03d}-{candidate_color}';directory.mkdir()
                seed=c['seed_prefix']+'-'+str(pair)
                cfg=re.sub(r'^(searchRandSeed|nnRandSeed)\s*=.*$',lambda m:m[1]+' = '+seed,base,flags=re.M)
                config=directory/'katago.cfg';config.write_text(cfg)
                candidate_argv=['taskset','-c',','.join(map(str,c['candidate_cpus'])),sys.executable,'-B',str(SOURCE/adapter),
                    '--candidate',str(candidate),'--native-receipt',str(native),'--artifacts-root',str(root),
                    '--simulations',str(c['candidate_simulations_excluding_root']),'--cpuct',str(c['candidate_cpuct']),*inference_args]
                if pinned.get('kind')=='visual_causal_checkpoint':
                    candidate_argv=['taskset','-c',','.join(map(str,c['candidate_cpus'])),sys.executable,'-B',str(SOURCE/adapter),
                        '--candidate',str(candidate),'--native-receipt',str(native),*inference_args]
                kata_argv=['taskset','-c',','.join(map(str,c['katago_cpus'])),str(kata_binary),'gtp','-model',str(kata_weights),'-config',str(config)]
                game={'pair':pair,'candidate_color':candidate_color,'opening':opening,'status':'running','moves':[],
                      'candidate_argv':candidate_argv,'katago_argv':kata_argv,'katago_config_sha256':sha256(config),'candidate_search_totals':{'simulations':0,'neural_evaluations':0,'terminal_evaluations':0},
                      'engine_seconds':{'candidate':0.0,'katago':0.0},'checked_positions':0}
                start=time.monotonic();deadline=start+c['game_timeout_seconds']
                def command(client,text):
                    remaining=deadline-time.monotonic()
                    if remaining<=0: raise TimeoutError('Match game deadline expired')
                    return client.command(text,timeout=min(remaining,60))
                try:
                    env={**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
                    with GTPClient(candidate_argv,directory/'candidate',environment=env) as model, GTPClient(kata_argv,directory/'katago') as kata:
                        game['candidate_version']=command(model,'version');game['katago_version']=command(kata,'version')
                        for client in (model,kata):
                            command(client,'boardsize 9');command(client,'clear_board');command(client,'komi 7.5')
                        rules=json.loads(command(kata,'kata-get-rules'));game['katago_rules']=rules
                        if rules['ko']!='POSITIONAL' or rules['scoring']!='AREA' or not rules['suicide']: raise ValueError('External rules mismatch')
                        def check_board():
                            if command(model,'showboard')!=kata_cells(command(kata,'showboard'),9): raise AssertionError('Candidate and KataGo boards differ')
                            game['checked_positions']+=1
                        passes=0
                        for ply in range(c['max_game_moves']):
                            color='B' if ply%2==0 else 'W'
                            if ply<len(opening):
                                move=opening[ply]
                                if move.lower() in ('pass','resign'): raise ValueError('Bootstrap openings require stone placements')
                                for client in (model,kata): command(client,'play '+color+' '+move)
                                entry={'color':color,'vertex':move,'opening':True}
                            else:
                                ours=color==candidate_color
                                player,opponent=(model,kata) if ours else (kata,model)
                                before=time.monotonic();move=command(player,'genmove '+color).strip();elapsed=time.monotonic()-before
                                if move.lower()=='resign': raise ValueError('Unexpected resignation with resignation disabled')
                                command(opponent,'play '+color+' '+move)
                                game['engine_seconds']['candidate' if ours else 'katago']+=elapsed
                                entry={'color':color,'vertex':move,'opening':False,'engine_seconds':elapsed}
                                if ours:
                                    stats=json.loads(command(model,'gozero-search-stats'));entry['search']=stats
                                    for key in game['candidate_search_totals']:game['candidate_search_totals'][key]+=stats[key]
                            game['moves'].append(entry);check_board()
                            passes=passes+1 if move.lower()=='pass' else 0
                            if passes==2:
                                score=command(model,'final_score');kata_score=command(kata,'final_score')
                                board=command(model,'showboard')
                                independent=score_string(direct_area(board,c['komi']))
                                if scoring=='raw_area' and score!=independent: raise AssertionError('Native and independent raw area scores differ')
                                if scoring=='pass_alive_area' and score!=kata_score: raise AssertionError('Native pass-alive and KataGo area scores differ')
                                game.update(status='completed',score=score,independent_raw_score=independent,katago_adjudicated_score=kata_score,
                                            score_margin_agrees_with_katago=score==kata_score,
                                            winner_agrees_with_katago=score[0]==kata_score[0],
                                            candidate_points=0.5 if score=='0' else float(score[0]==candidate_color))
                                if not game['winner_agrees_with_katago']:
                                    raise AssertionError('Scoring conventions change the winner; compatibility qualification fails')
                                break
                        else:game['status']='truncated'
                except Exception as error:
                    game.update(status='failed',error=repr(error))
                finally:
                    game['elapsed_seconds']=time.monotonic()-start
                    (directory/'result.json').write_bytes(canonical_json(game))
                    result='RE['+game['score']+']' if game['status']=='completed' else ''
                    record='(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RU[Tromp-Taylor]AP[gozero:paired-eval]C[candidate terminal scoring profile: '+scoring+']'+result
                    record+='PB['+('gozero' if candidate_color=='B' else 'KataGo')+']PW['+('gozero' if candidate_color=='W' else 'KataGo')+']'
                    record+=''.join(';'+m['color']+'['+sgf_vertex(m['vertex'],9)+']' for m in game['moves'])+')\n'
                    (directory/'game.sgf').write_text(record)
                    report['games'].append(game)
                    print(json.dumps({'kind':'match','pair':pair,'candidate_color':candidate_color,'status':game['status'],'score':game.get('score'),'plies':len(game['moves']),'error':game.get('error')}),flush=True)
        report['summary']=summarize(report['games'])
        report['status']='passed' if all(g['status']=='completed' for g in report['games']) else 'failed'
        verify(SOURCE)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time()
        (output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k not in ('games','candidate')},sort_keys=True),flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    def stop(signum,frame):
        # Unwind active GTP contexts so a panel deadline closes both engines.
        raise KeyboardInterrupt('match interrupted by signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visual-socket',type=Path)
    raise SystemExit(run(parser.parse_args()))
