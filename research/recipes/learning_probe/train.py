#!/usr/bin/env python3
"""Finite CPU integration run: native Go self-play -> replay -> plain-JAX updates.

This is a qualification recipe, not a tuned AlphaZero reproduction or a
restartable production trainer. It exports inference weights and complete games.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import read_json,verify


def sgf(game):
    size=game['size']
    text='(;GM[1]FF[4]CA[UTF-8]SZ[%d]KM[%s]RU[Tromp-Taylor]AP[gozero:learning_probe]'%(size,game['komi'])
    if game['white_score'] is not None:
        value=game['white_score'];result='0' if value==0 else ('W' if value>0 else 'B')+'+'+str(abs(value))
        text+='RE['+result+']'
    if game['truncated']:text+='C[move-limit truncation; excluded from training]'
    for ply,action in enumerate(game['actions']):
        vertex='' if action==size*size else chr(97+action%size)+chr(97+action//size)
        text+=';'+('B' if ply%2==0 else 'W')+'['+vertex+']'
    return text+')\n'


def run(args):
    verify(ROOT)
    config=read_json(args.config)
    if config['platform']!='cpu':raise ValueError('This initial integration recipe is CPU-only')
    if os.environ.get('JAX_PLATFORMS','cpu')!='cpu':raise ValueError('CPU qualification cannot claim TPU devices')
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import model
    from gozero.native import Actors
    receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=ROOT.name:raise ValueError('Native library was built from a different snapshot')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'games').mkdir()
    cfg_bytes=json.dumps(config,sort_keys=True,allow_nan=False).encode()
    (output/'resolved_config.json').write_bytes(cfg_bytes+b'\n')
    actor=config['actors'];learner=config['learner'];net=config['model']
    if not (1<=actor['size']<=26):raise ValueError('Qualification SGF writer requires sizes 1–26')
    size=actor['size'];channels=actor['history']*2+4;actions=size*size+1
    if learner['warmup_rows']<1 or learner['warmup_rows']>learner['replay_capacity'] or learner['batch_size']<1:
        raise ValueError('Invalid replay settings')
    params=model.initialize(config['seed'],channels,net)
    velocity=jax.tree.map(jnp.zeros_like,params)
    initial=[np.asarray(p).copy() for p in jax.tree.leaves(params)]
    model_shapes=[{'path':jax.tree_util.keystr(path),'shape':list(leaf.shape),'dtype':str(leaf.dtype)}
                  for path,leaf in jax.tree_util.tree_flatten_with_path(params)[0]]
    dummy=jnp.zeros((actor['games'],size,size,channels),jnp.float32)
    before=time.perf_counter();forward=jax.jit(lambda p,x:model.apply(p,x,net)).lower(params,dummy).compile()
    forward_compile=time.perf_counter()-before
    train_shape=(learner['batch_size'],size,size,channels)
    before=time.perf_counter()
    step=jax.jit(lambda p,m,x,pi,z,key:model.update(p,m,x,pi,z,key,net,learner)).lower(
        params,velocity,jnp.zeros(train_shape,jnp.float32),jnp.zeros((learner['batch_size'],actions),jnp.float32),
        jnp.zeros(learner['batch_size'],jnp.float32),jax.random.key(0)).compile()
    update_compile=time.perf_counter()-before
    capacity=learner['replay_capacity'];replay_x=np.empty((capacity,size,size,channels),np.float32)
    replay_pi=np.empty((capacity,actions),np.float32);replay_z=np.empty(capacity,np.float32)
    replay_meta=np.empty((capacity,6),np.uint64);count=0;cursor=0
    random=np.random.default_rng(config['seed']+1);training_key=jax.random.key(config['seed']+2)
    counters={'real_moves':0,'completed_games':0,'truncated_games':0,'eligible_rows':0,'updates':0,
              'neural_batches':0,'neural_slots':0,'active_neural_evaluations':0,
              'native_seconds':0.0,'inference_seconds':0.0,'learner_seconds':0.0}
    report={'schema_version':1,'kind':'learning_integration_qualification','status':'running','snapshot_id':ROOT.name,
            'config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),'native':receipt,'platform':jax.default_backend(),
            'jax_version':jax.__version__,'parameter_count':sum(x.size for x in initial),
            'forward_compile_seconds':forward_compile,'update_compile_seconds':update_compile,
            'claims_go_strength':False,'resumable_training_state':False,'counters':counters}
    started=time.perf_counter();last_metrics=None
    try:
        with Actors(actor,args.native_receipt.parent/receipt['filename'],receipt['binary_sha256']) as engine:
            with (output/'metrics.jsonl').open('w') as log:
                for turn in range(config['selfplay_turns']):
                    before=time.perf_counter();batch=engine.start(counters['updates']);counters['native_seconds']+=time.perf_counter()-before
                    while batch.active_count:
                        before=time.perf_counter()
                        logits,values=jax.device_get(forward(params,batch.features))
                        counters['inference_seconds']+=time.perf_counter()-before
                        counters['neural_batches']+=1;counters['neural_slots']+=actor['games']
                        counters['active_neural_evaluations']+=batch.active_count
                        before=time.perf_counter();batch=engine.evaluate(batch,logits,values);counters['native_seconds']+=time.perf_counter()-before
                    before=time.perf_counter();rows=engine.commit();counters['native_seconds']+=time.perf_counter()-before
                    counters['real_moves']+=actor['games'];counters['eligible_rows']+=len(rows.outcomes)
                    for game in rows.games:
                        counters['truncated_games' if game['truncated'] else 'completed_games']+=1
                        (output/'games'/('%016x.sgf'%game['game_id'])).write_text(sgf(game))
                        (output/'games'/('%016x.json'%game['game_id'])).write_text(json.dumps(game,sort_keys=True)+'\n')
                    n=min(capacity,len(rows.outcomes))
                    if n:
                        indices=(cursor+np.arange(n))%capacity
                        replay_x[indices]=rows.features[-n:];replay_pi[indices]=rows.policies[-n:]
                        replay_z[indices]=rows.outcomes[-n:];replay_meta[indices]=rows.metadata[-n:]
                        cursor=(cursor+n)%capacity;count=min(capacity,count+n)
                    if count>=learner['warmup_rows']:
                        indices=random.integers(0,count,size=learner['batch_size'])
                        training_key,key=jax.random.split(training_key)
                        before=time.perf_counter()
                        params,velocity,metrics=step(params,velocity,replay_x[indices],replay_pi[indices],replay_z[indices],key)
                        metrics=jax.device_get(metrics);counters['learner_seconds']+=time.perf_counter()-before
                        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite learner metrics')
                        counters['updates']+=1;last_metrics={key:float(value) for key,value in metrics.items()}
                    entry={'turn':turn+1,**counters,'replay_rows':count,'metrics':last_metrics}
                    log.write(json.dumps(entry,sort_keys=True)+'\n');log.flush()
                    if (turn+1)%16==0:print(json.dumps(entry,sort_keys=True),flush=True)
        if counters['updates']==0 or counters['completed_games']==0:raise RuntimeError('No completed-game learning was demonstrated')
        leaves,definition=jax.tree.flatten(params);final=[np.asarray(p) for p in leaves]
        change=max(float(np.max(np.abs(a-b))) for a,b in zip(initial,final))
        if not np.isfinite(change) or change==0:raise AssertionError('Parameters did not change finitely')
        np.savez(output/'model_export.npz',**{'p_%04d'%i:p for i,p in enumerate(final)})
        (output/'model_schema.json').write_text(json.dumps(model_shapes,indent=2)+'\n')
        with np.load(output/'model_export.npz',allow_pickle=False) as saved:
            restored=definition.unflatten([jnp.asarray(saved['p_%04d'%i]) for i in range(len(final))])
        probe=replay_x[np.arange(actor['games'])%count]
        original=jax.device_get(forward(params,probe))
        loaded=jax.device_get(forward(restored,probe))
        for a,b in zip(original,loaded):np.testing.assert_array_equal(a,b)
        np.savez(output/'replay_export.npz',features=replay_x[:count],policies=replay_pi[:count],outcomes=replay_z[:count],metadata=replay_meta[:count])
        report.update(status='passed',maximum_parameter_change=change,inference_restore_exact=True,last_metrics=last_metrics)
        verify(ROOT)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_training_seconds']=time.perf_counter()-started
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='native'},sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-receipt',type=Path,required=True)
    run(parser.parse_args())
