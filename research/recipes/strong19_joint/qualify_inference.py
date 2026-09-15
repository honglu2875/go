"""Legal 19x19 branch/cache qualification and real Rust MCTS leaf evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.native import load_library
from gozero.v7_replay import Replay
from gozero.v7_state import digest_from_native,row_digest
import inference
import joint
from qualify_joint import configs


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if jax.device_count()!=4 or jax.default_backend()!='cpu':raise ValueError('Requires four simulated CPU devices')
    started=time.monotonic();study=ROOT/'research/studies/joint_inference';plan=json.loads((study/'plan-001.json').read_text())
    boundary=study/'native-boundary-001.json'
    if json.loads(boundary.read_text())['status']!='passed':raise ValueError('Native boundary qualification is missing')
    nr=plan['rust_native'];receipt_path=ROOT/nr['receipt']
    if sha(receipt_path)!=nr['receipt_sha256']:raise ValueError('Native receipt changed')
    r=json.loads(receipt_path.read_text());native=load_library(receipt_path.parent/r['filename'],nr['binary_sha256'])
    binary=ROOT/'.gozero/build/katago-v7-stream-001/feature_stream';binary_hash=plan['feature_binary_sha256']
    data=ROOT/'research/studies/strong19_scaling/feature-loader-qualification-001.json'
    feature=json.loads(data.read_text());data=Path(feature['target']);m=json.loads((data/'manifest.json').read_text())
    if sha(data/'manifest.json')!=feature['manifest_sha256']:raise ValueError('Feature fixture changed')
    arrays={}
    for name in ('actions','expert_offsets','games'):
        item=m['shards'][0]['files'][name];path=data/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Fixture tape changed')
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    if not np.all(arrays['games']['split']==0):raise ValueError('Test targets must remain closed')
    tapes=[arrays['actions'][int(lo):int(lo)+20].tolist() for lo in arrays['expert_offsets'][:2]]
    records=[];max_errors={};search_events=[]
    with Replay(binary,binary_hash,size=19) as oracle:
        prefix=tapes[0][:6];row=oracle([prefix])[0]
        alternate=next(int(x) for x in np.flatnonzero(row['legal'][-1]) if x!=tapes[0][6] and x!=361)
        batches=[{0:tapes[0][:5],1:tapes[1][:7],2:[],3:[361]},
                 {0:tapes[0][:12],1:tapes[1][:4],2:[0,361,1],3:[0]},
                 {0:prefix+[alternate],1:tapes[1][:9],2:[0,361,1,361,2],3:[]},
                 {0:tapes[0][:10],1:tapes[1][:8],2:[0,361,1],3:[361,1]}]
        for name,small in configs().items():
            c={**small,'max_board_size':19,'max_positions':64};v=dict(hidden=7,spatial_channels=5)
            params=joint.initialize(7431,c,v)
            if name=='transformer':
                params['head.local.weight']=jnp.linspace(-.12,.08,16)[:,None]
                params['head.context.q.weight']=jnp.linspace(-.1,.2,64).reshape(16,4)
            def reference(rows,histories):
                length=1 if name=='cnn' else 1<<(max(len(h)+1 for h in histories)-1).bit_length()
                b=dict(spatial=np.zeros((4,length,19,19,22),np.float32),global_features=np.zeros((4,length,19),np.float32),
                       actions=np.zeros((4,length),np.int32),counts=np.zeros(4,np.int32))
                for i,(row,history) in enumerate(zip(rows,histories)):
                    n=1 if name=='cnn' else len(history)+1
                    b['spatial'][i,:n]=row['spatial'][-1:] if name=='cnn' else row['spatial']
                    b['global_features'][i,:n]=row['global_features'][-1:] if name=='cnn' else row['global_features']
                    b['counts'][i]=n
                    if name!='cnn':b['actions'][i,:len(history)]=history
                out=jax.device_get(oracle_forward(params,jax.tree.map(jnp.asarray,b)))
                return [{k:out[k][i,int(b['counts'][i])-1] for k in out} for i in range(len(histories))]
            oracle_forward=jax.jit(lambda p,b:joint.forward(p,b,c))
            runner=inference.Runner(params,c,Replay(binary,binary_hash,size=19),slots=4,network_version=41,max_block=4)
            comparisons=0;rejected=0
            def compare(actual,expected):
                nonlocal comparisons
                assert set(actual)==set(expected)=={'policy','value','value_logits'}
                for key in expected:
                    error=float(np.max(np.abs(actual[key]-expected[key])))
                    max_errors[name+'.'+key]=max(max_errors.get(name+'.'+key,0.),error)
                    np.testing.assert_allclose(actual[key],expected[key],atol=5e-5,rtol=5e-4,err_msg=name+'.'+key)
                comparisons+=1
            def requests(histories):
                rows=oracle(list(histories.values()))
                request={slot:(h,row_digest(row,h,size=19,komi=7.5)) for (slot,h),row in zip(histories.items(),rows)}
                return request,reference(rows,list(histories.values()))
            try:
                for histories in batches:
                    request,expected=requests(histories);out=runner.score(request)
                    for slot,e in zip(histories,expected):compare(out[slot],e)
                    before=runner.stats.copy();again=runner.score(dict(reversed(list(request.items()))))
                    assert runner.stats['dispatches']==before['dispatches'] and runner.stats['replay_calls']==before['replay_calls']
                    for slot,e in zip(histories,expected):compare(again[slot],e)
                    # Published arrays and dictionaries cannot poison retained
                    # results owned by the scorer.
                    out[0]['policy'][0]=1e6;out[0]['value']=np.asarray(1e6)
                    compare(runner.score({0:request[0]})[0],expected[0])
                request,expected=requests(batches[-1]);good=request[0]
                invalid=[{0:(good[0],'0'*64)}, {True:good}, {4:good}, {0:([362],good[1])},
                         {0:([0,0],good[1])}, {0:([361,361],good[1])}, {0:([0]*64,good[1])},
                         {0:good,1:(batches[-1][1]+[361],'0'*64)}]
                for bad in invalid:
                    previous=runner.stats['dispatches']
                    try:runner.score(bad)
                    except ValueError:rejected+=1
                    else:raise AssertionError('Invalid request accepted')
                    assert not runner.failed and runner.stats['dispatches']==previous
                    compare(runner.score({0:good})[0],expected[0])
                # Two independent real Rust searches, using complete pending
                # leaf histories and this same cached neural owner.
                games=[];pending={};root_values={}
                for slot,h in enumerate((tapes[0][:6],tapes[1][:7])):
                    game=native.Game(json.dumps(dict(size=19,komi=7.5,scoring='pass_alive_area',history=1,
                        simulations=2,cpuct=1.5,max_search_edges=10000)))
                    for i,action in enumerate(h):game.play(1+i%2,action)
                    games.append(game);pending[slot]=game.start(41)
                while pending:
                    req={};histories={}
                    for slot,(handle,features) in pending.items():
                        h=games[slot].request_history(handle).tolist();histories[slot]=h
                        req[slot]=(h,digest_from_native(features,history=h,size=19,komi=7.5))
                    rows=oracle(list(histories.values()));expected=reference(rows,list(histories.values()));prediction=runner.score(req)
                    for (slot,h),want,row in zip(histories.items(),expected,rows):
                        if req[slot][1]!=row_digest(row,h,size=19,komi=7.5):raise ValueError('Search leaf mismatch')
                        compare(prediction[slot],want)
                        if slot not in root_values:root_values[slot]=float(prediction[slot]['value'])
                        search_events.append(dict(architecture=name,slot=slot,depth=len(h),to_play=1+len(h)%2,
                            value_sent=float(prediction[slot]['value']),state_digest=req[slot][1]))
                        handle,_=pending[slot]
                        nxt=games[slot].evaluate(handle,np.ascontiguousarray(prediction[slot]['policy'],dtype=np.float32),float(prediction[slot]['value']))
                        if nxt[0] is None:del pending[slot]
                        else:pending[slot]=nxt
                searches=[]
                for slot,game in enumerate(games):
                    audit=json.loads(game.inspect_search())
                    chosen,policy,value,simulations,neural,terminal=game.finish()
                    assert chosen in game.legal() and simulations==2 and np.isclose(policy.sum(),1.)
                    np.testing.assert_allclose(audit['network_value'],root_values[slot],atol=1e-7,rtol=0)
                    searches.append(dict(chosen=chosen,completed_simulations=simulations,neural_evaluations=neural,
                                         terminal_evaluations=terminal,root_value=value,network_value=audit['network_value']))
                records.append(dict(architecture=name,config=c,value_config=v,comparisons=comparisons,rejections=rejected,
                                    stats=runner.stats,compilation=runner.compilation,searches=searches))
            finally:runner.close()
            print(json.dumps(dict(architecture=name,status='passed',comparisons=comparisons,rejections=rejected)),flush=True)
    paths=[*Path(__file__).parent.glob('*.py'),ROOT/'packages/gozero/src/gozero/v7_replay.py',ROOT/'packages/gozero/src/gozero/v7_state.py']
    result=dict(kind='joint19_native_cached_inference_cpu_qualification',status='passed',created=time.time(),seconds=time.monotonic()-started,
        plan_sha256=sha(study/'plan-001.json'),native_boundary_sha256=sha(boundary),records=records,max_absolute_errors=max_errors,
        search_events=search_events,test_targets_decoded=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        scope='Small complete joint backbones on real legal19 prefixes and synthetic passes; cached/full and branch/prefix equivalence, rejected inputs and Rust MCTS leaf evaluation. No trained19 model, full-size TPU throughput, RPC/GTP, or strength claim.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',seconds=result['seconds'],max_absolute_errors=max_errors)),flush=True)


if __name__=='__main__':main()
