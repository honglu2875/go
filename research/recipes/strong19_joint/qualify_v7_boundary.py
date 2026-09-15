"""Cross-check independent Rust and KataGo V7 replay on real 19x19 games."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.native import load_library
from gozero.v7_replay import Replay
from gozero.v7_state import digest_from_native,row_digest,state_digest


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    started=time.monotonic();study=ROOT/'research/studies/joint_inference'
    plan=json.loads((study/'plan-001.json').read_text());nr=plan['rust_native'];path=ROOT/nr['receipt']
    if sha(path)!=nr['receipt_sha256']:raise ValueError('Rust receipt changed')
    receipt=json.loads(path.read_text());native=load_library(path.parent/receipt['filename'],nr['binary_sha256'])
    if native.CAUSAL_GAME_ABI_VERSION!=1 or native.OBSERVATION_REPLAY_ABI_VERSION!=1:raise ValueError('Missing Rust history interface')
    fs=json.loads((ROOT/'research/studies/strong19_scaling/feature-loader-qualification-001.json').read_text())
    data=Path(fs['target']);mp=data/'manifest.json'
    if sha(mp)!=fs['manifest_sha256']:raise ValueError('Feature qualification data changed')
    manifest=json.loads(mp.read_text());files=manifest['shards'][0]['files'];arrays={}
    for name in ('actions','expert_offsets','games'):
        path=data/files[name]['path']
        if sha(path)!=files[name]['sha256']:raise ValueError('Source tape changed')
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    if not np.all(arrays['games']['split']==0):raise ValueError('Boundary qualification must use training histories only')
    offsets=arrays['expert_offsets'];histories=[arrays['actions'][int(lo):int(hi)-1].tolist() for lo,hi in zip(offsets[:-1],offsets[1:])]
    actions=np.concatenate([np.asarray(h+[361],np.int32) for h in histories])
    row_offsets=np.concatenate(([0],np.cumsum([len(h)+1 for h in histories]))).astype(np.int64)
    rules=dict(size=19,komi=7.5,scoring='pass_alive_area')
    stones,legal,_=native.replay_observations(json.dumps(rules),actions,row_offsets)
    stones=np.asarray(stones).reshape(-1,19,19);legal=np.asarray(legal).reshape(-1,362)
    binary=ROOT/'.gozero/build/katago-v7-stream-001/feature_stream'
    checked=0;pending_checked=0;rejections=0
    with Replay(binary,plan['feature_binary_sha256'],size=19) as replay:
        for begin in range(0,len(histories),8):
            rows=replay(histories[begin:begin+8])
            for index,row in enumerate(rows,begin):
                lo,hi=map(int,row_offsets[index:index+2]);history=histories[index]
                np.testing.assert_array_equal(row['stones'],stones[lo:hi])
                np.testing.assert_array_equal(row['legal'],legal[lo:hi]);checked+=hi-lo
                c={**rules,'history':1,'simulations':0,'cpuct':1.5,'max_search_edges':10000}
                game=native.Game(json.dumps(c));checkpoints={0,1,2,min(20,len(history)),len(history)}
                for position in range(len(history)+1):
                    if position in checkpoints:
                        request,features=game.start(17)
                        assert request is not None and game.request_history(request).tolist()==history[:position]
                        got=digest_from_native(features,history=history[:position],size=19,komi=7.5)
                        want=row_digest(row,history[:position],size=19,komi=7.5,index=position)
                        if got!=want:raise ValueError('Rust pending leaf and independent V7 state disagree')
                        request,_=game.evaluate(request,np.zeros(362,np.float32),0.)
                        assert request is None;game.finish();pending_checked+=1
                    if position<len(history):game.play(1+position%2,history[position])
        root=replay([[]])[0]
        # Representative legality and metadata corruption must not silently
        # become a plausible identity at the neural boundary.
        game=native.Game(json.dumps({**rules,'history':1,'simulations':0,'cpuct':1.5,'max_search_edges':10000}))
        _,features=game.start(17);features=np.asarray(features).reshape(19,19,6)
        for channel,value in ((2,0.),(3,0.),(4,.5),(0,np.nan)):
            bad=features.copy();bad[...,channel]=value
            try:digest_from_native(bad,history=[],size=19,komi=7.5)
            except ValueError:rejections+=1
            else:raise AssertionError('Invalid native metadata accepted')
        altered=root['stones'][0].copy();altered[0,0]=1
        assert state_digest(altered,root['legal'][0],history=[],size=19,komi=7.5)!=row_digest(root,[],size=19,komi=7.5)
        altered=root['legal'][0].copy();altered[0]=False
        assert state_digest(root['stones'][0],altered,history=[],size=19,komi=7.5)!=row_digest(root,[],size=19,komi=7.5)
    paths=[Path(__file__),ROOT/'packages/gozero/src/gozero/v7_replay.py',ROOT/'packages/gozero/src/gozero/v7_state.py',ROOT/'packages/gozero/src/gozero/native.py']
    result=dict(kind='joint_v7_rust_boundary_qualification',status='passed',created=time.time(),seconds=time.monotonic()-started,
        games=len(histories),positions=checked,native_pending_leaves=pending_checked,metadata_rejections=rejections,
        all_stones_and_legal_masks_equal=True,test_targets_decoded=False,
        plan_sha256=sha(study/'plan-001.json'),feature_manifest_sha256=fs['manifest_sha256'],
        native_receipt_sha256=nr['receipt_sha256'],native_binary_sha256=nr['binary_sha256'],feature_binary_sha256=sha(binary),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        scope='Independent Rust/C++ histories, stones, legal masks and native pending-leaf metadata. No JAX model, TPU, MCTS strength or scoring qualification.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({key:result[key] for key in ('status','games','positions','native_pending_leaves','seconds')}),flush=True)


if __name__=='__main__':main()
