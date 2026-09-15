"""Check native streaming replay against pinned offline features and histories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import numpy as np
from client import Replay,MAX_REQUEST

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def offline(binary,histories,size,parent):
    with tempfile.TemporaryDirectory(prefix='v7-offline-',dir=parent) as name:
        folder=Path(name);games=[dict(size=size,komi=7.5,actions=h+[size*size]) for h in histories]
        (folder/'games.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in games))
        paths=[folder/'spatial.u8',folder/'global.f32',folder/'audit.u8']
        result=subprocess.run([str(binary),str(folder/'games.jsonl'),*map(str,paths)],check=True,capture_output=True,text=True,timeout=60)
        count=sum(len(g['actions']) for g in games)
        assert json.loads(result.stdout)==dict(games=len(games),positions=count,status='passed')
        s=np.fromfile(paths[0],np.uint8).reshape(count,size,size,22)
        g=np.fromfile(paths[1],'<f4').reshape(count,19)
        a=np.fromfile(paths[2],np.uint8).reshape(count,2*size*size+1)
        offsets=np.concatenate(([0],np.cumsum([len(h)+1 for h in histories])))
        return [dict(spatial=s[lo:hi],global_features=g[lo:hi],
                     stones=a[lo:hi,:size*size].reshape(-1,size,size),legal=a[lo:hi,size*size:].astype(bool))
                for lo,hi in zip(offsets[:-1],offsets[1:])]


def equal(actual,expected):
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        assert set(a)==set(b)
        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def reject(fn,contains=None):
    try:fn()
    except ValueError as error:
        if contains:assert contains in str(error),str(error)
    else:raise AssertionError('Invalid request accepted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();started=time.time()
    build=json.loads(a.build_receipt.read_text());assert build['status']=='passed'
    binary=a.build_receipt.parent/'feature_stream';assert sha(binary)==build['binary_sha256']
    source=Path(__file__).with_name('feature_stream.cpp');assert sha(source)==build['source_sha256']
    feature_receipt=ROOT/'research/studies/strong19_scaling/feature-loader-qualification-001.json'
    feature=json.loads(feature_receipt.read_text());assert feature['status']=='passed' and feature['games']==30
    data=Path(feature['target']);manifest=json.loads((data/'manifest.json').read_text())
    assert sha(data/'manifest.json')==feature['manifest_sha256'] and manifest['qualification_only']
    fs=os.statvfs('/dev/shm');assert fs.f_bavail*fs.f_frsize>66*(1<<30)
    original=ROOT/'.gozero/build/katago-v7-12e944b6/feature_worker'
    assert sha(original)==feature['feature_worker_binary_sha256']
    arrays={}
    for name,item in manifest['shards'][0]['files'].items():
        path=data/item['path'];assert sha(path)==item['sha256']
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    offsets=arrays['expert_offsets'];games=arrays['games'];assert np.all(games['split']==0)
    histories=[];expected=[]
    for i in range(len(games)):
        lo,hi=map(int,offsets[i:i+2]);history=arrays['actions'][lo:hi].tolist()[:-1]
        histories.append(history)
        expected.append(dict(spatial=np.unpackbits(arrays['spatial'][lo:hi],axis=1,count=361*22,bitorder='little').reshape(-1,19,19,22),
                             global_features=np.asarray(arrays['global_features'][lo:hi]),
                             legal=np.unpackbits(arrays['legal'][lo:hi],axis=1,count=362,bitorder='little').astype(bool)))
    # Raw boards provide an independent engine audit; no policy/value labels are
    # decoded here. All selected games belong to the training split.
    inventory=json.loads((ROOT/'research/studies/strong19_scaling/inventory-001.json').read_text())
    records=inventory['feature_qualification_selection'];assert len(records)==len(histories)
    for i,row in enumerate(records):
        path=data/'raw-inputs'/(row['game_id']+'.npz');assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as raw:
            expected[i]['stones']=raw['stones'].reshape(-1,19,19)
            assert raw['actions'].tolist()[:-1]==histories[i]
    calls=[]
    with Replay(binary,build['binary_sha256'],size=19) as replay:
        empty=offline(original,[[]],19,data)
        equal(replay([[]]),empty)
        for start in range(0,len(histories),8):
            hs=histories[start:start+8];before=time.perf_counter();actual=replay(hs)
            calls.append(dict(kind='complete',games=len(hs),rows=sum(len(h)+1 for h in hs),seconds=time.perf_counter()-before))
            equal(actual,expected[start:start+8])
            begins=[max(0,len(h)-3) for h in hs]
            equal(replay(hs,begins),[{key:value[b:] for key,value in row.items()} for row,b in zip(expected[start:start+8],begins)])
        prefix=histories[0][:20]
        next_action=int(arrays['actions'][20]);legal=expected[0]['legal'][20]
        alternative=next(int(x) for x in np.flatnonzero(legal) if x!=next_action)
        branches=[prefix+[next_action],prefix+[alternative],prefix[:10],[]]
        exact=offline(original,branches,19,data);equal(replay(branches),exact)
        starts=[len(h) for h in branches]
        equal(replay(branches,starts),[{key:value[-1:] for key,value in row.items()} for row in exact])
        # Native application errors leave the framed stream usable.
        for invalid in ([[0,0]],[[361,361]],[[361,361,0]]):
            reject(lambda:replay(invalid),'Native history rejected')
            equal(replay([[]]),empty)
        for invalid in ([[362]],[[-1]],[[True]],[[0.0]],[list(range(2048))],[],[[]]*129):
            reject(lambda:replay(invalid))
        reject(lambda:replay([[0]],[2]))
        reject(lambda:replay([[0]],[-1]))
        reject(lambda:replay([[0]],[False]))
        reject(lambda:replay([[0]*1024]*128),'Output exceeds bound')
        bad=dict(version=1,id=9999,size=19,komi=7.5,histories=[[]],starts=[0],unknown=True)
        raw=json.dumps(bad).encode();replay.process.stdin.write(struct.pack('<I',len(raw))+raw)
        deadline=time.monotonic()+5;n=struct.unpack('<I',replay._read(4,deadline))[0]
        response=json.loads(replay._read(n,deadline));assert response['status']=='error'
        equal(replay([[]]),empty)
    for size in (3,9):
        hs=[[],[0],[size*size], [0,size*size,1]]
        with Replay(binary,build['binary_sha256'],size=size) as replay:
            equal(replay(hs),offline(original,hs,size,data))
    for payload in (b'\x01',struct.pack('<I',10)+b'{}',struct.pack('<I',MAX_REQUEST+1)):
        result=subprocess.run([str(binary)],input=payload,capture_output=True,timeout=5)
        assert result.returncode==1
    report=dict(kind='native_v7_inference_qualification',status='passed',created_unix=time.time(),seconds=time.time()-started,
                build_receipt_sha256=sha(a.build_receipt),binary_sha256=sha(binary),offline_binary_sha256=sha(original),
                feature_qualification_receipt_sha256=sha(feature_receipt),feature_manifest_sha256=feature['manifest_sha256'],
                complete_real_training_games=len(histories),positions=sum(len(h)+1 for h in histories),calls=calls,
                all_arrays_exact=True,board_sizes=[3,9,19],test_targets_decoded=False,
                checks=['Complete and suffix output matches packed independently qualified V7 features',
                        'Alternative legal branches and reversed/shortened histories match offline C++ replay',
                        'Native rejects illegal and terminal histories and then accepts a fresh valid request',
                        'Input types, action/pass bounds, context, output size and suffix bounds',
                        'Unknown fields and truncated/oversized framing rejected'],
                source_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()},
                scope='Feature interface only; no model, MCTS, cache, TPU or Go-strength qualification')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps({k:report[k] for k in ('status','complete_real_training_games','positions','seconds')}),flush=True)


if __name__=='__main__':main()
