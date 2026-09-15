"""Replay pinned training games and exercise the existing packed loader at 19x19."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path('/workspace/go')
STUDY = Path(__file__).resolve().parent
SNAPSHOT = 'ed15249055a11bac5b86806505d0d14ae75a4aee2ea327e86b6a422e49a7259c'
sys.path.insert(0,str(ROOT/'.gozero/snapshots'/SNAPSHOT/'packages/gozero/src'))
from gozero.snapshots import verify, canonical_json
from gozero.corpus_format import KIND, SPLITS, array_specs
from gozero.corpus_sequence_batches import Dataset, augment
from gozero.sequence_symmetry import action_map


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def features(binary, games, parent):
    count=sum(len(g['actions']) for g in games)
    with tempfile.TemporaryDirectory(prefix='.v7-',dir=parent) as name:
        tmp=Path(name)
        (tmp/'games.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in games))
        result=subprocess.run([str(binary),str(tmp/'games.jsonl'),str(tmp/'spatial.u8'),
            str(tmp/'global.f32'),str(tmp/'audit.u8')],capture_output=True,text=True,check=True,timeout=180)
        assert json.loads(result.stdout)==dict(games=len(games),positions=count,status='passed')
        return (np.fromfile(tmp/'spatial.u8',np.uint8).reshape(count,19,19,22),
                np.fromfile(tmp/'global.f32','<f4').reshape(count,19),
                np.fromfile(tmp/'audit.u8',np.uint8).reshape(count,723))


def main():
    started=time.time();verify(ROOT/'.gozero/snapshots'/SNAPSHOT)
    inventory_path=STUDY/'inventory-001.json';inventory=json.loads(inventory_path.read_text())
    records=inventory['feature_qualification_selection'];assert len(records)==30
    assert all(r['split']=='train' and r['terminal'] for r in records)
    target=Path('/dev/shm/gozero-datasets/strong19-feature-qualification-001')
    fs=os.statvfs('/dev/shm')
    assert fs.f_bavail*fs.f_frsize>66*(1<<30)
    target.mkdir(exist_ok=False);inputs=target/'raw-inputs';inputs.mkdir()
    worker=ROOT/'.gozero/build/katago-v7-12e944b6/receipt.json'
    assert sha(worker)=='9e7d8d4757c4214e25ea34a8ff3cbe57e1817abd7daf535117e0850eb0a6bd07'
    binary=worker.parent/'feature_worker';assert sha(binary)==json.loads(worker.read_text())['binary_sha256']
    originals=[]
    for row in records:
        source=Path(row['path']);assert source.is_relative_to('/dev/shm/go-corpus19/corpora')
        assert source.name==row['game_id']+'.npz'
        dest=inputs/source.name
        if row['host']==0:raw=source.read_bytes()
        else:
            raw=subprocess.check_output(['ssh','-F','/dev/null','-o','BatchMode=yes',
                f"go-user@worker-{row['host']}",'cat -- '+shlex.quote(str(source))],timeout=30)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        with dest.open('xb') as f:f.write(raw)
        dest.chmod(0o444)
        with np.load(dest,allow_pickle=False) as a:
            originals.append({name:a[name] for name in ('actions','stones','legal','raw_policy','raw_value')})
    games=[dict(size=19,komi=7.5,actions=d['actions'].tolist()) for d in originals]
    spatial,glob,audit=features(binary,games,target)
    count=len(spatial);boards=np.concatenate([d['stones'].reshape(-1,361) for d in originals])
    legal=np.concatenate([d['legal'] for d in originals])
    assert np.array_equal(audit[:,:361],boards) and np.array_equal(audit[:,361:].astype(bool),legal)
    assert not np.any(spatial>1) and np.all(spatial[...,0]==1) and np.isfinite(glob).all()
    directory=target/'shard-00000';directory.mkdir()
    arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',shape=shape,dtype=dtype)
            for name,(shape,dtype) in array_specs(19,count,len(records)).items()}
    arrays['spatial'][:]=np.packbits(spatial.reshape(count,-1),axis=1,bitorder='little')
    arrays['legal'][:]=np.packbits(legal,axis=1,bitorder='little')
    arrays['global_features'][:]=glob
    offsets=np.concatenate(([0],np.cumsum([r['rows'] for r in records],dtype=np.int64)))
    arrays['expert_offsets'][:]=offsets
    for name,original in (('actions','actions'),('policies','raw_policy'),('values','raw_value')):
        arrays[name][:]=np.concatenate([d[original] for d in originals])
    for i,(r,d) in enumerate(zip(records,originals)):
        # An int32 encoding is essential: pass=361 and many points exceed 255.
        trajectory=hashlib.sha256(min(action_map(19,s)[d['actions']].astype('<i4').tobytes() for s in range(8))).hexdigest()
        arrays['games'][i]=(r['game_id'],r['opening_family'],trajectory,SPLITS[r['split']],
                            r['opponent_index'],r['expert_color'],r['rows'])
    files={}
    for name,array in arrays.items():
        array.flush();path=directory/(name+'.npy');path.chmod(0o444)
        files[name]=dict(path=str(path.relative_to(target)),bytes=path.stat().st_size,sha256=sha(path),
                         shape=list(array.shape),dtype=str(array.dtype))
    manifest=dict(schema_version=1,kind=KIND,size=19,komi=7.5,max_game_moves=max(r['rows'] for r in records),
        shards=[dict(id=0,games=len(records),positions=count,files=files,all_boards_equal=True,
                     all_legal_masks_equal=True,targets_changed=False)],
        populations=dict(train=dict(games=len(records),positions=count),validation=dict(games=0,positions=0)),
        feature_version=7,spatial_channels=22,global_channels=19,targets_changed=False,
        test_arrays_included=False,complete_train_validation=False,
        qualification_only=True,inventory_sha256=sha(inventory_path),library_snapshot=SNAPSHOT,
        feature_worker_binary_sha256=sha(binary),source_game_sha256={r['game_id']:r['sha256'] for r in records},
        target_teacher_sha256=records[0]['teacher_sha256'],operator_sha256=sha(Path(__file__)))
    (target/'manifest.json').write_bytes(canonical_json(manifest));(target/'manifest.json').chmod(0o444)
    dataset=Dataset(target,sha(target/'manifest.json'),allow_partial=True)
    for i,(r,d) in enumerate(zip(records,originals)):
        batch=dataset.batch([('expert',0,i)],positions=r['rows'])
        start,end=map(int,offsets[i:i+2])
        assert np.array_equal(batch['spatial'][0],spatial[start:end])
        assert np.array_equal(batch['global_features'][0],glob[start:end])
        for name,key in (('actions','actions'),('policies','raw_policy'),('values','raw_value'),('legal','legal')):
            assert np.array_equal(batch[name][0],d[key]),name
    # All eight symmetries of two complete causal histories go back through
    # KataGo's C++ feature code, independently of Python's grid transforms.
    checks=[]
    for i in (0,len(records)-1):
        row=records[i];batch=dataset.batch([('expert',0,i)],positions=row['rows'])
        for code in range(8):
            changed=augment(batch,np.asarray([code],np.int32))
            x,g,state=features(binary,[dict(size=19,komi=7.5,actions=changed['actions'][0].tolist())],target)
            assert np.array_equal(x,changed['spatial'][0]),('spatial equivariance',i,code)
            assert np.array_equal(g,changed['global_features'][0]),('global equivariance',i,code)
            assert np.array_equal(state[:,361:].astype(bool),changed['legal'][0]),('legal equivariance',i,code)
            assert np.array_equal(changed['values'],batch['values'])
            assert np.array_equal(changed['policies'][0,:,-1],batch['policies'][0,:,-1])
            checks.append(dict(game=row['game_id'],symmetry=code,positions=row['rows']))
    result=dict(status='passed',games=len(records),positions=count,source_inventory_sha256=sha(inventory_path),
        target=str(target),manifest_sha256=sha(target/'manifest.json'),library_snapshot=SNAPSHOT,
        feature_worker_binary_sha256=sha(binary),packed_array_bytes=sum(r['bytes'] for r in files.values()),
        all_boards_and_legal_masks_equal=True,all_loaded_inputs_and_targets_exact=True,
        d4_native_equivariance=checks,stored_value_perspective='player to move, signed [-1,1]',
        largest_action=int(arrays['actions'].max()),pass_action=361,
        qualification_only=True,test_targets_read=False,elapsed_seconds=time.time()-started,
        operator_sha256=sha(Path(__file__)),completed=time.time())
    with (STUDY/'feature-loader-qualification-001.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    (STUDY/'feature-loader-qualification-001.json').chmod(0o444)
    print(json.dumps({k:result[k] for k in ('status','games','positions','packed_array_bytes','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
