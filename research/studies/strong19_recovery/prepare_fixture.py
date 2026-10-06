"""Create an explicitly non-scientific, reproducible recovery execution fixture.

Prefixes exercise both sequence buckets cheaply; these are NOT full-game
learning data. Test members are never opened. Production rejects this fixture.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.corpus_format import KIND,SPLITS,array_specs
from gozero.corpus_sequence_batches import Dataset
from gozero.durable_files import atomic_json,copy_verified,mkdir,sha256
from gozero.sequence_symmetry import action_map
from gozero.snapshots import freeze


def main():
    import numpy as np
    release=Path('/dev/shm/go9x9-release-v1/publish')
    selected={'train':[],'validation':[]}
    with gzip.open(release/'index/host-0.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);s=r['split']
            if s in selected and r['terminal'] and r['rows']>=24 and len(selected[s])<8:selected[s].append(r)
            if all(len(x)==8 for x in selected.values()):break
    rows=selected['train']+selected['validation'];assert len(rows)==16
    originals=[];selection=[]
    for i,r in enumerate(rows):
        n=16 if i%2==0 else 24
        with (release/r['shard']).open('rb') as stream:
            stream.seek(r['offset']);raw=stream.read(r['bytes'])
        assert hashlib.sha256(raw).hexdigest()==r['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as a:
            originals.append({k:a[k][:n] for k in ('actions','stones','legal','raw_policy','raw_value')})
        selection.append(dict(r,prefix_rows=n))
    target=Path('/dev/shm/gozero-datasets/recovery-fixture-20260925-001')
    target.mkdir(parents=True,exist_ok=False)
    worker=ROOT/'.gozero/build/katago-v7-12e944b6/feature_worker'
    assert sha256(worker)=='a380059106a3a7dd92f24e75805777f29855452785679273a7e80a3f1d84bab9'
    with tempfile.TemporaryDirectory(dir=target) as name:
        scratch=Path(name)
        (scratch/'games.jsonl').write_text(''.join(json.dumps(dict(size=9,komi=7.5,actions=d['actions'].tolist()))+'\n' for d in originals))
        subprocess.run([str(worker),str(scratch/'games.jsonl'),str(scratch/'spatial'),str(scratch/'global'),str(scratch/'audit')],check=True,timeout=60)
        spatial=np.fromfile(scratch/'spatial',np.uint8).reshape(-1,9,9,22)
        glob=np.fromfile(scratch/'global','<f4').reshape(-1,19)
        audit=np.fromfile(scratch/'audit',np.uint8).reshape(-1,163)
    boards=np.concatenate([d['stones'].reshape(-1,81) for d in originals]);legal=np.concatenate([d['legal'] for d in originals])
    np.testing.assert_array_equal(audit[:,:81],boards);np.testing.assert_array_equal(audit[:,81:].astype(bool),legal)
    count=len(spatial);folder=target/'shard-00000';folder.mkdir()
    arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',shape=s,dtype=t) for k,(s,t) in array_specs(9,count,16).items()}
    arrays['spatial'][:]=np.packbits(spatial.reshape(count,-1),axis=1,bitorder='little')
    arrays['legal'][:]=np.packbits(legal,axis=1,bitorder='little');arrays['global_features'][:]=glob
    arrays['expert_offsets'][:]=np.concatenate(([0],np.cumsum([len(d['actions']) for d in originals])))
    for k,old in [('actions','actions'),('policies','raw_policy'),('values','raw_value')]:arrays[k][:]=np.concatenate([d[old] for d in originals])
    for i,(r,d) in enumerate(zip(rows,originals)):
        trajectory=hashlib.sha256(min(action_map(9,s)[d['actions']].astype('<i4').tobytes() for s in range(8))).hexdigest()
        arrays['games'][i]=(r['game_id'],r['opening_family'],trajectory,SPLITS[r['split']],r['opponent_index'],r['expert_color'],len(d['actions']))
    files={}
    for k,a in arrays.items():
        a.flush();p=folder/(k+'.npy');p.chmod(0o444)
        files[k]=dict(path=str(p.relative_to(target)),sha256=sha256(p),bytes=p.stat().st_size,shape=list(a.shape),dtype=str(a.dtype))
    manifest=dict(schema_version=1,kind=KIND,size=9,max_game_moves=24,targets_changed=False,
        feature_version=7,spatial_channels=22,global_channels=19,test_arrays_included=False,
        complete_train_validation=True,complete_train_validation_scope='Complete declared qualification prefix view; not complete games.',
        qualification_only=True,source_release='go9x9:4f579b0a21c456f3bb568174d384056351124cd5',
        target_teacher_sha256=rows[0]['teacher_sha256'],selection=selection,
        populations={s:dict(games=8,positions=160) for s in selected},
        shards=[dict(id=0,games=16,positions=count,files=files,all_boards_equal=True,all_legal_masks_equal=True,targets_changed=False)])
    atomic_json(target/'manifest.json',manifest,replace=False);data=Dataset(target,sha256(target/'manifest.json'))
    durable=ROOT/'.gozero/recovery-fixture-20260925-001';mkdir(durable)
    for p in target.rglob('*'):
        if p.is_file():copy_verified(p,durable/p.relative_to(target),expected_sha256=sha256(p),expected_bytes=p.stat().st_size)
    snapshots={}
    for arm in ('flat','attention'):
        c=json.loads((ROOT/'research/studies/strong19_attention_pool'/f'{arm}-harness-config-001.json').read_text())
        c['dataset']=dict(path=str(target),manifest_sha256=sha256(target/'manifest.json'),buckets=[16,24],bucket_probabilities=[.5,.5],warmup_buckets=[16,24])
        c['model']['max_positions']=32;c['training']['chunk_frames']=4
        c['checkpoint_disk']=dict(peer=None,root=str(ROOT/'.gozero/checkpoint-recovery-qualification'),keep=2,floor_bytes=1<<30,peer_floor_bytes=1<<30)
        cfg=STUDY/f'{arm}-harness-config-001.json';atomic_json(cfg,c,replace=False)
        snap=freeze(ROOT,ROOT/'research/recipes/strong19_recovery',cfg,ROOT/'.gozero/snapshots')
        snapshots[arm]=snap.name
    plan=dict(status='prepared',snapshots=snapshots,fixture=str(target),durable_fixture=str(durable),
        fixture_manifest_sha256=sha256(target/'manifest.json'),operator_sha256=sha256(Path(__file__)),
        source_fixture_scope='16 real non-test game prefixes, 320 positions, 9x9; execution/recovery only.')
    atomic_json(STUDY/'harness-plan-001.json',plan,replace=False)
    print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
