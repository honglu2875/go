"""Reconstruct complete strong-teacher histories into lossless RAM arrays."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE/'packages/gozero/src'))
from gozero.corpus_format import KIND, SPLITS, array_specs
from gozero.snapshots import canonical_json, verify


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def capacity(config, path, additional=0):
    fs=os.statvfs(path)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    if (fs.f_bavail*fs.f_frsize < config['minimum_free_shm_gib']*(1<<30)+additional
            or available < config['minimum_memory_available_gib']*(1<<30)):
        raise ValueError('Corpus preparation RAM headroom exhausted')


def load_game(release, row):
    with (release/row['shard']).open('rb') as stream:
        stream.seek(row['offset']);raw=stream.read(row['bytes'])
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:
        raise ValueError('Source record differs: '+row['game_id'])
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        data={name:archive[name] for name in ('actions','stones','legal','raw_policy','raw_value')}
    n=row['rows']
    if (data['raw_policy'].dtype!=np.float32 or data['raw_value'].dtype!=np.float32
            or data['raw_value'].shape!=(n,) or data['actions'].shape!=(n,)):
        raise ValueError('Target precision or row count changed')
    return data


def convert_shard(config, release, binary, records, trajectories, output, index):
    size=config['size'];area=size*size;n=sum(r['rows'] for r in records)
    directory=output/f'shard-{index:05d}';directory.mkdir()
    arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,(shape,dtype) in array_specs(size,n,len(records)).items()}
    offsets=np.concatenate(([0],np.cumsum([r['rows'] for r in records],dtype=np.int64)))
    arrays['expert_offsets'][:]=offsets
    for game,row in enumerate(records):
        arrays['games'][game]=(row['game_id'],row['opening_family'],trajectories[row['game_id']],
            SPLITS[row['split']],row['opponent_index'],row['expert_color'],row['rows'])
    for begin in range(0,len(records),config['chunk_games']):
        chunk=records[begin:begin+config['chunk_games']]
        capacity(config,output,256*(1<<20))
        originals=[load_game(release,row) for row in chunk]
        count=sum(r['rows'] for r in chunk)
        with tempfile.TemporaryDirectory(prefix='.features-',dir=output) as temporary:
            tmp=Path(temporary)
            with (tmp/'games.jsonl').open('x') as f:
                for data in originals:
                    f.write(json.dumps(dict(size=size,komi=config['komi'],actions=data['actions'].tolist()))+'\n')
            completed=subprocess.run([str(binary),str(tmp/'games.jsonl'),str(tmp/'spatial.u8'),
                str(tmp/'global.f32'),str(tmp/'audit.u8')],capture_output=True,text=True,check=True,timeout=180)
            result=json.loads(completed.stdout)
            if result['positions']!=count or result['games']!=len(chunk):
                raise ValueError('Incomplete native feature replay')
            spatial=np.fromfile(tmp/'spatial.u8',np.uint8).reshape(count,area*22)
            glob=np.fromfile(tmp/'global.f32','<f4').reshape(count,19)
            audit=np.fromfile(tmp/'audit.u8',np.uint8).reshape(count,2*area+1)
            boards=np.concatenate([d['stones'].reshape(-1,area) for d in originals])
            legal=np.concatenate([d['legal'] for d in originals])
            if not np.array_equal(audit[:,:area],boards) or not np.array_equal(audit[:,area:].astype(bool),legal):
                raise ValueError('KataGo/stored board or legal-mask disagreement')
            if np.any(spatial>1) or not np.all(spatial[:,::22]==1) or not np.isfinite(glob).all():
                raise ValueError('Invalid neural input features')
            start,end=map(int,offsets[[begin,begin+len(chunk)]])
            arrays['spatial'][start:end]=np.packbits(spatial,axis=-1,bitorder='little')
            arrays['legal'][start:end]=np.packbits(legal,axis=-1,bitorder='little')
            arrays['global_features'][start:end]=glob
            for target,source in (('actions','actions'),('policies','raw_policy'),('values','raw_value')):
                data=np.concatenate([d[source] for d in originals])
                arrays[target][start:end]=data
                if not np.array_equal(arrays[target][start:end],data):
                    raise ValueError('Target conversion is not exact')
    files={}
    for name,array in arrays.items():
        array.flush()
        path=directory/(name+'.npy');path.chmod(0o444)
        files[name]=dict(path=str(path.relative_to(output)),bytes=path.stat().st_size,
                         sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype))
    summary=dict(id=index,games=len(records),positions=n,files=files,
        source_shard=records[0]['shard'],source_game_ids_sha256=hashlib.sha256(canonical_json([r['game_id'] for r in records])).hexdigest(),
        all_boards_equal=True,all_legal_masks_equal=True,targets_changed=False)
    (directory/'receipt.json').write_bytes(canonical_json(summary));(directory/'receipt.json').chmod(0o444)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-shards',type=int,default=20)
    a=p.parse_args();verify(SOURCE);config=json.loads(a.config.read_text());started=time.time()
    if config['kind']!='offline_corpus_transform' or not 1<=a.first_shards<=20 or config['splits']!=['train','validation']:
        raise ValueError('Unexpected bounded corpus transform')
    output=a.output.absolute()
    if not output.is_relative_to('/dev/shm/gozero-datasets') or output.is_symlink():
        raise ValueError('Derived corpus must be in its explicit RAM namespace')
    release=Path(config['release']);manifest=json.loads((release/'manifest.json').read_text())
    if manifest['release_id']!=config['release_id'] or sha(release/'manifest.json')!=config['release_manifest_sha256']:
        raise ValueError('Release identity differs')
    worker_path=Path(config['worker_receipt']);worker=json.loads(worker_path.read_text());binary=worker_path.parent/'feature_worker'
    if sha(worker_path)!=config['worker_receipt_sha256'] or sha(binary)!=worker['binary_sha256']:
        raise ValueError('Unpinned native feature adapter')
    qualification=json.loads(Path(config['feature_qualification']).read_text())
    if (sha(Path(config['feature_qualification']))!=config['feature_qualification_sha256']
            or qualification['status']!='passed' or qualification['binary_sha256']!=worker['binary_sha256']
            or qualification['release_id']!=config['release_id']):
        raise ValueError('Missing source feature qualification')
    trajectory_root=Path(config['trajectory_audit']);trajectory_report=json.loads((trajectory_root/'result.json').read_text())
    if (sha(trajectory_root/'result.json')!=config['trajectory_audit_sha256']
            or trajectory_report['status']!='passed' or sha(trajectory_root/'identities.jsonl.gz')!=trajectory_report['identities_sha256']):
        raise ValueError('Trajectory structural audit changed')
    trajectories={}
    with gzip.open(trajectory_root/'identities.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);trajectories[row['game_id']]=row['trajectory_sha256']
    grouped=defaultdict(list)
    for item in manifest['indices']:
        if sha(release/item['path'])!=item['sha256']:
            raise ValueError('Source index changed')
        with gzip.open(release/item['path'],'rt') as f:
            for line in f:
                row=json.loads(line)
                if row['split'] in config['splits']:
                    if row['game_id'] not in trajectories:
                        raise ValueError('Unaudited source game')
                    grouped[row['shard']].append(row)
    chosen=[grouped[key] for key in sorted(grouped)[:a.first_shards]]
    estimated=sum(sum(np.prod(shape)*dtype.itemsize for shape,dtype in array_specs(config['size'],sum(r['rows'] for r in rows),len(rows)).values())
                  for rows in chosen)
    if estimated>16*(1<<30):
        raise ValueError('Transform exceeds its 16-GiB output budget')
    output.parent.mkdir(parents=True,exist_ok=True);capacity(config,output.parent,int(estimated)+(1<<30))
    output.mkdir(exist_ok=False)
    (output/'operator.json').write_bytes(canonical_json(dict(snapshot=SOURCE.name,config=config,
        config_sha256=sha(a.config),estimated_array_bytes=int(estimated),started_unix=started)))
    entries=[]
    for index,rows in enumerate(chosen):
        entry=convert_shard(config,release,binary,rows,trajectories,output,index);entries.append(entry)
        progress=dict(shards=len(entries),games=sum(x['games'] for x in entries),positions=sum(x['positions'] for x in entries),elapsed_seconds=time.time()-started)
        temporary=output/'.progress.json.tmp';temporary.write_bytes(canonical_json(progress));temporary.replace(output/'progress.json')
        print(json.dumps(progress),flush=True)
    populations={split:dict(games=sum(r['split']==split for rows in chosen for r in rows),
        positions=sum(r['rows'] for rows in chosen for r in rows if r['split']==split)) for split in config['splits']}
    result=dict(schema_version=1,kind=KIND,size=config['size'],komi=config['komi'],
        max_game_moves=max(r['rows'] for rows in chosen for r in rows),shards=entries,
        source_release=dict(id=config['release_id'],repo='quintic/go9x9',revision='4f579b0a21c456f3bb568174d384056351124cd5',manifest_sha256=sha(release/'manifest.json')),
        source_qualification_sha256=sha(Path(config['feature_qualification'])),source_trajectory_audit_sha256=sha(trajectory_root/'result.json'),
        feature_worker_receipt_sha256=sha(worker_path),feature_worker_binary_sha256=worker['binary_sha256'],
        complete_train_validation=a.first_shards==len(grouped),populations=populations,
        spatial_channels=22,global_channels=19,feature_version=7,packing='little-endian bits; each position independently padded',
        policy_target='fixed teacher raw_policy; every pre-action position; float32 bytes preserved',
        value_target='fixed teacher raw_value; player-to-move perspective; float32 bytes preserved',
        target_teacher_sha256='a1298ce1adc1dad7bd868ca962b2384cc8388ed373a00e6bae1114fa6f9e2d61',
        behavior='Archived actions from mixed-strength games, never substituted for policy targets',
        targets_changed=False,test_arrays_included=False,operator_snapshot=SOURCE.name,
        started_unix=started,finished_unix=time.time())
    (output/'manifest.json').write_bytes(canonical_json(result));(output/'manifest.json').chmod(0o444)
    verify(SOURCE)
    print(json.dumps(dict(status='passed',output=str(output),manifest_sha256=sha(output/'manifest.json'),populations=populations,bytes=sum(v['bytes'] for x in entries for v in x['files'].values()))),flush=True)


if __name__=='__main__':main()
