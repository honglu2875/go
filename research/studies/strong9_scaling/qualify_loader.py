"""Check packed batches directly against source records and feature semantics."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.corpus_sequence_batches import Dataset,augment
from gozero.sequence_symmetry import action_map,transform_grid


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.time();data=Dataset(a.dataset,a.manifest_sha256,allow_partial=True)
    selected=[]
    for split in (0,1):
        indices=data.indices['expert',split]
        ordered=sorted(indices,key=lambda x:(int(data.shards[x[0]]['games'][x[1]]['length']),x))
        chosen=[*ordered[:8],*ordered[-8:],*indices[::max(1,len(indices)//16)][:16]]
        selected.extend(('expert',s,e) for s,e in dict.fromkeys(chosen))
    selected_ids={bytes(data.game_info(entry)['game_id']).decode() for entry in selected}
    originals={}
    for path in sorted((a.release/'index').glob('host-*.jsonl.gz')):
        with gzip.open(path,'rt') as f:
            for line in f:
                row=json.loads(line)
                if row['game_id'] not in selected_ids:continue
                with (a.release/row['shard']).open('rb') as stream:
                    stream.seek(row['offset']);raw=stream.read(row['bytes'])
                if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Original changed')
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    originals[row['game_id']]={name:z[name] for name in ('actions','legal','raw_policy','raw_value','stones')}
    weights=data.evaluation_family_weights(selected)
    batch=data.batch([*selected,None],positions=256,family_weights=weights)
    for i,entry in enumerate(selected):
        gid=bytes(data.game_info(entry)['game_id']).decode();source=originals[gid];n=len(source['actions'])
        if batch['counts'][i]!=n:raise ValueError('Length changed')
        for field,raw in (('actions','actions'),('legal','legal'),('policies','raw_policy'),('values','raw_value')):
            if not np.array_equal(batch[field][i,:n],source[raw]):raise ValueError('Batch changed '+field)
        to_play=(1+np.arange(n)%2)[:,None,None]
        if (not np.array_equal(batch['spatial'][i,:n,:,:,1],source['stones']==to_play)
                or not np.array_equal(batch['spatial'][i,:n,:,:,2],source['stones']==3-to_play)):
            raise ValueError('Feature orientation or perspective changed')
    family_masses=Counter()
    for entry in selected:
        info=data.game_info(entry);family_masses[bytes(info['opening_family'])]+=weights[tuple(entry)]*int(info['length'])
    if any(abs(m-1)>1e-12 for m in family_masses.values()):raise ValueError('Family weights do not sum to equal mass')
    if batch['counts'][-1]!=0 or np.any(batch['spatial'][-1]) or np.any(batch['policies'][-1]):
        raise ValueError('Padding contains observations or targets')
    for code in range(8):
        transformed=augment(batch,np.full(len(selected)+1,code,np.int32))
        if not np.array_equal(transformed['values'],batch['values']):raise ValueError('Symmetry changed value targets')
        for i,entry in enumerate(selected):
            n=int(batch['counts'][i]);mapping=action_map(9,code)
            if not np.array_equal(transformed['actions'][i,:n],mapping[batch['actions'][i,:n]]):raise ValueError('Action symmetry differs')
            if not np.array_equal(transformed['spatial'][i,:n],transform_grid(batch['spatial'][i,:n],code,(1,2))):raise ValueError('Spatial symmetry differs')
            for name in ('policies','legal'):
                if not np.array_equal(transformed[name][i,:n,-1],batch[name][i,:n,-1]):raise ValueError('Symmetry moved pass')
                if not np.array_equal(transformed[name][i,:n][:,mapping],batch[name][i,:n]):
                    raise ValueError('Target action remapping differs')
    try:data.batch([max(selected,key=lambda e:int(data.game_info(e)['length']))],positions=1)
    except ValueError:pass
    else:raise ValueError('Truncation was allowed')
    out=dict(status='passed',dataset=str(a.dataset),manifest_sha256=a.manifest_sha256,games=len(selected),
        positions=int(batch['counts'].sum()),all_source_targets_exact=True,feature_side_to_move_exact=True,
        d4_symmetries_checked=8,family_weight_mass_exact=True,test_arrays_accessed=False,
        started_unix=started,finished_unix=time.time(),operator_sha256=sha(Path(__file__)),
        numerical_sources={name:sha(ROOT/'packages/gozero/src/gozero'/name) for name in ('corpus_format.py','corpus_sequence_batches.py','katago_sequence_batches.py','sequence_symmetry.py')})
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps(out))


if __name__=='__main__':main()
