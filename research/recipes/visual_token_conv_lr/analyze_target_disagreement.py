#!/usr/bin/env python3
"""Training-only target ambiguity under exact CNN inputs and exact move prefixes.

Dictionary keys contain actual input bytes, so hash collisions are resolved by
byte equality. This is an empirical finite-training-set diagnostic under uniform
position weighting, not a population Bayes-risk estimate or a validation metric.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(SOURCE);started=time.time()
    if sha256(args.dataset/'manifest.json')!=args.manifest_sha256: raise ValueError('Input dataset differs')
    overlay=read_json(args.dataset/'manifest.json');base=Path(overlay['parent_dataset']['path'])
    if sha256(base/'manifest.json')!=overlay['parent_dataset']['manifest_sha256']: raise ValueError('Parent dataset differs')
    parent=read_json(base/'manifest.json');maps={'cnn_v7_inputs':{},'complete_action_prefix':{}}
    positions=0;episodes=0;phase_counts=np.zeros(5,np.int64);phase_bounds=(16,64,128,256,2048)
    def add(table,key,pi,entropy):
        item=table.get(key)
        if item is None: table[key]=[1,pi.copy(),entropy]
        else: item[0]+=1;item[1]+=pi;item[2]+=entropy
    for raw,feature in zip(parent['shards'],overlay['shards']):
        if raw['id']!=feature['id']: raise ValueError('Shard order differs')
        for directory,entry in ((base,raw),(args.dataset,feature)):
            if sha256(directory/entry['arrays'])!=entry['sha256']: raise ValueError('Shard bytes differ')
        with np.load(base/raw['arrays'],allow_pickle=False) as f:
            offsets=f['expert_offsets'];splits=f['expert_splits'];actions=f['expert_actions'];policies=f['expert_policies']
            eligible=f['expert_training_eligible'] if 'expert_training_eligible' in f else np.ones_like(splits,bool)
        with np.load(args.dataset/feature['arrays'],allow_pickle=False) as f:
            bits=np.packbits(f['spatial'].reshape(len(actions),-1),axis=-1,bitorder='little');glob=f['global_features']
        for episode in np.flatnonzero((splits==0)&eligible):
            begin,end=map(int,offsets[episode:episode+2]);prefix=b'';episodes+=1
            for t,index in enumerate(range(begin,end)):
                pi=policies[index].astype(np.float64)
                entropy=float(-np.sum(pi*np.log(np.maximum(pi,1e-300))))
                key=bits[index].tobytes()+glob[index].astype('<f4',copy=False).tobytes()
                add(maps['cnn_v7_inputs'],key,pi,entropy)
                add(maps['complete_action_prefix'],prefix,pi,entropy)
                prefix+=int(actions[index]).to_bytes(2,'little')
                positions+=1;phase_counts[np.searchsorted(phase_bounds,t,side='right')]+=1
        print(json.dumps({'shard':raw['id'],'positions':positions,'distinct_inputs':{k:len(v) for k,v in maps.items()}}),flush=True)
    result={'kind':'training_input_target_disagreement','operator_snapshot':SOURCE.name,'dataset_manifest_sha256':args.manifest_sha256,
        'split':'train only','training_positions':positions,'training_episodes':episodes,
        'phase_upper_bounds_exclusive':list(phase_bounds),'phase_counts':phase_counts.tolist(),
        'weighting':'Uniform stored training positions; differs from per-bucket batch-loss weighting.',
        'interpretation':'KL of each stored policy to the optimal mean target for identical available inputs. Unique inputs have empirical zero ambiguity, not proven population predictability.',
        'representations':{}}
    for name,table in maps.items():
        repeated_groups=0;repeated_positions=0;cost=0.;max_group=0;different_groups=0
        for count,total,entropy_sum in table.values():
            max_group=max(max_group,count)
            if count<2: continue
            mean=total/count
            gap=max(0.,count*float(-np.sum(mean*np.log(np.maximum(mean,1e-300))))-entropy_sum)
            cost+=gap;repeated_groups+=1;repeated_positions+=count;different_groups+=gap>1e-10
        result['representations'][name]={'distinct_inputs':len(table),'repeated_input_groups':repeated_groups,
            'positions_in_repeated_groups':repeated_positions,'groups_with_disagreeing_targets':different_groups,
            'largest_group':max_group,'mean_empirical_unavoidable_kl':cost/positions,
            'mean_kl_within_repeated_groups':cost/max(1,repeated_positions)}
    result['elapsed_seconds']=time.time()-started;result['status']='passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444);print(json.dumps({'status':'passed','sha256':sha256(args.output),'representations':result['representations']}),flush=True)
if __name__=='__main__':main()
