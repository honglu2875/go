#!/usr/bin/env python3
"""Independently reconstruct every episode/symmetry draw and checkpoint identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero import checkpoints
from gozero.katago_sequence_batches import Dataset

def digest(x): return hashlib.sha256(canonical_json(x)).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE)
    attempt=args.attempt.resolve();root=args.workspace_root.resolve()
    closed=read_json(attempt/'result.json')
    if closed['status']!='passed' or closed.get('resume_attempt'):
        raise ValueError('Expected a successful uninterrupted fixed-policy attempt')
    snapshot=root/'.gozero/snapshots'/closed['snapshot_id'];verify(snapshot)
    c=read_json(snapshot/'resolved_config.json')
    if c['kind']!='fixed_policy_learning': raise ValueError('Not a single-policy run')
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);buckets=c['dataset']['buckets']
    by_bucket=data.bucket_entries(buckets);reports=[];logs=[];files={}
    for host in range(c['expected_processes']):
        folder=attempt/f'rank-{host}/artifacts'
        for name in ('result.json','metrics.jsonl'):
            files[str((folder/name).relative_to(root))]=checkpoints.sha256(folder/name)
        report=read_json(folder/'result.json');log=[json.loads(row) for row in (folder/'metrics.jsonl').read_text().splitlines()]
        if report['status']!='passed' or report['turn']!=c['steps'] or len(log)!=c['steps']:
            raise ValueError('Incomplete learner')
        if report['config_sha256']!=digest(c) or report['dataset_manifest_sha256']!=c['dataset']['manifest_sha256']:
            raise ValueError('Learner provenance differs')
        reports.append(report);logs.append(log)
    expected_positions=np.zeros(c['steps'],np.int64)
    for host,(report,log) in enumerate(zip(reports,logs)):
        rank=report['jax_rank'];random=np.random.Generator(np.random.PCG64(c['seed']+1+104729*rank))
        bucket_random=np.random.Generator(np.random.PCG64(c['seed']+9143))
        aug=np.random.Generator(np.random.PCG64(c['seed']+400003+104729*rank))
        for i,row in enumerate(log):
            bucket=c['dataset']['warmup_buckets'][i] if i<len(c['dataset']['warmup_buckets']) else int(bucket_random.choice(buckets,p=c['dataset']['bucket_probabilities']))
            entries=[by_bucket['expert',bucket][int(x)] for x in random.integers(len(by_bucket['expert',bucket]),size=c['learner']['games_per_host'])]
            symmetry=aug.integers(0,8,len(entries)).tolist() if c['learner']['augmentation']=='d4' else [0]*len(entries)
            if row['turn']!=i+1 or row['bucket']!=bucket or row['local_entries_sha256']!=digest(entries) or row['local_symmetries']!=symmetry:
                raise ValueError('Episode/augmentation draw changed')
            for _,shard,episode in entries:
                off=data.shards[shard]['expert_offsets']; expected_positions[i]+=int(off[episode+1]-off[episode])
            if not row['accepted'] or any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):
                raise ValueError('Rejected/nonfinite update')
        saved=read_json(Path(report['latest_checkpoint']['path'])/'state.json')
        for name,expected in [('numpy_rng',random.bit_generator.state),('bucket_rng',bucket_random.bit_generator.state),('augmentation_rng',aug.bit_generator.state)]:
            if saved[name]!=expected: raise ValueError('Checkpoint sampling state differs')
    local_fields={'local_entries_sha256','local_symmetries','cumulative_learning_seconds','cumulative_sampling_seconds'}
    for i in range(c['steps']):
        master={k:v for k,v in logs[0][i].items() if k not in local_fields}
        if int(master['expert_positions'])!=int(expected_positions[i]): raise ValueError('Global exposure differs')
        for log in logs[1:]:
            if {k:v for k,v in log[i].items() if k not in local_fields}!=master: raise ValueError('Replicated update metrics differ')
    group_paths=[Path(r['latest_checkpoint']['path']).with_suffix('.group.json') for r in reports]
    identities=[checkpoints.sha256(x) for x in group_paths]
    if len(set(identities))!=1: raise ValueError('Checkpoint group differs across ranks')
    group=read_json(group_paths[0])
    for host,report in enumerate(reports):
        path=Path(report['latest_checkpoint']['path'])
        manifest=read_json(path/'manifest.json')
        if checkpoints.sha256(path/'manifest.json')!=group['host_manifests'][str(host)]: raise ValueError('Checkpoint manifest differs')
        for name,item in manifest['files'].items():
            if checkpoints.sha256(path/name)!=item['sha256']: raise ValueError('Checkpoint bytes differ')
    owner=Path(group['owner_checkpoint_path']);_,arrays,_=checkpoints.read(owner)
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=arrays[key]
        if not np.isfinite(a).all(): raise ValueError('Nonfinite saved model/optimizer')
        h.update(canonical_json([key,list(a.shape),str(a.dtype)]));h.update(a.tobytes(order='C'))
    if h.hexdigest()!=group['replicated_arrays_elements_sha256']: raise ValueError('Replicated array content differs')
    report={'kind':'single_policy_learning_audit','operator_snapshot':SOURCE.name,'status':'passed',
        'attempt':attempt.name,'closed_result_sha256':checkpoints.sha256(attempt/'result.json'),'training_snapshot':snapshot.name,
        'dataset_manifest_sha256':c['dataset']['manifest_sha256'],'steps':c['steps'],'expert_positions':int(expected_positions.sum()),
        'draws_reconstructed':c['steps']*c['expected_processes']*c['learner']['games_per_host'],
        'all_rank_metrics_equal':True,'checkpoint_array_count':len(arrays),'checkpoint_group_sha256':identities[0],
        'checkpoint_arrays_elements_sha256':h.hexdigest(),'input_files':files,
        'validation_curve':[reports[0]['initial_validation'],*reports[0]['validation_history']],
        'timing':reports[0]['segment_timing'],'reserved_chip_hours':closed['reserved_chip_hours'],
        'clipped_updates':sum(row['clip_scale']<1 for row in logs[0]),
        'scope':'Exact sampling/provenance/finite-update/checkpoint audit; no independent replay of model gradients.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:f.write(canonical_json(report))
    args.output.chmod(0o444)
    print(json.dumps({'status':'passed','positions':report['expert_positions'],'sha256':checkpoints.sha256(args.output)}),flush=True)
if __name__=='__main__':main()
