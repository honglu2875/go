"""Audit both registered architecture arms, all draws, metrics and checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,read_json,verify
from gozero.visual_sequence_batches import Dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cnn',required=True);p.add_argument('--transformer',required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
    if checkpoints.sha256(a.registration)!=a.registration_sha256 or a.output.exists():raise ValueError('Registration or output changed')
    root=SOURCE.parents[2];registration=read_json(a.registration);evidence={};arms={};draws={};configs={}
    def remember(path):
        evidence[str(path.relative_to(root))]=checkpoints.sha256(path);return read_json(path)
    for name,attempt in [('cnn',a.cnn),('transformer',a.transformer)]:
        d=root/'runs'/attempt;closed=remember(d/'result.json');pin=registration['arms'][name]
        if closed['status']!='passed' or closed['snapshot_id']!=pin['snapshot']:raise ValueError('Architecture run not complete')
        source=root/'.gozero/snapshots'/pin['snapshot'];verify(source);c=read_json(source/'resolved_config.json');configs[name]=c
        data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);by=data.bucket_entries(c['dataset']['buckets'])
        reports=[];local_draws={}
        for host in range(4):
            r=remember(d/f'rank-{host}/artifacts/result.json');reports.append(r)
            if r['status']!='passed' or not r['training_complete'] or r['turn']!=128 or r['parameter_count']!=pin['parameters']:raise ValueError('Incomplete rank or wrong architecture')
            rank=r['jax_rank'];random=np.random.Generator(np.random.PCG64(c['seed']+1+104729*rank))
            buckets=np.random.Generator(np.random.PCG64(c['seed']+9143))
            symmetry=np.random.Generator(np.random.PCG64(c['seed']+400003+104729*rank))
            path=d/f'rank-{host}/artifacts/metrics.jsonl';evidence[str(path.relative_to(root))]=checkpoints.sha256(path)
            records=[json.loads(line) for line in path.read_text().splitlines()]
            if len(records)!=128:raise ValueError('Missing updates')
            tape=[];counts={'expert':0,'behavior':0}
            for index,row in enumerate(records):
                warm=c['dataset']['warmup_buckets'];bucket=warm[index] if index<len(warm) else int(buckets.choice(c['dataset']['buckets'],p=c['dataset']['bucket_probabilities']))
                entries=[by[role,bucket][int(i)] for role in ('expert','behavior') for i in random.integers(len(by[role,bucket]),size=c['learner']['games_per_role'])]
                sym=symmetry.integers(0,8,len(entries)).tolist();digest=hashlib.sha256(canonical_json(entries)).hexdigest()
                if row['turn']!=index+1 or row['bucket']!=bucket or row['local_entries_sha256']!=digest or row['local_symmetries']!=sym or row['accepted']!=1.:raise ValueError('Episode, symmetry or update differs')
                if any(not math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):raise ValueError('Nonfinite recorded metric')
                for role in counts:counts[role]+=int(row[role+'_positions'])
                tape.append([bucket,digest,sym,row['expert_positions'],row['behavior_positions']])
            for role in counts:
                if counts[role]!=r['counters'][role+'_positions']:raise ValueError('Exposure counters differ')
            local_draws[rank]=tape
            for held in [r['initial_validation'],*r['validation_history'],r['test']]:
                raw=held['raw_totals'];m=held['metrics']
                expected={'expert_kl':(raw['expert_ce']-raw['expert_target_entropy'])/raw['expert_count'],
                          'behavior_ce':raw['behavior_ce']/raw['behavior_count'],'value_mse':raw['value_mse']/raw['value_count']}
                if any(not math.isclose(m[k],v,rel_tol=1e-10,abs_tol=1e-10) for k,v in expected.items()):raise ValueError('Held-out metric arithmetic differs')
        if len({r['latest_checkpoint']['group_sha256'] for r in reports})!=1:raise ValueError('Replicated owner checkpoint differs')
        if any(r['validation_history']!=reports[0]['validation_history'] or r['test']!=reports[0]['test'] for r in reports):raise ValueError('Collective metrics differ by rank')
        r=reports[0];cp=Path(r['latest_checkpoint']['owner_checkpoint_path'])
        state,arrays,_=checkpoints.read(cp,expected_manifest_sha256=r['latest_checkpoint']['manifest_sha256'])
        schema=r['model_schema'];expected={f'{kind}_{i:04d}' for kind in ('p','m','v') for i in range(len(schema))}
        if set(arrays)!=expected or state['turn']!=128:raise ValueError('Full Adam checkpoint coverage differs')
        digest=hashlib.sha256()
        for key in sorted(arrays):
            x=arrays[key];item=schema[int(key[2:])]
            if list(x.shape)!=item['shape'] or str(x.dtype)!=item['dtype'] or not np.isfinite(x).all():raise ValueError('Bad checkpoint array')
            digest.update(canonical_json([key,list(x.shape),str(x.dtype)]));digest.update(x.tobytes(order='C'))
        if digest.hexdigest()!=r['latest_checkpoint']['replicated_arrays_elements_sha256']:raise ValueError('Checkpoint content digest differs')
        del arrays
        arms[name]={'attempt':attempt,'parameters':r['parameter_count'],'initial_validation':r['initial_validation'],
            'validation_history':r['validation_history'],'test':r['test'],'global_exposures':r['counters'],
            'critical_learning_seconds':max(x['segment_timing']['learning_seconds'] for x in reports),
            'host_segment_timing':[x['segment_timing'] for x in reports],
            'attempt_seconds':closed['elapsed_seconds'],'attempt_chip_hours':closed['reserved_chip_hours'],
            'compiler_updates':r['compiled_updates'],'checkpoint_group_sha256':r['latest_checkpoint']['group_sha256']}
        draws[name]=local_draws
    if {k:v for k,v in configs['cnn'].items() if k!='model'}!={k:v for k,v in configs['transformer'].items() if k!='model'}:raise ValueError('Non-architecture training settings differ')
    if draws['cnn']!=draws['transformer']:raise ValueError('Architecture arms saw different episodes or augmentations')
    for split in ('initial_validation','test'):
        if arms['cnn'][split]['episode_ids_sha256']!=arms['transformer'][split]['episode_ids_sha256']:raise ValueError('Held-out coverage differs')
    result={'schema_version':1,'kind':'visual_cnn_transformer_learning_audit','status':'passed','operator_snapshot':SOURCE.name,
        'registration_sha256':a.registration_sha256,'all_1024_rank_update_draws_exact':True,'arms':arms,'evidence':evidence,
        'test_cnn_over_transformer':{k:arms['cnn']['test']['metrics'][k]/arms['transformer']['test']['metrics'][k] for k in ('expert_kl','behavior_ce','value_mse')},
        'scope':'128 updates from scratch, same whole-episode draws and D4, near-matched parameter count; CNN sees bounded history, transformer sees full history. Different FLOPs, parameter sharing and feature inductive biases. No KataGo strength or faster RL claim. Compiler costs may count loop bodies differently and are not measured MFU.'}
    a.output.write_bytes(canonical_json(result));print(json.dumps({k:v for k,v in result.items() if k not in ('arms','evidence')}))
if __name__=='__main__':main()
