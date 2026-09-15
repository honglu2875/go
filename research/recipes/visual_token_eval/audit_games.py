#!/usr/bin/env python3
"""Authenticate paired game outputs, reproduce logged metrics and bootstrap games."""
import argparse
import itertools
import math
from pathlib import Path
import sys
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,canonical_json,verify
from gozero.checkpoints import sha256
from gozero.katago_sequence_batches import Dataset
from paired_statistics import compare

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve();attempt=a.attempt.resolve();closed=read_json(attempt/'result.json')
    if closed['status']!='passed':raise ValueError('Evaluation attempt did not pass')
    snapshot=root/'.gozero/snapshots'/closed['snapshot_id'];verify(snapshot);c=read_json(snapshot/'resolved_config.json')
    if c['kind']!='paired_policy_evaluation':raise ValueError('Wrong evaluation kind')
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);selected=data.bucket_entries(c['dataset']['buckets'],split=1)
    expected=[entry for bucket in c['dataset']['buckets'] for entry in selected['expert',bucket][:10000]]
    expected_ids={tuple(x) for x in expected};files={};reports=[]
    for host in range(4):
        path=attempt/f'rank-{host}/artifacts/result.json';r=read_json(path);files[str(path.relative_to(root))]=sha256(path)
        if r['status']!='passed' or r['host_rank']!=host or r['snapshot_id']!=snapshot.name:raise ValueError('Host report differs')
        reports.append(r)
    if sorted(r['jax_rank'] for r in reports)!=list(range(4)):raise ValueError('Invalid host/JAX mapping')
    summaries=[];all_rows=[]
    for case in c['cases']:
        label=case['label'];rows=[];digests=[]
        for host in range(4):
            folder=attempt/f'rank-{host}/artifacts';path=folder/f'{label}.json';r=read_json(path)
            files[str(path.relative_to(root))]=sha256(path)
            if r['label']!=label or r['episode_ids_sha256']!=case['validation_episode_ids_sha256']:raise ValueError('Case identity differs')
            if {k:v for k,v in r.items() if k!='rows'}!=next(x for x in reports[host]['cases'] if x['label']==label):raise ValueError('Case summary changed')
            digests.append(r['parameter_elements_sha256']);rows.extend(r['rows'])
            for bucket,profile in r['profiles'].items():
                hlo=folder/f'{label}-{bucket}.hlo'
                if sha256(hlo)!=profile['hlo_sha256']:raise ValueError('Evaluation HLO changed')
                files[str(hlo.relative_to(root))]=sha256(hlo)
        if len(set(digests))!=1:raise ValueError('Broadcast parameter bytes differ across hosts')
        ids=[tuple(x['entry']) for x in rows]
        if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=expected_ids:raise ValueError('Game coverage differs')
        rows.sort(key=lambda x:tuple(x['entry']))
        for row in rows:
            role,shard,game=row['entry'];offsets=data.shards[shard]['expert_offsets'];n=int(offsets[game+1]-offsets[game])
            if row['count']!=n or any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):raise ValueError('Invalid game metric')
            if not 0<=row['top1']<=n or row['top1']!=int(row['top1']):raise ValueError('Invalid top-move count')
        totals={k:float(np.asarray([x[k] for x in rows],np.float64).sum()) for k in rows[0] if k not in ('entry','bucket')}
        observed={'expert_count':totals['count'],'expert_ce':totals['ce']/totals['count'],
            'expert_target_entropy':totals['target_entropy']/totals['count'],'expert_kl':(totals['ce']-totals['target_entropy'])/totals['count'],
            'expert_top1':totals['top1']/totals['count']}
        for name in totals:
            if name.startswith('phase_') and name.endswith('_count'):
                prefix=name[:-6];n=totals[name]
                observed[name]=n;observed[prefix+'_kl']=(totals[prefix+'_ce']-totals[prefix+'_target_entropy'])/max(1,n)
        for key,value in observed.items():
            tolerance=0 if key.endswith('_count') else 1e-7 if key=='expert_top1' else 3e-5
            if abs(value-case['expected_validation'][key])>tolerance:raise ValueError('Stored endpoint does not reproduce: '+label+' '+key)
        summaries.append({'label':label,'games':len(rows),'parameter_elements_sha256':digests[0],'reproduced_validation':observed,
            'original_training_snapshot':case['training_snapshot'],'audit_sha256':case['audit_sha256']})
        all_rows.append(rows)
    pairs=[{'left':c['cases'][i]['label'],'right':c['cases'][j]['label'],**compare(all_rows[i],all_rows[j])}
        for i,j in itertools.combinations(range(len(all_rows)),2)]
    result={'kind':'paired_game_validation_audit','status':'passed','operator_snapshot':SOURCE.name,'attempt':attempt.name,
        'closed_result_sha256':sha256(attempt/'result.json'),'evaluation_snapshot':snapshot.name,'input_files':files,
        'validation_games':len(expected),'arms':summaries,'pairs':pairs,'reserved_chip_hours':closed['reserved_chip_hours'],
        'scope':'Same fixed validation games and exact completed model implementations. Reproduces original aggregate metrics; test split remains closed. Bootstrap uncertainty covers validation games only, not training seeds or RL strength.'}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444);print(canonical_json({'status':'passed','sha256':sha256(a.output),'pairs':pairs}).decode().strip())

if __name__=='__main__':main()
