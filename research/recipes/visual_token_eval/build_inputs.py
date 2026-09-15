#!/usr/bin/env python3
"""Create evaluation inputs from authenticated, completed learning audits."""
import argparse
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,canonical_json,verify
from gozero.checkpoints import sha256
import evaluate_games

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--audit',nargs=3,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(SOURCE);root=a.workspace_root.resolve();cases=[];datasets=[]
    for label,path,digest in a.audit:
        if not label.replace('_','').isalnum():raise ValueError('Use simple unique labels')
        path=Path(path)
        if sha256(path)!=digest:raise ValueError('Audit changed')
        audit=read_json(path)
        if audit['status']!='passed' or audit['steps']!=1024:raise ValueError('Expected completed full learning audit')
        attempt=root/'runs'/audit['attempt'];closed=attempt/'result.json'
        if sha256(closed)!=audit['closed_result_sha256'] or read_json(closed)['status']!='passed':raise ValueError('Run changed')
        for name,wanted in audit['input_files'].items():
            if sha256(root/name)!=wanted:raise ValueError('Audited input changed')
        source=root/'.gozero/snapshots'/audit['training_snapshot'];manifest=verify(source);c=read_json(source/'resolved_config.json')
        result_path=attempt/'rank-0/artifacts/result.json';result=read_json(result_path);module=evaluate_games.implementation(c['model'])
        filename='katago.py' if c['model']['architecture']=='katago_nested_policy' else 'causal.py'
        code_sha=sha256(source/manifest['recipe']/filename)
        if code_sha!=sha256(Path(module.__file__)):raise ValueError('Evaluator model code differs from trained source')
        cases.append({'label':label,'audit_path':str(path.relative_to(root) if path.is_absolute() else path),'audit_sha256':digest,
            'training_snapshot':source.name,'training_result_path':str(result_path.relative_to(root)),'training_result_sha256':sha256(result_path),
            'model':c['model'],'model_schema':result['model_schema'],'model_code_sha256':code_sha,
            'validation_episode_ids_sha256':result['validation_history'][-1]['episode_ids_sha256'],
            'expected_validation':result['validation_history'][-1]['metrics']})
        datasets.append(c['dataset'])
    if len(cases)<2 or len({x['label'] for x in cases})!=len(cases) or any(x!=datasets[0] for x in datasets):raise ValueError('Case controls differ')
    if len({x['validation_episode_ids_sha256'] for x in cases})!=1:raise ValueError('Validation population differs')
    result={'schema_version':1,'kind':'paired_policy_evaluation','platform':'tpu','expected_devices':16,'expected_processes':4,
        'dataset':datasets[0],'cases':cases,'evaluation':{'split':1,'games_per_host':32,'games_per_bucket':10000}}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444);print(canonical_json({'path':str(a.output),'sha256':sha256(a.output),'cases':[x['label'] for x in cases]}).decode().strip())

if __name__=='__main__':main()
