"""Publish only a fully completed registered baseline endpoint."""
import argparse
import json
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--arm',choices=('cnn','transformer'),required=True)
    a=p.parse_args();verify(SOURCE);root=SOURCE.parents[2]
    registration=read_json(root/'research/studies/visual_causal/cnn_baseline_registration.json')
    d=root/'runs'/a.attempt;closed=read_json(d/'result.json')
    if closed['status']!='passed' or closed['snapshot_id']!=registration['arms'][a.arm]['snapshot']:raise ValueError('Unregistered or incomplete endpoint')
    reports=[read_json(d/f'rank-{h}/artifacts/result.json') for h in range(4)]
    if any(r['status']!='passed' or r['turn']!=128 or not r['training_complete'] for r in reports):raise ValueError('Missing completed rank')
    if len({r['latest_checkpoint']['group_sha256'] for r in reports})!=1:raise ValueError('Inconsistent checkpoint group')
    result=d/'rank-0/artifacts/result.json';r=reports[0];latest=r['latest_checkpoint']
    cp=Path(latest['owner_checkpoint_path']);group=read_json(cp.with_suffix('.group.json'))
    descriptor={'schema_version':1,'kind':'visual_causal_checkpoint','training_snapshot':r['snapshot_id'],
       'training_result':{'path':str(result.relative_to(root)),'sha256':sha256(result)},
       'checkpoint':{'path':str(cp.relative_to(root)),'manifest_sha256':group['host_manifests']['0'],'group_sha256':sha256(cp.with_suffix('.group.json'))},
       'network_version':128,'training_complete':True}
    destination=root/f'eval/visual_causal/baseline_{a.arm}.json'
    with destination.open('xb') as f:f.write(canonical_json(descriptor))
    print(json.dumps({'candidate':str(destination),'sha256':sha256(destination)}))
if __name__=='__main__':main()
