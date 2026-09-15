"""Pin both fixed final endpoints and the already selected KataGo panel."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cnn-snapshot',required=True);p.add_argument('--transformer-snapshot',required=True)
    a=p.parse_args();verify(SOURCE);root=SOURCE.parents[2];study=root/'research/studies/visual_causal'
    design_path=study/'cnn_katago_design.json';design=read_json(design_path);arms=[]
    for name,pin in [('cnn',a.cnn_snapshot),('transformer',a.transformer_snapshot)]:
        source=root/'.gozero/snapshots'/pin;verify(source);c=read_json(source/'resolved_config.json')
        candidate=root/f'eval/visual_causal/baseline_{name}.json';d=read_json(candidate)
        if c['mode']!='matches' or c['candidate']!=str(candidate.relative_to(root)) or d['network_version']!=128 or not d['training_complete']:raise ValueError('Endpoint selection differs')
        seen={}
        for host,specifications in c['matches_by_host'].items():
            if len(specifications)!=16:raise ValueError('Incomplete fixed panel host')
            for specification in specifications:
                spec=read_json(source/specification);key=(spec['katago_weights'],spec['seed_prefix'])
                if spec['openings'][0] not in design['openings'] or spec['katago_weights'] not in design['anchors'] or key in seen:raise ValueError('Opening/anchor selection changed')
                seen[key]=spec['openings']
        coverage={(anchor,tuple(opening[0])) for (anchor,_),opening in seen.items()}
        expected={(anchor,tuple(opening)) for anchor in design['anchors'] for opening in design['openings']}
        if coverage!=expected:raise ValueError('Panel opening coverage differs')
        if len(seen)!=64:raise ValueError('Missing paired jobs')
        arms.append({'arm':name,'snapshot_id':pin,'config_sha256':sha256(source/'resolved_config.json'),'candidate_sha256':sha256(candidate)})
    result={'schema_version':1,'kind':'visual_cnn_transformer_fixed_endpoint_katago_panel','registered_at':datetime.now(timezone.utc).isoformat(),
            'design_sha256':sha256(design_path),'arms':arms,'anchors':design['anchors'],'pairs_per_anchor_per_arm':32,'games_per_arm':128,
            'maximum_attempts':2,'maximum_seconds_per_attempt':900,
            'scope':'Final128-update checkpoints selected before endpoint evaluation; identical fresh paired openings and explicit16sim/1visit search budgets. CNN history8, transformer complete history. No equal-compute, general architecture superiority, Elo or faster-RL claim.'}
    output=study/'cnn_katago_registration.json'
    with output.open('xb') as f:f.write(canonical_json(result))
    print(sha256(output))
if __name__=='__main__':main()
