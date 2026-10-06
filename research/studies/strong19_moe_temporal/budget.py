"""Abstract tensor and cached-move arithmetic budgets; no device allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
RECIPE=ROOT/'research/recipes/strong19_moe_temporal'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
import joint,encoder
from gozero.snapshots import canonical_json


def budget(c,v):
    schema=joint.parameter_schema(c,v)
    d,L,f=c['width'],c['layers'],c['mlp_hidden']
    kv=c['kv_heads']*(d//c['heads']);points=c['max_board_size']**2
    temporal=4*L*(2*d*d+2*d*kv+3*d*f)
    if 'moe' in c: temporal+=4*L*d*c['moe']['experts']
    context=c['policy_context_dim'];w=c['encoder_width']
    policy=2*((points+1)*d+points*w+d*context+points*w*context+points*context)
    value=2*(d*v['hidden']+v['hidden']*3)
    decode={}
    for history in (0,128,512,768,1534):
        work=dict(encoder=encoder.flops(c,c['max_board_size']),temporal_dense=temporal,
                  temporal_attention=4*L*d*(2*history+3),policy=policy,value=value)
        decode[str(history)]={**work,'total_matrix_flops':sum(work.values())}
    return dict(parameters=sum(x['elements'] for x in schema),
                encoder_parameters=sum(x['elements'] for x in schema if x['path'].startswith('encoder.')),
                expert_parameters=sum(x['elements'] for x in schema if '.moe.' in x['path']),
                decode=decode,schema=schema)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=STUDY/'proposal-config-001.json';c=json.loads(source.read_text())
    model=c['model'];dense={k:v for k,v in model.items() if k!='moe'}
    full={**model,'moe':{**model['moe'],'scope':'all'}}
    budgets={name:budget(cfg,c['value_model']) for name,cfg in [('dense',dense),('all_experts',full),('temporal_experts',model)]}
    old=json.loads((ROOT/'research/studies/strong19_moe/budget-001.json').read_text())
    for name,reference in [('dense','dense'),('all_experts','moe')]:
        for key in ('parameters','encoder_parameters','expert_parameters','decode'):
            assert budgets[name][key]==old['budgets'][reference][key],(name,key)
    temporal=budgets['temporal_experts'];base=budgets['dense']
    assert temporal['encoder_parameters']==base['encoder_parameters']
    assert temporal['parameters']==317001492
    assert not any('.moe.' in x['path'] for x in temporal['schema'] if x['path'].startswith('encoder.'))
    result=dict(status='passed',budgets=budgets,
                active_flop_ratio=temporal['decode']['128']['total_matrix_flops']/base['decode']['128']['total_matrix_flops'],
                parameter_ratio=temporal['parameters']/base['parameters'],
                operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='Logical cached-move matrix FLOPs including dense encoder passes, both temporal tokens, routers, attention and heads. Excludes sort, dispatch, activations, tile padding, and training-only auxiliary tokens. Not measured throughput or MFU.')
    with a.output.open('xb') as stream: stream.write(canonical_json(result))
    print(json.dumps({k:v for k,v in result.items() if k!='budgets'}))


if __name__=='__main__':main()
