"""Execute unmodified pinned source functions without launching its trainer."""
import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pin_path=ROOT/'research/studies/katago_muon/reference-source-001.json'
    pin=json.loads(pin_path.read_text());source=ROOT/pin['source']/'python/train.py'
    if sha(source)!=pin['files']['python/train.py']:
        raise ValueError('Pinned source changed')
    text=source.read_text();tree=ast.parse(text)
    names={'get_is_muon_suitable','get_weight_decay','update_and_return_lr_and_wd'}
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions}!=names or len(functions)!=3:
        raise ValueError('Reference function coverage differs')
    # Extract the original clipping branch and its three scaling statements.
    clip_nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.If,ast.Assign,ast.AugAssign))
                and n.lineno in (1589,1598,1599,1613)]
    if len(clip_nodes)!=4:
        raise ValueError('Pinned clipping source spans differ')
    clip_nodes.sort(key=lambda n:n.lineno)
    function_code=compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec')
    clipping_code=compile(ast.Module(body=clip_nodes,type_ignores=[]),str(source),'exec')
    groups=('normal','normal_attn','normal_gab','gab_mlp','tab_module','input','input_noreg',
            'normal_gamma','noreg','output','output_noreg')
    factors=dict(head_lr_factor=.5,noreg_lr_factor=1.,muon_adam_lr_factor=1.,input_wd_factor=1.,
                 normal_wd_factor=1.,normal_attn_wd_factor=1.,gnorm_clip_scale=1.)
    profiles=[dict(effective_lr_scale=1.,factors=factors,norm_ratios=None,lookahead_alpha=None,no_lr_warmup=False),
              dict(effective_lr_scale=12.,factors=factors,norm_ratios=dict(input=1.25,normal=.8),lookahead_alpha=.5,no_lr_warmup=False),
              dict(effective_lr_scale=.2,factors={**factors,'head_lr_factor':.8,'noreg_lr_factor':.7,
                   'muon_adam_lr_factor':.6,'input_wd_factor':.3,'normal_wd_factor':1.4,
                   'normal_attn_wd_factor':.2,'gnorm_clip_scale':.4},norm_ratios=dict(input=.2,normal=4.),lookahead_alpha=.7,no_lr_warmup=False),
              dict(effective_lr_scale=3.,factors={**factors,'input_wd_factor':0.},
                   norm_ratios=dict(input=0.,normal=100.),lookahead_alpha=1.,no_lr_warmup=True)]
    samples=[0]+[x+d for x in range(250000,2000001,250000) for d in (-1,0)]+[250000000,4000000000,12000000000]
    cases=[]
    for profile in profiles:
        for count in samples:
            for batch in (64,256,1024,16384):
                args_row=dict(samples=count,global_batch=batch,**profile)
                norms=profile['norm_ratios'];running={}
                if norms is not None:
                    running=dict(sums={'norm_input_batch':norms['input']*2.,'norm_normal_batch':norms['normal']*3.},
                                 weights={'norm_input_batch':2.,'norm_normal_batch':3.})
                optimizer=SimpleNamespace(param_groups=[dict(group_name=g,lr=0.,weight_decay=0.) for g in groups])
                env=dict(math=math,logging=logging,use_muon=True,use_adamw=False,world_size=4,batch_size=batch//4,
                         optimizer=optimizer,raw_model=SimpleNamespace(get_norm_kind=lambda:'fixscaleonenorm'),
                         model_config=dict(norm_kind='fixscaleonenorm'),
                         train_state=dict(global_step_samples=count,modelnorm_input_baseline=1.,modelnorm_normal_baseline=1.),
                         running_metrics=running,get_effective_lr_scale=lambda state:profile['effective_lr_scale'],
                         lookahead_alpha=None if profile['lookahead_alpha']==1. else profile['lookahead_alpha'],
                         no_lr_warmup=profile['no_lr_warmup'],**profile['factors'])
                exec(function_code,env)
                per_sample,normal_decay=env['update_and_return_lr_and_wd'](log_if='never')
                exec(clipping_code,env)
                cases.append(dict(inputs=args_row,expected=dict(per_sample_lr=per_sample,
                    rates={g['group_name']:g['lr'] for g in optimizer.param_groups},
                    decays={g['group_name']:g['weight_decay'] for g in optimizer.param_groups},
                    source_sum_gradient_clip_cap=env['gnorm_cap'])))
    report=dict(kind='katago_source_schedule_reference',status='passed',created=time.time(),
                source_revision=pin['revision'] if 'revision' in pin else '92ee95c0a4b25fec214da00951ab69e97e207729',
                source_sha256=sha(source),pin_sha256=sha(pin_path),operator_sha256=sha(Path(__file__)),
                extracted_source=[dict(first_line=n.lineno,last_line=n.end_lineno,
                    sha256=hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()) for n in functions+clip_nodes],
                cases=cases,scope='Actual pinned source scalar LR/decay/clipping functions for standard Muon and FSON. No trainer launch or claim about unpublished historical arguments, norm-update cadence or loss labels.')
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed',cases=len(cases),sha256=sha(args.output))))


if __name__=='__main__':main()
