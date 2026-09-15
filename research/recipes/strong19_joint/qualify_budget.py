"""Trace complete joint decoders and differentiated full-size training shapes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.jaxpr_cost import analyze
import causal
import compute_budget
import joint
import training_arithmetic


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def batch_shape(batch,length,size):
    s=jax.ShapeDtypeStruct
    return dict(spatial=s((batch,length,size,size,22),jnp.float32),global_features=s((batch,length,19),jnp.float32),
        actions=s((batch,length),jnp.int32),counts=s((batch,),jnp.int32),
        policies=s((batch,length,size*size+1),jnp.float32),legal=s((batch,length,size*size+1),jnp.bool_),
        values=s((batch,length),jnp.float32))


def largest_shapes(graph,limit=5):
    found={};seen=set()
    def walk(g):
        g=getattr(g,'jaxpr',g)
        if not hasattr(g,'eqns') or id(g) in seen:return
        seen.add(id(g))
        for eq in g.eqns:
            for v in eq.outvars:
                a=v.aval
                if hasattr(a,'shape'):
                    found[(eq.primitive.name,tuple(a.shape),str(a.dtype))]=math.prod(a.shape)*np.dtype(a.dtype).itemsize
            for value in eq.params.values():
                for child in value if isinstance(value,(tuple,list)) else (value,):walk(child)
    walk(graph)
    return [dict(primitive=k[0],shape=list(k[1]),dtype=k[2],bytes=n)
            for k,n in sorted(found.items(),key=lambda kv:kv[1],reverse=True)[:limit]]


def decode(c,v,params,*,past=128,batch=128):
    size=c['max_board_size'];s=jax.ShapeDtypeStruct
    if c['architecture']=='katago_nested_policy':
        graph=jax.make_jaxpr(lambda p,b:joint.forward(p,b,c))(params,batch_shape(batch,1,size))
        base=compute_budget.cnn(c,size,batch)
        head=2*batch*(size*size*c['width']*v['spatial_channels']+3*v['spatial_channels']*v['hidden']+v['hidden']*3)
    else:
        spatial=s((batch,size,size,22),jnp.float32);glob=s((batch,19),jnp.float32)
        if past==0:
            graph=jax.make_jaxpr(lambda p,x,g:joint.first_move(p,x,g,c))(params,spatial,glob)
        else:
            shape=(batch,c['layers'],2*c['max_positions'],c['kv_heads'],c['width']//c['heads'])
            cache=dict(keys=s(shape,jnp.bfloat16),values=s(shape,jnp.bfloat16),lengths=s((batch,),jnp.int32),
                       valid=s((batch,),jnp.bool_),network_version=s((),jnp.uint32))
            graph=jax.make_jaxpr(lambda p,ca,a,x,g:joint.append_move(p,ca,a,x,g,c,attention_positions=past+1))(
                params,cache,s((batch,),jnp.int32),spatial,glob)
        base=compute_budget.transformer(c,size,batch,past)
        head=2*batch*(c['width']*v['hidden']+v['hidden']*3)
    traced=analyze(graph);expected=base['multiply_add_flops_per_batch']+head
    if traced['unaccounted_primitives'] or traced['counts']['multiply_add_flops']!=expected:
        raise ValueError('Full joint decode differs from independent component count: '+str(traced))
    return dict(past_moves=past,batch=batch,policy_only=base,value_head_matrix_flops=head,
                multiply_add_flops_per_batch=expected,multiply_add_flops_per_move=expected/batch,traced=traced)


def main():
    p=argparse.ArgumentParser();p.add_argument('--transformer-config',type=Path,required=True)
    p.add_argument('--cnn-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--positions',type=int,nargs='+',default=[128,512,1536]);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if any(type(t) is not int or not 1<=t<=1536 for t in a.positions):raise ValueError('Invalid history extent')
    started=time.monotonic()
    cnn={**json.loads(a.cnn_config.read_text())['model'],'max_board_size':19,'max_positions':1536}
    transformer={**json.loads(a.transformer_config.read_text())['model'],'max_board_size':19,'max_positions':1536,'connector_channels':16}
    configs=dict(cnn=cnn,transformer=transformer);v=dict(hidden=256,spatial_channels=256)
    params={name:jax.eval_shape(lambda:joint.initialize(0,c,v)) for name,c in configs.items()}
    schemas={name:joint.parameter_schema(c,v) for name,c in configs.items()}
    counts={name:dict(trainable=sum(row['elements'] for row in schema),
                      inference=sum(row['elements'] for row in schema if row['inference'])) for name,schema in schemas.items()}
    decoder_cnn=decode(cnn,v,params['cnn'])
    decoders={str(past):decode(transformer,v,params['transformer'],past=past) for past in (0,32,128,256,1024,1535)}
    selected=decoders['128']
    differences=dict(parameters=counts['transformer']['trainable']/counts['cnn']['trainable']-1,
        dense_flops=selected['multiply_add_flops_per_batch']/decoder_cnn['multiply_add_flops_per_batch']-1,
        unit_cost_floating_operations=selected['traced']['floating_operations_unit_cost']/decoder_cnn['traced']['floating_operations_unit_cost']-1)
    if any(abs(x)>.01 for x in differences.values()):raise ValueError('Joint 128-past reference budget exceeds 1%: '+str(differences))
    cases=[]
    for length in a.positions:
        b=batch_shape(8,length,19)
        for name,c in configs.items():
            before=time.monotonic()
            loss=lambda p,b:joint.losses(p,b,c,value_weight=.7,chunk_frames=32,inner_rematerialize=False)[0]
            f=jax.make_jaxpr(loss)(params[name],b)
            g=jax.make_jaxpr(jax.value_and_grad(loss))(params[name],b)
            kw=dict(draft_valid_length=3*length if name=='transformer' else None)
            row=dict(architecture=name,positions=length,local_batch=8,
                forward=training_arithmetic.count(f,**kw),differentiated=training_arithmetic.count(g,**kw),
                largest_differentiated_arrays=largest_shapes(g),trace_seconds=time.monotonic()-before)
            cases.append(row)
            print(json.dumps(dict(architecture=name,positions=length,largest_abstract_array_bytes=row['largest_differentiated_arrays'][0]['bytes'],
                                  trace_seconds=row['trace_seconds'])),flush=True)
    result=dict(kind='joint19_abstract_budget_qualification',status='passed',created=time.time(),elapsed_seconds=time.monotonic()-started,
        configs=configs,value_config=v,parameter_counts=counts,parameter_schemas=schemas,
        cnn_decode=decoder_cnn,transformer_decode=decoders,relative_differences_at_128_past=differences,training_shapes=cases,
        config_sha256={str(p):sha(p) for p in (a.cnn_config,a.transformer_config)},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'packages/gozero/src/gozero/jaxpr_cost.py']},
        scope='Abstract logical work and shapes only; no full arrays, TPU execution, peak HBM, MFU or training. All trainable CNN helpers count toward parameters; deployed policy and value count toward complete decoder FLOPs. Training includes rematerialization and padded Splash matrices but excludes optimizer/SPMD/physical MXU layout.',
        required_gates=['Compiled TPU optimizer step and measured peak memory for both architectures','Training horizon/data/optimizer registration','Actual cached model serving and native-search evaluation'])
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',parameter_counts=counts,relative_differences=differences,seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
