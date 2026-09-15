#!/usr/bin/env python3
"""Abstract full-size shape and matrix-work qualification, without allocation.

This traces the actual differentiated path and the complete cached decoder.
Shape sizes are not compiled peak-memory measurements.
"""
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
from gozero.snapshots import canonical_json
import causal
import compact
import compute_budget
import katago
import policy_model
import qualify_shared
import training_arithmetic


def shapes(batch,positions,size):
    s=jax.ShapeDtypeStruct
    return dict(spatial=s((batch,positions,size,size,22),jnp.float32),
                global_features=s((batch,positions,19),jnp.float32),
                actions=s((batch,positions),jnp.int32),counts=s((batch,),jnp.int32),
                policies=s((batch,positions,size*size+1),jnp.float32),
                legal=s((batch,positions,size*size+1),jnp.bool_))


def nbytes(x):return math.prod(x.shape)*np.dtype(x.dtype).itemsize


def largest_values(graph,limit=8):
    """Static individual array shapes across nested JAXPRs; no liveness claim."""
    found={}
    seen=set()
    def walk(g):
        g=getattr(g,'jaxpr',g)
        if not hasattr(g,'eqns') or id(g) in seen:return
        seen.add(id(g))
        for eq in g.eqns:
            for v in eq.outvars:
                a=v.aval
                if not hasattr(a,'shape'):continue
                key=(eq.primitive.name,tuple(a.shape),str(a.dtype))
                found[key]=nbytes(a)
            for value in eq.params.values():
                if isinstance(value,(tuple,list)):
                    for child in value:walk(child)
                else:walk(value)
    walk(graph)
    return [dict(primitive=k[0],shape=list(k[1]),dtype=k[2],bytes=v)
            for k,v in sorted(found.items(),key=lambda x:x[1],reverse=True)[:limit]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--cnn-config',type=Path,required=True)
    parser.add_argument('--positions',type=int,nargs='+',default=[128,512,1024,1536])
    parser.add_argument('--local-batch',type=int,default=8)
    parser.add_argument('--chunk-frames',type=int,default=32)
    parser.add_argument('--inner-rematerialization',choices=('keep','disable'),default='keep')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output already exists')
    if not 1<=args.local_batch<=128 or any(not 1<=t<=1536 for t in args.positions):
        raise ValueError('Invalid qualification batch/history')
    started=time.time()
    reference=json.loads(args.reference_config.read_text())
    c={**reference['model'],'max_board_size':19,'max_positions':1536,'connector_channels':16}
    inner=None if args.inner_rematerialization=='keep' else False
    compact.validate(c,args.chunk_frames,inner)
    base=json.loads(args.cnn_config.read_text())['model']
    params=jax.eval_shape(lambda:causal.initialize(0,c))
    schema=causal.parameter_schema(c)
    decoder=qualify_shared.trace(c,128,batch=128)
    cnn_params=jax.eval_shape(lambda:katago.initialize(0,base))
    s=jax.ShapeDtypeStruct
    cnn_trace=analyze(jax.make_jaxpr(lambda p,x,g:katago.forward(p,x,g,base))(
        cnn_params,s((128,19,19,22),jnp.float32),s((128,19),jnp.float32)))
    cnn_formula=compute_budget.cnn(base,19,128)
    if cnn_trace['unaccounted_primitives'] or cnn_trace['counts']['multiply_add_flops']!=cnn_formula['multiply_add_flops_per_batch']:
        raise ValueError('CNN complete decode count differs')
    differences=dict(parameters=decoder['analytical']['trainable_parameters']/cnn_formula['trainable_parameters']-1,
                     dense_flops=decoder['analytical']['multiply_add_flops_per_batch']/cnn_formula['multiply_add_flops_per_batch']-1,
                     unit_cost_floating_operations=decoder['traced']['floating_operations_unit_cost']/cnn_trace['floating_operations_unit_cost']-1)
    if any(abs(v)>.01 for v in differences.values()):raise ValueError('Policy-only decoder budget exceeds 1%: '+str(differences))
    cases=[]
    for length in args.positions:
        b=shapes(args.local_batch,length,19)
        encoded=jax.eval_shape(lambda p,s,g:compact.encode(p,s,g,c,chunk_frames=args.chunk_frames,inner_rematerialize=inner),params,b['spatial'],b['global_features'])
        retention=compact.retained_feature_bytes(batch=args.local_batch,positions=length,size=19,c=c)
        if sum(nbytes(x) for x in jax.tree.leaves(encoded))!=retention['compact_outputs_bytes']:
            raise ValueError('Abstract output bytes differ from memory formula')
        row=dict(positions=length,local_batch=args.local_batch,chunk_frames=args.chunk_frames,
                 compact_inner_rematerialization=args.inner_rematerialization,
                 input_bytes=sum(nbytes(x) for x in b.values()),retained_outputs=retention,
                 extra_encoder_frames=(-args.local_batch*length)%args.chunk_frames)
        for name,loss in [('reference',lambda p,b:policy_model.losses(p,b,c)[0]),
                          ('compact',lambda p,b:compact.losses(p,b,c,chunk_frames=args.chunk_frames,inner_rematerialize=inner)[0])]:
            before=time.perf_counter()
            forward=jax.make_jaxpr(loss)(params,b)
            grad=jax.make_jaxpr(jax.value_and_grad(loss))(params,b)
            fwd_count=training_arithmetic.count(forward,draft_valid_length=3*length)
            grad_count=training_arithmetic.count(grad,draft_valid_length=3*length)
            row[name]=dict(forward=fwd_count,differentiated=grad_count,
                           largest_differentiated_values=largest_values(grad),
                           trace_seconds=time.perf_counter()-before)
            print(json.dumps(dict(kind='compact_shape_case',positions=length,path=name,
                                  forward_flops=fwd_count['total_matrix_flops'],
                                  differentiated_flops=grad_count['total_matrix_flops'],
                                  trace_seconds=row[name]['trace_seconds'])),flush=True)
        if not row['extra_encoder_frames'] and row['reference']['forward']['total_matrix_flops']!=row['compact']['forward']['total_matrix_flops']:
            raise ValueError('Unpadded forward work changed')
        row['differentiated_work_ratio']=row['compact']['differentiated']['total_matrix_flops']/row['reference']['differentiated']['total_matrix_flops']
        cases.append(row)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    result=dict(kind='compact_grid_19_abstract_qualification',status='passed',created_unix=time.time(),
                seconds=time.time()-started,jax_version=jax.__version__,backend=jax.default_backend(),
                reference_config_sha256=hashlib.sha256(args.reference_config.read_bytes()).hexdigest(),
                cnn_config_sha256=hashlib.sha256(args.cnn_config.read_bytes()).hexdigest(),model=c,
                parameter_count=sum(x['elements'] for x in schema),parameter_schema=schema,
                policy_only_decoder=dict(transformer=decoder,cnn=dict(analytical=cnn_formula,traced=cnn_trace),relative_differences=differences),
                cases=cases,source_sha256=files,
                scope='Abstract JAX tracing only. No complete arrays allocated, no model trained, no TPU HBM or MFU measured. Logical MAC work includes differentiated rematerialization and actual padded Splash tiles; elementwise, optimizer, SPMD and physical MXU layout excluded from training counts.',
                remaining_gates=['Full-shape TPU peak memory, forward/gradient equivalence and latency','Value heads and all deployed outputs need fresh budget accounting','A 19x19 model/data/optimizer comparison has not been registered'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed',output=str(args.output),seconds=result['seconds'],relative_differences=differences)),flush=True)


if __name__=='__main__':main()
