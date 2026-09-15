#!/usr/bin/env python3
"""Count differentiated model matrices and qualified causal Splash block work."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256
import policy_model


def tile_graph_flops(graph):
    """Check matrix bodies of one active tile directly in the Pallas JAX graph."""
    graph=getattr(graph,'jaxpr',graph);total=0
    def child(x):return hasattr(x,'eqns') or hasattr(getattr(x,'jaxpr',None),'eqns')
    for eq in graph.eqns:
        name=eq.primitive.name;p=eq.params
        if name=='dot_general':
            shape=eq.invars[0].aval.shape
            total+=2*math.prod(eq.outvars[0].aval.shape)*math.prod(shape[d] for d in p['dimension_numbers'][0][0])
        elif name=='scan':total+=p['length']*tile_graph_flops(p['jaxpr'])
        elif name=='cond':
            branches=[tile_graph_flops(x) for x in p['branches']]
            if sum(x>0 for x in branches)>1:raise ValueError('Ambiguous matrix work in tile branches')
            total+=max(branches)
        else:
            children=[]
            for value in p.values():
                if child(value):children.append(value)
                elif isinstance(value,(list,tuple)):children.extend(x for x in value if child(x))
            subtotal=sum(tile_graph_flops(x) for x in children)
            if name=='while' and subtotal:raise ValueError('Unbounded tile matrix loop')
            total+=subtotal
    return total


def count(graph):
    work=Counter();calls=[];seen=Counter()
    def shape(v):return tuple(v.aval.shape)
    def walk(graph,m=1):
        graph=getattr(graph,'jaxpr',graph)
        for eq in graph.eqns:
            name=eq.primitive.name;p=eq.params;seen[name]+=m
            if name=='scan':walk(p['jaxpr'],m*p['length'])
            elif name in ('jit','pjit','remat2','custom_jvp_call','custom_vjp_call','eval_jaxpr'):
                child=p.get('jaxpr',p.get('call_jaxpr',p.get('fun_jaxpr')))
                if child is None:raise ValueError('Unknown nested operation '+name)
                walk(child,m)
            elif name=='dot_general':
                k=math.prod(shape(eq.invars[0])[d] for d in p['dimension_numbers'][0][0])
                work['outside_splash_matrices']+=m*2*math.prod(shape(eq.outvars[0]))*k
            elif name=='conv_general_dilated':
                k=math.prod(shape(eq.invars[1])[d] for d in p['dimension_numbers'].rhs_spec[1:])
                work['convolutions']+=m*2*math.prod(shape(eq.outvars[0]))*k
            elif name=='pallas_call':
                label=p['name'];meta=json.loads(p['metadata']['xprof_metadata'])
                q=next(v for v in eq.invars if str(v.aval.dtype)=='bfloat16' and len(shape(v))==4)
                batch,heads,length,dim=shape(q)
                if label=='splash_mha_fwd_residuals':phase='fwd';tile=meta['block_q'];other=meta['block_kv'];dots=2
                elif label=='splash_mha_dkv_no_residuals':phase='dkv';tile=meta['block_q_dkv'];other=meta['block_kv_dkv'];dots=4
                elif label=='splash_mha_dq_no_residuals':phase='dq';tile=meta['block_q_dq'];other=meta['block_kv_dq'];dots=3
                else:raise ValueError('Unqualified Pallas kernel '+label)
                if tile!=other or length%tile or meta.get('use_fused_bwd_kernel',False):raise ValueError('Unsupported kernel tiling')
                tile_work=tile_graph_flops(p['jaxpr'])
                if tile_work!=dots*2*tile*tile*dim:raise ValueError(f'Pallas tile body differs: {label}, {tile_work}')
                n=length//tile;blocks=n*(n+1)//2
                flops=m*dots*2*batch*heads*blocks*tile*tile*dim
                work['splash_'+phase]+=flops
                calls.append({'name':label,'multiplicity':m,'q_shape':list(shape(q)),
                    'square_tile':tile,'causal_blocks_per_head':blocks,'matrix_products_per_block':dots,
                    'tile_body_jaxpr_flops':tile_work,'dense_mac_flops':flops})
            elif name in ('add','add_any','sub','mul','div','neg','pow','integer_pow','square','sqrt','rsqrt','exp','log','log1p',
                    'sin','cos','tanh','logistic','sign','rem','max','min','abs','reduce_sum','reduce_max','reduce_min',
                    'reduce_and','reduce_or','and','or','not','xor','lt','le','gt','ge','eq','ne','select_n','argmax',
                    'convert_element_type','pad','reshape','transpose','broadcast_in_dim','slice','squeeze','iota','gather',
                    'scatter-add','scatter','stack','unstack','split','concatenate','stop_gradient','copy','device_put',
                    'dynamic_slice','dynamic_update_slice','sharding_constraint','rev'):
                pass
            else:raise ValueError('Unaudited differentiated primitive '+name)
    walk(graph)
    return {'matrix_flops':dict(work),'total_matrix_flops':sum(work.values()),'splash_calls':calls,'primitive_occurrences':dict(seen)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-snapshot',type=Path,required=True)
    p.add_argument('--tile-decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE);verify(args.baseline_snapshot)
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as kernel_source
    decision=read_json(args.tile_decision)
    if sha256(Path(kernel_source.__file__))!=decision['kernel_source_sha256']:raise ValueError('Attention implementation differs from numerical qualification')
    c=read_json(SOURCE/'resolved_config.json');base=read_json(args.baseline_snapshot/'resolved_config.json');batch=128;cases={}
    for label,model in [('cnn',base['model']),('transformer',c['model'])]:
        params=jax.eval_shape(lambda:policy_model.initialize(0,model))
        for length in c['dataset']['buckets']:
            s=jax.ShapeDtypeStruct
            b={'spatial':s((batch,length,9,9,22),jnp.float32),'global_features':s((batch,length,19),jnp.float32),
                'actions':s((batch,length),jnp.int32),'counts':s((batch,),jnp.int32),
                'policies':s((batch,length,82),jnp.float32),'legal':s((batch,length,82),jnp.bool_)}
            for remat in (False,True):
                net={**model,'rematerialize':remat}
                graph=jax.make_jaxpr(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,net)[0]))(params,b)
                row=count(graph);cases[f'{label}-{length}-remat{int(remat)}']=row
                print(json.dumps({'kind':'training_arithmetic','model':label,'bucket':length,'remat':remat,'matrix_flops':row['total_matrix_flops']}),flush=True)
    result={'kind':'differentiated_training_matrix_arithmetic','operator_snapshot':SOURCE.name,'baseline_snapshot':args.baseline_snapshot.name,
        'tile_decision_sha256':sha256(args.tile_decision),'kernel_source_sha256':sha256(Path(kernel_source.__file__)),
        'cases':cases,'global_sequences':batch,'board_size':9,'fma_flops':2,
        'scope':'Logical differentiated model graph with static scans expanded. Includes full encoder, helper/head, backward matrices, and explicit rematerialization. Causal Splash counts complete executed square tiles, including padding/diagonal masked work and score recomputation in the separate dQ/dKV kernels.',
        'exclusions':'Elementwise/nonlinear work, optimizer work, SPMD communication, MXU physical tile layout and memory transfers. Counts are arithmetic accounting, not measured MFU or hardware FLOP counters.',
        'remat_false':'Hypothetical same differentiated model without layer rematerialization; not executed at this batch because it may not fit HBM. Flash backward score recomputation remains included.',
        'splash_audit':'Pinned qualified source uses two forward matmuls, three dQ matmuls and four dKV matmuls per active causal square tile. The block count n*(n+1)/2 includes the full diagonal tile.'}
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444);print(json.dumps({'status':'passed','sha256':sha256(args.output)}),flush=True)


if __name__=='__main__':main()
