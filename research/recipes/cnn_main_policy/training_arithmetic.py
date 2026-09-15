#!/usr/bin/env python3
"""Count differentiated model matrices and qualified causal Splash block work."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256
import policy_model


@lru_cache(maxsize=32)
def active_attention_blocks(length,tile,observation_stride=0,valid_length=None):
    if not observation_stride:
        n=length//tile;return n*(n+1)//2
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as masks
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as mask_info
    from observation_attention import ObservationMask
    one=ObservationMask((length,length),observation_stride,valid_length)
    # Independent frame enumeration: known board/readout, or completed frame
    # on an action token. Every query sees a contiguous prefix of keys.
    expected=0
    for start in range(0,length,tile):
        query=start+tile-1;frame,offset=divmod(query,observation_stride)
        end=min(valid_length,frame*observation_stride+(observation_stride if offset==observation_stride-1 else observation_stride-1))
        expected+=math.ceil(end/tile)
    mask=masks.MultiHeadMask([one])
    for processor in (mask_info.process_mask,mask_info.process_mask_dkv):
        info,_=processor(mask,(tile,tile),head_shards=1,q_seq_shards=1)
        if np.count_nonzero(info.block_mask)!=expected:raise ValueError('Attention block schedule differs from frame enumeration')
    return expected


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


def count(graph,*,observation_stride=0,valid_length=None):
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
                blocks=active_attention_blocks(length,tile,observation_stride,valid_length)
                flops=m*dots*2*batch*heads*blocks*tile*tile*dim
                work['splash_'+phase]+=flops
                calls.append({'name':label,'multiplicity':m,'q_shape':list(shape(q)),
                    'square_tile':tile,'active_blocks_per_head':blocks,'observation_stride':observation_stride,'matrix_products_per_block':dots,
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


