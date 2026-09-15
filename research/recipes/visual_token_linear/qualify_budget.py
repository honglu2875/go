#!/usr/bin/env python3
"""Freeze complete neural decoding counts before observing transformer learning."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256
from gozero.jaxpr_cost import analyze
from gozero.katago_sequence_batches import Dataset
import causal
import katago
import compute_budget
import policy_model


def abstract(shape,dtype=jnp.float32):return jax.ShapeDtypeStruct(shape,dtype)


def trace_transformer(c,size,past,batch=128):
    p=jax.eval_shape(lambda:causal.initialize(0,c));s=abstract((batch,size,size,22));g=abstract((batch,19))
    spec=causal.layout(size,c);dim=c['width']//c['heads']
    if past==0:
        graph=jax.make_jaxpr(lambda p,s,g:causal.first_move(p,s,g,c))(p,s,g)
    else:
        shape=(batch,c['layers'],spec['capacity'],c['kv_heads'],dim)
        cache={'keys':abstract(shape,jnp.bfloat16),'values':abstract(shape,jnp.bfloat16),
            'lengths':abstract((batch,),jnp.int32),'valid':abstract((batch,),jnp.bool_),
            'network_version':abstract((),jnp.uint32)}
        graph=jax.make_jaxpr(lambda p,ca,a,s,g:causal.append_move(p,ca,a,s,g,c,attention_positions=past+1))(
            p,cache,abstract((batch,),jnp.int32),s,g)
    counts=analyze(graph);analytic=compute_budget.transformer(c,size,batch,past)
    if counts['unaccounted_primitives'] or counts['counts']['multiply_add_flops']!=analytic['multiply_add_flops_per_batch']:
        raise ValueError('Transformer analytic/traced arithmetic disagrees: '+str(counts['unaccounted_primitives']))
    return {'analytic':analytic,'traced':counts,'allocated_cache_bytes':batch*c['layers']*spec['capacity']*c['kv_heads']*dim*2*2}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE);verify(a.baseline_snapshot)
    c=read_json(SOURCE/'resolved_config.json');base=read_json(a.baseline_snapshot/'resolved_config.json')
    if sha256(Path(katago.__file__))!=sha256(a.baseline_snapshot/'research/recipes/visual_katago/katago.py'):
        raise ValueError('CNN reference implementation differs')
    n=c['model'];cn=base['model'];batch=128;size=9
    cp=jax.eval_shape(lambda:katago.initialize(0,cn))
    graph=jax.make_jaxpr(lambda p,s,g:katago.forward(p,s,g,cn))(cp,abstract((batch,size,size,22)),abstract((batch,19)))
    ca=analyze(graph);cb=compute_budget.cnn(cn,size,batch)
    if ca['unaccounted_primitives'] or ca['counts']['multiply_add_flops']!=cb['multiply_add_flops_per_batch']:
        raise ValueError('CNN analytic/traced arithmetic disagrees')
    cases={f'{board}x{board}-past{past}':trace_transformer(n,board,past) for board in (9,19) for past in (0,32,128,256)}
    chosen=cases['9x9-past128']
    ratios={'parameters':chosen['analytic']['trainable_parameters']/cb['trainable_parameters']-1,
            'dense_multiply_add_flops':chosen['analytic']['multiply_add_flops_per_batch']/cb['multiply_add_flops_per_batch']-1,
            'unit_cost_floating_operations':chosen['traced']['floating_operations_unit_cost']/ca['floating_operations_unit_cost']-1}
    if any(abs(x)>.01 for x in ratios.values()):raise ValueError('Candidate exceeds matching tolerance')
    rejected=trace_transformer({**n,'layers':35,'mlp_hidden':2176},9,128)
    # Position-weighted history cost uses only offsets/splits, never new labels.
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);weighted={}
    for split,name in [(0,'train'),(1,'validation')]:
        lengths=[]
        for shard,game in data.indices['expert',split]:
            off=data.shards[shard]['expert_offsets'];lengths.append(int(off[game+1]-off[game]))
        counts=np.asarray([sum(length>past for length in lengths) for past in range(max(lengths))],np.int64)
        work=np.asarray([compute_budget.transformer(n,9,1,past)['multiply_add_flops_per_move'] for past in range(len(counts))])
        weighted[name]={'positions':int(counts.sum()),'mean_past_moves':float(np.dot(counts,np.arange(len(counts)))/counts.sum()),
            'mean_dense_decode_flops_per_move':float(np.dot(counts,work)/counts.sum())}
    report={'kind':'complete_neural_decode_budget_qualification','status':'passed','operator_snapshot':SOURCE.name,
        'baseline_snapshot':a.baseline_snapshot.name,'config_sha256':hashlib.sha256(canonical_json(c)).hexdigest(),
        'cnn':{'analytic':cb,'traced':ca},'transformer_model':n,'cases':cases,'relative_differences_at_reference':ratios,
        'position_weighted_decode':weighted,'previous_untrained_shape':{'model':{**n,'layers':35,'mlp_hidden':2176},
            'case':rejected,'rejection':'Unit-cost floating operation difference is above 1% once softmax and normalization are included. No learning run used this shape.'},
        'scope':'Complete logical pre-SPMD inference. FMA=2. Transcendentals counted separately and as one operation in the auxiliary unit-cost total; not a hardware FLOP-equivalence claim.',
        'physical_work':'Compiler padding, memory aliasing and latency are qualified on TPU before training. Full-sequence training arithmetic is not equated to cached decoding.',
        'opening':'No fictitious previous action at past=0; its distinct graph includes the board/readout and returns the KV cache.',
        'match_scope':'9x9, global batch 128, 128 previous moves only. Other histories and 19x19 are deliberately reported as unmatched.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical_json(report))
    a.output.chmod(0o444)
    print(json.dumps({'status':'passed','relative_differences':ratios,'sha256':sha256(a.output)}),flush=True)


if __name__=='__main__':main()
