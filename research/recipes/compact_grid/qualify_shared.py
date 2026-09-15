"""Independent traced decoding and differentiated training arithmetic ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.jaxpr_cost import analyze
from gozero.snapshots import canonical_json
import causal
import katago
import compute_budget
import policy_model
import training_arithmetic


def trace(c,past,batch=128):
    size=c['max_board_size'];s=jax.ShapeDtypeStruct
    p=jax.eval_shape(lambda:causal.initialize(0,c))
    spatial=s((batch,size,size,22),jnp.float32);glob=s((batch,19),jnp.float32)
    if past==0:graph=jax.make_jaxpr(lambda p,x,g:causal.first_move(p,x,g,c))(p,spatial,glob)
    else:
        shape=(batch,c['layers'],2*c['max_positions'],c['kv_heads'],c['width']//c['heads'])
        cache={'keys':s(shape,jnp.bfloat16),'values':s(shape,jnp.bfloat16),'lengths':s((batch,),jnp.int32),'valid':s((batch,),jnp.bool_),'network_version':s((),jnp.uint32)}
        graph=jax.make_jaxpr(lambda p,ca,a,x,g:causal.append_move(p,ca,a,x,g,c,attention_positions=past+1))(p,cache,s((batch,),jnp.int32),spatial,glob)
    traced=analyze(graph);formula=compute_budget.transformer(c,size,batch,past)
    if traced['unaccounted_primitives'] or traced['counts']['multiply_add_flops']!=formula['multiply_add_flops_per_batch']:
        raise ValueError('Decoder trace differs from analytical model: '+str(traced))
    return {'analytical':formula,'traced':traced}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();c=json.loads(a.config.read_text());base=json.loads((a.baseline/'resolved_config.json').read_text())
    model=c['model'];s=jax.ShapeDtypeStruct;bp=jax.eval_shape(lambda:katago.initialize(0,base['model']))
    baseline=analyze(jax.make_jaxpr(lambda p,x,g:katago.forward(p,x,g,base['model']))(bp,s((128,9,9,22),jnp.float32),s((128,19),jnp.float32)))
    bc=compute_budget.cnn(base['model'],9,128)
    if baseline['unaccounted_primitives'] or baseline['counts']['multiply_add_flops']!=bc['multiply_add_flops_per_batch']:raise ValueError('CNN trace failed')
    cases={str(past):trace(model,past) for past in (0,32,128,256)}
    ref=cases['128'];ratios={
        'parameters':ref['analytical']['trainable_parameters']/bc['trainable_parameters']-1,
        'dense_flops':ref['analytical']['multiply_add_flops_per_batch']/bc['multiply_add_flops_per_batch']-1,
        'unit_cost_floating_operations':ref['traced']['floating_operations_unit_cost']/baseline['floating_operations_unit_cost']-1}
    if any(abs(v)>.01 for v in ratios.values()):raise ValueError('Reference budget tolerance exceeded: '+str(ratios))
    training={};p=jax.eval_shape(lambda:causal.initialize(0,model))
    for length in c['dataset']['buckets']:
        b={'spatial':s((128,length,9,9,22),jnp.float32),'global_features':s((128,length,19),jnp.float32),'actions':s((128,length),jnp.int32),'counts':s((128,),jnp.int32),'policies':s((128,length,82),jnp.float32),'legal':s((128,length,82),jnp.bool_)}
        graph=jax.make_jaxpr(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,model)[0]))(p,b)
        training[str(length)]=training_arithmetic.count(graph,draft_valid_length=3*length if model.get('first_pass_aux_weight',0) else None)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    result={'status':'passed','kind':'single_board_shared_cpu_budget_qualification','config':c,'source_sha256':files,'cnn':{'analytical':bc,'traced':baseline},'cases':cases,'relative_differences':ratios,'training_arithmetic':training,
        'scope':'Exact traced logical decoding; FMA=2, plus unit-cost nonlinear accounting. Training matrices include rematerialization and full padded Splash tile work. TPU compilation, latency and physical MXU padding require separate qualification.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical_json(result))
    print(json.dumps({'status':'passed','relative_differences':ratios,'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
