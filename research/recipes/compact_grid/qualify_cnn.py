"""Trace the complete CNN decode and whole-game differentiated training work."""
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
import compute_budget
import katago
import policy_config
import policy_model
import training_arithmetic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    c=policy_config.validate(json.loads(a.config.read_text()));net=c['model'];s=jax.ShapeDtypeStruct
    if net['architecture']!='katago_nested_policy':raise ValueError('Expected CNN')
    p=jax.eval_shape(lambda:katago.initialize(0,net))
    graph=jax.make_jaxpr(lambda p,x,g:katago.forward(p,x,g,net))(p,s((128,9,9,22),jnp.float32),s((128,19),jnp.float32))
    traced=analyze(graph);analytical=compute_budget.cnn(net,9,128)
    if traced['unaccounted_primitives'] or traced['counts']['multiply_add_flops']!=analytical['multiply_add_flops_per_batch']:
        raise ValueError('CNN complete decoding accounting differs')
    training={}
    for length in c['dataset']['buckets']:
        b={'spatial':s((128,length,9,9,22),jnp.float32),'global_features':s((128,length,19),jnp.float32),
           'actions':s((128,length),jnp.int32),'counts':s((128,),jnp.int32),
           'policies':s((128,length,82),jnp.float32),'legal':s((128,length,82),jnp.bool_)}
        graph=jax.make_jaxpr(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,net)[0]))(p,b)
        training[str(length)]=training_arithmetic.count(graph)
    result=dict(status='passed',kind='strong9_cnn_cpu_budget',config=c,
        cnn=dict(analytical=analytical,traced=traced),training_arithmetic=training,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    with a.output.open('xb') as f:f.write(canonical_json(result))
    print(json.dumps(dict(status='passed',parameters=analytical['trainable_parameters'],flops=analytical['multiply_add_flops_per_move'])))


if __name__=='__main__':main()
