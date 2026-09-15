"""Numerically qualify draft-mask Splash outputs and Q/K/V gradients on the pod."""
import gc
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding,PartitionSpec as P
import draft_model


def run(c,buckets,mesh,replica):
    from gozero.checkpoints import sha256
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as source
    if len(jax.devices())!=16 or jax.process_count()!=4:raise ValueError('Expected the full pod')
    sharding=NamedSharding(mesh,P('data'));heads=c['heads'];kv=c['kv_heads'];dim=c['width']//heads
    rows=[]
    for bucket in buckets:
        length=3*bucket
        def create(key):
            keys=jax.random.split(key,4)
            return tuple(.3*jax.random.normal(k,(16,length,h,dim),jnp.bfloat16) for k,h in zip(keys,(heads,kv,kv,heads)))
        inputs=jax.jit(create,out_shardings=(sharding,)*4)(jax.random.key(28371+length))
        def functions(backend):
            attend=lambda q,k,v:draft_model.attention(q,k,v,{**c,'attention_backend':backend})
            output=jax.shard_map(attend,mesh=mesh,in_specs=(P('data'),)*3,out_specs=P('data'),check_vma=False)
            def loss(q,k,v,do):return jax.lax.psum(jnp.sum(attend(q,k,v)*do),'data')/(16*length*heads*dim)
            objective=jax.shard_map(loss,mesh=mesh,in_specs=(P('data'),)*4,out_specs=P(),check_vma=False)
            return jax.jit(output),jax.jit(jax.value_and_grad(objective,argnums=(0,1,2)))
        dense,dense_grad=functions('xla');splash,splash_grad=functions('splash')
        expected=dense(*inputs[:3]);_,reference_grad=dense_grad(*inputs)
        actual=splash(*inputs[:3]);_,grad=splash_grad(*inputs)
        @jax.jit
        def errors(a,b):
            a=a.astype(jnp.float32);b=b.astype(jnp.float32)
            return jnp.max(jnp.abs(a-b)),jnp.sqrt(jnp.sum((a-b)**2)/jnp.maximum(jnp.sum(b*b),1e-30))
        discrepancies=[list(map(float,replica(errors(x,y)))) for x,y in zip((actual,*grad),(expected,*reference_grad))]
        if any(not np.isfinite(v) for pair in discrepancies for v in pair) or any(a>.05 or r>.03 for a,r in discrepancies):
            raise ValueError('Draft Splash output/gradient mismatch: '+str(discrepancies))
        row={'bucket':bucket,'tokens':length,'padded_tokens':((length+511)//512)*512,'global_batch':16,
             'output_q_k_v_errors':discrepancies,'absolute_tolerance':.05,'relative_l2_tolerance':.03}
        rows.append(row);print(json.dumps({'kind':'draft_kernel_qualification',**row}),flush=True)
        del inputs,actual,expected,grad,reference_grad,dense,dense_grad,splash,splash_grad
        gc.collect()
    return {'status':'passed','cases':rows,'kernel_source_sha256':sha256(Path(source.__file__)),
            'scope':'Dynamic BF16 synthetic output and Q/K/V-gradient checks at every actual training length; global batch16. Full-model/global-batch128 execution is qualified separately.'}
