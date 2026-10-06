"""Compare live outputs, full gradients and AdamW state across padding paths."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import adamw
import compact
import compact_reference
import joint
import learner


def fixture(dtype='float32'):
    config=json.loads((Path(__file__).parent/'cpu_transformer_qualification.json').read_text())
    c={**config['model'],'dtype':dtype,'max_positions':16}
    rng=np.random.default_rng(731);n,t,size=4,6,9
    counts=np.asarray([6,3,1,0],np.int32)
    live=np.arange(t)[None,:]<counts[:,None]
    b=dict(spatial=jnp.asarray(rng.normal(size=(n,t,size,size,22))*live[...,None,None,None],jnp.float32),
        global_features=jnp.asarray(rng.normal(size=(n,t,19))*live[...,None],jnp.float32),
        actions=jnp.asarray(rng.integers(0,size*size+1,(n,t)),jnp.int32),counts=jnp.asarray(counts),
        policies=jnp.asarray(rng.dirichlet(np.ones(size*size+1),n*t).reshape(n,t,-1),jnp.float32),
        legal=jnp.ones((n,t,size*size+1),bool),
        values=jnp.asarray(np.where(live,rng.uniform(-1,1,(n,t)),np.nan),jnp.float32))
    return c,joint.initialize(731,c,config['value_model']),b


def compare(a,b,*,rtol=1e-5,atol=2e-6):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    error=0.;squared=0.;reference=0.;count=0
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_allclose(x,y,rtol=rtol,atol=atol)
        delta=x.astype(np.float64)-y.astype(np.float64)
        if delta.size:error=max(error,float(np.max(np.abs(delta))))
        squared+=float(np.sum(delta*delta));reference+=float(np.sum(x.astype(np.float64)**2));count+=x.size
    return dict(max_abs=error,relative_l2=(squared/max(reference,1e-30))**.5,elements=count)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert jax.default_backend()=='cpu' and jax.device_count()==4
    assert not args.output.exists();checks=[];started=time.time()
    for dtype in ('float32','bfloat16'):
        c,p,b=fixture(dtype);tol=dict(rtol=2e-5,atol=3e-6) if dtype=='float32' else dict(rtol=2e-2,atol=3e-4)
        # The retained implementation is independent of the new conditional.
        for chunk in (4,5):
            old=jax.jit(lambda p,b:compact_reference.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,chunk_frames=chunk,inner_rematerialize=False,with_hidden=True))(p,b)
            baseline=jax.jit(lambda p,b:compact.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,chunk_frames=chunk,inner_rematerialize=False,with_hidden=True))(p,b)
            candidate=jax.jit(lambda p,b:compact.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,chunk_frames=chunk,inner_rematerialize=False,with_hidden=True,skip_padding=True))(p,b)
            reference_check=compare(old,baseline,**tol);forward=compare(old,candidate,**tol)
            def loss(skip):return lambda p,b:joint.losses(p,b,c,value_weight=.7,chunk_frames=chunk,skip_padding=skip)
            full=jax.jit(jax.value_and_grad(loss(False),has_aux=True))(p,b)
            skip=jax.jit(jax.value_and_grad(loss(True),has_aux=True))(p,b)
            gradient=compare(full,skip,**tol)
            checks.append(dict(dtype=dtype,chunk_frames=chunk,reference=reference_check,forward=forward,gradient=gradient))
            print(json.dumps(dict(kind='padding_equivalence',**checks[-1])),flush=True)
    c,p,b=fixture()
    opt=dict(learning_rate=.001,end_learning_rate=.0003,warmup_steps=2,horizon_steps=8,
             beta1=.9,beta2=.95,epsilon=1e-8,weight_decay=.01,max_grad_norm=1.)
    mesh=Mesh(np.asarray(jax.devices()),('data',))
    for distributed in (False,True):
        original=learner.step(c,opt,value_weight=.7,chunk_frames=2,mesh=mesh if distributed else None)
        candidate=learner.step(c,opt,value_weight=.7,chunk_frames=2,mesh=mesh if distributed else None,skip_padding=True)
        original=jax.jit(original);candidate=jax.jit(candidate)
        a=(p,adamw.initialize(p));z=(p,adamw.initialize(p));records=[]
        for update in range(2):
            pa,sa,ma=original(*a,b);pz,sz,mz=candidate(*z,b)
            records.append(compare((pa,sa,ma),(pz,sz,mz),rtol=5e-4,atol=2e-5))
            assert float(ma['accepted'])==float(mz['accepted'])==1.
            a=(pa,sa);z=(pz,sz)
        checks.append(dict(kind='two_adamw_updates',distributed=distributed,comparisons=records))
        print(json.dumps(checks[-1]),flush=True)
    for counts in ([6,6,6,6],[0,0,0,0]):
        c,p,b=fixture();b={**b,'counts':jnp.asarray(counts,jnp.int32),'values':jnp.zeros((4,6))}
        fn=lambda skip:jax.jit(jax.value_and_grad(lambda p:joint.losses(p,b,c,value_weight=.7,chunk_frames=4,skip_padding=skip),has_aux=True))(p)
        checks.append(dict(kind='population_boundary',counts=counts,comparison=compare(fn(False),fn(True))))
        print(json.dumps(checks[-1]),flush=True)
    result=dict(status='passed',kind='padding_runtime_cpu_equivalence',created=time.time(),seconds=time.time()-started,
        checks=checks,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        scope='Small complete model, FP32/BF16, partial/empty chunks, empty shard, four-device collectives, full gradients and two AdamW updates. Full-size TPU timing/equivalence is separate.')
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',checks=len(checks),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
