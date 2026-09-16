"""Paired head-only repair on frozen full-model features; not a learning-scale claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PARENT=ROOT/'.gozero/snapshots/5e3978d62bc6d8611cd703f5ac894bb70a73289b7a6d62fe615c44e2727a5173'
sys.path.insert(0,str(PARENT/'research/recipes/strong19_train'))
import heads
import adamw
from analyze_diagnostic import rows,sha,summary


def cross_entropy(logits,target):
    target=target.astype(jnp.float32)
    q=jnp.stack(((1+target)/2,(1-target)/2,jnp.zeros_like(target)),axis=-1)
    return -jnp.sum(q*jax.nn.log_softmax(logits.astype(jnp.float32)),axis=-1)


def qualify():
    z=jnp.asarray([0.,0.,20.]);y=jnp.asarray(-1.)
    mse=lambda z,y:(heads.signed_value(z)-y)**2
    mg=jax.grad(mse)(z,y);cg=jax.grad(cross_entropy)(z,y)
    assert float(jnp.linalg.norm(mg))<1e-7 and float(jnp.linalg.norm(cg))>1.
    for y in (-.95,-.4,0.,.3,.9):
        q=jnp.asarray([(1+y)/2,(1-y)/2,1e-12]);z=jnp.log(q)
        assert abs(float(heads.signed_value(z))-y)<1e-6
        assert float(jnp.linalg.norm(jax.grad(cross_entropy)(z,jnp.asarray(y))))<1e-6
    z=jnp.asarray([1.2,-.7,.3]);y=jnp.asarray(.7)
    np.testing.assert_allclose(jax.grad(cross_entropy)(z,y),
        jax.nn.softmax(z)-jnp.asarray([.85,.15,0]),rtol=1e-6,atol=1e-7)
    return dict(saturated_mse_gradient=np.asarray(mg).tolist(),saturated_ce_gradient=np.asarray(cg).tolist(),
                expectation_consistency=True,analytic_gradient_checked=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--attempt',type=Path,required=True)
    a.add_argument('--head',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if jax.default_backend()!='cpu':raise ValueError('Use CPU for this small frozen-feature probe')
    qualification=qualify();start=time.time()
    sets={}
    for mode in ('training','evaluation'):
        parts=[rows(args.attempt/f'rank-{r}/artifacts/{mode}.npz') for r in range(4)]
        sets[mode]={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    train=sets['training'];ev=sets['evaluation'];chosen=train['split']==0
    h=jnp.asarray(train['latent'][chosen]);a=jnp.asarray(train['aux_latent'][chosen]);y=jnp.asarray(train['values'][chosen])
    original={k:jnp.asarray(v) for k,v in dict(np.load(args.head,allow_pickle=False)).items()}
    original_logits=jax.jit(heads.temporal)(original,jnp.asarray(ev['latent']))
    max_abs=float(np.max(abs(np.asarray(original_logits)-ev['logits'])))
    value_max_abs=float(np.max(abs(np.asarray(heads.signed_value(original_logits))-np.asarray(heads.signed_value(ev['logits'])))))
    # TPU default FP32 dot precision differs from CPU accumulation. The repair
    # comparison uses CPU arithmetic for BOTH arms and the initial baseline.
    if max_abs>.05 or value_max_abs>.01:raise ValueError('CPU and TPU head outputs differ beyond numerical tolerance')
    rng=np.random.default_rng(927402);draws=jnp.asarray(rng.integers(0,len(y),(256,256)))
    history={}
    for mode in ('mse','ce'):
        def objective(p,indices):
            z=heads.temporal(p,h[indices]);az=heads.temporal(p,a[indices]);target=y[indices]
            loss=(lambda z:jnp.mean((heads.signed_value(z)-target)**2)) if mode=='mse' else (lambda z:jnp.mean(cross_entropy(z,target)))
            return .75*loss(z)+.25*loss(az)
        def step(carry,indices):
            p,s=carry;value,g=jax.value_and_grad(objective)(p,indices)
            p,s,m=adamw.apply_gradient(p,s,g,value,learning_rate=.0003,architecture='causal_visual_policy')
            return (p,s),(value,m['accepted'])
        run=jax.jit(lambda p,s,d:jax.lax.scan(step,(p,s),d))
        p=original;s=adamw.initialize(p);history[mode]=[];previous=0
        for turn in (0,1,4,16,64,128,256):
            if turn:
                (p,s),(_,accepted)=run(p,s,draws[previous:turn]);jax.block_until_ready(p)
                if not np.asarray(accepted).all():raise ValueError('Rejected probe update')
            z=np.asarray(jax.jit(heads.temporal)(p,jnp.asarray(ev['latent'])))
            row=dict(turn=turn,**{name:summary(z[ev['split']==i],ev['values'][ev['split']==i]) for i,name in ((0,'train'),(1,'validation'))})
            history[mode].append(row);previous=turn
            print(json.dumps(dict(mode=mode,turn=turn,train_mse=row['train']['value_mse'],validation_mse=row['validation']['value_mse'],validation_probabilities=row['validation']['probability_mean'])),flush=True)
    result=dict(status='passed',kind='frozen_feature_value_repair',qualification=qualification,
        backbone_updated=False,seed=927402,updates=256,batch_positions=256,learning_rate=.0003,
        objective_auxiliary_weight=.25,head_sha256=sha(args.head),operator_sha256=sha(Path(__file__)),
        diagnostic_attempt=args.attempt.name,train_positions=int(chosen.sum()),validation_positions=int((ev['split']==1).sum()),
        initial_cpu_tpu_max_abs_logits=max_abs,initial_cpu_tpu_max_abs_value=value_max_abs,
        histories=history,elapsed_seconds=time.time()-start,
        caveat='Eight train and eight validation games, stratified by opponent; diagnostic of recoverable frozen representations, not end-to-end learning or strength evidence.')
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
