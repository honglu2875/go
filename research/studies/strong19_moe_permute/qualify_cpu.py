"""Qualify routing permutation VJPs independently of the ongoing learner."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'research/recipes/strong19_moe_permute'),str(ROOT/'packages/gozero/src')]
import numpy as np
import jax
import jax.numpy as jnp
import joint,learner,adamw,optimizer_io
from gozero import moe,checkpoints
from gozero.snapshots import canonical_json


def close(a,b,*,precision):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    maximum=0.;squared=0.;norm=0.;dot=0.;actual_norm=0.
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        maximum=max(maximum,float(np.max(np.abs(x-y))))
        squared+=float(np.sum((x-y)**2));norm+=float(np.sum(y*y));actual_norm+=float(np.sum(x*x));dot+=float(np.sum(x*y))
        np.testing.assert_allclose(x,y,rtol=3e-4 if precision=='float32' else .02,atol=3e-5 if precision=='float32' else .01)
    relative=(squared/max(norm,1e-30))**.5;cosine=dot/max((norm*actual_norm)**.5,1e-30)
    assert relative<(1e-4 if precision=='float32' else .003)
    return dict(max_abs=maximum,relative_l2=relative,cosine=cosine)


def main():
    start=time.time();checks=[]
    base=json.loads((STUDY/'proposal-config-001.json').read_text())
    c=dict(base['model'],width=16,encoder_width=16,layers=2,mlp_hidden=32,heads=2,kv_heads=1,max_board_size=3,max_positions=8,
        encoder_blocks=2,encoder_attention_blocks=1,encoder_attention_heads=2,encoder_attention_mlp_hidden=16,connector_channels=2,
        policy_context_dim=4,encoder_expansion=2,attention_backend='xla')
    c['moe']={**c['moe'],'backend':'ragged'}
    rng=np.random.default_rng(48)
    batch=dict(spatial=rng.normal(size=(4,6,3,3,22)).astype('float32'),global_features=rng.normal(size=(4,6,19)).astype('float32'),
        actions=rng.integers(0,10,(4,6),dtype=np.int32),counts=np.array([6,3,2,0],np.int32),policies=np.full((4,6,10),.1,np.float32),
        legal=np.ones((4,6,10),bool),values=rng.uniform(-1,1,(4,6)).astype('float32'))
    opt={k:v for k,v in base['learner'].items() if k not in ('games_per_host','augmentation')};opt['horizon_steps']=512
    for precision in ('float32','bfloat16'):
        configs=[{**c,'dtype':precision,'moe':{**c['moe'],'permutation_vjp':flag}} for flag in (False,True)]
        p=joint.initialize(44,configs[0],dict(hidden=8,spatial_channels=8));state=adamw.initialize(p)
        gradients=[]
        for flag,chunk in ((False,2),(True,2)):
            cfg=configs[int(flag)]
            fn=jax.jit(jax.value_and_grad(lambda p,b:learner.loss(p,b,cfg,value_weight=.7,path='bounded',chunk_frames=chunk,skip_padding=True),has_aux=True))
            observed=fn(p,batch)
            if not gradients:gradients.append(observed)
            else:checks.append(dict(name=precision+' complete loss/metrics/gradient',remat=flag,chunk=chunk,status='passed',error=close(observed,gradients[0],precision=precision)))
        updates=[jax.jit(learner.step(cfg,opt,value_weight=.7,chunk_frames=2,skip_padding=True)) for cfg in configs]
        states=[(p,state),(p,state)]
        for step in range(3):
            result=[]
            for i,fn in enumerate(updates):
                pp,ss,metrics=fn(*states[i],batch);assert float(metrics['accepted'])==1;states[i]=(pp,ss);result.append((pp,ss,metrics))
            error=close(result[1],result[0],precision=precision)
            checks.append(dict(name=precision+' same-state optimizer trajectory',step=step+1,status='passed',error=error))
        pp,ss=states[1];meta,arrays=optimizer_io.flatten(pp,ss,configuration_sha256='1'*64,source_sha256='2'*64)
        with tempfile.TemporaryDirectory(prefix='gozero-permutation-vjp-') as tmp:
            path=Path(tmp)/'checkpoint';digest=checkpoints.write(path,state=dict(optimizer=meta),arrays=arrays,actors='{}')
            saved,payload,_=checkpoints.read(path,expected_manifest_sha256=digest)
            restored,rs=optimizer_io.restore(saved['optimizer'],payload,schema=meta['parameter_schema'],configuration_sha256='1'*64,source_sha256='2'*64)
        a=updates[1](pp,ss,batch);b=updates[1](jax.tree.map(jnp.asarray,restored),jax.tree.map(jnp.asarray,rs),batch)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
        checks.append(dict(name=precision+' optimizer checkpoint continuation exact',status='passed'))
        print(json.dumps(dict(precision=precision,status='passed',seconds=time.time()-start)),flush=True);jax.clear_caches()
    result=dict(status='passed',checks=checks,seconds=time.time()-start,operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'packages/gozero/src/gozero/moe.py',*Path(ROOT/'research/recipes/strong19_moe_permute').glob('*.py')]},
        scope='Small-model CPU correctness and optimizer behavior only; no TPU speed or compiled memory claim.')
    with (STUDY/'cpu-qualification-001.json').open('xb') as f:f.write(canonical_json(result))

if __name__=='__main__':main()
