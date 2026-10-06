"""Verify the new pool is shared consistently by packed training and decoding."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time
os.environ['JAX_PLATFORMS']='cpu'
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
RECIPE=ROOT/'research/recipes/strong19_attention_pool'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
import numpy as np
import jax
import jax.numpy as jnp
import joint
from gozero.snapshots import canonical_json


def main():
    start=time.time();checks=[]
    config=json.loads((STUDY/'attention-config-001.json').read_text())
    c=dict(config['model'],width=16,encoder_width=16,layers=2,mlp_hidden=32,heads=2,kv_heads=1,
        max_board_size=3,max_positions=4,encoder_blocks=2,encoder_attention_blocks=1,
        encoder_attention_heads=2,encoder_attention_mlp_hidden=16,connector_channels=2,
        policy_context_dim=4,board_pool_heads=2,board_pool_mlp_hidden=32,encoder_expansion=2,
        attention_backend='xla')
    rng=np.random.default_rng(42)
    batch=dict(spatial=rng.normal(size=(2,3,3,3,22)).astype('float32'),
        global_features=rng.normal(size=(2,3,19)).astype('float32'),
        actions=rng.integers(0,10,(2,3),dtype=np.int32),counts=np.full(2,3,np.int32))
    for precision in ('float32','bfloat16'):
        cfg={**c,'dtype':precision};p=joint.initialize(44,cfg,dict(hidden=8,spatial_channels=8))
        full=jax.jit(lambda p,b:joint.forward(p,b,cfg,training=False))(p,batch)
        packed=jax.jit(lambda p,b:joint.forward(p,b,cfg,training=True,chunk_frames=2,skip_padding=True))(p,batch)
        output,cache=jax.jit(lambda p,s,g:joint.first_move(p,s,g,cfg))(p,batch['spatial'][:,0],batch['global_features'][:,0])
        outputs=[output]
        append=jax.jit(lambda p,cache,a,s,g:joint.append_move(p,cache,a,s,g,cfg,attention_positions=3))
        for t in (1,2):
            output,cache=append(p,cache,batch['actions'][:,t-1],batch['spatial'][:,t],batch['global_features'][:,t])
            outputs.append(output)
        assert np.asarray(cache['valid']).all()
        np.testing.assert_array_equal(cache['lengths'],np.full(2,5,np.int32))
        tolerance=3e-5 if precision=='float32' else .02
        errors={}
        for name in ('policy','value','value_logits'):
            decode=jnp.stack([x[name] for x in outputs],axis=1)
            for label,actual in (('incremental',decode),('packed_main',packed[name])):
                np.testing.assert_allclose(actual,full[name],rtol=tolerance,atol=tolerance)
                errors[label+'/'+name]=float(jnp.max(jnp.abs(actual-full[name])))
        checks.append(dict(precision=precision,status='passed',max_abs=errors,rtol=tolerance,atol=tolerance))
        print(json.dumps(checks[-1]),flush=True)
        jax.clear_caches()
    result=dict(status='passed',checks=checks,seconds=time.time()-start,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RECIPE.glob('*.py')},
        scope='Small-model complete-history vs packed-main vs opening+cached appends; CPU only, both precisions.')
    with (STUDY/'decode-qualification-001.json').open('xb') as f:f.write(canonical_json(result))


if __name__=='__main__':main()
