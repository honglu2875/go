"""Small complete-model correctness checks before sparse TPU experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
os.environ['JAX_PLATFORMS']='cpu'
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--dense-reference',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
RECIPE=args.dense_reference or ROOT/'research/recipes/strong19_moe'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
import numpy as np
import jax
import jax.numpy as jnp
import joint,learner,adamw,optimizer_io
from gozero import checkpoints
from gozero.snapshots import canonical_json


def close(a,b,rtol=2e-4,atol=3e-5):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    error=0.
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_allclose(x,y,rtol=rtol,atol=atol)
        error=max(error,float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)))))
    return error


def main():
    start=time.time();checks=[]
    base=json.loads((ROOT/'research/studies/strong19_recovery/flat-config-001.json').read_text())
    c=dict(base['model'],width=16,encoder_width=16,layers=2,mlp_hidden=32,heads=2,kv_heads=1,
        max_board_size=3,max_positions=4,encoder_blocks=2,encoder_attention_blocks=1,
        encoder_attention_heads=2,encoder_attention_mlp_hidden=16,connector_channels=2,
        policy_context_dim=4,encoder_expansion=2,attention_backend='xla',dtype='float32')
    v=dict(hidden=8,spatial_channels=8)
    opt={k:x for k,x in base['learner'].items() if k not in ('games_per_host','augmentation')};opt['horizon_steps']=512
    rng=np.random.default_rng(42)
    batch=dict(spatial=rng.normal(size=(4,4,3,3,22)).astype('float32'),
        global_features=rng.normal(size=(4,4,19)).astype('float32'),
        actions=rng.integers(0,10,(4,4),dtype=np.int32),counts=np.array([4,3,1,0],np.int32),
        policies=np.full((4,4,10),.1,np.float32),legal=np.ones((4,4,10),bool),
        values=rng.uniform(-1,1,(4,4)).astype('float32'))
    if args.dense_reference:
        params=joint.initialize(44,c,v)
        f=jax.jit(jax.value_and_grad(lambda p:learner.loss(p,batch,c,value_weight=.7,path='bounded',chunk_frames=2,skip_padding=True),has_aux=True))
        a=f(params)
        np.savez(args.output,**{'p/'+k:np.asarray(x) for k,x in params.items()},
                 **{'g/'+k:np.asarray(x) for k,x in a[1].items()},loss=np.asarray(a[0][0]))
        return
    params=joint.initialize(44,c,v)
    f=jax.jit(jax.value_and_grad(lambda p:learner.loss(p,batch,c,value_weight=.7,path='bounded',chunk_frames=2,skip_padding=True),has_aux=True))
    a=f(params)
    reference=np.load(STUDY/'dense-reference-001.npz')
    for k,x in params.items():np.testing.assert_array_equal(x,reference['p/'+k])
    for k,x in a[1].items():np.testing.assert_array_equal(x,reference['g/'+k])
    np.testing.assert_array_equal(a[0][0],reference['loss'])
    checks.append(dict(name='dense parent clone initialization loss and all gradients exact',status='passed'))
    for precision in ('float32','bfloat16'):
        cfg={**c,'dtype':precision,'moe':dict(experts=4,top_k=2,backend='ragged',balance_weight=.01,z_weight=.001)}
        params=joint.initialize(44,cfg,v)
        for k in set(params)&set(joint.initialize(44,c,v)):
            np.testing.assert_array_equal(params[k],joint.initialize(44,c,v)[k])
        assert adamw.decay('blocks.moe.router') and adamw.decay('encoder.blocks.moe.up')
        assert not adamw.decay('encoder.blocks.moe.up_bias')
        def objective(p,b,path):return learner.loss(p,b,cfg,value_weight=.7,path=path,chunk_frames=2,skip_padding=True)
        bounded=jax.jit(jax.value_and_grad(lambda p,b:objective(p,b,'bounded'),has_aux=True))
        actual=bounded(params,batch)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actual))
        assert all(float(jnp.linalg.norm(g))>0 for k,g in actual[1].items() if k.endswith('.moe.router'))
        material=jax.jit(jax.value_and_grad(lambda p,b:objective(p,b,'materialized'),has_aux=True))(params,batch)
        tolerance=3e-4 if precision=='float32' else .04
        error=close(actual,material,rtol=tolerance,atol=tolerance)
        checks.append(dict(name=precision+' bounded/materialized loss and all gradients',status='passed',max_abs=error))
        # Same executable with/without skipping wholly padded chunks.
        skipped=bounded(params,{**batch,'encoder_counts':batch['counts']})
        noskip=bounded(params,{**batch,'encoder_counts':np.full(4,4,np.int32)})
        close(skipped,noskip,rtol=0,atol=0)
        forward=jax.jit(lambda p,b:joint.forward(p,b,cfg,training=True,chunk_frames=2,skip_padding=True))
        changed={k:np.array(x,copy=True) for k,x in batch.items()}
        changed['actions'][:,1:]=(changed['actions'][:,1:]+1)%10
        changed['spatial'][:,2:]+=3;changed['global_features'][:,2:]-=2
        before,after=forward(params,batch),forward(params,changed)
        close({k:x[:,:2] for k,x in before.items() if k!='router_statistics'},
              {k:x[:,:2] for k,x in after.items() if k!='router_statistics'},rtol=0,atol=0)
        checks.append(dict(name=precision+' padding and no future/current-action leakage',status='passed'))
        decode_batch={k:x[:2,:3] if x.ndim>1 else np.full(2,3,np.int32) for k,x in batch.items()}
        full=jax.jit(lambda p,b:joint.forward(p,b,cfg))(params,decode_batch)
        output,cache=jax.jit(lambda p,s,g:joint.first_move(p,s,g,cfg))(params,decode_batch['spatial'][:,0],decode_batch['global_features'][:,0])
        outputs=[output]
        append=jax.jit(lambda p,cache,a,s,g:joint.append_move(p,cache,a,s,g,cfg,attention_positions=3))
        for t in (1,2):
            output,cache=append(params,cache,decode_batch['actions'][:,t-1],decode_batch['spatial'][:,t],decode_batch['global_features'][:,t]);outputs.append(output)
        assert np.asarray(cache['valid']).all()
        for name in ('policy','value','value_logits'):
            close(jnp.stack([x[name] for x in outputs],1),full[name],rtol=tolerance,atol=tolerance)
        checks.append(dict(name=precision+' complete causal history and incremental decode',status='passed'))
        update=jax.jit(learner.step(cfg,opt,value_weight=.7,chunk_frames=2,skip_padding=True))
        state=adamw.initialize(params)
        params,state,metrics=update(params,state,batch)
        assert float(metrics['accepted'])==1
        metadata,arrays=optimizer_io.flatten(params,state,configuration_sha256='1'*64,source_sha256='2'*64)
        with tempfile.TemporaryDirectory(prefix='gozero-moe-') as tmp:
            path=Path(tmp)/'checkpoint';digest=checkpoints.write(path,state={'optimizer':metadata},arrays=arrays,actors='{}')
            saved,payload,_=checkpoints.read(path,expected_manifest_sha256=digest)
            rp,rs=optimizer_io.restore(saved['optimizer'],payload,schema=metadata['parameter_schema'],configuration_sha256='1'*64,source_sha256='2'*64)
        close(update(params,state,batch),update(jax.tree.map(jnp.asarray,rp),jax.tree.map(jnp.asarray,rs),batch),rtol=0,atol=0)
        checks.append(dict(name=precision+' actual optimizer checkpoint continuation and semantic decay',status='passed'))
        if precision=='float32':
            from jax.sharding import Mesh,PartitionSpec as P
            assert len(jax.devices())==4
            mesh=Mesh(np.array(jax.devices()),('data',))
            sharded=jax.shard_map(lambda p,b:learner.loss(p,b,cfg,value_weight=.7,path='bounded',chunk_frames=2,skip_padding=True,axis_name='data'),mesh=mesh,in_specs=(P(),P('data')),out_specs=P(),check_vma=False)
            distributed=jax.jit(jax.value_and_grad(sharded,has_aux=True))(params,batch)
            unsharded=bounded(params,batch)
            error=close(distributed,unsharded,rtol=3e-4,atol=3e-5)
            checks.append(dict(name='four-device global router objective/gradients including empty shard',status='passed',max_abs=error))
        print(json.dumps(dict(precision=precision,status='passed',checks=len(checks),seconds=time.time()-start)),flush=True)
        jax.clear_caches()
    result=dict(status='passed',checks=checks,seconds=time.time()-start,operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*RECIPE.glob('*.py'),ROOT/'packages/gozero/src/gozero/moe.py']})
    with args.output.open('xb') as stream:stream.write(canonical_json(result))
    print(json.dumps(dict(status='passed',checks=len(checks),seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
