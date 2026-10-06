"""CPU qualification for the new connector and complete training integration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
RECIPE = ROOT/'research/recipes/strong19_attention_pool'
sys.path[:0] = [str(RECIPE), str(ROOT/'packages/gozero/src')]
import numpy as np
import jax
import jax.numpy as jnp
import board_pool
import encoder
import causal
import joint
import learner
import adamw
import optimizer_io
import train_config
from gozero import checkpoints
from gozero.snapshots import canonical_json


def exact(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def main():
    started = time.time()
    assert all(d.platform == 'cpu' for d in jax.devices())
    base = json.loads((STUDY/'flat-config-001.json').read_text())
    candidate = json.loads((STUDY/'attention-config-001.json').read_text())
    train_config.validate(base); train_config.validate(candidate)
    c = dict(candidate['model'], width=16, encoder_width=16, layers=2, mlp_hidden=32,
        heads=2, kv_heads=1, max_board_size=3, max_positions=8, encoder_blocks=2,
        encoder_attention_blocks=1, encoder_attention_heads=2, encoder_attention_mlp_hidden=16,
        connector_channels=2, policy_context_dim=4, board_pool_heads=2, board_pool_mlp_hidden=32,
        encoder_expansion=2, attention_backend='xla', dtype='float32')
    flat = {k:v for k,v in c.items() if not k.startswith('board_pool_')}
    checks=[]
    # Compare the independent, projected K/V attention reference, including all
    # parameter and feature derivatives, before introducing reduced precision.
    p = board_pool.initialize(732, c)
    x = jax.random.normal(jax.random.key(16), (2,3,3,3,16))
    weights = jax.random.normal(jax.random.key(18), (2,3,16))
    def objective(p, x, reference):
        y = board_pool.forward(p, x, c, reference=reference)
        return jnp.sum(y*weights), y
    a = jax.jit(jax.value_and_grad(lambda p,x:objective(p,x,False), (0,1), has_aux=True))(p,x)
    b = jax.jit(jax.value_and_grad(lambda p,x:objective(p,x,True), (0,1), has_aux=True))(p,x)
    for av,bv in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_allclose(av,bv,rtol=2e-4,atol=2e-5)
    checks.append(dict(name='FP32 attention algebra and complete gradients',status='passed',
        max_absolute_error=max(float(jnp.max(jnp.abs(av-bv))) for av,bv in zip(jax.tree.leaves(a),jax.tree.leaves(b)))))
    # No stochastic draw in a common tensor may move when the connector changes.
    ep0,ep1=encoder.initialize(82,flat),encoder.initialize(82,c)
    common=sorted(set(ep0)&set(ep1))
    exact({k:ep0[k] for k in common},{k:ep1[k] for k in common})
    checks.append(dict(name='Common encoder initialization identity',status='passed',tensors=len(common)))
    rng=np.random.default_rng(17)
    batch=dict(spatial=rng.normal(size=(4,4,3,3,22)).astype('float32'),
        global_features=rng.normal(size=(4,4,19)).astype('float32'),
        actions=rng.integers(0,10,(4,4),dtype=np.int32), counts=np.array([4,3,1,0],np.int32),
        policies=np.full((4,4,10),.1,np.float32),legal=np.ones((4,4,10),bool),
        values=rng.uniform(-1,1,(4,4)).astype('float32'))
    value_config=dict(hidden=8,spatial_channels=8)
    opt={k:v for k,v in base['learner'].items() if k not in ('games_per_host','augmentation')}
    opt.update(horizon_steps=512)
    for precision in ('float32','bfloat16'):
        cfg={**c,'dtype':precision}
        params=joint.initialize(83,cfg,value_config)
        # A shared conditional executable is an exact padding control, including
        # partial chunks and an entirely empty game, without compiler confounds.
        def objective(p,b):
            return learner.loss(p,b,cfg,value_weight=.7,path='bounded',chunk_frames=2,skip_padding=True)
        lossgrad=jax.jit(jax.value_and_grad(objective,has_aux=True))
        dense={**batch,'encoder_counts':np.full_like(batch['counts'],4)}
        skipped={**batch,'encoder_counts':batch['counts']}
        dense_result=lossgrad(params,dense);skip_result=lossgrad(params,skipped)
        exact(dense_result,skip_result)
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(skip_result))
        assert float(jnp.linalg.norm(skip_result[1]['encoder.pool.key.weight']))>0
        checks.append(dict(name=precision+' integrated loss/gradient and padding control',status='passed'))
        # Changing this turn's played action and all subsequent observations must
        # leave both policy paths and value paths up to this turn unchanged.
        forward=jax.jit(lambda p,b:joint.forward(p,b,cfg,training=True,chunk_frames=2,skip_padding=True))
        changed={k:np.array(v,copy=True) for k,v in batch.items()}
        changed['actions'][:,1:]=(changed['actions'][:,1:]+1)%10
        changed['spatial'][:,2:]+=3
        changed['global_features'][:,2:]-=2
        before,after=forward(params,batch),forward(params,changed)
        exact({k:v[:,:2] for k,v in before.items()},{k:v[:,:2] for k,v in after.items()})
        checks.append(dict(name=precision+' no current-action/future leakage',status='passed'))
        update=jax.jit(learner.step(cfg,opt,value_weight=.7,chunk_frames=2,skip_padding=True))
        state=adamw.initialize(params)
        for _ in range(2):params,state,metrics=update(params,state,batch)
        metadata,arrays=optimizer_io.flatten(params,state,configuration_sha256='1'*64,source_sha256='2'*64)
        with tempfile.TemporaryDirectory(prefix='attention-pool-') as tmp:
            path=Path(tmp)/'checkpoint'
            digest=checkpoints.write(path,state={'optimizer':metadata},arrays=arrays,actors='{}')
            saved,payload,_=checkpoints.read(path,expected_manifest_sha256=digest)
            rp,rs=optimizer_io.restore(saved['optimizer'],payload,schema=metadata['parameter_schema'],
                configuration_sha256='1'*64,source_sha256='2'*64)
        uninterrupted=update(params,state,batch)
        restored=update(jax.tree.map(jnp.asarray,rp),jax.tree.map(jnp.asarray,rs),batch)
        exact(uninterrupted,restored)
        assert float(uninterrupted[2]['accepted'])==1
        checks.append(dict(name=precision+' actual checkpoint optimizer continuation',status='passed'))
        print(json.dumps(dict(precision=precision,status='passed',checks=len(checks))),flush=True)
        jax.clear_caches()
    # Full size shape accounting requires no accelerator or full-size weights.
    budgets={}
    for name,config in (('flat',base),('attention',candidate)):
        model=config['model'];schema=joint.parameter_schema(model,config['value_model'])
        w,d,area=model['encoder_width'],model['width'],19**2
        kv=model['kv_heads']*(d//model['heads']);f=model['mlp_hidden'];layers=model['layers']
        # Two cached tokens per appended move: previous action, current board.
        temporal_dense=2*2*layers*(2*d*d+2*d*kv+3*d*f)
        policy=2*d*(area+1)+2*area*w+2*d*64+2*area*w*64+2*area*64
        value=2*d*256+2*256*3
        e=encoder.flops(model,19)
        decode={}
        for context in (0,128,512,768,1534):
            # Current append attends to K cached tokens, then K+1 and K+2.
            attn=4*layers*d*(2*context+3)
            decode[str(context)]=dict(encoder=e,temporal_dense=temporal_dense,temporal_attention=attn,
                policy=policy,value=value,total_matrix_flops=e+temporal_dense+attn+policy+value)
        budgets[name]=dict(parameters=sum(v['elements'] for v in schema),
            encoder_parameters=sum(v['elements'] for v in schema if v['path'].startswith('encoder.')),
            connector_parameters=sum(v['elements'] for v in schema if v['path'].startswith(('encoder.pool.',
                'encoder.compress.','encoder.flat.'))),decode=decode,parameter_schema=schema)
    assert budgets['flat']['parameters']==232011540
    assert abs(budgets['attention']['parameters']/budgets['flat']['parameters']-1)<.001
    assert all(abs(budgets['attention']['decode'][k]['total_matrix_flops']/v['total_matrix_flops']-1)<.001
               for k,v in budgets['flat']['decode'].items())
    result=dict(status='passed',started=started,finished=time.time(),checks=checks,budgets=budgets,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RECIPE.glob('*.py')},
        scope='CPU numerical and integration checks; full-size TPU resource/finite-update guards remain live.')
    (STUDY/'cpu-qualification-002.json').write_bytes(canonical_json(result))
    print(json.dumps(dict(status='passed',checks=len(checks),seconds=time.time()-started,
        parameters={k:v['parameters'] for k,v in budgets.items()})),flush=True)


if __name__=='__main__':main()
