"""Independent gradient-unit/clipping checks for the frozen Muon bridge."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import read_json,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=verify(a.snapshot)
    os.environ['JAX_PLATFORMS']='cpu'
    sys.path.insert(0,str(a.snapshot/manifest['recipe']))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optimizer
    import optimizer_io
    import muon_groups
    c=read_json(a.snapshot/'resolved_config.json')
    opt={k:v for k,v in c['learner'].items() if k not in ('games_per_host','augmentation')}
    opt.update(horizon_steps=4,max_grad_norm=64.)
    # These explicitly chosen names cover all six CNN groups and both eligible
    # and ineligible spatial kernels. Expected centers are specified separately.
    shapes={'conv_spatial.weight':(3,3,2,4),'linear_global.weight':(19,4),
        'cycles.a.normactconv1.conv.weight':(2,3,3,2,4),
        'tail.0.normactconv2.conv.weight':(3,3,2,4),
        'tail.0.normactconv1.conv.weight':(1,1,2,4),
        'tail.0.normactconv1.norm.gamma':(4,),
        'tail.0.normactconv1.norm.beta':(4,),
        'value_head.linear2.weight':(4,3),'value_head.linear2.bias':(3,)}
    params={k:jnp.asarray(np.arange(np.prod(s),dtype=np.float32).reshape(s)/512) for k,s in shapes.items()}
    gradients={k:jnp.full(s,(i+1)/128,dtype=jnp.float32) for i,(k,s) in enumerate(shapes.items())}
    state=optimizer.initialize(params)
    update=jax.jit(lambda p,s,g,loss,positions:optimizer.apply_gradient(p,s,g,loss,positions,opt))
    started=time.monotonic();report=dict(kind='joint_muon_gradient_bridge_qualification',snapshot_id=a.snapshot.name,status='running')
    try:
        p1,s1,m=update(params,state,gradients,jnp.float32(2),jnp.float32(1024))
        jax.block_until_ready((p1,s1,m))
        expected={k:np.asarray(g)*np.float32(1024) for k,g in gradients.items()}
        expected['cycles.a.normactconv1.conv.weight'][:,1,1,:,:]*=2
        expected['tail.0.normactconv2.conv.weight'][1,1,:,:]*=2
        norm=np.sqrt(sum(np.sum(g.astype(np.float64)**2) for g in expected.values()))
        scale=min(1.,64./(norm+1e-6))
        if not bool(m['accepted']) or not scale<1:raise ValueError('Clipped update was not accepted')
        np.testing.assert_allclose(float(m['source_sum_gradient_norm']),norm,rtol=2e-6,atol=0)
        np.testing.assert_allclose(float(m['clip_scale']),scale,rtol=2e-6,atol=0)
        for k,g in expected.items():
            clipped=g.astype(np.float64)*scale
            np.testing.assert_allclose(np.asarray(s1['first'][k]),.05*clipped,rtol=3e-6,atol=1e-7,err_msg=k)
            if k in s1['second']:
                np.testing.assert_allclose(np.asarray(s1['second'][k]),.005*clipped**2,rtol=4e-6,atol=1e-7,err_msg=k)
        def unchanged(p,s):
            for x,y in zip(jax.tree.leaves((p,s)),jax.tree.leaves((p1,s1))):np.testing.assert_array_equal(x,y)
        badgrad=dict(gradients);badgrad['value_head.linear2.bias']=jnp.full((3,),jnp.inf)
        cases=[(gradients,jnp.float32(2),jnp.float32(0)),
               (gradients,jnp.float32(2),jnp.float32(-1)),
               (gradients,jnp.float32(2),jnp.float32(jnp.nan)),
               (gradients,jnp.float32(jnp.nan),jnp.float32(1024)),
               (badgrad,jnp.float32(2),jnp.float32(1024))]
        for g,loss,positions in cases:
            pr,sr,mr=update(p1,s1,g,loss,positions)
            if bool(mr['accepted']):raise ValueError('Invalid bridge update accepted')
            unchanged(pr,sr)
        expired={**s1,'step':jnp.int32(4)}
        pe,se,me=update(p1,expired,gradients,jnp.float32(2),jnp.float32(1024))
        if bool(me['accepted']):raise ValueError('Expired horizon advanced')
        for x,y in zip(jax.tree.leaves((pe,se)),jax.tree.leaves((p1,expired))):np.testing.assert_array_equal(x,y)
        np_params=jax.tree.map(np.asarray,p1);np_state=jax.tree.map(np.asarray,s1)
        meta,arrays=optimizer_io.flatten(np_params,np_state,configuration_sha256='a'*64,source_sha256=a.snapshot.name)
        schema=[dict(path=k,shape=list(v.shape),dtype=str(v.dtype)) for k,v in sorted(np_params.items())]
        restored=optimizer_io.restore(meta,arrays,schema=schema,configuration_sha256='a'*64,source_sha256=a.snapshot.name)
        for x,y in zip(jax.tree.leaves(restored),jax.tree.leaves((np_params,np_state))):np.testing.assert_array_equal(x,y)
        specs=muon_groups.cnn(np_params)
        if set(x['group'] for x in specs.values())!=set(opt['rates']):raise ValueError('Not every CNN group was checked')
        report.update(status='passed',groups=sorted(opt['rates']),parameter_leaves=len(params),
            moment_arrays_checked=len(s1['first'])+len(s1['second']),rejected_updates=6,
            checkpoint_arrays=len(arrays),state_codec_exact=True,source_sum_gradient_norm=float(m['source_sum_gradient_norm']),
            clip_scale=float(m['clip_scale']),reference_arithmetic='Independent NumPy float64 global-sum, explicit eligible centers and norm/clipping; compare actual first/second moments.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report.update(seconds=time.monotonic()-started,operator_sha256=checkpoints.sha256(Path(__file__)))
        with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        a.output.chmod(0o444)
        print(json.dumps(dict(status=report['status'],sha256=checkpoints.sha256(a.output),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
