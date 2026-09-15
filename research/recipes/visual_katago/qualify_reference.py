#!/usr/bin/env python3
"""Compare primary/helper outputs and every active gradient to official KataGo."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
import katago
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); verify(SOURCE)
    if sha256(args.reference/'manifest.json') != args.reference_sha256: raise ValueError('Reference identity differs')
    meta=read_json(args.reference/'manifest.json')
    if sha256(args.reference/'reference.npz') != meta['arrays_sha256']: raise ValueError('Reference arrays differ')
    c=dict(architecture='katago_nested_policy',width=32,mid_width=16,gpool_width=4,policy_width=8,
        layers=len(meta['config']['block_kind']),dtype='float32',rematerialize=True,microbatch=2,
        max_board_size=5,max_positions=512,norm_epsilon=1e-4)
    params=katago.initialize(0,c); names=katago.reference_names(params,c)
    with np.load(args.reference/'reference.npz',allow_pickle=False) as f: data=dict(f)
    def converted(kind):
        result={}
        for key,items in names.items():
            values=[]
            for name,index in items:
                x=data[kind+'.'+name]
                if name.endswith(('conv2p.weight','linear_pass2.weight')): x=x[:1]
                if name.endswith(('.gamma','.beta')): x=x.reshape(-1)
                elif x.ndim==4: x=x.transpose(2,3,1,0)
                elif x.ndim==2: x=x.T
                values.append(x)
            result[key]=np.stack(values) if items[0][1] is not None else values[0]
            if result[key].shape != params[key].shape: raise ValueError('Reference shape differs: '+key)
        return result
    params=jax.tree.map(jnp.asarray,converted('p')); expected=converted('g')
    spatial=jnp.asarray(data['spatial'].transpose(0,2,3,1)); glob=jnp.asarray(data['global_features'])
    def objective(p,c):
        main,helper=katago.forward(p,spatial,glob,c,training=True)
        ce=lambda x: -jnp.sum(data['target']*jax.nn.log_softmax(x,-1),-1)
        return jnp.sum((.2*ce(main)+.8*ce(helper))*data['weights'])/data['weights'].sum(),(main,helper)
    (loss,(main,helper)),gradient=jax.jit(jax.value_and_grad(lambda p:objective(p,c),has_aux=True))(params)
    checks={}
    def compare(name,a,b,atol=3e-5,rtol=3e-4):
        a,b=np.asarray(a),np.asarray(b)
        passed=bool(np.allclose(a,b,atol=atol,rtol=rtol))
        checks[name]={'passed':passed,'max_absolute_error':float(np.max(np.abs(a-b))),
                      'relative_l2_error':float(np.linalg.norm(a-b)/max(1e-15,np.linalg.norm(b))),
                      'atol':atol,'rtol':rtol}
    compare('loss',loss,data['loss']); compare('main',main,data['main']); compare('helper',helper,data['helper'])
    for key,g in gradient.items(): compare('gradient.'+key,g,expected[key])
    # Changing trunk microbatch boundaries must leave global helper statistics
    # AND their derivatives unchanged. This catches stop-gradient BN substitutes.
    (other_loss,_),other_g=jax.jit(jax.value_and_grad(lambda p:objective(p,{**c,'microbatch':5}),has_aux=True))(params)
    compare('microbatch_loss',other_loss,loss)
    for key,g in gradient.items(): compare('microbatch_gradient.'+key,other_g[key],g)
    single=katago.forward(params,spatial[:1],glob[:1],c)
    compare('primary_batch_independent',single,main[:1])
    independent=katago.forward(params,spatial,glob,c)
    compare('helper_excluded_inference',independent,main)
    result={'kind':'katago_jax_reference_qualification','source_snapshot':SOURCE.name,'reference_manifest_sha256':args.reference_sha256,
            'config':c,'jax_version':jax.__version__,'checks':checks,
            'status':'passed' if all(x['passed'] for x in checks.values()) else 'failed'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f: f.write(canonical_json(result))
    args.output.chmod(0o444)
    print(json.dumps({'status':result['status'],'checks':len(checks),'failures':[k for k,v in checks.items() if not v['passed']],
                      'largest_absolute_error':max(v['max_absolute_error'] for v in checks.values())}),flush=True)
    if result['status']!='passed': raise AssertionError('Official reference qualification failed')
if __name__=='__main__': main()
