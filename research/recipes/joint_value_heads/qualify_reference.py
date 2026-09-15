"""Compare every active head/input gradient and output to official KataGo."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import heads


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sha(a.reference/'manifest.json')!=a.reference_sha256:raise ValueError('Reference identity changed')
    m=json.loads((a.reference/'manifest.json').read_text())
    if sha(a.reference/'arrays.npz')!=m['array_sha256']:raise ValueError('Reference arrays changed')
    with np.load(a.reference/'arrays.npz',allow_pickle=False) as source:d=dict(source)
    def convert(name,x):
        if name=='conv1.weight':return x.transpose(2,3,1,0)
        if name=='bias1.beta':return x.reshape(-1)
        return x.T if x.ndim==2 else x
    p={name:convert(name,d['p.'+name]) for name in m['parameter_shapes']}
    initialized=jax.eval_shape(lambda:heads.initialize(0,kind='cnn'))
    if set(initialized)!=set(p) or any(initialized[k].shape!=p[k].shape for k in p):raise ValueError('Value parameter tree differs')
    p=jax.tree.map(jnp.asarray,p);x=jnp.asarray(d['x'].transpose(0,2,3,1));mask=jnp.asarray(d['mask'].transpose(0,2,3,1))
    def objective(p,x):
        logits=heads.cnn(p,x,mask);value=heads.signed_value(logits)
        loss=jnp.sum((value-d['target'])**2*d['weights'])/d['weights'].sum()
        return loss,(logits,value)
    (loss,(logits,value)),(gradient,input_gradient)=jax.jit(jax.value_and_grad(objective,argnums=(0,1),has_aux=True))(p,x)
    checks={}
    def check(name,actual,expected,atol=2e-6,rtol=3e-4):
        actual,expected=np.asarray(actual),np.asarray(expected)
        checks[name]=dict(passed=bool(np.allclose(actual,expected,atol=atol,rtol=rtol)),
                          max_absolute_error=float(np.max(np.abs(actual-expected))),atol=atol,rtol=rtol)
    check('loss',loss,d['loss']);check('logits',logits,d['logits']);check('signed_value',value,d['value'])
    check('input_gradient',input_gradient,d['input_gradient'].transpose(0,2,3,1))
    for name,g in gradient.items():check('gradient.'+name,g,convert(name,d['g.'+name]))
    # Frame padding is an explicit extension; it must be finite and isolated.
    padded=heads.cnn(p,x,jnp.zeros_like(mask));check('all_padding_logits',padded,np.zeros((5,3),np.float32))
    _,g=jax.jit(jax.value_and_grad(lambda p:jnp.sum(heads.cnn(p,x,jnp.zeros_like(mask)))))(p)
    for name,value in g.items():check('padding_gradient.'+name,value,np.zeros(value.shape,np.float32))
    check('batch_independence',heads.cnn(p,x[:1],mask[:1]),logits[:1])
    report=dict(kind='pure_jax_value_head_reference_qualification',status='passed' if all(x['passed'] for x in checks.values()) else 'failed',
                reference_manifest_sha256=a.reference_sha256,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                checks=checks,jax_version=jax.__version__,parameter_count=sum(v.size for v in p.values()),
                scope='Full-width CNN value-branch CPU arithmetic and all active/input gradients; not joint training or TPU qualification')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps(dict(status=report['status'],checks=len(checks),failures=[k for k,v in checks.items() if not v['passed']])) ,flush=True)
    if report['status']!='passed':raise AssertionError('Official value head differs')


if __name__=='__main__':main()
