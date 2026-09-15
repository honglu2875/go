"""Cross-check host scalar normalization against qualified JAX metric math."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/recipes/strong19_train'))
import jax
import jax.numpy as jnp
import numpy as np
import evaluation
import heads
import host_metrics
import policy_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    started=time.monotonic();rng=np.random.default_rng(952);maximum=0.;records=[]
    for counts in ([4,3,1,0],[1,0,0,0],[0,0,0,0],[4,4,4,4]):
        live=np.arange(4)[None,:]<np.asarray(counts)[:,None]
        values=rng.uniform(-1,1,(4,4)).astype(np.float32);values[~live]=np.nan
        b=jax.tree.map(jnp.asarray,dict(actions=np.zeros((4,4),np.int32),counts=np.asarray(counts,np.int32),
            policies=rng.dirichlet(np.ones(82),size=(4,4)).astype(np.float32),legal=np.ones((4,4,82),bool),
            values=values,family_weights=np.asarray([.01,.05,.003,.01],np.float32),opponent=np.arange(4,dtype=np.int32)))
        raw={**policy_model.total_metrics(jnp.asarray(rng.normal(size=(4,4,82)),jnp.float32),b,stratify=True),
             **heads.value_totals(jnp.asarray(rng.uniform(-1,1,(4,4)),jnp.float32),b,stratify=True)}
        expected=evaluation.averages(raw);actual=host_metrics.averages(raw)
        if set(expected)!=set(actual):raise ValueError('Metric coverage differs')
        error=0.
        for key in actual:
            np.testing.assert_allclose(actual[key],float(expected[key]),atol=1e-6,rtol=1e-5,err_msg=key)
            error=max(error,abs(actual[key]-float(expected[key])))
        maximum=max(maximum,error);records.append(dict(counts=counts,metric_count=len(actual),max_abs=error))
    for bad in ({**raw,'expert_count':float('nan')},{**raw,'value_count':-1},{**raw,'expert_count':1}):
        try:host_metrics.averages(bad)
        except ValueError:pass
        else:raise ValueError('Invalid metric totals accepted')
    paths=[Path(__file__),ROOT/'research/recipes/strong19_train/host_metrics.py',ROOT/'research/recipes/strong19_train/evaluation.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(kind='joint_host_metric_normalization_qualification',status='passed',created=time.time(),
        seconds=time.monotonic()-started,cases=records,rejections=3,max_abs=maximum,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        scope='Scalar policy/value population normalization, including fractional family mass and empty groups, against qualified JAX equations. Host normalization dispatches no JAX operations.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps(dict(status='passed',max_abs=maximum,sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
