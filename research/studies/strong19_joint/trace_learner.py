"""Verify full-size joint optimizer schemas without allocating full arrays."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
sys.path.insert(0,str(ROOT/'research/recipes/strong19_joint'))
import jax
import jax.numpy as jnp
import numpy as np
import adamw
import joint
import learner
from qualify_budget import batch_shape,largest_shapes


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def nbytes(tree):
    return sum(math.prod(v.shape)*np.dtype(v.dtype).itemsize for v in jax.tree.leaves(tree))


def schema(tree):
    return jax.tree.map(lambda v:(tuple(v.shape),str(v.dtype)),tree)


def checked(root,path,expected):
    if sha(path)!=expected:raise ValueError('Reference identity changed')
    r=json.loads(path.read_text())
    if r['status']!='passed':raise ValueError('Reference has not passed')
    for name,digest in r['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Qualified source changed: '+name)
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--qualification-sha256',required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    q=checked(ROOT,a.qualification,a.qualification_sha256)
    r=checked(ROOT,a.reference,a.reference_sha256)
    if q['kind']!='joint_optimizer_cpu_qualification' or r['kind']!='joint19_abstract_budget_qualification':
        raise ValueError('Wrong qualification kind')
    if sha(a.plan)!=q['plan_sha256']:raise ValueError('Optimizer fixture plan differs')
    plan=json.loads(a.plan.read_text());started=time.monotonic();rows=[]
    for name,c in r['configs'].items():
        params=jax.eval_shape(lambda:joint.initialize(0,c,r['value_config']))
        state=jax.eval_shape(adamw.initialize,params)
        if any(x.dtype!=jnp.float32 for x in jax.tree.leaves(params)):
            raise ValueError('Expected float32 master parameters')
        if nbytes(params)!=4*r['parameter_counts'][name]['trainable']:
            raise ValueError('Qualified parameter budget differs')
        operation=learner.step(c,plan['fixture_optimizer'],value_weight=plan['fixture_value_coefficient'],chunk_frames=32)
        for length in (128,512,1536):
            before=time.monotonic();b=batch_shape(8,length,19)
            graph,out=jax.make_jaxpr(operation,return_shape=True)(params,state,b)
            if schema(out[0])!=schema(params) or schema(out[1])!=schema(state):
                raise ValueError('Update changes parameter or optimizer schemas')
            if any(x.shape!=() for x in jax.tree.leaves(out[2])):
                raise ValueError('Expected scalar global diagnostics')
            if out[2]['accepted'].dtype!=jnp.bool_ or out[1]['step'].dtype!=jnp.int32:
                raise ValueError('Update gate or step dtype differs')
            row=dict(architecture=name,local_sequences=8,positions=length,chunk_frames=32,
                master_parameter_bytes=nbytes(params),optimizer_state_bytes=nbytes(state),input_batch_bytes=nbytes(b),
                parameter_leaves=len(params),metric_scalars=len(out[2]),
                largest_update_graph_arrays=largest_shapes(graph),trace_seconds=time.monotonic()-before)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('architecture','positions','master_parameter_bytes','optimizer_state_bytes','trace_seconds')}),flush=True)
    # Recheck the qualification dependency identities before issuing a receipt.
    checked(ROOT,a.qualification,a.qualification_sha256)
    checked(ROOT,a.reference,a.reference_sha256)
    report=dict(kind='joint_full_size_optimizer_schema_qualification',status='passed',created=time.time(),
        seconds=time.monotonic()-started,qualification_sha256=a.qualification_sha256,reference_sha256=a.reference_sha256,
        plan_sha256=sha(a.plan),operator_sha256=sha(Path(__file__)),jax_version=jax.__version__,cases=rows,
        scope='Abstract complete optimizer updates for qualified full-size 19x19 models, local batch 8 and histories through 1536. Includes both moments, master parameters, finite-update gate and scalar diagnostics. No full arrays allocated, no TPU compilation/execution, peak-HBM, latency or MFU measurement. Replication and transient buffer overlap are not summed into a peak-memory estimate.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',cases=len(rows),seconds=report['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
