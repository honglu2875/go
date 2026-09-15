"""Account for the extra trunk pass and bounded abstract CNN activation shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import joint
import cnn_chunks
import compute_budget
import training_arithmetic
from qualify_budget import batch_shape,largest_shapes

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.reference)!=a.reference_sha256:raise ValueError('Reference budget changed')
    r=json.loads(a.reference.read_text())
    if r['status']!='passed' or r['kind']!='joint19_abstract_budget_qualification':raise ValueError('Wrong reference')
    for name,digest in r['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified parent source changed: '+name)
    started=time.monotonic();c=r['configs']['cnn'];v=r['value_config']
    params=jax.eval_shape(lambda:joint.initialize(0,c,v))
    policy_formula=compute_budget.cnn(c,19,1)
    trunk_per_frame=sum(policy_formula['multiply_add_flops_by_component'][k] for k in ('encoder','trunk'))
    reference={x['positions']:x for x in r['training_shapes'] if x['architecture']=='cnn'}
    cases=[]
    for length in (128,512,1536):
        b=batch_shape(8,length,19);old=reference[length]
        for chunk in (8,16,32):
            before=time.monotonic()
            objective=lambda p,b:cnn_chunks.losses(p,b,c,value_weight=.7,chunk_frames=chunk)[0]
            f=training_arithmetic.count(jax.make_jaxpr(objective)(params,b))
            graph=jax.make_jaxpr(jax.value_and_grad(objective))(params,b)
            grad=training_arithmetic.count(graph);largest=largest_shapes(graph)
            extra=f['total_matrix_flops']-old['forward']['total_matrix_flops']
            if extra!=trunk_per_frame*8*length:raise ValueError('Additional forward work is not exactly one trunk pass')
            row=dict(positions=length,local_batch=8,chunk_frames=chunk,forward=f,differentiated=grad,
                forward_work_ratio=f['total_matrix_flops']/old['forward']['total_matrix_flops'],
                differentiated_work_ratio=grad['total_matrix_flops']/old['differentiated']['total_matrix_flops'],
                additional_forward_matrix_flops=extra,largest_differentiated_arrays=largest,
                largest_reference_array_bytes=old['largest_differentiated_arrays'][0]['bytes'],trace_seconds=time.monotonic()-before)
            cases.append(row)
            print(json.dumps(dict(positions=length,chunk=chunk,forward_ratio=row['forward_work_ratio'],gradient_ratio=row['differentiated_work_ratio'],
                largest_array_bytes=largest[0]['bytes'],reference_largest=row['largest_reference_array_bytes'])),flush=True)
    result=dict(kind='bounded_cnn_abstract_memory_work_qualification',status='passed',created=time.time(),seconds=time.monotonic()-started,
        reference_sha256=a.reference_sha256,model=c,value_config=v,cases=cases,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        scope='No full arrays allocated. Individual abstract array shapes are not peak HBM; logical differentiated matrices are not MFU or runtime. One extra trunk forward pass is independently accounted in every case. No change to inference parameters or graph.',
        adoption='Requires measured full-size TPU memory, optimizer/gradient equivalence and latency before a registered training run.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',cases=len(cases),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
