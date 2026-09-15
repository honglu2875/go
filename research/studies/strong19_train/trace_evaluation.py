"""Compare full-size validation arithmetic and retained array shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/recipes/strong19_train'))
sys.path.append(str(ROOT/'research/recipes/strong19_joint'))
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
import jax
import evaluation
import joint
import training_arithmetic
from qualify_budget import batch_shape,largest_shapes


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--qualification-sha256',required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    for path,digest in [(a.qualification,a.qualification_sha256),(a.reference,a.reference_sha256)]:
        if sha(path)!=digest:raise ValueError('Qualified reference changed')
        r=json.loads(path.read_text())
        if r['status']!='passed':raise ValueError('Unqualified reference')
        for name,h in r['source_sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Qualified source changed: '+name)
    r=json.loads(a.reference.read_text());start=time.monotonic();cases=[]
    for name,c in r['configs'].items():
        params=jax.eval_shape(lambda:joint.initialize(0,c,r['value_config']))
        for length in (128,512,1536):
            b=batch_shape(8,length,19)
            before=jax.make_jaxpr(lambda p,b:joint.forward(p,b,c))(params,b)
            after=jax.make_jaxpr(lambda p,b:evaluation.forward(p,b,c,chunk_frames=32))(params,b)
            old=training_arithmetic.count(before);new=training_arithmetic.count(after)
            if old['total_matrix_flops']!=new['total_matrix_flops']:
                raise ValueError('Validation path changes logical matrix work')
            row=dict(architecture=name,positions=length,local_sequences=8,chunk_frames=32,
                reference=old,bounded=new,reference_largest_arrays=largest_shapes(before),
                bounded_largest_arrays=largest_shapes(after))
            cases.append(row)
            print(json.dumps(dict(model=name,positions=length,matrix_work_ratio=1.,
                reference_largest_bytes=row['reference_largest_arrays'][0]['bytes'],
                bounded_largest_bytes=row['bounded_largest_arrays'][0]['bytes'])),flush=True)
    report=dict(kind='joint_bounded_validation_abstract_qualification',status='passed',created=time.time(),seconds=time.monotonic()-start,
        qualification_sha256=a.qualification_sha256,reference_sha256=a.reference_sha256,operator_sha256=sha(Path(__file__)),
        cases=cases,scope='Both full-size 19x19 main policy/value graphs at three history extents. Logical matrix work is unchanged; individual array shapes are not peak-HBM or measured latency. No TPU execution or MFU claim.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps(dict(status='passed',seconds=report['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
