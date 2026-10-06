"""Summarize measured full-learner performance after its complete state audit."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.audit.read_text())
if r['status']!='passed':raise ValueError('Complete execution audit must pass')
rows=[]
for bucket in (512,768):
    cases={name:[next(c for c in host['cases'] if c['name']==name and c['bucket']==bucket) for host in r['reports']]
           for name in ('original8','candidate8')}
    seconds={name:max(c['updates'][1]['seconds'] for c in group) for name,group in cases.items()}
    drift={part:max(values['relative_l2'] for c in cases['candidate8'] for u in c['updates']
                    for key,values in u['comparison']['groups'].items() if key.split('/')[0]==part)
           for part in ('params','first','second')}
    rows.append(dict(bucket=bucket,second_update_seconds=seconds,speedup=seconds['original8']/seconds['candidate8'],
                     peak_gib={name:max(max(c['all_rank_peak_upper_bytes']) for c in group)/(1<<30) for name,group in cases.items()},
                     maximum_group_relative_l2=drift))
result=dict(status='passed',attempt=r['attempt'],snapshot=r['snapshot'],buckets=rows,
            audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
            operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Synchronized fresh-state full-model updates on repeated real batches. Complete two-update AdamW state qualification passed; not long-horizon numerical identity, a learning result, or a playing-strength result.')
with a.output.open('x') as stream:json.dump(result,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')
print(json.dumps(result,indent=2))
