"""Review actual full-size timings, rank metrics, and the historical dense draw."""
import json
import argparse
import math
from pathlib import Path
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json
parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
r=json.loads(args.audit.read_text());assert r['status']=='passed'
reports=r['reports'];reference=reports[0]
for report in reports[1:]:
    assert len(report['cases'])==len(reference['cases'])
    for a,b in zip(report['cases'],reference['cases']):
        assert (a['name'],a['bucket'],a['parameters'])==(b['name'],b['bucket'],b['parameters'])
        assert [x['metrics'] for x in a['updates']]==[x['metrics'] for x in b['updates']]
old=ROOT/'runs/pod-20260928T030525Z-44bb05aa/rank-0/artifacts/metrics.jsonl'
old_first=json.loads(old.read_text().splitlines()[0])
dense=next(row for row in reference['cases'] if row['name']=='dense')['updates'][0]['metrics']
common=sorted(set(dense)&set(old_first));diff={k:[old_first[k],dense[k]] for k in common if old_first[k]!=dense[k]}
assert common and all(math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6) for a,b in diff.values()),diff
summary=[]
for case in reference['cases']:
    peer_cases=[next(x for x in report['cases'] if (x['name'],x['bucket'])==(case['name'],case['bucket'])) for report in reports]
    summary.append(dict(name=case['name'],bucket=case['bucket'],parameters=case['parameters'],
        peak_gib=max(x['global_peak_bytes'] for x in peer_cases)/(1<<30),
        warmup_seconds=max(x['updates'][0]['seconds'] for x in peer_cases),
        timed_seconds=max(x['updates'][1]['seconds'] for x in peer_cases)))
result=dict(status='passed',kind='moe_full_system_review',audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
    all_rank_metrics_identical=True,historical_dense_first_update=dict(common_metrics=len(common),exactly_equal=not diff,differences=diff,source_sha256=hashlib.sha256(old.read_bytes()).hexdigest()),
    cases=summary,scope='Same registered first dense draw and cloned model reproduce historical metrics; two repeated-batch finite updates per systems case. Not a learning curve.')
with args.output.open('xb') as f:f.write(canonical_json(result))
print(json.dumps(result,indent=2))
