"""Seal an all-rank systems/kernel qualification without consuming model data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import verify,canonical_json
p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
verify(ROOT/'.gozero/snapshots'/a.snapshot)
read=lambda path:json.loads(path.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
r=read(a.attempt/'result.json')
assert r['status']=='passed' and r['snapshot_id']==a.snapshot
reports=[];inputs={str(a.attempt/'result.json'):sha(a.attempt/'result.json')}
for host in range(4):
 d=a.attempt/f'rank-{host}'
 p=read(d/'result.json');report=read(d/'artifacts/result.json')
 assert p['status']=='passed' and p['returncode']==0 and p['source_integrity'] and not p['cancelled'] and not p['timed_out']
 assert report['status']=='passed' and report['snapshot_id']==a.snapshot and report['host_rank']==host and all(c['status']=='passed' for c in report['cases'])
 inputs[str(d/'result.json')]=sha(d/'result.json');inputs[str(d/'artifacts/result.json')]=sha(d/'artifacts/result.json')
 reports.append(report)
assert sorted(r['jax_rank'] for r in reports)==list(range(4))
config=read(ROOT/'.gozero/snapshots'/a.snapshot/'resolved_config.json')
assert config['kind']=='moe_system_qualification'
for report in reports:
    assert report['kind']==config['kind']
    keys=set(config['cases'][0])
    assert [{k:row[k] for k in keys} for row in report['cases']]==config['cases']
    for row in report['cases']:
        assert len(row['updates'])==config['updates_per_case']==2
        assert row['global_peak_bytes']<=config['compiled_memory_limit_bytes']
        assert all(x['metrics']['accepted']==1 for x in row['updates'])
for report in reports[1:]:
    for x,y in zip(reports[0]['cases'],report['cases'],strict=True):
        assert x['parameters']==y['parameters']
        assert [u['metrics'] for u in x['updates']]==[u['metrics'] for u in y['updates']]
if config.get('partition_cases_by_rank',False):
    keys=set(config['cases'][0])
    actual=[{k:row[k] for k in keys} for r in reports for row in r['cases']]
    assert sorted(map(canonical_json,actual))==sorted(map(canonical_json,config['cases']))
result=dict(status='passed',kind='all_rank_moe_qualification_audit',snapshot=a.snapshot,attempt=a.attempt.name,inputs=inputs,reports=reports,
 operator_sha256=sha(Path(__file__)))
with a.output.open('xb') as f:f.write(canonical_json(result))
print(json.dumps(dict(status='passed',kind=reports[0]['kind'],cases_per_host=len(reports[0]['cases']))))
