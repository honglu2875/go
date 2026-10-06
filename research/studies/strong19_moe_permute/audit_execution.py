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
assert config['kind']=='moe_execution_qualification'
expected=[(name,bucket) for bucket in config['buckets'] for name in config['variants']]
for report in reports:
    assert report['kind']==config['kind'] and report['parameters']==420991764
    assert [(c['name'],c['bucket']) for c in report['cases']]==expected
    assert len({c['initial_state_sha256'] for c in report['cases']})==1
    for case in report['cases']:
        assert case['chunk_frames']==8 and len(case['updates'])==2
        assert max(case['all_rank_peak_upper_bytes'])<=config['compiled_memory_limit_bytes']
        for update in case['updates']:
            assert update['metrics']['accepted']==1 and update['metrics']['moe_dropped_tokens']==0
            if case['name']=='candidate8':
                assert update['comparison']['gate']['status']=='passed'
                for key,values in update['comparison']['groups'].items():
                    part=key.split('/')[0]
                    if part in config['group_relative_l2_limits']:
                        assert values['relative_l2']<=config['group_relative_l2_limits'][part]
                        if part=='first' and values['cosine'] is not None:
                            assert values['cosine']>=config['first_moment_min_cosine']
                    elif part=='counter':assert values['exactly_equal']
for report in reports[1:]:
    for left,right in zip(reports[0]['cases'],report['cases'],strict=True):
        assert left['initial_state_sha256']==right['initial_state_sha256']
        assert [x['metrics'] for x in left['updates']]==[x['metrics'] for x in right['updates']]
        assert [x.get('comparison') for x in left['updates']]==[x.get('comparison') for x in right['updates']]
if config.get('partition_cases_by_rank',False) or config.get('partition_by_assignment',False):
    keys=set(config['cases'][0])
    actual=[{k:row[k] for k in keys} for r in reports for row in r['cases']]
    assert sorted(map(canonical_json,actual))==sorted(map(canonical_json,config['cases']))
result=dict(status='passed',kind='all_rank_moe_qualification_audit',snapshot=a.snapshot,attempt=a.attempt.name,inputs=inputs,reports=reports,
 operator_sha256=sha(Path(__file__)))
with a.output.open('xb') as f:f.write(canonical_json(result))
print(json.dumps(dict(status='passed',kind=reports[0]['kind'],cases_per_host=len(reports[0]['cases']))))
