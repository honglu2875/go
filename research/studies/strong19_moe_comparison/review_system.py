"""Review the new-corpus CNN systems gate without reusing old-corpus losses."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.audit.read_text());assert report['status']=='passed'
    assert len(report['reports'])==4
    rows=[]
    for bucket in (512,768):
        cases=[next(c for c in h['cases'] if c['name']=='cnn' and c['bucket']==bucket) for h in report['reports']]
        assert all(c['parameters']==233220870 and c['status']=='passed' for c in cases)
        assert all([x['metrics'] for x in c['updates']]==[x['metrics'] for x in cases[0]['updates']] for c in cases)
        rows.append(dict(bucket=bucket,parameters=233220870,
            timed_seconds=max(c['updates'][1]['seconds'] for c in cases),
            warmup_seconds=max(c['updates'][0]['seconds'] for c in cases),
            peak_gib=max(c['global_peak_bytes'] for c in cases)/2**30,
            first_update_metrics=cases[0]['updates'][0]['metrics']))
    result=dict(status='passed',cases=rows,all_rank_metrics_identical=True,
        audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
        scope='Fresh full-size CNN systems qualification on the current corpus. Its previous different-corpus loss values are not a numerical control.')
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status='passed',cases=[{k:v for k,v in r.items() if k!='first_update_metrics'} for r in rows])))


if __name__=='__main__':main()
