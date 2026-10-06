"""Execute the pinned six-checkpoint relocation with two exact retained copies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan_path=STUDY/'storage-plan-001.json'
    if sha(plan_path)!='c9a9d54d95c3bf7623b42fc959470051176098e5a334e8fb77e45ea382983120':
        raise ValueError('Storage plan changed')
    if any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json')):
        raise ValueError('Accelerator attempt is open')
    plan=json.loads(plan_path.read_text());results=[]
    for index,row in enumerate(plan['cases']):
        import eligible_long_pair
        eligible_long_pair.inspect(row['attempt'])
        extra=STUDY/f'storage-{index+1:02d}-extra-peer.json'
        evicted=STUDY/f'storage-{index+1:02d}-owner-eviction.json'
        def run(argv,label):
            with (STUDY/f'storage-{index+1:02d}-{label}.log').open('xb') as f:
                subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,check=True,timeout=600)
        run([sys.executable,'-B',ROOT/'research/studies/strong19_source_muon/replicate_full_size.py',
            '--workspace-root',ROOT,'--attempt',row['attempt'],'--peer',row['new_peer'],'--output',extra],'copy')
        run([sys.executable,'-B',STUDY/'storage.py','--attempt',row['attempt'],
            '--peer-receipts',ROOT/row['replica'],extra,'--operation','evict','--output',evicted],'evict')
        result=json.loads(evicted.read_text());assert result['status']=='passed'
        results.append(dict(attempt=row['attempt'],eviction_sha256=sha(evicted),reclaimed_bytes=result['reclaimed_local_bytes']))
        print(json.dumps(results[-1]),flush=True)
    result=dict(status='passed',kind=plan['kind'],completed=time.time(),plan_sha256=sha(plan_path),
        operator_sha256=sha(Path(__file__)),results=results,reclaimed_owner_bytes=sum(r['reclaimed_bytes'] for r in results),
        durability='Two exact root-sealed peer RAM copies per checkpoint; all metadata and restore locators retained.')
    with (STUDY/'storage-result-001.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',reclaimed_owner_bytes=result['reclaimed_owner_bytes'])),flush=True)


if __name__=='__main__':
    main()
