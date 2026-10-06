"""Independently recheck a trainer's already fsynced disk peer checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts,SSH_OPTIONS

REMOTE=r'''import hashlib,json,os,sys
from pathlib import Path
c=json.load(sys.stdin);root=Path(c['target'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
assert sha(root/'mirror.json')==c['receipt_sha256']
m=json.loads((root/'mirror.json').read_text());assert m['bundle_sha256']==c['bundle_sha256'] and m['status']=='passed'
for name,r in m['files'].items():
 p=root/name;assert not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
print(json.dumps(dict(status='passed',files=len(m['files']),bytes=sum(r['bytes'] for r in m['files'].values()),bundle_sha256=m['bundle_sha256'])))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--peer',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.workspace_root;attempt=root/'runs'/args.attempt
    closed=json.loads((attempt/'result.json').read_text());assert closed['status']=='passed'
    report=json.loads((attempt/'rank-0/artifacts/result.json').read_text());cp=report['latest_checkpoint'];d=cp['disk']
    assert d['status']=='passed' and d['all_rank_states'] and d['turn']==report['turn']
    assert sha256(Path(cp['path'])/'manifest.json')==cp['manifest_sha256']
    assert sha256(Path(cp['path']).with_suffix('.group.json'))==d['group_sha256']==cp['group_sha256']
    source=root/'.gozero/snapshots'/closed['snapshot_id'];config=json.loads((source/'resolved_config.json').read_text())
    assert config['checkpoint_disk']['peer']==args.peer
    hosts=load_hosts(source/'ops/hosts.json');assert d['peer']==hosts[args.peer].ssh
    python=root/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
    r=subprocess.run(['ssh',*SSH_OPTIONS,hosts[args.peer].ssh,shlex.join(['taskset','-c','0,1',str(python),'-B','-c',REMOTE])],
        input=json.dumps(d),capture_output=True,text=True,check=True,timeout=600)
    result=dict(status='passed',attempt=attempt.name,snapshot=closed['snapshot_id'],
        disk_primary=str(cp['path']),disk_peer=d,remote_verification=json.loads(r.stdout),verified=time.time())
    atomic_json(args.output,result,replace=False);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
