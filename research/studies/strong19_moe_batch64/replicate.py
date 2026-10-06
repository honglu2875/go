"""Verify two extra RAM copies of a newly created, audited trial checkpoint."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.pod import load_hosts,SSH_OPTIONS
from ram_copy import BASE,REMOTE,remote


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--peer',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(a.workspace_root.resolve()==ROOT and a.peer==1,'Unexpected RAM replication scope')
    attempt=ROOT/'runs'/a.attempt;closed=read(attempt/'result.json');r=read(attempt/'rank-0/artifacts/result.json')
    require(closed['status']==r['status']=='passed','Unclosed trial')
    cp=r['latest_checkpoint'];source=Path(cp['path']);require(source.is_relative_to('/dev/shm/gozero-staged-checkpoints'),'Only new RAM trial state')
    require(sha(source/'manifest.json')==cp['manifest_sha256'] and sha(source.with_suffix('.group.json'))==cp['group_sha256'],'Checkpoint identity changed')
    files={n:dict(bytes=(source/n).stat().st_size,sha256=sha(source/n)) for n in ('manifest.json','state.json','actors.json','arrays.npz')}
    for n,v in read(source/'manifest.json')['files'].items():require(files[n]==v,'Trial bytes changed')
    files['group.json']=dict(bytes=source.with_suffix('.group.json').stat().st_size,sha256=cp['group_sha256'])
    snapshot=ROOT/'.gozero/snapshots'/r['snapshot_id'];hosts={h.rank:h for h in load_hosts(snapshot/'ops/hosts.json')}
    target=BASE/a.attempt/'rank-0/artifacts/checkpoints'/source.name
    settings=dict(path=str(target),files=files,bytes=sum(v['bytes'] for v in files.values()),library=str(snapshot/'packages/gozero/src'))
    copies=[]
    for rank in (1,3):
        host=hosts[rank];remote(host,REMOTE,dict(settings,action='prepare'))
        subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(['ssh',*SSH_OPTIONS]),str(source)+'/',host.ssh+':'+str(target)+'/'],check=True,timeout=600)
        subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),str(source.with_suffix('.group.json')),host.ssh+':'+str(target.with_suffix('.group.json'))],check=True,timeout=60)
        copies.append(dict(peer_rank=rank,**remote(host,REMOTE,dict(settings,action='seal'))))
    answer=dict(status='passed',attempt=a.attempt,snapshot=r['snapshot_id'],files=files,copies=copies,created=time.time(),
                durability='Three volatile RAM copies including the owner. Selected endpoints require separate fsynced disk promotion. No source file is removed.')
    publish(a.output,answer);print(json.dumps(dict(status='passed',extra_ram_copies=2,bytes=settings['bytes'])))


if __name__=='__main__':main()
