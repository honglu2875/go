"""Move only rejected new trial payloads to two verified, explicitly volatile copies."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.pod import load_hosts,SSH_OPTIONS

BASE=Path('/dev/shm/gozero-archived-checkpoints')

REMOTE=r'''import hashlib,json,os,sys
from pathlib import Path
c=json.load(sys.stdin);p=Path(c['path']);base=Path('/dev/shm/gozero-archived-checkpoints')
assert p.is_relative_to(base) and '..' not in p.parts and not p.is_symlink()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
if c['action']=='prepare':
 assert not p.exists()
 v=os.statvfs(base.parent);assert v.f_bavail*v.f_frsize>c['bytes']+(64<<30)
 p.mkdir(parents=True)
else:
 for name,r in c['files'].items():
  q=p/name if name!='group.json' else p.with_suffix('.group.json')
  assert q.is_file() and not q.is_symlink() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256']
 if c['action']=='seal':
  sys.path.insert(0,c['library']);from gozero.ram_checkpoints import seal
  files=[p/name if name!='group.json' else p.with_suffix('.group.json') for name in c['files']]
  for q in files:
   q.chmod(0o444)
   with q.open('rb') as f:os.fsync(f.fileno())
  seal(files)
 for name in c['files']:
  q=p/name if name!='group.json' else p.with_suffix('.group.json')
  assert q.stat().st_uid==0 and not q.stat().st_mode&0o222
print(json.dumps(dict(status='passed',path=str(p),action=c['action'],bytes=c.get('bytes'))))
'''

DELETE_PEER=r'''import hashlib,json,os,sys
from pathlib import Path
c=json.load(sys.stdin);p=Path(c['path']);base=Path(c['root'])
assert p.is_relative_to(base) and '..' not in p.parts and not p.is_symlink() and p.name=='arrays.npz'
h=hashlib.sha256()
with p.open('rb') as f:
 while b:=f.read(8<<20):h.update(b)
assert h.hexdigest()==c['sha256'] and p.stat().st_size==c['bytes']
marker=p.parent/'payload-retired-to-ram.json'
with marker.open('x') as f:json.dump(c,f);f.flush();os.fsync(f.fileno())
p.unlink();fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='passed',bytes=c['bytes'])))
'''


def remote(host,code,value):
    python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python')
    argv=['ssh',*SSH_OPTIONS,host.ssh,shlex.join(['taskset','-c','0,1',python,'-B','-c',code])]
    r=subprocess.run(argv,input=json.dumps(value),capture_output=True,text=True,check=True,timeout=600)
    return json.loads(r.stdout)


def retire(record, *, reason, closure_backup):
    require(record.get('eligible_retirement') is True,'Reference/old checkpoints are protected')
    stage=ROOT/record['stage_result']
    require(stage.is_relative_to(STUDY) or stage.is_relative_to(ROOT/'research/studies/strong19_dense_lr'),
            'Only trials from this tuning campaign may be retired')
    require(sha(stage)==record['stage_result_sha256'] and read(stage)['status']=='passed','Trial not closed')
    audit=ROOT/record['audit'];require(sha(audit)==record['audit_sha256'] and read(audit)['status']=='passed','Trial not audited')
    require(closure_backup['status']=='passed' and len(closure_backup['copies'])==2,'Metadata must be durably mirrored first')
    require(not any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json')),
            'Never retire while a pod job is open')
    attempt=record['attempt'];result=read(ROOT/'runs'/attempt/'rank-0/artifacts/result.json');cp=result['latest_checkpoint']
    source=ROOT/'.gozero/snapshots'/result['snapshot_id'];cfg=read(source/'resolved_config.json')
    require(read(source/'manifest.json')['recipe'] in ('research/recipes/strong19_dense_lr','research/recipes/strong19_dense_batch'),
            'Only new LR/batch recipe checkpoints may be retired')
    path=Path(cp['path']);expected=ROOT/'runs'/attempt/'rank-0/artifacts/checkpoints'/f"turn-{result['turn']:09d}"
    require(path==expected and not path.is_symlink(),'Unexpected primary payload path')
    files={name:dict(bytes=(path/name).stat().st_size,sha256=sha(path/name))
           for name in ('manifest.json','state.json','actors.json','arrays.npz')}
    files['group.json']=dict(bytes=path.with_suffix('.group.json').stat().st_size,sha256=sha(path.with_suffix('.group.json')))
    require(files['manifest.json']['sha256']==cp['manifest_sha256'] and files['group.json']['sha256']==cp['group_sha256'],
            'Checkpoint identity changed')
    for name,r in read(path/'manifest.json')['files'].items():require(files[name]==r,'Primary checkpoint bytes changed')
    hosts={h.rank:h for h in load_hosts(source/'ops/hosts.json')};copies=[]
    target=BASE/attempt/'rank-0/artifacts/checkpoints'/path.name
    total=sum(x['bytes'] for x in files.values());settings=dict(path=str(target),bytes=total,files=files,
        library=str(source/'packages/gozero/src'))
    for rank in (1,3):
        host=hosts[rank];remote(host,REMOTE,dict(settings,action='prepare'))
        subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(['ssh',*SSH_OPTIONS]),
            str(path)+'/',host.ssh+':'+str(target)+'/'],check=True,timeout=600,stdout=subprocess.DEVNULL)
        subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),str(path.with_suffix('.group.json')),
            host.ssh+':'+str(target.with_suffix('.group.json'))],check=True,timeout=60,stdout=subprocess.DEVNULL)
        copies.append(dict(peer_rank=rank,**remote(host,REMOTE,dict(settings,action='seal'))))
    folder=STUDY/'retention';folder.mkdir(exist_ok=True)
    receipt=folder/(attempt+'-ram-copies.json')
    payload=dict(status='passed',created=time.time(),attempt=attempt,files=files,copies=copies,reason=reason,
                 source_audit_sha256=record['audit_sha256'],closure_backup=closure_backup,
                 durability='Two verified read-only RAM copies. Volatile across pod restart; metrics/configuration/source remain on disk peers.')
    publish(receipt,payload)
    # Verify both independent RAM copies again before releasing either disk payload.
    for rank in (1,3):remote(hosts[rank],REMOTE,dict(settings,action='verify'))
    d=cp['disk'];peer=cfg['checkpoint_disk']['peer'];require(d['status']=='passed' and peer==2,'Unexpected checkpoint peer')
    peerpath=Path(d['target'])/'host-0/arrays.npz'
    deletion=dict(path=str(peerpath),root=cfg['checkpoint_disk']['root'],**files['arrays.npz'],
                  ram_receipt=str(receipt.relative_to(ROOT)),ram_receipt_sha256=sha(receipt),copies=copies)
    require(sha(path/'arrays.npz')==files['arrays.npz']['sha256'],'Owner payload changed')
    marker=path.with_suffix('.retired-to-ram.json');publish(marker,dict(status='prepared',**deletion))
    remote(hosts[peer],DELETE_PEER,deletion)
    (path/'arrays.npz').unlink();fd=os.open(path,os.O_RDONLY);os.fsync(fd);os.close(fd)
    result=dict(status='passed',attempt=attempt,ram_receipt_sha256=sha(receipt),bytes_released_per_disk=files['arrays.npz']['bytes'],
                kept_on_disk='All small rank states, manifests, diagnostics, configurations and source; reference and incumbent payloads protected.')
    publish(folder/(attempt+'-retirement.json'),result)
    return result


def restore(record):
    """Restore a selected provisional batch winner, then recheck its disk peer."""
    attempt=record['attempt'];folder=STUDY/'retention'
    receipt=read(folder/(attempt+'-ram-copies.json'))
    require(read(folder/(attempt+'-retirement.json'))['status']=='passed','No completed retirement to restore')
    result=read(ROOT/'runs'/attempt/'rank-0/artifacts/result.json');cp=result['latest_checkpoint'];path=Path(cp['path'])
    source=ROOT/'.gozero/snapshots'/result['snapshot_id'];cfg=read(source/'resolved_config.json')
    hosts={h.rank:h for h in load_hosts(source/'ops/hosts.json')}
    data=receipt['files']['arrays.npz'];v=os.statvfs(ROOT)
    require(v.f_bavail*v.f_frsize>data['bytes']+(2<<30),'Insufficient owner restore reserve')
    target=BASE/attempt/'rank-0/artifacts/checkpoints'/path.name
    settings=dict(path=str(target),bytes=sum(r['bytes'] for r in receipt['files'].values()),files=receipt['files'])
    for rank in (1,3):remote(hosts[rank],REMOTE,dict(settings,action='verify'))
    require(not (path/'arrays.npz').exists(),'Primary payload already exists')
    temporary=path/'.arrays.restore.partial'
    subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(['ssh',*SSH_OPTIONS]),
                    hosts[1].ssh+':'+str(target/'arrays.npz'),str(temporary)],check=True,timeout=600)
    require(sha(temporary)==data['sha256'] and temporary.stat().st_size==data['bytes'],'Restored primary differs')
    with temporary.open('rb') as f:os.fsync(f.fileno())
    temporary.rename(path/'arrays.npz');fd=os.open(path,os.O_RDONLY);os.fsync(fd);os.close(fd)
    peerpath=Path(cp['disk']['target'])/'host-0/arrays.npz'
    admission="import json,os,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['path']);v=os.statvfs(p.parent);assert not p.exists() and v.f_bavail*v.f_frsize>c['bytes']+(8<<30);print(json.dumps(dict(status='passed')))"
    remote(hosts[2],admission,dict(path=str(peerpath),bytes=data['bytes']))
    subprocess.run(['rsync','-a','--rsync-path=taskset -c 0,1 rsync','-e',shlex.join(['ssh',*SSH_OPTIONS]),
                    str(path/'arrays.npz'),hosts[2].ssh+':'+str(peerpath)],check=True,timeout=600)
    from replicate import REMOTE as VERIFY_DISK
    proof=remote(hosts[2],VERIFY_DISK,cp['disk'])
    # Hash verification plus fsync makes the newly restored peer durable again.
    remote(hosts[2],"import os,json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['path']);f=p.open('rb');os.fsync(f.fileno());f.close();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);print(json.dumps(dict(status='passed')))",dict(path=str(peerpath)))
    answer=dict(status='passed',attempt=attempt,created=time.time(),primary_sha256=sha(path/'arrays.npz'),peer_verification=proof,
                durability='Selected endpoint restored to disk primary and checksum-verified fsynced disk peer.')
    publish(folder/(attempt+'-restored.json'),answer)
    return answer
