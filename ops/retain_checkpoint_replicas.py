#!/usr/bin/env python3
"""Retain verified persistent replicas before removing redundant local arrays.

The old manifests and exact NPZ bytes survive. Historical readers require an
explicit restore of the NPZ to its original path, as with checkpoint archival.
This provides pod-local replication, not external checkpoint durability.
Two retained copies are the default. Explicit single-owner relocation keeps
one complete persistent copy, matching a checkpoint that initially had one owner.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify

ATTEMPTS = [
    'pod-20260912T030317Z-37c338d6', 'pod-20260912T032049Z-7b898166',
    'pod-20260912T034717Z-cbea8a84', 'pod-20260912T064811Z-eb080156',
    'pod-20260912T070151Z-e95fe209',
]

REMOTE = r'''
import hashlib,json,os
from pathlib import Path
root=Path('/workspace/go')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
rows=[]
for entry in payload['entries']:
 path=root/entry['path']
 if payload['operation']=='evict' and not path.exists():
  rows.append({'path':entry['path'],'already_absent':True});continue
 info=path.stat()
 if path.is_symlink() or info.st_mode&0o222 or info.st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
  raise ValueError('Replica differs: '+str(path))
 row={'path':entry['path'],'sha256':entry['sha256'],'bytes':info.st_size}
 if payload['operation']=='retain':
  target=root/'.gozero/replica-archives'/entry['sha256'];target.mkdir(parents=True,exist_ok=True)
  for name,wanted in entry['checkpoint_files'].items():
   original=path.with_name(name);out=target/name
   if original.is_symlink() or sha(original)!=wanted:raise ValueError('Checkpoint metadata differs')
   if not out.exists():os.link(original,out)
   if sha(out)!=wanted or out.stat().st_mode&0o222:raise ValueError('Retained copy differs')
  fd=os.open(target,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  row['retained_checkpoint']=str(target)
 elif payload['operation']=='evict':
  marker=path.with_name('arrays.remote-replicas.json')
  raw=(json.dumps({'schema_version':1,'kind':'retained_checkpoint_replicas',
    'array_sha256':entry['sha256'],'original_path':entry['path'],
    'retained_hosts':payload['retained_hosts'],'retained_directory':'.gozero/replica-archives/'+entry['sha256'],
    'qualification_sha256':payload['qualification_sha256']},sort_keys=True)+'\n').encode()
  with marker.open('xb') as f:f.write(raw);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
  path.unlink();fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  row['redundant_array_removed']=True
 else:raise ValueError('Unknown operation')
 rows.append(row)
s=os.statvfs(root)
print(json.dumps({'host':os.uname().nodename,'operation':payload['operation'],
 'entries':rows,'free_bytes':s.f_bavail*s.f_frsize}))
'''


def remote(host, operation, entries, qualification=None, retained_hosts=(2,3)):
    payload = {'operation': operation, 'entries': entries,
               'qualification_sha256': qualification, 'retained_hosts':list(retained_hosts)}
    code = 'payload=json.loads(' + repr(json.dumps(payload)) + ')\n'
    code = 'import json\n' + code + REMOTE
    command = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               f'go-user@worker-{host}.example.invalid', 'python3', '-']
    done = subprocess.run(command, input=code, text=True, capture_output=True, timeout=600)
    if done.returncode:raise RuntimeError(f'Replica operation on host {host} failed: {done.stderr[-4000:]}')
    return json.loads(done.stdout)


def copy_missing_replicas(root, host, entries):
    """Newer checkpoints have one owner; create verified replicas before retention."""
    destination=f'go-user@worker-{host}.example.invalid'
    directories=[str((root/entry['path']).parent) for entry in entries]
    code='from pathlib import Path\n'
    code+='directories='+repr(directories)+'\n'
    code+='for name in directories:\n p=Path(name)\n if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError("Replica path is a symlink")\n p.mkdir(parents=True,exist_ok=True)\n'
    subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes',destination,'python3','-'],
                   input=code,text=True,check=True,timeout=60)
    for entry in entries:
        parent=(root/entry['path']).parent
        # Existing bytes are never replaced; the subsequent remote hash check
        # rejects any mismatch before either original copy can be removed.
        subprocess.run(['rsync','-a','--protect-args','--ignore-existing','-e','ssh -F /dev/null -o BatchMode=yes','--',
            *[str(parent/name) for name in entry['checkpoint_files']],destination+':'+str(parent)+'/'],check=True,timeout=600)
    return remote(host,'retain',entries)


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fchmod(stream.fileno(), 0o444); os.fsync(stream.fileno())
    checkpoints._sync_directory(path.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--attempt',action='append',help='Explicit closed attempt to retain; repeat for multiple checkpoints')
    parser.add_argument('--copy-missing-replicas',action='store_true',help='Create replicas of single-owner checkpoints before verification')
    parser.add_argument('--retained-hosts',type=int,nargs='+',default=[2,3],help='Non-owner hosts for verified copies (default: 2 3)')
    parser.add_argument('--relocate-single-owner',action='store_true',help='Explicitly relocate one single-owner checkpoint to one persistent host after restore verification')
    args = parser.parse_args(); verify(SOURCE)
    replicas=1 if args.relocate_single_owner else 2
    if len(args.retained_hosts)!=replicas or len(set(args.retained_hosts))!=replicas or any(h not in (1,2,3) for h in args.retained_hosts):
        raise ValueError('Select the required number of distinct hosts from 1, 2 and 3')
    if args.relocate_single_owner and (not args.copy_missing_replicas or not args.attempt or len(args.attempt)!=1):
        raise ValueError('Single-owner relocation requires one explicit attempt and copying its missing replica')
    eviction_hosts=[h for h in (0,1) if h not in args.retained_hosts]
    root = args.workspace_root.resolve()
    if root != Path('/workspace/go') or args.output.exists():
        raise ValueError('Wrong workspace or existing receipt')
    if any(not (p.parent/'result.json').exists() for p in (root/'runs').glob('pod-*/launch.json')):
        raise ValueError('A pod attempt remains open')
    entries = []
    selected_attempts=args.attempt or ATTEMPTS
    if len(set(selected_attempts))!=len(selected_attempts) or any(Path(x).name!=x or not x.startswith('pod-') for x in selected_attempts):
        raise ValueError('Expected distinct local pod attempt names')
    for attempt in selected_attempts:
        closed = read_json(root/'runs'/attempt/'result.json')
        if closed['status'] not in ('passed', 'failed') or 'end_unix_time' not in closed:
            raise ValueError('Selected checkpoint is not closed')
        paths = [p for p in (root/'runs'/attempt).glob('rank-0/artifacts/checkpoints/turn-*/arrays.npz')
                 if p.stat().st_size > 100_000_000]
        if len(paths) != 1:
            raise ValueError('Selected archive is ambiguous')
        path = paths[0]; manifest = read_json(path.with_name('manifest.json'))
        wanted = manifest['files']['arrays.npz']
        if path.stat().st_mode & 0o222 or checkpoints.sha256(path) != wanted['sha256']:
            raise ValueError('Original array changed')
        files = {name: checkpoints.sha256(path.with_name(name))
                 for name in ('arrays.npz', 'manifest.json', 'state.json', 'actors.json')}
        entries.append({'path':str(path.relative_to(root)), **wanted,
                        'checkpoint_files':files,'closed_result_sha256':checkpoints.sha256(root/'runs'/attempt/'result.json')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version':1, 'kind':'single_owner_checkpoint_relocation' if args.relocate_single_owner else 'retained_checkpoint_replica_maintenance',
              'operator_snapshot':SOURCE.name, 'start_unix':time.time(), 'entries':entries,
              'status':'running', 'apply':args.apply, 'retained_hosts':args.retained_hosts, 'eviction_hosts':eviction_hosts,
              'persistent_replica_count':replicas,
              'restore_contract':'Copy arrays.npz from either retained directory to original_path, verify SHA256, then use the unchanged historical reader.'}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_missing_replicas,root,host,entries) if args.copy_missing_replicas
                       else pool.submit(remote,host,'retain',entries) for host in args.retained_hosts]
            report['retained_replicas'] = [future.result() for future in futures]
        print(json.dumps({'phase':'persistent_replicas_verified','replicas':replicas,'arrays':len(entries)}),flush=True)
        selected = entries[0]
        with tempfile.TemporaryDirectory(prefix='go-checkpoint-replica-restore-', dir='/dev/shm') as directory:
            destination = Path(directory)
            for name in ('manifest.json','state.json','actors.json'):
                shutil.copyfile(root/selected['path'].rsplit('/',1)[0]/name, destination/name)
            source_path = '/workspace/go/.gozero/replica-archives/'+selected['sha256']+'/arrays.npz'
            with (destination/'arrays.npz').open('xb') as stream:
                subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes',f'go-user@worker-{args.retained_hosts[0]}',
                                'cat',source_path],stdout=stream,check=True,timeout=300)
            if checkpoints.sha256(destination/'arrays.npz') != selected['sha256']:
                raise ValueError('Restored bytes differ')
            _, arrays, _ = checkpoints.read(destination, expected_manifest_sha256=selected['checkpoint_files']['manifest.json'])
            report['restore_qualification'] = {'array_count':len(arrays),'original_npz_sha256':selected['sha256'],
                'scratch':'/dev/shm','original_preserved':(root/selected['path']).is_file()}
            del arrays
        qualification_path = args.output.with_name(args.output.stem+'-qualification.json')
        report['status'] = 'qualified'; publish(qualification_path, report)
        qualification_sha = checkpoints.sha256(qualification_path)
        if args.apply:
            report['qualification_sha256'] = qualification_sha
            # Each host rechecks every SHA before removing only its redundant copy.
            report['evictions'] = [remote(host,'evict',entries,qualification_sha,args.retained_hosts) for host in eviction_hosts]
        report['status']='passed'
    except BaseException as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:
        report['end_unix']=time.time();verify(SOURCE);publish(args.output,report)
        print(json.dumps({'status':report['status'],'receipt':str(args.output),'sha256':checkpoints.sha256(args.output)}),flush=True)


if __name__ == '__main__':
    main()
