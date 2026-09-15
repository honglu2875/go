#!/usr/bin/env python3
"""Stage a complete joint checkpoint before distributed initialization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.joint_resume import collect
from gozero.pod import SSH_OPTIONS, load_hosts
from gozero.snapshots import canonical_json, verify

REMOTE = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
c=json.load(sys.stdin);files=c['files'];mode=c['mode'];missing=0;ram=[]
for name,record in files.items():
 p=Path(name)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Invalid stage path')
 for parent in (p,*p.parents):
  if parent.is_symlink():raise ValueError('Symlink in checkpoint destination')
 if p.exists() and not p.is_file():raise ValueError('Occupied checkpoint path')
 if name.startswith('/dev/shm/'):
  ram.append(p)
  if not p.exists():missing+=record['bytes']
fs=os.statvfs('/dev/shm')
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
if fs.f_bavail*fs.f_frsize<missing+64*(1<<30) or mem['MemAvailable']<96*(1<<30):
 raise ValueError('Checkpoint staging RAM floor')
if mode=='verify':
 for name,record in files.items():
  p=Path(name)
  if not p.is_file() or p.stat().st_size!=record['bytes']:raise ValueError('Missing or wrong-size checkpoint member')
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
  if h.hexdigest()!=record['sha256']:raise ValueError('Checkpoint bytes differ')
 for p in ram:
  if p.stat().st_mode&0o222 or p.stat().st_nlink!=1 or p.stat().st_uid not in (0,os.getuid()):
   raise ValueError('Unexpected mutable or foreign RAM checkpoint file')
 if ram:
  subprocess.run(['sudo','-n','--','chown','--no-dereference','0:0','--',*map(str,ram)],check=True,timeout=30)
  if any(p.stat().st_uid!=0 or p.stat().st_gid!=0 for p in ram):raise ValueError('RAM checkpoint protection failed')
print(json.dumps(dict(status='passed',mode=mode,files=len(files),missing_ram_bytes=missing,
 shm_free=fs.f_bavail*fs.f_frsize,memory_available=mem['MemAvailable'])))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-list', required=True)
    args = parser.parse_args(); verify(SOURCE)
    if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', args.cpu_list):
        raise ValueError('Explicit staging CPU placement is required')
    os.sched_setaffinity(0, {int(value) for value in args.cpu_list.split(',')})
    root = args.workspace_root.resolve()
    files = collect(root, args.manifest, args.expected_sha256, snapshot=SOURCE.name)
    if sum(record['bytes'] for record in files.values()) > 8 * (1 << 30):
        raise ValueError('Checkpoint closure exceeds eight GiB')
    args.output.mkdir(parents=True, exist_ok=False)
    checks = {str(path): record for path, record in files.items()}
    report = dict(kind='joint_replicated_checkpoint_staging', status='running', started=time.time(),
                  snapshot=SOURCE.name, group_sha256=args.expected_sha256, files=checks)
    listing = b'\0'.join(str(path).lstrip('/').encode() for path in sorted(files)) + b'\0'
    def stage(host):
        started = time.monotonic()
        records = []
        with (args.output / f'host-{host.rank}.log').open('x') as log:
            for mode in ('preflight', 'verify'):
                if mode == 'verify':
                    subprocess.run(['rsync', '-rlpt', '--ignore-existing', '--relative', '--no-implied-dirs',
                        '--omit-dir-times', '--from0', '--files-from=-',
                        '--rsync-path=taskset -c ' + args.cpu_list + ' rsync',
                        '-e', shlex.join(['ssh', *SSH_OPTIONS]), '/', host.ssh + ':/'], input=listing,
                        check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                command = ['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['taskset', '-c', args.cpu_list, 'python3', '-c', REMOTE])]
                output = subprocess.check_output(command, input=canonical_json(dict(mode=mode, files=checks)),
                                                 stderr=log, timeout=180)
                records.append(json.loads(output)); log.write(output.decode()); log.flush()
        return dict(host=host.rank, status='passed', seconds=time.monotonic()-started, checks=records)
    try:
        with ThreadPoolExecutor(4) as pool:
            report['hosts'] = list(pool.map(stage, load_hosts(SOURCE / 'ops/hosts.json')))
        if collect(root, args.manifest, args.expected_sha256, snapshot=SOURCE.name) != files:
            raise ValueError('Source checkpoint changed during staging')
        verify(SOURCE); report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished'] = time.time()
        with (args.output / 'receipt.json').open('xb') as stream:
            stream.write(canonical_json(report))
        (args.output / 'receipt.json').chmod(0o444)
    print(json.dumps(dict(status=report['status'], files=len(files),
                         receipt_sha256=sha256(args.output / 'receipt.json'))), flush=True)


if __name__ == '__main__':
    main()
