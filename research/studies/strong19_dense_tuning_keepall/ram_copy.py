"""Copy and verify new temporary checkpoints; never remove source payloads."""
import json
import os
from pathlib import Path
import shlex
import subprocess
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.pod import load_hosts,SSH_OPTIONS
BASE=Path('/dev/shm/gozero-archived-checkpoints')

REMOTE="import hashlib,json,os,sys\nfrom pathlib import Path\nc=json.load(sys.stdin);p=Path(c['path']);base=Path('/dev/shm/gozero-archived-checkpoints')\nassert p.is_relative_to(base) and '..' not in p.parts and not p.is_symlink()\ndef sha(p):\n h=hashlib.sha256()\n with p.open('rb') as f:\n  while b:=f.read(8<<20):h.update(b)\n return h.hexdigest()\nif c['action']=='prepare':\n assert not p.exists()\n v=os.statvfs(base.parent);assert v.f_bavail*v.f_frsize>c['bytes']+(64<<30)\n p.mkdir(parents=True)\nelse:\n for name,r in c['files'].items():\n  q=p/name if name!='group.json' else p.with_suffix('.group.json')\n  assert q.is_file() and not q.is_symlink() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256']\n if c['action']=='seal':\n  sys.path.insert(0,c['library']);from gozero.ram_checkpoints import seal\n  files=[p/name if name!='group.json' else p.with_suffix('.group.json') for name in c['files']]\n  for q in files:\n   q.chmod(0o444)\n   with q.open('rb') as f:os.fsync(f.fileno())\n  seal(files)\n for name in c['files']:\n  q=p/name if name!='group.json' else p.with_suffix('.group.json')\n  assert q.stat().st_uid==0 and not q.stat().st_mode&0o222\nprint(json.dumps(dict(status='passed',path=str(p),action=c['action'],bytes=c.get('bytes'))))\n"

def remote(host,code,value):
    python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python')
    argv=['ssh',*SSH_OPTIONS,host.ssh,shlex.join(['taskset','-c','0,1',python,'-B','-c',code])]
    r=subprocess.run(argv,input=json.dumps(value),capture_output=True,text=True,check=True,timeout=600)
    return json.loads(r.stdout)
