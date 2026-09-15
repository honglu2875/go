"""Release only verified all-zero reservations from a completed study."""
import argparse, hashlib, json, os, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',type=int,choices=(0,2,3),required=True)
p.add_argument('--source-sha256',required=True);p.add_argument('--allocation-sha256',required=True)
a=p.parse_args();root=Path('/workspace/go')
if os.uname().nodename!='worker-'+str(a.host):raise ValueError('Wrong host')
path=root/'.gozero/checkpoint-part-reservations'/('encoder-overnight-best-'+str(a.host))
expected={0:900000000,2:1200000000,3:900000000}[a.host]
s=path.lstat()
if path.is_symlink() or s.st_nlink!=1 or s.st_size!=expected or not s.st_mode&0o222:raise ValueError('Not the expected unused reservation')
h=hashlib.sha256()
with path.open('rb') as f:
 for block in iter(lambda:f.read(8*2**20),b''):
  if block.strip(b'\0'):raise ValueError('Reservation contains nonzero data; preserve it')
  h.update(block)
now=path.lstat()
if (now.st_ino,now.st_size,now.st_mtime_ns,now.st_nlink)!=(s.st_ino,s.st_size,s.st_mtime_ns,1):raise ValueError('Reservation changed')
folder=root/'research/studies/spatial_followups';folder.mkdir(parents=True,exist_ok=True)
record={'kind':'released_unused_zero_reservation','host':a.host,'path':str(path),'bytes':s.st_size,
 'allocated_bytes':s.st_blocks*512,'sha256':h.hexdigest(),'source_sha256':a.source_sha256,
 'historical_allocation_sha256':a.allocation_sha256,'created_unix':time.time(),
 'content':'Entire file verified zero; no checkpoint or optimizer bytes.',
 'restore':'Recreate exactly bytes zero bytes with posix_fallocate at the same path if reserving again.',
 'reason':'Old encoder study is closed. Supersede its unused storage reservation for the new authorized experiments.'}
receipt=folder/('zero_reservation_release_host'+str(a.host)+'_001.json')
with receipt.open('x') as f:json.dump(record,f,sort_keys=True);f.write('\n');f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
path.unlink();fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'released','host':a.host,'receipt':str(receipt),'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'released_allocated_bytes':s.st_blocks*512}),flush=True)
