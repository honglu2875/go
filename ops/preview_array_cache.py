"""Reclaim/restore an unused exact preview duplicate; retain both manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT=Path('/workspace/go')
PREVIEW=Path('/dev/shm/gozero-datasets/strong9-v7-preview-0f78b62d')
FULL=Path('/dev/shm/gozero-datasets/strong9-v7-0f78b62d')
INSPECTION=ROOT/'research/studies/storage_cleanup_20260915/preview-inspection-001.json'
INSPECTION_SHA='426e429529589e72bbce8a94abee3cd18aaa1439183cb2a39a43b049fbe7613f'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def regular(path):
    if any(x.is_symlink() for x in (path,*path.parents)):raise ValueError('Unexpected symlink')
    s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_mode&0o222:raise ValueError('Expected unique read-only regular file')
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--operation',choices=('evict','restore'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    if sha(INSPECTION)!=INSPECTION_SHA:raise ValueError('Inspection changed')
    proof=json.loads(INSPECTION.read_text())
    if sha(PREVIEW/'manifest.json')!=proof['preview_manifest_sha256'] or sha(FULL/'manifest.json')!=proof['full_manifest_sha256']:
        raise ValueError('Corpus identity changed')
    expected=json.loads((PREVIEW/'manifest.json').read_text())['shards'][0]['files']
    if len(proof['files'])!=8 or {Path(r['preview']) for r in proof['files']}!={PREVIEW/v['path'] for v in expected.values()}:
        raise ValueError('Incomplete preview array set')
    files=[];reclaimed=0
    for row in proof['files']:
        source=Path(row['retained']);target=Path(row['preview'])
        if source!=FULL/target.relative_to(PREVIEW):raise ValueError('Unexpected restore source')
        s=regular(source)
        if s.st_size!=row['bytes'] or sha(source)!=row['sha256']:raise ValueError('Retained full data changed')
        if a.operation=='evict':
            old=regular(target)
            if old.st_size!=row['bytes'] or sha(target)!=row['sha256']:raise ValueError('Preview changed')
            reclaimed+=old.st_blocks*512
        elif target.exists():raise ValueError('Preview arrays already present')
        files.append((source,target,row))
    if a.operation=='evict':
        for launch in (ROOT/'runs').glob('pod-*/launch.json'):
            if (launch.parent/'result.json').exists():continue
            snap=json.loads(launch.read_text())['snapshot_id']
            config=json.loads((ROOT/'.gozero/snapshots'/snap/'resolved_config.json').read_text())
            if config.get('dataset',{}).get('path')==str(PREVIEW):raise ValueError('A live learner uses this preview')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:maps=(proc/'maps').read_text()
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if str(PREVIEW) in maps:raise ValueError('Preview arrays remain mapped by PID '+proc.name)
    else:
        fs=os.statvfs('/dev/shm')
        if fs.f_bavail*fs.f_frsize<sum(r['bytes'] for _,_,r in files)+64*(1<<30):raise ValueError('Restore violates RAM floor')
    result=dict(kind='preview_array_cache',operation=a.operation,inspection_sha256=INSPECTION_SHA,
                operator_sha256=sha(Path(__file__)),files=proof['files'],preview_manifest_sha256=proof['preview_manifest_sha256'],
                full_manifest_sha256=proof['full_manifest_sha256'],started=time.time())
    verified=a.output.with_suffix('.verified.json')
    with verified.open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    verified.chmod(0o444)
    for source,target,row in files:
        if a.operation=='evict':target.unlink()
        else:
            temporary=target.with_suffix('.restoring.npy')
            with temporary.open('xb') as stream,source.open('rb') as original:shutil.copyfileobj(original,stream,8*1024*1024)
            temporary.chmod(0o444)
            if sha(temporary)!=row['sha256']:raise ValueError('Restored preview changed')
            os.link(temporary,target);temporary.unlink()
    for source,target,row in files:
        if sha(source)!=row['sha256']:raise ValueError('Retained full data changed during cache operation')
        if a.operation=='evict':assert not target.exists()
        else:assert sha(target)==row['sha256']
    result.update(status='passed',finished=time.time(),reclaimed_allocated_bytes=reclaimed,
                  arrays_present=a.operation=='restore',manifests_retained=True,
                  restore='Run this operator with --operation restore and a new output path; it copies exact bytes from the retained full corpus.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({k:result[k] for k in ('status','operation','arrays_present','reclaimed_allocated_bytes')}))


if __name__=='__main__':main()
