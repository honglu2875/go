"""Protect only this study's read-only RAM checkpoint files from logout cleanup.

Some pod images remove user-owned /dev/shm files after the final SSH session
ends. Root ownership of completed checkpoint files avoids that cleanup without
altering any system policy. The arrays are still volatile across reboot.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gozero.checkpoint_archive import FILES,safe_path,publish,verify_files,ROOT
from gozero import checkpoints,checkpoint_archive
from gozero.snapshots import read_json

BASES=(Path('/dev/shm/gozero-archived-checkpoints'),Path('/dev/shm/gozero-staged-checkpoints'))


def seal(paths):
    paths=list(map(Path,paths))
    for path in paths:
        base=next((b for b in BASES if path.is_relative_to(b)),None)
        if base is None:raise ValueError('Only study RAM checkpoint files may be protected')
        safe_path(path,base)
        rel=path.relative_to(base)
        checkpoint_file=(len(rel.parts)==6 and rel.parts[1:4]==('rank-0','artifacts','checkpoints')
                         and re.fullmatch('turn-[0-9]{9}',rel.parts[4]) and rel.parts[5] in FILES)
        group_file=(len(rel.parts)==5 and rel.parts[1:4]==('rank-0','artifacts','checkpoints')
                    and re.fullmatch('turn-[0-9]{9}\\.group\\.json',rel.parts[4]))
        if (not (checkpoint_file or group_file) or not path.is_file() or path.stat().st_nlink!=1
                or path.stat().st_mode&0o222 or path.stat().st_uid not in (0,os.getuid())):
            raise ValueError('Unexpected mutable or foreign RAM file')
    before={str(p):checkpoints.sha256(p) for p in paths}
    subprocess.run(['sudo','-n','--','chown','--no-dereference','0:0','--',*map(str,paths)],check=True,timeout=30)
    if any(p.stat().st_uid!=0 or p.stat().st_gid!=0 or checkpoints.sha256(p)!=before[str(p)] for p in paths):
        raise ValueError('RAM checkpoint protection failed or changed bytes')
    return {'kind':'root_owned_readonly_ram_files','files':before,
            'scope':'Only these new checkpoint files changed ownership. RAM remains volatile; system logout policy is unchanged.'}


def recover_archived(attempt):
    """Run inside a live owner SSH session, then the cache survives its logout."""
    attempt=safe_path(attempt,ROOT/'runs')
    report=read_json(attempt/'rank-0/artifacts/result.json')
    latest=report['latest_checkpoint'];record=latest['archive']
    if checkpoints.sha256(Path(record['receipt']))!=record['receipt_sha256']:raise ValueError('Archive receipt changed')
    cache=Path(record['cache_path']);restored=False
    if cache.exists():
        verify_files(cache,record)
    else:
        checkpoint_archive.restore(record,cache);restored=True
    group=cache.with_suffix('.group.json')
    original=Path(record['logical_path']).with_suffix('.group.json')
    if checkpoints.sha256(original)!=latest['group_sha256']:raise ValueError('Persistent checkpoint group changed')
    if not group.exists():
        with group.open('xb') as f:
            f.write(original.read_bytes());f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    if checkpoints.sha256(group)!=latest['group_sha256']:raise ValueError('RAM checkpoint group changed')
    protection=seal([cache/name for name in sorted(FILES)]+[group])
    return {'kind':'archived_checkpoint_cache_recovery','status':'passed','attempt':attempt.name,
            'restored_from_peer':restored,'manifest_sha256':record['manifest_sha256'],
            'receipt_sha256':record['receipt_sha256'],'protection':protection}


def main():
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=recover_archived(a.attempt);publish(a.output,result)
    print(json.dumps({'status':'passed','manifest_sha256':result['manifest_sha256'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
