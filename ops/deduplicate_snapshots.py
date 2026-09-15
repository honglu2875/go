"""Share identical read-only evaluation fixtures across verified snapshots.

Every logical path and byte hash survives. Temporary links live outside all
snapshots so concurrent readers never observe an extra or missing source file.
Run between TPU attempts. Per-path original timestamps are retained in a
compressed receipt; this changes storage allocation, not source contents.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import uuid

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,canonical_json,SnapshotError
from gozero.checkpoints import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if root!=Path('/workspace/go'):raise ValueError('Unexpected workspace')
    if any(not (p.parent/'result.json').exists() for p in (root/'runs').glob('pod-*/launch.json')):
        raise ValueError('Run fixture deduplication between TPU attempts')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();groups=defaultdict(list);pins={};skipped=[]
    for folder in sorted((root/'.gozero/snapshots').iterdir()):
        if not re.fullmatch('[0-9a-f]{64}',folder.name) or not folder.is_dir():continue
        try:
            manifest=verify(folder)
        except (SnapshotError,FileNotFoundError) as error:
            # Historical peer staging can be incomplete. Never modify a path
            # whose complete source copy has not independently verified.
            skipped.append({'snapshot':folder.name,'reason':repr(error)})
            print(json.dumps({'phase':'leave_unqualified_snapshot_unchanged',**skipped[-1]}),flush=True)
            continue
        pins[folder.name]=sha256(folder/'manifest.json')
        for name,entry in manifest['files'].items():
            if not name.startswith('eval/'):continue
            path=folder/name;s=path.stat()
            if path.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_mode&0o222:
                raise ValueError('Evaluation fixture must be regular and read-only')
            if s.st_size:
                groups[(entry['sha256'],s.st_size,s.st_dev,stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid)].append((path,s))
    pending=[(key,paths) for key,paths in groups.items() if len({s.st_ino for _,s in paths})>1]
    staging=root/'.gozero/snapshot-dedup-staging'/uuid.uuid4().hex;staging.mkdir(parents=True,exist_ok=False)
    modified=0;released=0;directories=set();touched=set();status='running';error=None
    print(json.dumps({'phase':'qualified_snapshot_fixtures','snapshots':len(pins),'duplicate_groups':len(pending)}),flush=True)
    records=a.output/'links.jsonl.gz'
    try:
        with records.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as log:
                for key,paths in pending:
                    digest=key[0]
                    # Prefer the inode already retaining the most logical paths.
                    canonical=max(paths,key=lambda pair:pair[1].st_nlink)[0]
                    if sha256(canonical)!=digest:raise ValueError('Canonical fixture changed')
                    for path,original in paths:
                        now=path.stat();source=canonical.stat()
                        if now.st_ino==source.st_ino:continue
                        if (now.st_ino,now.st_dev,now.st_size,now.st_mtime_ns,now.st_mode)!=(
                                original.st_ino,original.st_dev,original.st_size,original.st_mtime_ns,original.st_mode):
                            raise ValueError('Fixture changed after qualification')
                        temporary=staging/str(modified)
                        os.link(canonical,temporary,follow_symlinks=False)
                        os.replace(temporary,path)
                        if path.stat().st_ino!=source.st_ino:raise ValueError('Atomic fixture replacement failed')
                        freed=now.st_blocks*512 if now.st_nlink==1 else 0
                        row={'path':str(path.relative_to(root)),'source':str(canonical.relative_to(root)),
                             'sha256':digest,'bytes':now.st_size,'original_mode':stat.S_IMODE(now.st_mode),
                             'original_atime_ns':now.st_atime_ns,'original_mtime_ns':now.st_mtime_ns,
                             'released_allocated_bytes':freed}
                        log.write(canonical_json(row));modified+=1;released+=freed
                        directories.add(path.parent);touched.add(path.relative_to(root).parts[2])
                        if modified%25000==0:print(json.dumps({'phase':'deduplicate','files':modified,'released_bytes':released}),flush=True)
            raw.flush();os.fchmod(raw.fileno(),0o444);os.fsync(raw.fileno())
        for directory in directories:
            fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        for snapshot in touched:
            folder=root/'.gozero/snapshots'/snapshot;verify(folder)
            if sha256(folder/'manifest.json')!=pins[snapshot]:raise ValueError('Snapshot identity changed')
        verify(SOURCE);status='passed'
    except BaseException as exc:
        error=repr(exc);status='failed';raise
    finally:
        for p in staging.iterdir():p.unlink()
        staging.rmdir()
        result={'status':status,'kind':'immutable_evaluation_fixture_deduplication','operator_snapshot':SOURCE.name,
            'host':os.uname().nodename,'started_unix':started,'ended_unix':time.time(),'error':error,
            'modified_paths':modified,'released_allocated_bytes':released,'fixture_records_sha256':sha256(records) if records.exists() else None,
            'snapshot_manifest_sha256':pins,'touched_snapshots':sorted(touched),'unmodified_unqualified_snapshots':skipped,
            'scope':'Only identical read-only eval/ fixtures. Logical paths, source bytes, permissions and manifests preserved. Original per-path timestamps in compressed receipt; hard links share current inode timestamps.'}
        with (a.output/'result.json').open('xb') as f:f.write(canonical_json(result));f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
        print(json.dumps({k:result[k] for k in ('status','modified_paths','released_allocated_bytes','error')}),flush=True)


if __name__=='__main__':main()
