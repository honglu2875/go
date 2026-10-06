"""Hard-link identical immutable snapshot files, preserving every path and hash."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from execute_run import ROOT, STUDY, read, require, sha, publish
from gozero.snapshots import canonical_json


def main():
    require(not any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json')),
            'Do not deduplicate during an open pod attempt')
    store=ROOT/'.gozero/snapshots';groups=defaultdict(list);manifests={}
    for snapshot in sorted(store.iterdir()):
        path=snapshot/'manifest.json'
        if not path.is_file():continue
        m=read(path);contract={k:v for k,v in m.items() if k!='snapshot_id'}
        require(hashlib.sha256(canonical_json(contract)).hexdigest()==m['snapshot_id']==snapshot.name,
                'Snapshot manifest identity differs')
        manifests[str(path.relative_to(ROOT))]=sha(path)
        for name,r in m['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid member path')
            if r['bytes']<1024:continue
            groups[(r['sha256'],r['bytes'],r['executable'])].append(snapshot/name)
    free=os.statvfs(ROOT);before=free.f_bavail*free.f_frsize;target=8*(1<<30)
    needed=max(0,target-before)+(64<<20);pairs=[];potential=0
    for (digest,size,executable),paths in sorted(groups.items(),key=lambda kv:kv[0][1]*(len(kv[1])-1),reverse=True):
        if len(paths)<2 or potential>=needed:continue
        canonical=paths[0];cs=canonical.lstat()
        require(stat.S_ISREG(cs.st_mode) and not cs.st_mode&0o222 and bool(cs.st_mode&stat.S_IXUSR)==executable
                and cs.st_size==size and sha(canonical)==digest,'Canonical file changed')
        for path in paths[1:]:
            if potential>=needed:break
            st=path.lstat()
            require(stat.S_ISREG(st.st_mode) and st.st_mode==cs.st_mode and st.st_size==size
                    and st.st_dev==cs.st_dev and st.st_uid==cs.st_uid and st.st_gid==cs.st_gid,'Duplicate identity differs')
            if st.st_ino==cs.st_ino or st.st_nlink!=1:continue
            require(sha(path)==digest,'Duplicate content differs')
            pairs.append(dict(source=str(canonical.relative_to(ROOT)),target=str(path.relative_to(ROOT)),
                sha256=digest,bytes=size,mode=stat.S_IMODE(st.st_mode),device=st.st_dev,
                old_inode=st.st_ino,old_blocks=st.st_blocks,source_inode=cs.st_ino))
            potential+=st.st_blocks*512
    require(before+potential>=(7.5*(1<<30)), 'Immutable deduplication cannot provide the planned checkpoint reserve')
    plan=dict(status='prepared',created=time.time(),operator_sha256=sha(Path(__file__)),
              disk_free_before=before,estimated_reclaimed_bytes=potential,pairs=pairs,manifest_sha256=manifests)
    plan_path=STUDY/'snapshot-dedup-plan-001.json';publish(plan_path,plan)
    print(json.dumps(dict(kind='dedup_prepared',files=len(pairs),estimated_reclaimed_bytes=potential)),flush=True)
    touched=set()
    for i,row in enumerate(pairs):
        source=ROOT/row['source'];path=ROOT/row['target'];s=source.lstat();p=path.lstat()
        require((s.st_dev,s.st_ino,s.st_size)==(row['device'],row['source_inode'],row['bytes'])
                and (p.st_dev,p.st_ino,p.st_size,p.st_nlink)==(row['device'],row['old_inode'],row['bytes'],1)
                and stat.S_IMODE(s.st_mode)==stat.S_IMODE(p.st_mode)==row['mode']
                and not s.st_mode&0o222 and sha(source)==sha(path)==row['sha256'],'Planned identity changed')
        temporary=path.with_name('.dedup-'+path.name+'-'+str(os.getpid()))
        require(not temporary.exists(),'Unexpected temporary member')
        os.link(source,temporary);os.replace(temporary,path);touched.add(path.parent)
        require(path.stat().st_ino==source.stat().st_ino and sha(path)==row['sha256'],'Link verification failed')
        if (i+1)%5000==0:print(json.dumps(dict(kind='dedup_progress',files=i+1)),flush=True)
    for directory in touched:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    for name,digest in manifests.items():require(sha(ROOT/name)==digest,'Snapshot manifest changed')
    for row in pairs:
        path=ROOT/row['target'];st=path.stat()
        require(st.st_size==row['bytes'] and stat.S_IMODE(st.st_mode)==row['mode']
                and sha(path)==row['sha256'],'Final snapshot member differs')
    v=os.statvfs(ROOT)
    result=dict(status='passed',created=time.time(),plan_sha256=sha(plan_path),files=len(pairs),
        disk_free_before=before,disk_free_after=v.f_bavail*v.f_frsize,
        observed_reclaimed_bytes=v.f_bavail*v.f_frsize-before,
        scope='Only byte-identical read-only snapshot files share inodes. All paths, contents, permissions and manifest identities preserved. No model checkpoint or dataset removed.')
    publish(STUDY/'snapshot-dedup-result-001.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
