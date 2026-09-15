"""Seal the combined source optimizer and its completed CPU evidence."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json,verify


def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
    archive=STUDY/'source-001.tar';output=STUDY/'source-bundle-001.json'
    if archive.exists() or output.exists():raise FileExistsError('Bundle already exists')
    files={}
    def add(path,name=None,expected=None):
        if path.is_symlink() or not path.is_file():raise ValueError('Expected regular retained source/evidence')
        raw=path.read_bytes();key=name or str(path.relative_to(ROOT))
        if expected is not None and digest(raw)!=expected:raise ValueError('Pinned source/evidence changed')
        if key in files and files[key]!=raw:raise ValueError('Archive source conflict')
        files[key]=raw
    evidence={'runtime-cpu-001.json':'26d4f9d22e1ebd613f6e1a361ffe1e3fe736b06ae1064075b8346ff9367b4aaa',
        'harness-result-001.json':'0cab9d71e6aa105c7189bf0569cd573150698c8005ad2228d23ca5d3ebcad7e5',
        'inference-cpu-001/result.json':'3678c8735a4e95a3263727015d9aca1c592237cd0ed92bdba44da99ab7b774fc'}
    for name,identity in evidence.items():
        add(STUDY/name,expected=identity)
        if json.loads((STUDY/name).read_text())['status']!='passed':raise ValueError('Required qualification failed')
    snapshots=('3a34720e97f82b7026e18c5247804e8e591145823368c6c89b2eb8e0169aba3b',
               '3bd93c2a8fb2ee67fa8c071ebd435b9325bb921eae3a218c14d8092ad2f9cb70',
               '2c2b102a31cf140f076dba8cb92c858d333167c47a8eaf4f0d8a59db9e343f2d')
    for identity in snapshots:
        source=ROOT/'.gozero/snapshots'/identity;manifest=verify(source)
        for name in ('manifest.json','resolved_config.json'):
            add(source/name,name='qualification-snapshots/'+identity+'/'+name)
        for name,entry in manifest['files'].items():
            if name.startswith(('packages/gozero/src/',manifest['recipe']+'/')) or name in ('uv.lock','pyproject.toml','.python-version'):
                add(source/name,name='qualification-snapshots/'+identity+'/'+name,expected=entry['sha256'])
    parent=ROOT/'research/studies/strong19_train/qualify_harness.py'
    add(parent,expected='88389cf66f567a2610430c98f2a23218432b492dd0276b115be626459751e2a9')
    parent_bundles={
        'research/studies/strong19_muon/source-bundle-001.json':'13ba5d7321e2fb7f52aeacbe10ef6f55b787ea5c4728bceb036921ed4e87a4cf',
        'research/studies/katago_lookahead/source-bundle-001.json':'f55ba9a0d44a98444332e96d962fa85000ccc23fcdd7b723e1b537a9cddde4de'}
    for name,identity in parent_bundles.items():add(ROOT/name,expected=identity)
    for path in sorted(STUDY.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.md','.json','.jsonl','.log','.npz'):add(path)
    with tarfile.open(archive,'x',format=tarfile.PAX_FORMAT) as tar:
        for name,raw in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(raw);item.mtime=0;item.mode=0o444
            tar.addfile(item,io.BytesIO(raw))
    with tarfile.open(archive,'r') as tar:
        if tar.getnames()!=sorted(files):raise ValueError('Archive membership differs')
        for item in tar:
            if tar.extractfile(item).read()!=files[item.name]:raise ValueError('Archive bytes differ')
    result=dict(kind='joint_source_muon_integration_source_bundle',status='passed',created=time.time(),
        operator_sha256=digest(Path(__file__).read_bytes()),evidence=evidence,snapshots=list(snapshots),
        parent_bundles=parent_bundles,archive=str(archive.relative_to(ROOT)),archive_sha256=digest(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,
        files={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in sorted(files.items())},
        scope='Private local bundle of frozen model/optimizer/serving source, small complete CPU checkpoints, readback and recovery evidence. Every member read back. Plan001 was superseded before execution; plan002 supplied the executed trainer. Parent qualification bundles are separately retained; this is not a hermetic runtime, public export or full-size TPU result.')
    with output.open('xb') as f:f.write(canonical_json(result))
    archive.chmod(0o444);output.chmod(0o444)
    print(json.dumps(dict(status='passed',files=len(files),archive_bytes=result['archive_bytes'],archive_sha256=result['archive_sha256'],receipt_sha256=digest(output.read_bytes()))))


if __name__=='__main__':main()
