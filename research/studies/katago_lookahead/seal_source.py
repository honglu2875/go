"""Seal both Lookahead revisions and their complete small CPU evidence."""
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
    evidence={
        'cpu-result-001.json':'c0c316b4efe88e211f162f4ce9e22a07fc6bb30606ec30ee55d5692e09091390',
        'cpu-result-002.json':'8727b231309c5b6c24fcfff9a37e27d359678fe6b983e5400a7df52e74d9b15b',
        'reference-001/manifest.json':'07794f410ea4f34f6c7df133200c3f9c7cd09222083440d42d71a4a26c4046d2'}
    for name,identity in evidence.items():add(STUDY/name,expected=identity)
    if json.loads((STUDY/'cpu-result-002.json').read_text())['status']!='passed':raise ValueError('Qualification failed')
    snapshots=('7d9bc6bc2235c0b5d88a01ba042ef8117f8e0a461cd6d6a72ca5d608039bc9c7',
               '7f996718dcc8195562f0adfafd242b1a67c7a495c553fbdb7bf4f6f4b882474d')
    for identity in snapshots:
        source=ROOT/'.gozero/snapshots'/identity;manifest=verify(source)
        for name in ('manifest.json','resolved_config.json'):
            add(source/name,name='qualification-snapshots/'+identity+'/'+name)
        for name,entry in manifest['files'].items():
            if name.startswith(('packages/gozero/src/',manifest['recipe']+'/')) or name in ('uv.lock','pyproject.toml','.python-version'):
                add(source/name,name='qualification-snapshots/'+identity+'/'+name,expected=entry['sha256'])
    source=ROOT/'.gozero/external/katago-source/92ee95c0a4b25fec214da00951ab69e97e207729/KataGo-92ee95c0a4b25fec214da00951ab69e97e207729/python/train.py'
    add(source,name='pinned-katago/train.py',expected='991cbfb2d5cea8180cd3daf818ef2a516aca048e0ae0eab68865e9a125967e9e')
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
    result=dict(kind='katago_lookahead_source_bundle',status='passed',created=time.time(),
        operator_sha256=digest(Path(__file__).read_bytes()),evidence=evidence,snapshots=list(snapshots),
        archive=str(archive.relative_to(ROOT)),archive_sha256=digest(archive.read_bytes()),archive_bytes=archive.stat().st_size,
        files={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in sorted(files.items())},
        scope='Private local bundle of both frozen component revisions, original Torch trainer, reference tensors, failed alias case, complete small CPU checkpoints and exact continuation evidence. Every member read back; no full neural/TPU integration claim.')
    with output.open('xb') as stream:stream.write(canonical_json(result))
    archive.chmod(0o444);output.chmod(0o444)
    print(json.dumps(dict(status='passed',files=len(files),archive_bytes=result['archive_bytes'],archive_sha256=result['archive_sha256'],receipt_sha256=digest(output.read_bytes()))))


if __name__=='__main__':main()
