"""Seal the qualified joint Muon integration and its small CPU evidence."""
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
    evidence={'bridge-cpu-001.json':'69063ffb7127260adcc73ecc488e2bc3a6688257bde1fdebb09b8567b5038894',
        'harness-cpu-001.json':'3c2efba0d788948f774670779e1efdb819a58508cea6769fd694bfecd66290ec',
        'inference-cpu-001/result.json':'9c67559954a0b8dca2d0630d5dfde341cc4780b0d9c0f96d61fdb603c1f7b44a'}
    for name,identity in evidence.items():
        add(STUDY/name,expected=identity)
        if json.loads((STUDY/name).read_text())['status']!='passed':raise ValueError('Required qualification failed')
    snapshots=('50fcb6ddef844d67dbc3b9ef595299c169659757504869d49acefcbba5b9853a',
               '741a7c0f89b771e9281334d05dde68b8dc6fab81faa470d8fdbbab389095aeb3')
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
        'research/studies/katago_muon/source-bundle-001.json':'bad11e110cd0c17ddb8045a7b6db25b12420b77f6c395e53ae82d4ef217452de',
        'research/studies/strong19_train/source-bundle-002.json':'e2fb86a620b24fa6dc84a80554a4a9d754e7a9bf8166a410fd580dc76dab4e22',
        'research/studies/strong19_eval/source-bundle-001.json':'5f5f30ae2aab4e377015f072e8b6de81cecc226f070e327bf585f2f994699d5c'}
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
    result=dict(kind='joint_muon_integration_source_bundle',status='passed',created=time.time(),
        operator_sha256=digest(Path(__file__).read_bytes()),evidence=evidence,snapshots=list(snapshots),
        parent_bundles=parent_bundles,archive=str(archive.relative_to(ROOT)),archive_sha256=digest(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,
        files={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in sorted(files.items())},
        scope='Private local bundle of frozen model/optimizer/serving source, small complete CPU checkpoints, readback and recovery evidence. Every member read back. Parent qualification bundles are separately retained; this is not a hermetic runtime, public export or full-size TPU result.')
    with output.open('xb') as f:f.write(canonical_json(result))
    archive.chmod(0o444);output.chmod(0o444)
    print(json.dumps(dict(status='passed',files=len(files),archive_bytes=result['archive_bytes'],archive_sha256=result['archive_sha256'],receipt_sha256=digest(output.read_bytes()))))


if __name__=='__main__':main()
