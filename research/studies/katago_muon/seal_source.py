"""Seal the qualified optimizer and retain the failed numerical evidence."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json, verify


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    archive, output = STUDY/'source-001.tar', STUDY/'source-bundle-001.json'
    if archive.exists() or output.exists():
        raise FileExistsError('Evidence bundle already exists')
    result_path = STUDY/'cpu-result-002/result.json'
    expected = '68271563efa4072ed12e1c0669047b635ffdada2b8c247005ab67c2ab03f4afa'
    if sha(result_path.read_bytes()) != expected or json.loads(result_path.read_text())['status'] != 'passed':
        raise ValueError('Required qualification changed or failed')
    files = {}

    def add(path, name=None, expected=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected a regular source/evidence file')
        raw = path.read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError('Pinned source changed')
        key = name or str(path.relative_to(ROOT))
        if key in files and files[key] != raw:
            raise ValueError('Conflicting archive member')
        files[key] = raw

    snapshots = [json.loads((STUDY/f'cpu-plan-{i:03}.json').read_text())['snapshot'] for i in (1,2)]
    for identity in snapshots:
        folder = ROOT/'.gozero/snapshots'/identity
        manifest = verify(folder)
        for name in ('manifest.json','resolved_config.json'):
            add(folder/name,name=f'qualification-snapshots/{identity}/{name}')
        if identity == snapshots[-1]:
            for name, item in manifest['files'].items():
                if name.startswith(('research/recipes/katago_muon/','packages/gozero/src/')) or name in ('pyproject.toml','uv.lock','.python-version'):
                    add(folder/name,name=name,expected=item['sha256'])
                    if sha((ROOT/name).read_bytes()) != item['sha256']:
                        raise ValueError('Working source differs from the qualified source')
    for path in sorted(STUDY.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.md','.json','.jsonl','.log','.npz','.npy'):
            add(path)
    with tarfile.open(archive,'x',format=tarfile.PAX_FORMAT) as tar:
        for name,raw in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(raw);item.mtime=0;item.mode=0o444
            tar.addfile(item,io.BytesIO(raw))
    with tarfile.open(archive,'r') as tar:
        if tar.getnames()!=sorted(files):
            raise ValueError('Archive membership differs')
        for item in tar:
            if tar.extractfile(item).read()!=files[item.name]:
                raise ValueError('Archive readback differs')
    report=dict(kind='katago_muon_source_bundle',status='passed',created=time.time(),
                operator_sha256=sha(Path(__file__).read_bytes()),qualification_sha256=expected,
                snapshots=snapshots,archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive.read_bytes()),
                archive_bytes=archive.stat().st_size,
                files={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in sorted(files.items())},
                scope='Qualified JAX optimizer, reference arrays, checkpoint continuation and retained failed evidence. Every archive member read back. Private operational paths are present; this is not a public export or a full-size TPU qualification.')
    with output.open('xb') as f:f.write(canonical_json(report))
    archive.chmod(0o444);output.chmod(0o444)
    print(json.dumps(dict(status='passed',files=len(files),archive_bytes=report['archive_bytes'],archive_sha256=report['archive_sha256'],receipt_sha256=sha(output.read_bytes()))))


if __name__=='__main__':
    main()
