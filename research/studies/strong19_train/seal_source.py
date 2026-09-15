"""Archive the qualified joint trainer and its evidence without live run arrays."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import verify


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    archive = STUDY / 'source-001.tar'
    output = STUDY / 'source-bundle-001.json'
    if archive.exists() or output.exists():
        raise FileExistsError('Source bundle already exists')
    parent_path = ROOT / 'research/studies/strong19_joint/source-bundle-003.json'
    parent_raw = parent_path.read_bytes()
    if sha(parent_raw) != 'ac4afd1143b2ba4f7f20c3b4af998fcc0fb27f4a132f3bad20dff002e5d656a9':
        raise ValueError('Parent receipt changed')
    parent = json.loads(parent_raw)
    parent_archive = ROOT / parent['archive']
    if sha(parent_archive.read_bytes()) != parent['archive_sha256']:
        raise ValueError('Parent archive changed')
    files = {}
    with tarfile.open(parent_archive, 'r') as tar:
        if set(tar.getnames()) != set(parent['files']):
            raise ValueError('Parent archive membership differs')
        for member in tar:
            raw = tar.extractfile(member).read()
            if sha(raw) != parent['files'][member.name]['sha256']:
                raise ValueError('Parent member changed')
            files[member.name] = raw

    def add(name, digest=None, base=ROOT):
        path = base / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected a regular source/evidence file')
        raw = path.read_bytes()
        if digest is not None and sha(raw) != digest:
            raise ValueError('Qualified source changed: ' + name)
        if name in files and files[name] != raw:
            raise ValueError('Source differs from included parent: ' + name)
        files[name] = raw

    evidence = {}
    for name in ('evaluation-cpu-001.json', 'evaluation-schema-001.json',
                 'host-metrics-cpu-001.json', 'harness-result-001.json'):
        path = STUDY / name
        record = json.loads(path.read_text())
        if record['status'] != 'passed':
            raise ValueError('Incomplete qualification')
        evidence[str(path.relative_to(ROOT))] = sha(path.read_bytes())
        for source, digest in record.get('source_sha256', {}).items():
            add(source, digest)
    plan = json.loads((STUDY / 'harness-plan-001.json').read_text())
    for name, identity in plan['snapshots'].items():
        snapshot = ROOT / '.gozero/snapshots' / identity
        manifest = verify(snapshot)
        for source, record in manifest['files'].items():
            if source.startswith(('research/recipes/strong19_train/', 'packages/gozero/src/')):
                add(source, record['sha256'], base=snapshot)
                if sha((ROOT / source).read_bytes()) != record['sha256']:
                    raise ValueError('Current trainer/library differs from qualified snapshot: ' + source)
        files['qualification-snapshots/' + name + '/manifest.json'] = (snapshot / 'manifest.json').read_bytes()
        files['qualification-snapshots/' + name + '/resolved_config.json'] = (snapshot / 'resolved_config.json').read_bytes()
    for path in sorted(STUDY.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md', '.log'):
            add(str(path.relative_to(ROOT)))
    add('research/recipes/strong19_train/README.md')
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw); member.mode = 0o444; member.mtime = 0
            tar.addfile(member, io.BytesIO(raw))
    with tarfile.open(archive, 'r') as tar:
        if tar.getnames() != sorted(files):
            raise ValueError('Archive membership differs')
        for member in tar:
            if tar.extractfile(member).read() != files[member.name]:
                raise ValueError('Archive read-back differs')
    result = dict(kind='qualified_joint_trainer_source_bundle', status='passed', created=time.time(),
        parent_bundle_sha256=sha(parent_raw), archive=str(archive.relative_to(ROOT)),
        archive_sha256=sha(archive.read_bytes()), archive_bytes=archive.stat().st_size,
        snapshots=plan['snapshots'], evidence=evidence,
        files={name: dict(sha256=sha(raw), bytes=len(raw)) for name, raw in sorted(files.items())},
        scope='Qualified joint trainer/model, common library, parent numerical/inference dependencies, locks and CPU/abstract qualification evidence. Every archived byte read back. Data, model arrays, wheels and external compiler objects are excluded. Snapshot manifests describe their full external source closures; this selected source bundle is not a replacement snapshot or a hermetic runtime.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    archive.chmod(0o444); output.chmod(0o444)
    print(json.dumps(dict(status='passed', files=len(files), archive_bytes=result['archive_bytes'],
                         archive_sha256=result['archive_sha256'], receipt_sha256=sha(output.read_bytes()))))


if __name__ == '__main__':
    main()
