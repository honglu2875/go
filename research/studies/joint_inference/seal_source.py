"""Seal the qualified inference sources without altering their parent archive."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    archive = STUDY / 'source-001.tar'
    output = STUDY / 'source-bundle-001.json'
    if archive.exists() or output.exists():
        raise ValueError('Source archive already exists')
    parent_path = ROOT / 'research/studies/strong19_joint/source-bundle-002.json'
    parent_raw = parent_path.read_bytes()
    if sha(parent_raw) != 'bb781f295c55d63388624559df2495464ec370db9bdb913131cb93afa4f97f66':
        raise ValueError('Parent source receipt changed')
    parent = json.loads(parent_raw)
    if sha((ROOT / parent['archive']).read_bytes()) != parent['archive_sha256']:
        raise ValueError('Parent source archive changed')
    files = {}

    def add(name, digest=None):
        path = ROOT / name
        if path.resolve() != path or not path.is_file():
            raise ValueError('Expected a regular workspace source: ' + name)
        raw = path.read_bytes()
        if digest is not None and sha(raw) != digest:
            raise ValueError('Qualified source changed: ' + name)
        if name in files and files[name] != raw:
            raise ValueError('Source changed during sealing: ' + name)
        files[name] = raw

    for name, record in parent['files'].items():
        add(name, record['sha256'])
    evidence = {}
    for name in ('native-boundary-001.json', 'model-cpu-002.json', 'pod-stream-staging-001.json'):
        path = STUDY / name
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if receipt['status'] != 'passed':
            raise ValueError('Incomplete qualification')
        evidence[str(path.relative_to(ROOT))] = sha(raw)
        for source, digest in receipt.get('source_sha256', {}).items():
            add(source, digest)
    for path in sorted((ROOT / 'research/recipes/v7_inference').iterdir()):
        if path.suffix in ('.py', '.cpp', '.md', '.json') and path.is_file():
            add(str(path.relative_to(ROOT)))
    for name in ('stage_stream.py', 'seal_source.py', 'README.md'):
        add(str((STUDY / name).relative_to(ROOT)))
    native_snapshot = ROOT / '.gozero/snapshots/7527dae25b6dac79bb6396ce5bc570eb53ac33088c31f357ffb88ced46805f3c'
    rust = [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock']
    rust += sorted((ROOT / 'crates').rglob('*.rs'))
    rust += sorted((ROOT / 'crates').rglob('Cargo.toml'))
    for path in rust:
        name = str(path.relative_to(ROOT))
        add(name, sha((native_snapshot / name).read_bytes()))
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o444
            member.mtime = 0
            tar.addfile(member, io.BytesIO(raw))
    with tarfile.open(archive, 'r') as tar:
        if tar.getnames() != sorted(files):
            raise ValueError('Archive membership differs')
        for member in tar:
            if tar.extractfile(member).read() != files[member.name]:
                raise ValueError('Archive read-back differs')
    record = dict(kind='joint_inference_preparation_source_archive', status='passed', created=time.time(),
                  parent_bundle_sha256=sha(parent_raw), archive=str(archive.relative_to(ROOT)),
                  archive_sha256=sha(archive.read_bytes()), archive_bytes=archive.stat().st_size,
                  files={name: dict(sha256=sha(raw), bytes=len(raw)) for name, raw in sorted(files.items())},
                  evidence=evidence,
                  scope='Qualified inference source, shared library, matching Rust source, native feature operators and dependency locks. Read-back verified. External KataGo objects, compilers and runtime wheels are referenced by their separate receipts; this is not a hermetic runtime or a registered learning snapshot.')
    with output.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    archive.chmod(0o444)
    output.chmod(0o444)
    print(json.dumps(dict(files=len(files), rust_files=len(rust), archive_bytes=record['archive_bytes'],
                          archive_sha256=record['archive_sha256'], receipt_sha256=sha(output.read_bytes()))))


if __name__ == '__main__':
    main()
