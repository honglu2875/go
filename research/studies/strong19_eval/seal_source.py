"""Seal tested joint serving sources, portable fixtures and retained CPU evidence."""
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
from gozero.snapshots import canonical_json, verify


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    archive, output = STUDY / 'source-001.tar', STUDY / 'source-bundle-001.json'
    if archive.exists() or output.exists():
        raise FileExistsError('This evidence bundle already exists')
    files = {}

    def add(path, name=None, expected=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected regular retained source/evidence')
        raw = path.read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError('Source/evidence identity changed: ' + str(path))
        key = name or str(path.relative_to(ROOT))
        if key in files and files[key] != raw:
            raise ValueError('Conflicting archive member')
        files[key] = raw

    passed = {
        'artifact-cpu-003.json': '78ca67d3bcb86b46fbac577e30552852f9de8d6e0eaf7e3b325655373147e5a1',
        'feature-equivalence-001.json': '199b37fc95593f8e8907220e2e63a3337430639df86a09b42d226364bfc70f41',
        'cpu-result-002.json': '11ae55c1b62f71db8d66d20efd55f7d67de6b039de7db051a6a38a43bad18bd4',
        'rpc-unit-001.json': '53ee9e15926db6ac4ec6fc4b8f5b6c61fa67b05715229bc05d9e3f916df18c4a',
        'rpc-cpu-result-001.json': '3999c18bd230e63cedbefb95ef438c8a274a29db2e9d3d1b71778d95d564948f',
        'match-cpu-result-001.json': '215d775c02102dc24be4186d3e0edeadfaaf0a4ad6ff4ad9fee0e7aaf36f724c',
    }
    for name, expected in passed.items():
        add(STUDY / name, expected=expected)
        if json.loads((STUDY / name).read_text())['status'] != 'passed':
            raise ValueError('Required qualification did not pass')
    parent = ROOT / 'research/studies/strong19_train/source-bundle-002.json'
    add(parent, expected='e2fb86a620b24fa6dc84a80554a4a9d754e7a9bf8166a410fd580dc76dab4e22')
    snapshots = set()
    for name in ('cpu-plan-001.json', 'cpu-plan-002.json', 'rpc-cpu-plan-001.json', 'match-cpu-plan-001.json'):
        snapshots.update(json.loads((STUDY / name).read_text())['snapshots'].values())
    latest = json.loads((STUDY / 'match-cpu-plan-001.json').read_text())['snapshots']['transformer']
    selected_eval = {'eval/joint_gtp.py', 'eval/joint_match.py', 'eval/learned_gtp.py',
                     'eval/match.py', 'eval/qualify_katago.py', 'eval/katago_build.json'}
    for identity in sorted(snapshots):
        folder = ROOT / '.gozero/snapshots' / identity
        manifest = verify(folder)
        for name in ('manifest.json', 'resolved_config.json'):
            add(folder / name, name='qualification-snapshots/' + identity + '/' + name)
        if identity == latest:
            for name, entry in manifest['files'].items():
                if (name.startswith(('research/recipes/strong19_eval/', 'packages/gozero/src/'))
                        or name in selected_eval or name in ('tests/test_joint_artifacts.py', 'tests/test_joint_rpc.py',
                                                            'pyproject.toml', 'uv.lock', '.python-version')):
                    add(folder / name, expected=entry['sha256'], name=name)
                    if sha((ROOT / name).read_bytes()) != entry['sha256']:
                        raise ValueError('Current qualified source changed: ' + name)
    for path in sorted(STUDY.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.jsonl', '.log', '.sgf', '.cfg', '.npz'):
            add(path)
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        for name, raw in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size, item.mtime, item.mode = len(raw), 0, 0o444
            tar.addfile(item, io.BytesIO(raw))
    with tarfile.open(archive, 'r') as tar:
        if tar.getnames() != sorted(files):
            raise ValueError('Archive membership differs')
        for item in tar:
            if tar.extractfile(item).read() != files[item.name]:
                raise ValueError('Archive bytes differ')
    result = dict(kind='trained_joint_serving_source_bundle', status='passed', created=time.time(),
                  operator_sha256=sha(Path(__file__).read_bytes()), evidence=passed, snapshots=sorted(snapshots),
                  parent_training_bundle_sha256=sha(parent.read_bytes()), archive=str(archive.relative_to(ROOT)),
                  archive_sha256=sha(archive.read_bytes()), archive_bytes=archive.stat().st_size,
                  files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in sorted(files.items())},
                  scope='Tested serving recipe/library/adapters, qualification identities, small portable trained fixtures and retained CPU failure/success evidence. Every archive member read back. Parent training bundle is separately retained. This private local archive contains operational paths; it is not a public export, hermetic runtime or full-size TPU qualification.')
    with output.open('xb') as stream:
        stream.write(canonical_json(result))
    archive.chmod(0o444)
    output.chmod(0o444)
    print(json.dumps(dict(status='passed', files=len(files), archive_bytes=result['archive_bytes'],
                          archive_sha256=result['archive_sha256'], receipt_sha256=sha(output.read_bytes()))))


if __name__ == '__main__':
    main()
