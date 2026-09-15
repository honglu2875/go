"""Seal current execution-qualified sources while retaining the previous bundle."""
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
    output = STUDY / 'source-bundle-002.json'; archive = STUDY / 'source-002.tar'
    if output.exists() or archive.exists():
        raise FileExistsError('Source bundle already exists')
    files = {}

    def add(path, name=None, expected=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected regular source or evidence')
        raw = path.read_bytes()
        if expected is not None and sha(raw) != expected:
            raise ValueError('Source/evidence identity differs: ' + str(path))
        key = name or str(path.relative_to(ROOT))
        if key in files and files[key] != raw:
            raise ValueError('Conflicting archive member: ' + key)
        files[key] = raw

    parent_path = STUDY / 'source-bundle-001.json'
    add(parent_path, expected='6821363f544a014e9507f3b98c385dacae205ca0bfad3b53201a2d50de1c1c72')
    parent = json.loads(parent_path.read_text())
    add(ROOT / parent['archive'], expected=parent['archive_sha256'])
    prep_path = STUDY / 'real19-snapshots-002.json'
    add(prep_path, expected='985b758d4c3ac4655b0e0fd8e7754fd4ba5a41c5022eec60299782fd5f9fdb4c')
    prep = json.loads(prep_path.read_text())
    if prep['status'] != 'prepared' or prep['tpu_jobs_launched']:
        raise ValueError('Unexpected preparation scope')
    evidence = dict(prep['evidence'])
    for name, digest in evidence.items():
        add(STUDY / name, expected=digest)
        if json.loads((STUDY / name).read_text())['status'] != 'passed':
            raise ValueError('Qualification is not passed')
    for mapping in ('recipe_sources_equal_to_real19_cpu_harness',
                    'recovery_sources_equal_to_four_host_transfer'):
        for source, digest in prep[mapping].items():
            add(ROOT / source, expected=digest)

    identities = set(prep['snapshots'].values())
    cpu = json.loads((STUDY / 'harness-real19-result-001.json').read_text())
    identities.update(row['snapshot'] for row in cpu['models'].values())
    identities.add(json.loads((STUDY / 'resume-transfer-result-002.json').read_text())['snapshot'])
    selected_ops = {'ops/pod_run.py', 'ops/run_host.py', 'ops/stage_joint_checkpoint.py',
                    'ops/stage_ram_corpus.py', 'tests/test_joint_resume.py', 'tests/test_pod.py'}
    for identity in sorted(identities):
        snapshot = ROOT / '.gozero/snapshots' / identity
        manifest = verify(snapshot)
        for name in ('manifest.json', 'resolved_config.json'):
            add(snapshot / name, name='qualification-snapshots/' + identity + '/' + name)
        if identity in prep['snapshots'].values():
            for source, item in manifest['files'].items():
                if (source.startswith(('research/recipes/strong19_train/', 'packages/gozero/src/'))
                        or source in selected_ops or source in ('uv.lock', 'pyproject.toml', '.python-version')):
                    add(snapshot / source, name=source, expected=item['sha256'])
                    if sha((ROOT / source).read_bytes()) != item['sha256']:
                        raise ValueError('Current source differs from sealed preparation: ' + source)

    for path in sorted(STUDY.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md', '.log'):
            add(path)
    for folder in ('fixture-staging-001', 'resume-transfer-staging-001', 'resume-transfer-staging-002',
                   'harness-real19-cpu-001'):
        for path in sorted((STUDY / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                add(path)
    add(ROOT / 'research/studies/strong19_scaling/inventory-002.json')

    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        for name, raw in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(raw); info.mtime = 0; info.mode = 0o444
            tar.addfile(info, io.BytesIO(raw))
    with tarfile.open(archive, 'r') as tar:
        if tar.getnames() != sorted(files):
            raise ValueError('Archive membership differs')
        for item in tar:
            if tar.extractfile(item).read() != files[item.name]:
                raise ValueError('Archive read-back differs')
    result = dict(kind='joint_trainer_execution_source_bundle', status='passed', created=time.time(),
                  operator_sha256=sha(Path(__file__).read_bytes()),
                  parent_bundle_sha256=sha(parent_path.read_bytes()),
                  preparation_sha256=sha(prep_path.read_bytes()), snapshots=prep['snapshots'],
                  evidence=evidence, archive=str(archive.relative_to(ROOT)),
                  archive_bytes=archive.stat().st_size, archive_sha256=sha(archive.read_bytes()),
                  files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in sorted(files.items())},
                  scope='Selected qualified joint recipe/library/recovery sources, snapshot identities, real19 CPU harness and complete four-host transport evidence, with the prior source archive retained verbatim. Every member read back. Private operational paths remain inside this local bundle. Dataset/checkpoint arrays, wheels and external compiled objects are separately retained; this is not a hermetic runtime, public export or TPU qualification.')
    with output.open('xb') as stream:
        stream.write(canonical_json(result))
    archive.chmod(0o444); output.chmod(0o444)
    print(json.dumps(dict(status='passed', files=len(files), archive_bytes=result['archive_bytes'],
                         archive_sha256=result['archive_sha256'], receipt_sha256=sha(output.read_bytes()))))


if __name__ == '__main__':
    main()
