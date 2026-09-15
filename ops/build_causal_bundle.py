#!/usr/bin/env python3
"""Create a verified, deterministic bundle of a causal model's original paths."""
import argparse
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256, _sync_directory
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--descriptor', required=True); p.add_argument('--expected-descriptor-sha256', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); verify(SOURCE)
    root = a.workspace_root.resolve(); path = artifact(SOURCE, a.descriptor)
    if sha256(path) != a.expected_descriptor_sha256:
        raise ValueError('Descriptor changed')
    descriptor = read_json(path); validate(root, descriptor)
    result = artifact(root, descriptor['training_result_path']); checkpoint = artifact(root, descriptor['checkpoint']['path'])
    paths = [result, result.parent / 'model_schema.json', artifact(root, descriptor['model_export_path']),
             checkpoint.with_suffix('.group.json'), *(checkpoint / n for n in ('manifest.json', 'state.json', 'arrays.npz', 'actors.json'))]
    if len(set(paths)) != 8 or any(not x.is_relative_to(result.parent) or x.is_symlink() for x in paths):
        raise ValueError('Only regular files below the original artifact directory are supported')
    if sum(x.stat().st_size for x in paths) > 512 * 2**20:
        raise ValueError('Inference bundle exceeds 512 MiB')
    output = a.output.resolve()
    if not output.is_relative_to(root / '.gozero/candidate-bundles') or output.exists():
        raise ValueError('Use a new directory under the local candidate bundle store')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bundle-', dir=output.parent)); files = {}
    with tarfile.open(temporary / 'payload.tar', 'w', format=tarfile.USTAR_FORMAT) as archive:
        for entry in sorted(paths):
            name = str(entry.relative_to(root)); contents = entry.read_bytes(); files[name] = sha256(entry)
            if __import__('hashlib').sha256(contents).hexdigest() != files[name]:
                raise ValueError('Artifact changed during bundle read')
            info = tarfile.TarInfo(name); info.size = len(contents); info.mode = 0o444; info.mtime = 0
            archive.addfile(info, io.BytesIO(contents))
    manifest = {'schema_version': 1, 'kind': 'portable_causal_artifact_paths', 'descriptor': descriptor,
                'descriptor_sha256': sha256(path), 'operator_snapshot': SOURCE.name, 'files': files}
    (temporary / 'manifest.json').write_bytes(canonical_json(manifest)); validate(root, descriptor); verify(SOURCE)
    for entry in temporary.iterdir():
        with entry.open('rb') as stream:
            os.fsync(stream.fileno())
        entry.chmod(0o444)
    _sync_directory(temporary); temporary.rename(output); _sync_directory(output.parent)
    print(canonical_json({'status': 'passed', 'directory': str(output), 'files': len(files),
                          'manifest_sha256': sha256(output / 'manifest.json'), 'archive_sha256': sha256(output / 'payload.tar')}).decode().strip())


if __name__ == '__main__':
    main()
