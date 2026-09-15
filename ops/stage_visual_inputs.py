#!/usr/bin/env python3
"""Stage pinned visual datasets or shared checkpoints on the declared hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.pod import SSH_OPTIONS, load_hosts
from gozero.snapshots import canonical_json, read_json, verify


def collect(root, path, expected, kind):
    files = {}
    def add(p, digest=None):
        p = artifact(root, str(p))
        actual = sha256(p)
        if digest is not None and digest != actual:
            raise ValueError('Artifact identity differs: ' + str(p))
        files[p] = {'sha256': actual, 'bytes': p.stat().st_size}
    add(path, expected); manifest = read_json(path)
    if kind == 'candidate':
        if manifest.get('kind') != 'visual_causal_checkpoint' or manifest.get('schema_version') != 1:
            raise ValueError('Expected a pinned visual checkpoint candidate')
        source = artifact(root, '.gozero/snapshots/' + manifest['training_snapshot'])
        source_manifest = verify(source)
        add(source / 'manifest.json')
        for name, item in source_manifest['files'].items():
            add(source / name, item['sha256'])
        result = manifest['training_result']
        add(artifact(root, result['path']), result['sha256'])
        checkpoint = artifact(root, manifest['checkpoint']['path'])
        files.update(collect(root, checkpoint.with_suffix('.group.json'),
                             manifest['checkpoint']['group_sha256'], 'checkpoint'))
        config = read_json(source / 'resolved_config.json')
        dataset = config['dataset']
        add(artifact(root, str(Path(dataset['path']) / 'manifest.json')), dataset['manifest_sha256'])
    elif kind == 'dataset':
        if manifest['kind'] not in ('visual_causal_teacher_dataset', 'katago_v7_expert_input_overlay'):
            raise ValueError('Expected visual dataset manifest')
        if manifest.get('spec', {}).get('kind') == 'visual_online_dataset':
            qualification = path.parent.parent / 'qualified.json'
            qualified = read_json(qualification)
            if qualified.get('status') != 'passed' or qualified.get('visual_manifest_sha256') != expected:
                raise ValueError('Online data has no successful final qualification')
            add(qualification)
            add(qualification.parent / 'receipt.json', qualified['receipt_sha256'])
        parent = manifest['parent_dataset']; base = artifact(root, parent['path'])
        add(base / 'manifest.json', parent['manifest_sha256'])
        original = read_json(base / 'manifest.json')
        if original['kind'] != 'causal_teacher_dataset':
            raise ValueError('Unexpected parent dataset')
        for directory, m in [(path.parent, manifest), (base, original)]:
            for shard in m['shards']:
                for field, digest in [('arrays', 'sha256'), ('evidence', 'evidence_sha256')]:
                    if field in shard:
                        if Path(shard[field]).name != shard[field]:
                            raise ValueError('Dataset shard escapes its directory')
                        add(directory / shard[field], shard[digest])
    elif kind == 'assets':
        if (set(manifest) != {'schema_version', 'kind', 'files', 'purpose'} or manifest['schema_version'] != 1
                or manifest['kind'] != 'visual_inference_assets' or not manifest['files']):
            raise ValueError('Invalid pinned inference asset closure')
        for name, item in manifest['files'].items():
            p = artifact(root, name); add(p, item['sha256'])
            if p.stat().st_size != item['bytes']:
                raise ValueError('Inference asset size differs')
    else:
        if manifest['kind'] != 'visual_replicated_checkpoint_group' or len(manifest['host_manifests']) != 4:
            raise ValueError('Expected four-host replicated checkpoint group')
        owner = artifact(root, manifest['owner_checkpoint_path'])
        attempt = owner.parents[3]
        if owner != attempt / 'rank-0/artifacts/checkpoints' / owner.name or not attempt.name.startswith('pod-'):
            raise ValueError('Shared checkpoint path does not belong to a pod attempt')
        for host, digest in manifest['host_manifests'].items():
            if host not in ('0', '1', '2', '3'):
                raise ValueError('Invalid checkpoint host')
            directory = attempt / ('rank-' + host) / 'artifacts/checkpoints' / owner.name
            add(directory / 'manifest.json', digest)
            saved = read_json(directory / 'manifest.json')
            if set(saved['files']) != {'arrays.npz', 'state.json', 'actors.json'}:
                raise ValueError('Unexpected checkpoint members')
            for name, item in saved['files'].items():
                add(directory / name, item['sha256'])
                if (directory / name).stat().st_size != item['bytes']:
                    raise ValueError('Checkpoint member size differs')
            add(directory.with_suffix('.group.json'), expected)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--kind', choices=['dataset', 'checkpoint', 'assets', 'candidate'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    files = collect(root, a.manifest.resolve(), a.expected_sha256, a.kind)
    if sum(v['bytes'] for v in files.values()) > 8 * 2**30:
        raise ValueError('Staging exceeds bounded eight-GiB input set')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    receipt = {'schema_version': 1, 'kind': 'visual_input_staging', 'operator_snapshot': SOURCE.name,
               'input_kind': a.kind, 'input_sha256': a.expected_sha256, 'started_unix': time.time(),
               'files': {str(k.relative_to(root)): v for k, v in files.items()}, 'status': 'running'}
    checks = {str(k): v for k, v in files.items()}
    remote_check = """import hashlib,json,os,sys
for name,item in json.load(sys.stdin).items():
 if os.path.islink(name) or not os.path.isfile(name):raise ValueError('Missing staged input: '+name)
 if os.path.getsize(name)!=item['bytes']:raise ValueError('Staged size differs: '+name)
 with open(name,'rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('Staged bytes differ: '+name)
print('all pinned input files verified')
"""
    python = '/home/go-user/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12'
    def stage(host):
        start = time.time()
        with (a.output / f'host-{host.rank}.log').open('x') as log:
            if a.kind in ('assets', 'candidate'):
                relative = b'\0'.join(str(p.relative_to(root)).encode() for p in sorted(files)) + b'\0'
                subprocess.run(['rsync', '-a', '--ignore-existing', '--relative', '--from0', '--files-from=-',
                    '-e', shlex.join(['ssh', *SSH_OPTIONS]), str(root) + '/', host.ssh + ':' + str(root) + '/'],
                    input=relative, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                subprocess.run(['ssh', *SSH_OPTIONS, host.ssh, shlex.join([python, '-B', '-c', remote_check])],
                    input=canonical_json(checks), check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                return {'host': host.ssh, 'rank': host.rank, 'status': 'passed', 'elapsed_seconds': time.time() - start}
            directories = sorted({p.parent for p in files})
            subprocess.run(['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['mkdir', '-p', *map(str, directories)])],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            for directory in directories:
                # Existing paths are never overwritten. Verification below
                # accepts identical files and rejects conflicting contents.
                subprocess.run(['rsync', '-a', '--ignore-existing', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
                                *[str(p) for p in files if p.parent == directory], host.ssh + ':' + str(directory) + '/'],
                               check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            subprocess.run(['ssh', *SSH_OPTIONS, host.ssh, shlex.join([python, '-B', '-c', remote_check])],
                           input=canonical_json(checks), check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        return {'host': host.ssh, 'rank': host.rank, 'status': 'passed', 'elapsed_seconds': time.time() - start}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipt['hosts'] = list(pool.map(stage, load_hosts(SOURCE / 'ops/hosts.json')))
        collect(root, a.manifest.resolve(), a.expected_sha256, a.kind); verify(SOURCE)
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error)); raise
    finally:
        receipt['finished_unix'] = time.time()
        (a.output / 'receipt.json').write_bytes(canonical_json(receipt))
    print(json.dumps({'status': receipt['status'], 'files': len(files), 'sha256': sha256(a.output / 'receipt.json')}))


if __name__ == '__main__':
    main()
