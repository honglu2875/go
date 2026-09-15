"""Reversibly move a closed checkpoint's exact bytes into two persistent parts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints, checkpoint_parts as parts
from gozero.checkpoint_archive import FILES, SSH, publish, verify_files
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('archive', 'restore'))
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path); p.add_argument('--audit-sha256')
    p.add_argument('--receipt', type=Path); p.add_argument('--receipt-sha256')
    p.add_argument('--runtime-key', required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    if root != Path('/workspace/go'): raise ValueError('Unexpected workspace')
    if any(not (x.parent / 'result.json').exists() for x in (root / 'runs').glob('pod-*/launch.json')):
        raise ValueError('Partition storage only between TPU attempts')
    a.output.mkdir(parents=True, exist_ok=False); started = time.time(); result = {'status': 'running'}
    runtime = root / '.gozero/environments' / a.runtime_key / 'bin/python'
    temporary = Path(tempfile.mkdtemp(prefix='gozero-closed-partition-', dir='/dev/shm'))
    try:
        if a.operation == 'restore':
            if a.receipt is None or checkpoints.sha256(a.receipt) != a.receipt_sha256:
                raise ValueError('Pinned archive receipt required')
            receipt = read_json(a.receipt)
            if receipt['status'] != 'passed' or receipt['operation'] != 'archive':
                raise ValueError('Uncommitted archive receipt')
            descriptor = Path(receipt['descriptor'])
            if checkpoints.sha256(descriptor) != receipt['descriptor_sha256']:
                raise ValueError('Composition descriptor changed')
            owner = Path(receipt['original_checkpoint']); owner.relative_to(root / 'runs')
            if (owner / 'arrays.npz').exists(): raise FileExistsError(owner / 'arrays.npz')
            parts.restore(descriptor, temporary / 'restored', python=runtime)
            for name in FILES - {'arrays.npz'}:
                if checkpoints.sha256(owner / name) != receipt['files'][name]['sha256']:
                    raise ValueError('Original checkpoint metadata changed')
            staging = owner / '.arrays-restoring.npz'
            with staging.open('xb') as out, (temporary / 'restored/arrays.npz').open('rb') as inp:
                shutil.copyfileobj(inp, out, 8 * 2**20); out.flush(); os.fsync(out.fileno()); os.fchmod(out.fileno(), 0o444)
            if checkpoints.sha256(staging) != receipt['files']['arrays.npz']['sha256']:
                raise ValueError('Restored checkpoint bytes differ')
            staging.rename(owner / 'arrays.npz'); checkpoints._sync_directory(owner)
            checkpoints.read(owner, expected_manifest_sha256=receipt['manifest_sha256'])
            result = {'status': 'passed', 'original_checkpoint': str(owner), 'archive_receipt_sha256': a.receipt_sha256,
                      'restored_arrays_sha256': checkpoints.sha256(owner / 'arrays.npz')}
        else:
            if a.audit is None or checkpoints.sha256(a.audit) != a.audit_sha256:
                raise ValueError('Pinned closed learning audit required')
            audit = read_json(a.audit)
            if audit['status'] != 'passed': raise ValueError('Learning audit failed')
            attempt = root / 'runs' / audit['attempt']; closed = read_json(attempt / 'result.json')
            if closed['status'] != 'passed' or checkpoints.sha256(attempt / 'result.json') != audit['closed_result_sha256']:
                raise ValueError('Attempt did not close successfully')
            for path, wanted in audit['input_files'].items():
                if checkpoints.sha256(root / path) != wanted: raise ValueError('Audited input changed')
            report = read_json(attempt / 'rank-0/artifacts/result.json')
            owner = Path(report['latest_checkpoint']['owner_checkpoint_path']); owner.relative_to(attempt)
            group_path = Path(report['latest_checkpoint']['path']).with_suffix('.group.json')
            if checkpoints.sha256(group_path) != report['latest_checkpoint']['group_sha256']:
                raise ValueError('Checkpoint group differs')
            manifest_sha = read_json(group_path)['host_manifests']['0']
            if checkpoints.sha256(owner / 'manifest.json') != manifest_sha:
                raise ValueError('Owner checkpoint manifest differs')
            files = {name: {'sha256': checkpoints.sha256(owner / name), 'bytes': (owner / name).stat().st_size}
                     for name in sorted(FILES)}
            if any((owner / name).is_symlink() or (owner / name).stat().st_mode & 0o222 for name in FILES):
                raise ValueError('Only regular immutable checkpoint files may be archived')
            cache = temporary / 'copy'; cache.mkdir()
            for name in FILES:
                shutil.copyfile(owner / name, cache / name); (cache / name).chmod(0o444)
            copy_receipt = {'kind': 'temporary_ram_checkpoint', 'status': 'temporary', 'cache_path': str(cache),
                'logical_path': str(owner), 'manifest_sha256': manifest_sha, 'files': files,
                'source_snapshot': closed['snapshot_id'], 'origin_kind': 'copy_of_persistent_closed_checkpoint',
                'scope': 'The original disk checkpoint remains authoritative until persistent parts have passed full reassembly/readback.'}
            verify_files(cache, copy_receipt)
            publish(a.output / 'ram-copy-receipt.json', copy_receipt)
            copy_receipt['receipt_sha256'] = checkpoints.sha256(a.output / 'ram-copy-receipt.json')
            total = files['arrays.npz']['bytes']; first = min(1_400_000_000, total)
            if total <= first: raise ValueError('This placement is for a checkpoint larger than 1.4 GB')
            allocations = [
                {'host': 2, 'bytes': first, 'reservation': '.gozero/checkpoint-part-reservations/spatial-followups-cnn-main-2'},
                {'host': 3, 'bytes': total - first, 'reservation': '.gozero/checkpoint-part-reservations/spatial-followups-cnn-main-3'}]
            offset = 0; planned = []
            for allocation in allocations:
                planned.append({**allocation, 'offset': offset,
                    'sha256': parts.digest_range(cache / 'arrays.npz', offset, allocation['bytes'])})
                offset += allocation['bytes']
            publish(a.output / 'placement-plan.json', {'kind': 'closed_checkpoint_partition_plan',
                'original_checkpoint': str(owner), 'audit_sha256': a.audit_sha256, 'files': files, 'parts': planned})
            for allocation in allocations:
                target = f"go-user@worker-{allocation['host']}"
                subprocess.run(['rsync', '-a', '--ignore-existing', '--protect-args', '-e', shlex.join(SSH),
                    '--', str(SOURCE) + '/', target + ':' + str(SOURCE) + '/'], check=True, timeout=180)
                reserve = root / allocation['reservation']
                script = ('import os,json;from pathlib import Path;'
                    'p=Path(' + repr(str(reserve)) + ');p.parent.mkdir(parents=True,exist_ok=True);'
                    'fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);'
                    'os.posix_fallocate(fd,0,' + str(allocation['bytes']) + ');os.fsync(fd);os.close(fd);'
                    'print(json.dumps({"path":str(p),"bytes":p.stat().st_size}))')
                command = [str(runtime), '-B', '-c', script]
                with (a.output / f"host{allocation['host']}-reservation.log").open('xb') as log:
                    subprocess.run([*SSH, target, shlex.join(command)], stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=120)
            promoted = parts.promote(copy_receipt, allocations, python=runtime)
            publish(a.output / 'composition.json', promoted)
            restored = temporary / 'restored'
            identity = parts.restore(promoted['descriptor'], restored, python=runtime)
            if identity != manifest_sha or checkpoints.sha256(restored / 'arrays.npz') != files['arrays.npz']['sha256']:
                raise ValueError('Real checkpoint reassembly differs')
            if any(not (x.parent / 'result.json').exists() for x in (root / 'runs').glob('pod-*/launch.json')):
                raise ValueError('A TPU attempt opened during maintenance; retain the original')
            if any(checkpoints.sha256(owner / name) != files[name]['sha256'] for name in FILES):
                raise ValueError('Original checkpoint changed before relocation')
            marker = owner.with_suffix('.partitioned.json')
            marker_record = {'kind': 'closed_checkpoint_partition_locator', 'status': 'committed',
                'original_checkpoint': str(owner), 'descriptor': promoted['descriptor'],
                'descriptor_sha256': promoted['descriptor_sha256'], 'files': files,
                'manifest_sha256': manifest_sha, 'operator_snapshot': SOURCE.name,
                'restore_contract': 'Use this operator restore with the pinned completed archive result; exact original arrays are restored beside unchanged metadata. The ordinary historical reader is then usable.'}
            publish(marker, marker_record)
            released = (owner / 'arrays.npz').stat().st_blocks * 512
            (owner / 'arrays.npz').unlink(); checkpoints._sync_directory(owner)
            result = {'status': 'passed', 'audit_sha256': a.audit_sha256, 'original_checkpoint': str(owner),
                'descriptor': promoted['descriptor'], 'descriptor_sha256': promoted['descriptor_sha256'],
                'manifest_sha256': manifest_sha, 'files': files, 'parts': promoted['parts'],
                'locator': str(marker), 'locator_sha256': checkpoints.sha256(marker),
                'roundtrip_manifest_sha256': identity, 'released_local_allocated_bytes': released,
                'scope': 'Exact compressed NPZ bytes and complete optimizer/state metadata retained in persistent parts on hosts2 and3. Ordinary-reader roundtrip passed before local-array removal; no unique checkpoint state deleted.'}
    except BaseException as error:
        result = {'status': 'failed', 'error': repr(error)}; raise
    finally:
        result.update(kind='closed_checkpoint_partition_operation', operation=a.operation,
            operator_snapshot=SOURCE.name, started_unix=started, ended_unix=time.time())
        publish(a.output / 'result.json', result)
        if result['status'] == 'passed': shutil.rmtree(temporary)
        else: print(json.dumps({'retained_temporary_copy': str(temporary)}), flush=True)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
