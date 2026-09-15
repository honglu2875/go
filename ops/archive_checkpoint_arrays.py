#!/usr/bin/env python3
"""Plan, qualify and apply reversible compression of closed legacy checkpoints."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoint_archives as archives, checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['plan', 'qualify', 'apply', 'restore'])
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--expected-plan-sha256')
    p.add_argument('--qualification', type=Path)
    p.add_argument('--expected-qualification-sha256')
    p.add_argument('--receipt')
    p.add_argument('--expected-receipt-sha256')
    args = p.parse_args()
    verify(SOURCE)
    root = args.workspace_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'operator_snapshot': SOURCE.name,
              'operation': args.operation, 'start_unix': time.time(), 'status': 'running'}
    try:
        if args.operation == 'plan':
            entries, evidence = [], {}
            for path in sorted(root.glob('runs/pod-*/rank-*/artifacts/checkpoints/turn-*/arrays.npz')):
                if path.stat().st_size < 100_000_000:
                    continue
                with zipfile.ZipFile(path) as z:
                    if any(x.compress_type != zipfile.ZIP_STORED for x in z.infolist()):
                        continue
                result = path.parents[4] / 'result.json'
                closed = read_json(result)
                if closed.get('status') not in ['passed', 'failed'] or 'end_unix_time' not in closed:
                    raise ValueError('Cannot archive an unfinished attempt')
                if path.stat().st_mode & 0o222:
                    raise ValueError('Candidate arrays are writable')
                manifest = path.with_name('manifest.json')
                expected = read_json(manifest)['files']['arrays.npz']
                if expected['bytes'] != path.stat().st_size:
                    raise ValueError('Candidate size differs')
                evidence[str(result.relative_to(root))] = checkpoints.sha256(result)
                entries.append({'path': str(path.relative_to(root)),
                                'manifest_sha256': checkpoints.sha256(manifest), **expected})
            report.update(kind='checkpoint_array_archival_plan', entries=entries, evidence_sha256=evidence,
                          original_logical_bytes=sum(x['bytes'] for x in entries),
                          restoration_contract='Original manifests and NPZ bytes preserved. Restore arrays before historical checkpoint readers run.')
        elif args.operation == 'restore':
            if not args.receipt or not args.expected_receipt_sha256:
                raise ValueError('Pinned receipt required')
            path = archives.restore(root, args.receipt, args.expected_receipt_sha256)
            state, arrays, actors = checkpoints.read(path.parent)
            report.update(path=str(path.relative_to(root)), array_count=len(arrays), original_sha256=checkpoints.sha256(path))
        else:
            if not args.plan or checkpoints.sha256(args.plan) != args.expected_plan_sha256:
                raise ValueError('Pinned plan required')
            plan = read_json(args.plan)
            if plan['kind'] != 'checkpoint_array_archival_plan' or plan['status'] != 'passed':
                raise ValueError('Invalid plan')
            for name, expected in plan['evidence_sha256'].items():
                if checkpoints.sha256(root / name) != expected:
                    raise ValueError('Closed attempt evidence changed')
            report['plan_sha256'] = args.expected_plan_sha256
            if args.operation == 'qualify':
                selected = plan['entries'][0]
                row = archives.archive(root, selected['path'], selected['manifest_sha256'])
                original = root / selected['path']
                target = args.output.parent / (args.output.stem + '-restored')
                target.mkdir(exist_ok=False)
                for name in ['manifest.json', 'state.json', 'actors.json']:
                    shutil.copyfile(original.with_name(name), target / name)
                archives.restore(root, row['receipt_path'], row['receipt_sha256'],
                                 destination=str((target / 'arrays.npz').relative_to(root)))
                state, arrays, actors = checkpoints.read(target, expected_manifest_sha256=selected['manifest_sha256'])
                report.update(archive=row, restored_checkpoint=str(target.relative_to(root)),
                              original_preserved=original.exists(), array_count=len(arrays),
                              restored_npz_sha256=checkpoints.sha256(target / 'arrays.npz'))
                if not report['original_preserved'] or report['restored_npz_sha256'] != selected['sha256']:
                    raise ValueError('Real checkpoint round trip failed')
            else:
                if not args.qualification or checkpoints.sha256(args.qualification) != args.expected_qualification_sha256:
                    raise ValueError('Pinned round-trip qualification required before removal')
                qualification = read_json(args.qualification)
                if (qualification['status'] != 'passed' or qualification['plan_sha256'] != args.expected_plan_sha256
                        or not qualification['original_preserved']):
                    raise ValueError('Round-trip qualification differs or failed')
                report['qualification_sha256'] = args.expected_qualification_sha256
                report['free_bytes_before'] = shutil.disk_usage(root).free
                report['entries'] = []
                journal = args.output.with_suffix('.progress.jsonl')
                with journal.open('x') as log:
                    for selected in plan['entries']:
                        row = archives.archive(root, selected['path'], selected['manifest_sha256'], remove_original=True)
                        report['entries'].append(row)
                        log.write(canonical_json(row).decode()); log.flush()
                        print(json.dumps({'archived': len(report['entries']), 'of': len(plan['entries']),
                                          'original_bytes': row['original_bytes'], 'archive_bytes': row['archive_bytes']}), flush=True)
                report['free_bytes_after'] = shutil.disk_usage(root).free
                report['released_original_allocated_bytes'] = sum(x['released_original_allocated_bytes'] for x in report['entries'])
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        verify(SOURCE)
        report['end_unix'] = time.time()
        archives._publish(args.output, report)
        print(json.dumps({'status': report['status'], 'output': str(args.output), 'sha256': checkpoints.sha256(args.output)}), flush=True)


if __name__ == '__main__':
    main()
