#!/usr/bin/env python3
"""Seal references to the closed state-expert study and all retained raw attempts."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--expected-result-sha256', required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    study = root / 'research/studies/state_expert_distillation'; result_path = study / 'result.json'; result = read_json(result_path)
    if sha256(result_path) != a.expected_result_sha256 or result['status'] != 'passed':
        raise ValueError('Final independent audit differs')
    qualification = read_json(study / 'tpu_qualification_result.json'); training = read_json(study / 'training_result.json')
    cpu = read_json(study / 'cpu_spec.json'); gtp = [read_json(study / p) for p in ('gtp_spec.json', 'gtp_retry_1_spec.json')]
    spec = read_json(study / 'pilot_spec.json'); qspec = read_json(study / 'tpu_qualification_spec.json')
    folders = [root / cpu['output'], root / qspec['output'], *[root / s['output'] for s in gtp],
               root / 'runs/qualification/state-expert-raw-policy-7b89a249',
               root / 'runs/qualification/state-expert-models-b64444d7',
               (root / spec['external_criterion_tests_path']).parent, (root / qspec['preflight']).parent,
               *[root / 'runs' / row['attempt'] for row in qualification['attempts']],
               *[root / 'runs' / row['attempt'] for row in training['arms'].values()],
               *[root / row['directory'] for row in result['arms'].values()]]
    files = {}; snapshots = {SOURCE.name}

    def discover(value):
        if isinstance(value, dict):
            for child in value.values():
                discover(child)
        elif isinstance(value, list):
            for child in value:
                discover(child)
        elif isinstance(value, str):
            candidates = re.findall(r'(?<![a-zA-Z0-9])[a-f0-9]{64}(?![a-zA-Z0-9])', value)
            for identity in candidates:
                if (root / '.gozero/snapshots' / identity).is_dir():
                    snapshots.add(identity)

    def include(path):
        if path.name == '.operator.lock':
            return
        files[str(path.relative_to(root))] = sha256(path)
        if path.suffix == '.json':
            discover(read_json(path))

    for folder in folders:
        if read_json(folder / 'result.json')['status'] not in ('passed', 'failed'):
            raise ValueError('A referenced attempt remains open')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                include(path)
    for path in sorted(study.glob('*.json')):
        if path.name != 'artifacts.json':
            include(path)
    for path in sorted((root / 'eval/state_expert_distillation').glob('*.json')):
        include(path)
    for name in ('.gozero/datasets/board-causal-13bcd5cd/manifest.json', '.gozero/datasets/causal-5c5f2c0d/manifest.json',
                 '.gozero/native/1bc92cbd36c8de3afe5e64987f9b08a2c4ed29320de3395c46c96a85442b6441/receipt.json',
                 '.gozero/native/1bc92cbd36c8de3afe5e64987f9b08a2c4ed29320de3395c46c96a85442b6441/lib_gozero_native.so'):
        include(root / name)
    for identity in sorted(snapshots):
        source = root / '.gozero/snapshots' / identity; verify(source)
        files[str((source / 'manifest.json').relative_to(root))] = sha256(source / 'manifest.json')
    report = {'schema_version': 1, 'kind': 'state_expert_study_artifact_index', 'status': 'passed', 'operator_snapshot': SOURCE.name,
              'result_sha256': sha256(result_path), 'source_snapshots': sorted(snapshots), 'files_sha256': dict(sorted(files.items())),
              'retains_failed_gtp_parent': True, 'scheduled_competitive_games': result['scheduled_games'],
              'caps_unassigned': result['capped_games'], 'cpu_continuation_arrays_exact': 147,
              'tpu_continuation_arrays_exact': sum(r['arrays_exact'] for r in qualification['arms'].values()),
              'recorded_new_attempt_chip_hours': qualification['recorded_attempt_chip_hours'] + training['recorded_pilot_attempt_chip_hours'],
              'registered_combined_criterion_met': result['registered_combined_criterion_met'], 'production_promotion': False,
              'scope': 'Study receipts, complete closed raw attempts, model descriptors and verified source manifests. Prior teacher/observer lineage remains referenced by their immutable manifests; external durability is not established.'}
    output = study / 'artifacts.json'
    with output.open('xb') as stream:
        stream.write(canonical_json(report))
    output.chmod(0o444)
    print(json.dumps({'path': str(output), 'sha256': sha256(output), 'files': len(files), 'sources': len(snapshots),
                      'recorded_new_attempt_chip_hours': report['recorded_new_attempt_chip_hours']}), flush=True)


if __name__ == '__main__':
    main()
