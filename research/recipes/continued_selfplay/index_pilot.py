#!/usr/bin/env python3
"""Seal the closed continuation pilot without hiding failed diagnostic games."""
import argparse
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value: raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--expected-result-sha256', required=True); a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    study = root / 'research/studies/online_annealing'; spec = read_json(study / 'pilot_spec.json')
    result = read_json(study / 'result.json'); training = read_json(study / 'training_result.json')
    require(sha256(study / 'result.json') == a.expected_result_sha256 and result['status'] == training['status'] == 'passed'
            and result['spec_sha256'] == training['spec_sha256'] == sha256(study / 'pilot_spec.json'), 'Final audit or registration differs')
    folders = [root / spec['training_output'], *[root / 'runs' / row['attempt'] for row in training['arms'].values()],
               *[root / row['directory'] for row in result['arms'].values()],
               root / 'runs/qualification/online-annealing-preflight-715e3f48',
               root / 'runs/qualification/online-annealing-paired-score-fc162064',
               root / 'runs/qualification/online-annealing-raw-audit-fc162064',
               root / 'runs/qualification/online-annealing-timeout-audit-a254168b']
    files = {}; snapshots = {SOURCE.name}
    def include(path):
        files[str(path.relative_to(root))] = sha256(path)
        if path.suffix == '.json':
            for identity in re.findall(r'(?<![a-zA-Z0-9])[a-f0-9]{64}(?![a-zA-Z0-9])', path.read_text()):
                if (root / '.gozero/snapshots' / identity).is_dir(): snapshots.add(identity)
    for folder in folders:
        require(read_json(folder / 'result.json')['status'] in ('passed', 'failed'), 'Referenced attempt is still open')
        for path in sorted(folder.rglob('*')):
            if path.is_file(): include(path)
    controller = root / 'runs/online-annealing-eval-control-59a7bfed'
    require(len(read_json(controller / 'result.json')['arms']) == 2, 'Evaluation orchestration is incomplete')
    for path in sorted(controller.rglob('*')):
        if path.is_file(): include(path)
    for item in spec['prerequisites']:
        require(sha256(root / item['path']) == item['sha256'], 'Prerequisite changed'); include(root / item['path'])
    for path in sorted(study.glob('*.json')):
        if path.name != 'artifacts.json': include(path)
    for path in sorted((root / 'eval/online_annealing').glob('*')):
        if path.is_file(): include(path)
    for name in ('research/studies/continued_selfplay/result.json', 'research/studies/continued_selfplay/artifacts.json',
                 '.gozero/native/08b8996848f5f08ace154f1d99fe4ed9f37c4cb8c2965a4a96b15f726f483295/receipt.json',
                 '.gozero/native/08b8996848f5f08ace154f1d99fe4ed9f37c4cb8c2965a4a96b15f726f483295/lib_gozero_native.so'):
        include(root / name)
    for identity in sorted(snapshots):
        source = root / '.gozero/snapshots' / identity; verify(source); include(source / 'manifest.json')
    index = {'schema_version': 1, 'kind': 'online_annealing_artifact_index', 'status': 'passed',
             'operator_snapshot': SOURCE.name, 'result_sha256': sha256(study / 'result.json'),
             'files_sha256': dict(sorted(files.items())), 'source_snapshots': sorted(snapshots),
             'scheduled_games': result['scheduled_games'], 'capped_games': result['capped_games'],
             'process_failures': result['process_failures'], 'complete_study_execution': result['complete_study_execution'],
             'registered_combined_criterion_met': result['registered_combined_criterion_met'], 'production_promotion': False,
             'scope': 'Complete new training/evaluation/qualification artifacts and source/native references, including failed diagnostic attempts. Hashes establish integrity, not external durability.'}
    with (study / 'artifacts.json').open('xb') as stream: stream.write(canonical_json(index))
    print(canonical_json({'path': str((study / 'artifacts.json').relative_to(root)), 'sha256': sha256(study / 'artifacts.json'),
                          'files': len(files), 'snapshots': len(snapshots)}).decode(), flush=True)


if __name__ == '__main__': main()
