#!/usr/bin/env python3
"""Seal the closed fork qualification, including both retained failed checks."""
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
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); study = root / 'research/studies/continued_selfplay'
    paths = {
        'failed_fixture': 'runs/qualification/continued-selfplay-contract-0c33297c/result.json',
        'initial_contract': 'runs/qualification/continued-selfplay-contract-071c7832/result.json',
        'cpu': 'runs/qualification/continued-selfplay-cpu-e3b08940/result.json',
        'preflight': 'runs/qualification/continued-selfplay-preflight-82451357/result.json',
        'tpu': 'runs/qualification/continued-selfplay-tpu-82451357/result.json',
        'evaluation_contract': 'runs/qualification/continued-selfplay-eval-contract-0b25622e/result.json',
        'multithreaded_games': 'runs/qualification/continued-selfplay-gtp-528fd20d/result.json',
        'failed_multithreaded_exactness': 'research/studies/continued_selfplay/gtp_result.json',
        'recorded_history': 'runs/qualification/continued-selfplay-recorded-gtp-cd25fc84/result.json',
        'single_thread_games': 'runs/qualification/continued-selfplay-single-thread-gtp-991d0faa/result.json',
        'single_thread_exactness': 'research/studies/continued_selfplay/single_thread_gtp_result.json'}
    records = {name: read_json(root / path) for name, path in paths.items()}
    for name, record in records.items():
        require(record['status'] == ('failed' if name in ('failed_fixture', 'failed_multithreaded_exactness') else 'passed'), 'Closed qualification status differs: ' + name)
    comparisons = {}
    for name in ('cpu', 'tpu'):
        comparisons[name] = {}
        for label, item in records[name]['comparisons'].items():
            require(sha256(root / item['path']) == item['sha256'], 'Comparison changed')
            compared = read_json(root / item['path']); require(compared['status'] == 'passed', 'Comparison failed')
            comparisons[name][label] = compared
    require(sum(r['arrays_exact'] for r in comparisons['cpu']['fork_comparison']['hosts']) == 82
            and sum(r['arrays_exact'] for r in comparisons['tpu']['fork_comparison']['hosts']) == 648
            and sum(r['subsequent_games_exact'] for r in comparisons['tpu']['fork_comparison']['hosts']) == 328
            and sum(r['compared_subsequent_games'] for r in comparisons['tpu']['resume_comparison']['hosts']) == 135,
            'Qualified state/game counts differ')
    report = {'schema_version': 1, 'kind': 'complete_state_fork_qualification', 'status': 'passed', 'analysis_snapshot': SOURCE.name,
              'checks': {name: {'path': path, 'sha256': sha256(root / path), 'status': records[name]['status']} for name, path in paths.items()},
              'cpu_saved_arrays_exact': 82, 'cpu_original_continuation_games_exact': 20, 'cpu_child_resume_games_exact': 12,
              'tpu_saved_arrays_exact_each_comparison': 648, 'tpu_original_continuation_games_exact': 328, 'tpu_child_resume_games_exact': 135,
              'recorded_gtp_search_comparisons': records['recorded_history']['search_comparisons'],
              'recorded_gtp_board_comparisons': records['recorded_history']['board_comparisons'],
              'single_thread_fresh_game_boards_exact': records['single_thread_exactness']['checked_boards'],
              'single_thread_fresh_games': 8, 'initial_multithreaded_fresh_games': 8,
              'recorded_qualification_attempt_chip_hours': sum(row['reserved_chip_hours'] for row in records['tpu']['attempts'].values()),
              'retained_failures': ['Initial data-only test fixture omitted train.py.', 'Four-thread KataGo games differed despite fixed seeds; strict external exactness failed.'],
              'qualified_changes': ['stop turn', 'checkpoint/log cadence', 'explicit constant learning-rate override'],
              'production_promotion': False,
              'scope': 'Explicit complete-state forks preserve the exact model/native/runtime/topology and learner/replay/actor state. Same-source child resume remains strict. Recorded-history and single-search-thread real-KataGo adapter checks passed. No injected failure, external durability, strength or utilization claim.'}
    with (study / 'result.json').open('xb') as stream: stream.write(canonical_json(report))
    files = {}; snapshots = {SOURCE.name}; folders = {(root / path).parent for name, path in paths.items() if not path.startswith('research/')}
    folders.update(root / 'runs' / row['attempt'] for row in records['tpu']['attempts'].values())
    def include(path):
        files[str(path.relative_to(root))] = sha256(path)
        if path.suffix == '.json':
            for identity in re.findall(r'(?<![a-zA-Z0-9])[a-f0-9]{64}(?![a-zA-Z0-9])', path.read_text()):
                if (root / '.gozero/snapshots' / identity).is_dir(): snapshots.add(identity)
    for folder in sorted(folders):
        require(read_json(folder / 'result.json')['status'] in ('passed', 'failed'), 'A qualification attempt remains open')
        for path in sorted(folder.rglob('*')):
            if path.is_file(): include(path)
    for path in sorted(study.glob('*.json')):
        if path.name != 'artifacts.json': include(path)
    for path in sorted((root / 'eval/continued_selfplay').glob('*')):
        if path.is_file(): include(path)
    for sid in ('0f2cf041bc75abdc3f3063ac332f43d6d92acd2ec6be82d333138d44feb267b6', '08b8996848f5f08ace154f1d99fe4ed9f37c4cb8c2965a4a96b15f726f483295'):
        for name in ('receipt.json', 'lib_gozero_native.so'): include(root / '.gozero/native' / sid / name)
    for identity in sorted(snapshots):
        source = root / '.gozero/snapshots' / identity; verify(source); include(source / 'manifest.json')
    index = {'schema_version': 1, 'kind': 'complete_state_fork_qualification_artifact_index', 'status': 'passed',
             'operator_snapshot': SOURCE.name, 'result_sha256': sha256(study / 'result.json'),
             'files_sha256': dict(sorted(files.items())), 'source_snapshots': sorted(snapshots),
             'scope': 'All new closed qualification raw artifacts and source/native identities, including failed parents. Earlier full training remains referenced by its committed checkpoint/source hashes.'}
    with (study / 'artifacts.json').open('xb') as stream: stream.write(canonical_json(index))
    print(canonical_json({'result_sha256': sha256(study / 'result.json'), 'index_sha256': sha256(study / 'artifacts.json'),
                          'files': len(files), 'snapshots': len(snapshots)}).decode(), flush=True)


if __name__ == '__main__': main()
