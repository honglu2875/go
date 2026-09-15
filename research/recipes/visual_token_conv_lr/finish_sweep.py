#!/usr/bin/env python3
"""Publish a reproducible final LR report after the bounded serial runner closes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def publish(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=0)
    a = p.parse_args()
    verify(SOURCE)
    root = a.workspace_root.resolve()
    if sha256(a.execution) != a.execution_sha256:
        raise ValueError('Execution receipt changed')
    execution = read_json(a.execution)
    for name, digest in execution['inputs'].items():
        if sha256(root / name) != digest:
            raise ValueError('Execution prerequisite changed: ' + name)
    verify(root / '.gozero/snapshots' / execution['operator_snapshot'])
    closed_path = root / execution['completion_receipt']
    deadline = time.monotonic() + a.wait_seconds
    while not closed_path.exists() or closed_path.stat().st_mode & 0o222:
        if time.monotonic() >= deadline:
            raise TimeoutError('Sweep is not closed; no report published')
        time.sleep(min(30, max(.1, deadline - time.monotonic())))
    closed = read_json(closed_path)
    if closed['status'] != 'passed' or closed['operator_snapshot'] != execution['operator_snapshot']:
        raise ValueError('Sweep did not finish successfully; inspect preserved runner logs')
    a.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': '/tmp/gozero-mpl'}
    registration = root / 'research/studies/visual_katago/lr_sweep_registration.json'
    analysis = a.output / 'analysis.json'
    command = [sys.executable, '-B', str(Path(__file__).with_name('analyze_lr.py')),
               '--workspace-root', str(root), '--registration', str(registration),
               '--registration-sha256', closed['registration_sha256'], '--output', str(analysis)]
    for triple in closed['audits']:
        command.extend(['--audit', *triple])
    subprocess.run(command, cwd=root, env=env, check=True)
    results = read_json(analysis)
    if not results['complete'] or results['pending_rates']:
        raise ValueError('All registered rates must finish before final selection')
    rows = results['completed_arms']
    figures = a.output / 'figures'
    command = [str(root / '.gozero/analysis-environments/plotting/bin/python'), '-B',
               str(Path(__file__).with_name('plot_learning.py')), '--workspace-root', str(root),
               '--output', str(figures)]
    for row in rows:
        command.extend(['--audit', format(row['peak_learning_rate'], '.0e'), row['audit_path'], row['audit_sha256']])
    subprocess.run(command, cwd=root, env=env, check=True)
    best = next(r for r in rows if r['peak_learning_rate'] == results['lowest_completed_endpoint_rate'])
    reference = next(r for r in rows if r['peak_learning_rate'] == 1e-4)
    difference = best['endpoint']['expert_kl'] - reference['endpoint']['expert_kl']
    rate = format(best['peak_learning_rate'], '.0e')
    lines = [f"The lowest final validation KL in this CNN-encoder transformer sweep is **{best['endpoint']['expert_kl']:.6f} at peak LR {rate}**. All four arms completed 1,024 accepted updates and passed independent checkpoint and data-draw audits.", '',
             '| Peak LR | Final LR | Validation KL ↓ | Top-move agreement ↑ | Training minutes | Clipped updates |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        end = row['endpoint']
        lines.append(f"| {row['peak_learning_rate']:.0e} | {row['end_learning_rate']:.0e} | {end['expert_kl']:.6f} | {100*end['expert_top1']:.2f}% | {row['timing']['learning_seconds']/60:.2f} | {row['clipped_updates']}/1024 |")
    lines.extend(['', f"The selected endpoint differs from the existing 1e-4 reference by {difference:+.6f} KL. This selects a candidate within the registered range and schedule; it does not establish a globally optimal LR or convergence.", '',
        'All non-LR settings, initial parameter bytes, episode and D4 draws, validation games and evaluation cadence matched exactly. Each arm saw 11,469,333 position exposures from the same 836,486 training positions. Validation contains 102,339 positions from 1,170 games; the test split remains closed.', '',
        'The architecture is unchanged: two 3×3 convolutions with 64 channels, per-point RMSNorm and SiLU, 36 overlapping visual tokens, and a width-768, 34-layer causal transformer with one policy head. Encoder and complete neural decode arithmetic are unchanged across rates. Warmup remains 64 updates; each cosine schedule ends at 30% of its peak.', '',
        'Learning dynamics at the registered validation points:', '',
        '| Peak LR | KL at 128 | KL at 512 | KL at 896 | KL at 1024 |',
        '|---|---:|---:|---:|---:|'])
    for row in rows:
        curve = {v['turn']: v['expert_kl'] for v in row['curve']}
        lines.append(f"| {row['peak_learning_rate']:.0e} | {curve[128]:.6f} | {curve[512]:.6f} | {curve[896]:.6f} | {curve[1024]:.6f} |")
    lines.extend(['', 'This is a one-seed, fixed weak-teacher policy-learning screen. LR selection uses validation data and carries selection uncertainty. CNN and linear-patch transformer learning rates were not swept here, so their earlier common-schedule comparison and this tuned result answer different questions. These losses do not measure Go strength or RL improvement.', '',
                  '[Learning curves](figures/learning.png) · [Phase errors](figures/phases.png) · [Audited analysis](analysis.json)', ''])
    publish(a.output / 'REPORT.md', '\n'.join(lines).encode())
    ledger = root / 'research/studies/runtime_qualification/reservation_ledger.json'
    subprocess.run([sys.executable, '-B', str(SOURCE / 'ops/update_reservation_ledger.py'),
                    '--workspace-root', str(root), '--expected-previous-sha256', sha256(ledger)],
                   cwd=root, env=env, check=True)
    manifest = {'kind': 'completed_lr_sweep_publication', 'status': 'passed', 'operator_snapshot': SOURCE.name,
                'execution_sha256': a.execution_sha256, 'runner_result_sha256': sha256(closed_path),
                'selected_rate': best['peak_learning_rate'], 'ledger_sha256': sha256(ledger),
                'files': {str(p.relative_to(a.output)): sha256(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    verify(SOURCE)
    publish(a.output / 'manifest.json', canonical_json(manifest))
    print(canonical_json({'status': 'passed', 'report': str(a.output / 'REPORT.md'),
                         'manifest_sha256': sha256(a.output / 'manifest.json')}).decode().strip(), flush=True)


if __name__ == '__main__':
    main()
