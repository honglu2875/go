#!/usr/bin/env python3
"""Publish both architectures' complete, identically sampled LR comparisons."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def publish(path, data):
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fchmod(stream.fileno(), 0o444); os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--cnn-execution', type=Path, required=True)
    p.add_argument('--transformer-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve(); recipe = Path(__file__).parent
    if sha256(a.registration) != a.registration_sha256:
        raise ValueError('CNN registration changed')
    closed = read_json(a.cnn_execution)
    if closed['status'] != 'passed' or closed['registration_sha256'] != a.registration_sha256:
        raise ValueError('CNN sweep did not finish')
    a.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': '/tmp/gozero-mpl'}
    analysis = a.output / 'cnn-analysis.json'
    command = [sys.executable, '-B', str(recipe / 'analyze_lr.py'), '--workspace-root', str(root),
               '--registration', str(a.registration), '--registration-sha256', a.registration_sha256,
               '--output', str(analysis)]
    for triple in closed['audits']:
        command.extend(['--audit', *triple])
    subprocess.run(command, cwd=root, env=env, check=True)
    cnn = read_json(analysis); transformer = read_json(a.transformer_analysis)
    if any(r['status'] != 'passed' or not r['complete'] or r['pending_rates'] for r in (cnn, transformer)):
        raise ValueError('Both full grids must be complete')
    crows = {r['peak_learning_rate']: r for r in cnn['completed_arms']}
    trows = {r['peak_learning_rate']: r for r in transformer['completed_arms']}
    if set(crows) != set(trows):
        raise ValueError('The learning-rate grids differ')
    pairs = []
    for rate in sorted(crows):
        left, right = crows[rate], trows[rate]
        path = a.output / f'paired-{rate:.0e}.json'
        command = [sys.executable, '-B', str(recipe / 'compare_learning.py'),
                   '--workspace-root', str(root), '--audit', 'CNN', left['audit_path'], left['audit_sha256'],
                   '--audit', 'Transformer', right['audit_path'], right['audit_sha256'], '--output', str(path)]
        subprocess.run(command, cwd=root, env=env, check=True)
        pairs.append({'rate': rate, 'path': path.name, 'sha256': sha256(path)})
    cbest = crows[cnn['lowest_completed_endpoint_rate']]
    tbest = trows[transformer['lowest_completed_endpoint_rate']]
    def plot(name, selections):
        command = [str(root / '.gozero/analysis-environments/plotting/bin/python'), '-B',
                   str(recipe / 'plot_audited.py'), '--workspace-root', str(root), '--output', str(a.output / name)]
        for label, row in selections:
            command.extend(['--audit', label, row['audit_path'], row['audit_sha256']])
        subprocess.run(command, cwd=root, env=env, check=True)
    plot('cnn-figures', [(f'{rate:.0e}', crows[rate]) for rate in sorted(crows)])
    plot('selected-figures', [(f"CNN {cbest['peak_learning_rate']:.0e}", cbest),
                              (f"Transformer {tbest['peak_learning_rate']:.0e}", tbest)])
    lines = [f"Both models completed the same four-rate AdamW sweep. The CNN's lowest final validation KL is **{cbest['endpoint']['expert_kl']:.6f} at {cbest['peak_learning_rate']:.0e}**; the CNN-encoder transformer's is **{tbest['endpoint']['expert_kl']:.6f} at {tbest['peak_learning_rate']:.0e}**.", '',
        '| Peak LR | CNN KL ↓ | Transformer KL ↓ | CNN top-move agreement | Transformer top-move agreement |',
        '|---|---:|---:|---:|---:|']
    for rate in sorted(crows):
        c, t = crows[rate]['endpoint'], trows[rate]['endpoint']
        lines.append(f"| {rate:.0e} | {c['expert_kl']:.6f} | {t['expert_kl']:.6f} | {100*c['expert_top1']:.2f}% | {100*t['expert_top1']:.2f}% |")
    lines.extend(['', 'Each cell completed 1,024 accepted updates. Independent audits verified checkpoint arrays, identical initial weights within each architecture, identical episode/D4 draws across both architectures, and the same complete validation population. Each run saw 11,469,333 position exposures from 836,486 training positions; validation contains 102,339 positions from 1,170 games. The test split remains closed.', '',
        'Only peak LR and its proportional endpoint changed within each architecture. Warmup remains 64 updates, cosine ends at 30% of peak, and the remaining AdamW settings are unchanged. Each architecture reuses its existing 1e-4 endpoint. The CNN retains its same-target training helper; the transformer has one policy head.', '',
        'Our shared AdamW training recipe is an adaptation, not KataGo\'s official production optimizer schedule. This comparison tunes both models over the same LR grid before further encoder-capacity changes. Individually selected endpoints include validation-selection uncertainty and do not establish a global optimum, convergence, Go playing strength or RL improvement.', '',
        '[CNN learning curves](cnn-figures/learning.png) · [Selected endpoint learning curves](selected-figures/learning.png) · [Selected phase errors](selected-figures/phases.png) · [CNN analysis](cnn-analysis.json)', ''])
    publish(a.output / 'REPORT.md', '\n'.join(lines).encode())
    manifest = {'kind': 'equal_grid_cnn_transformer_lr_comparison', 'status': 'passed', 'operator_snapshot': SOURCE.name,
        'cnn_registration_sha256': a.registration_sha256, 'cnn_execution_sha256': sha256(a.cnn_execution),
        'transformer_analysis_path': str(a.transformer_analysis), 'transformer_analysis_sha256': sha256(a.transformer_analysis),
        'cnn_selected_rate': cbest['peak_learning_rate'], 'transformer_selected_rate': tbest['peak_learning_rate'],
        'same_rate_pair_audits': pairs, 'files': {str(p.relative_to(a.output)): sha256(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    verify(SOURCE); publish(a.output / 'manifest.json', canonical_json(manifest))
    print(canonical_json({'status': 'passed', 'report': str(a.output / 'REPORT.md'), 'manifest_sha256': sha256(a.output / 'manifest.json')}).decode().strip())


if __name__ == '__main__':
    main()
