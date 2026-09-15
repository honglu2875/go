#!/usr/bin/env python3
"""Render audited learning curves; requires an isolated Matplotlib environment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if (data['status'] != 'analyzed'
            or data['protocol_sha256'] != 'd7dbd7461cdc76e2222609971b8895f17d6baa7cfcec82d2515449130a61b6d3'
            or len(data['checkpoints']) != 16):
        raise ValueError('Complete audited registered result required')
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': data['protocol_sha256']})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1), sharey=True)
    for ax, seed in zip(axes, (27, 28)):
        for architecture, color in (('cnn', '#2465a7'), ('attention', '#c05716')):
            rows = sorted((r for r in data['checkpoints'] if r['seed'] == seed and r['architecture'] == architecture),
                          key=lambda r: r['turn'])
            x = np.asarray([r['global_real_moves'] / 1e6 for r in rows])
            lower, upper = np.asarray([r['scheduled_score_bounds'] for r in rows]).T * 100
            ax.errorbar(x, lower, yerr=np.stack((np.zeros_like(lower), upper - lower)),
                        fmt='o-', color=color, linewidth=1.7, capsize=5, markersize=5,
                        label='CNN' if architecture == 'cnn' else 'Attention')
            for xx, yy, row in zip(x, lower, rows):
                if row['completion_coverage'] < .9:
                    ax.annotate(f"{row['summary']['completed_games']}/64 complete*", (xx, yy), xytext=(-6, -24),
                                textcoords='offset points', ha='center', fontsize=8, color='#a52323')
        ax.set_title(f'Training seed {seed}', loc='left', fontsize=12, pad=12)
        ax.set_xticks(x, [f'{v:.2f}' for v in x])
        ax.set_xlabel('Real self-play moves (millions)', labelpad=9)
        ax.set_ylim(0, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=.18)
        ax.legend(loc='lower right', frameon=False)
    axes[0].set_ylabel('Possible score on all scheduled games (%)', labelpad=9)
    fig.suptitle('Registered 9×9 architecture learning curves', x=.085, y=.98, ha='left', fontsize=16)
    fig.text(.085, .916, 'Gumbel search: 16 simulations   ·   Historical KataGo level 3: 1 visit   ·   64 games per point',
             fontsize=9, color='#444444')
    fig.text(.085, .091, 'Lines join conservative score bounds; vertical bars cover unresolved games, not sampling uncertainty.', fontsize=8.5)
    fig.text(.085, .054, '*The registered 90% completion requirement fails at this checkpoint. The full study criterion is not met.',
             fontsize=8.5, color='#a52323')
    fig.subplots_adjust(left=.085, right=.975, bottom=.23, top=.82, wspace=.16)
    fig.savefig(args.output / 'learning-curves.svg', metadata={'Date': None})
    fig.savefig(args.output / 'learning-curves.png', dpi=180)
    plt.close(fig)
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'schema_version': 1, 'kind': 'audited_architecture_curve_figure',
               'result_sha256': sha(args.result), 'script_sha256': sha(Path(__file__)),
               'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
               'artifacts': {name: sha(args.output / name) for name in ('learning-curves.svg', 'learning-curves.png')},
               'unresolved_games_have_no_assigned_outcome': True}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    for path in args.output.iterdir():
        path.chmod(0o444)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
