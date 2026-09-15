"""Plot the late learning-only curves from the closed, audited observation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or hashlib.sha256(args.input.read_bytes()).hexdigest() != args.input_sha256:
        raise ValueError('Use a new output and the pinned observation')
    data = json.loads(args.input.read_text())
    if data['status'] != 'passed' or data['kind'] != 'closed9_learning_time_observation':
        raise ValueError('Expected the completed learning-time observation')
    colors = {'CNN': '#2065b0', 'Transformer': '#bc581e', 'Transformer scale0.01': '#4c885b'}
    labels = dict(colors); labels.update({key: key for key in colors})
    labels['Transformer scale0.01'] = 'Encoder scale 0.01 (confirmation pending)'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .10, 1, .82))
    legend = {}
    for column, seed in enumerate(('seed1', 'seed2')):
        for name, arm in data['arms'].items():
            if not name.startswith(seed + '/'):
                continue
            label = name.split('/', 1)[1]
            rows = [row for row in arm['curve'] if row['maximum_rank_learning_seconds'] >= 45 * 60]
            for index, metric in enumerate(('policy_kl', 'family_kl')):
                axis = axes[index, column]
                line, = axis.plot([row['maximum_rank_learning_seconds'] / 60 for row in rows],
                                  [row[metric] for row in rows], color=colors[label], marker='o',
                                  markersize=3, linewidth=1.7, label=labels[label])
                legend[label] = line
                axis.scatter(rows[-1]['maximum_rank_learning_seconds'] / 60, rows[-1][metric],
                             color=colors[label], marker='D', s=28, zorder=5)
                axis.set_xlabel('Measured learning-update time (minutes)')
                axis.set_ylabel(('Position-weighted' if index == 0 else 'Equal-family') + ' validation KL')
                axis.set_title('Paired seed ' + str(arm['seed']))
                axis.grid(alpha=.2)
                axis.set_xlim(44, 128)
    fig.suptitle('9×9 validation by measured learning time · later observations', fontsize=14)
    fig.legend(list(legend.values()), [labels[key] for key in legend], loc='upper center',
               bbox_to_anchor=(.5, .96), ncol=3, fontsize=9)
    fig.text(.5, .025,
        'Learning time excludes compilation, sampling, validation, checkpointing and data generation.\n'
        'The transformer processes more training positions at this time budget. Same-update selection rules are unchanged.',
        ha='center', fontsize=9, color='#555555')
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
