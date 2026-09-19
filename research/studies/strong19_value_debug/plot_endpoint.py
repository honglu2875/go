"""Plot the audited repair with update and measured learning-time clocks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = {
    'source_cnn': ('CNN · source optimizer · MSE', '#7656a4'),
    'cnn_adamw': ('CNN · AdamW · MSE', '#2171b5'),
    'transformer_mse': ('Transformer · AdamW · MSE', '#cf772d'),
    'transformer_ce': ('Transformer · AdamW · CE', '#19835c'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output.with_suffix(s) for s in ('.png', '.pdf', '.csv', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Use a fresh artifact prefix')
    raw = args.input.read_bytes()
    data = json.loads(raw)
    if data['status'] != 'passed' or data['kind'] != 'completed_joint19_objective_repair_review':
        raise ValueError('Expected the complete audited endpoint comparison')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .075, 1, .82))
    handles = []
    for row_index, (metric, title) in enumerate([
        ('expert_kl', 'Policy KL'), ('value_mse', 'Value MSE')]):
        for column, (clock, xlabel, scale) in enumerate([
            ('turn', 'Accepted updates', 1), ('learning_seconds', 'Measured learning time (minutes)', 60)]):
            axis = axes[row_index, column]
            for arm, (label, color) in ARMS.items():
                for split, style in [('validation_history', '-'), ('training_probe_history', '--')]:
                    points = [r for r in data['curves'] if r['arm'] == arm and r['split'] == split]
                    line, = axis.plot([r[clock] / scale for r in points], [r[metric] for r in points],
                                      style, color=color, label=label, linewidth=1.7)
                    if row_index == column == 0 and split == 'validation_history':
                        handles.append(line)
            axis.set(title=title, xlabel=xlabel, ylabel='Lower is better', xlim=(0, None))
            axis.grid(alpha=.18)
    fig.suptitle('19×19 value-objective repair · fixed-data, single-seed pilot', fontsize=14)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.text(.5, .025,
        'Solid: full validation (60,284 positions) · Dashed: fixed training probe (53,743 positions)\n'
        'Each arm: 108 updates / 5,787,025 training-position exposures. CE time sums both segments.\n'
        'Learning time excludes compilation, evaluation and checkpointing. These are supervised losses, not playing strength.',
        ha='center', fontsize=9)
    fig.savefig(outputs[0], dpi=160)
    fig.savefig(outputs[1])
    plt.close(fig)
    with outputs[2].open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['curves'][0]))
        writer.writeheader()
        writer.writerows(data['curves'])
    with outputs[3].open('x') as stream:
        json.dump(dict(kind='joint19_repair_comparison_plot', input_sha256=hashlib.sha256(raw).hexdigest(),
            operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:3]}), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='passed', outputs=[str(p) for p in outputs])))


if __name__ == '__main__':
    main()
