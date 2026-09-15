"""Export the observed step/time curves; no smoothing or interpolation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if sha(args.result) != args.sha256:
        raise ValueError('Diagnostic identity differs')
    record = json.loads(args.result.read_text())
    if record['status'] != 'passed' or record['kind'] != 'audited_learning_time_diagnostic':
        raise ValueError('Expected an audited learning-time diagnostic')
    png = args.result.parent / 'curves-review-001.png'
    receipt = args.result.parent / 'plot-review-001.json'
    if png.exists() or receipt.exists():
        raise FileExistsError('Plot outputs already exist')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7), layout='constrained')
    colors = {'cnn': '#2869a6', 'transformer': '#c86523'}
    labels = {'cnn': 'CNN', 'transformer': 'Transformer'}
    for row, (metric, title) in enumerate((('expert_kl', 'Position-weighted validation KL'),
                                          ('family_kl', 'Equal-family validation KL'))):
        for col, clock in enumerate(('turn', 'learning_seconds')):
            ax = axes[row, col]
            for label, arm in record['arms'].items():
                points = [p for p in arm['curve'] if p['turn'] > 0]
                x = [p[clock] / (60 if col else 1) for p in points]
                ax.plot(x, [p[metric] for p in points], label=labels[label], color=colors[label],
                        marker='o', markersize=3.3, linewidth=1.65)
            ax.set_title(title)
            ax.set_xlabel('Updates (paired data draws)' if col == 0 else 'Measured learning time (minutes)')
            ax.set_ylabel('KL (log scale)')
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.grid(True, which='both', alpha=.18)
            if col:
                ax.axvline(record['common_learning_seconds'] / 60, color='#777777',
                           linestyle=':', linewidth=1)
            if row == 0:
                ax.legend(frameon=False)
    fig.suptitle('Larger fixed 9×9 corpus · paired seed 1', fontsize=16, y=.98)
    fig.get_layout_engine().set(rect=(0, .085, 1, .845))
    fig.text(.02, .015,
             'Observed validations every 256 updates; initial validation is retained in CSV. Lines connect observations.\n'
             'Learning time excludes setup, sampling, validation and checkpointing. Dotted line: common observed time limit.\n'
             'Secondary time diagnostic; the registered comparison uses the 4,096-update endpoints.', fontsize=9)
    fig.savefig(png, dpi=170)
    plt.close(fig)
    receipt.write_text(json.dumps(dict(kind='learning_time_diagnostic_plot',
        diagnostic_sha256=args.sha256, operator_sha256=sha(Path(__file__)),
        png_sha256=sha(png), matplotlib_version=matplotlib.__version__), indent=2) + '\n')
    png.chmod(0o444); receipt.chmod(0o444)
    print(json.dumps(dict(png=str(png), png_sha256=sha(png))))


if __name__ == '__main__':
    main()
