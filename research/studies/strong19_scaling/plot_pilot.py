"""Plot saved live observations or completed audits from the fixed joint19 pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('expert_kl', 'Position-weighted policy KL'), ('family_kl', 'Equal-family policy KL'),
          ('value_mse', 'Position-weighted value MSE'), ('value_family_mse', 'Equal-family value MSE')]
COLORS = ['#7257a3', '#2065b0', '#c56924']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, help='Label=JSON-path')
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new plot path')
    inputs = []; rows = []; populations = set(); provisional = False
    for item in args.input:
        label, filename = item.split('=', 1); path = Path(filename); raw = path.read_bytes(); data = json.loads(raw)
        if data['kind'] == 'live_joint19_pilot_observation':
            histories = data['histories']; provisional = True
        elif data['kind'] == 'complete_joint_learner_audit' and data['status'] == 'passed' and data['purpose'] == 'learning':
            histories = dict(validation=data['validation_history'], training_probe=data['training_probe_history'])
        else:
            raise ValueError('Expected a scientific pilot observation or audit')
        inputs.append(dict(label=label, sha256=hashlib.sha256(raw).hexdigest(), snapshot=data['snapshot']))
        for split, history in histories.items():
            for row in history:
                if split == 'validation':
                    populations.add(row['episode_ids_sha256'])
                for metric, _ in PANELS:
                    rows.append(dict(label=label, split=split, turn=row['turn'], metric=metric, value=row['metrics'][metric]))
    if len(populations) != 1:
        raise ValueError('Validation populations differ or no observations exist')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .065, 1, .86))
    handles = []; labels = []
    for (metric, title), axis in zip(PANELS, axes.flat):
        for index, source in enumerate(inputs):
            for split, style in [('validation','-'), ('training_probe','--')]:
                chosen = [x for x in rows if x['label'] == source['label'] and x['metric'] == metric and x['split'] == split]
                if not chosen:
                    continue
                line, = axis.plot([x['turn'] for x in chosen], [x['value'] for x in chosen],
                    style, color=COLORS[index % len(COLORS)], marker='o', markersize=3, linewidth=1.6)
                if metric == PANELS[0][0] and split == 'validation':
                    handles.append(line); labels.append(source['label'])
        axis.set_title(title); axis.set_xlabel('Accepted updates'); axis.set_ylabel('Lower is better')
        axis.grid(alpha=.2); axis.set_xlim(left=0)
    fig.suptitle('Fixed 19×19 joint-learning pilot' + (' · provisional observations' if provisional else ''), fontsize=14)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.96), ncol=3)
    fig.text(.5,.018,'Solid: full fixed validation · Dashed: fixed training probe · Same draws and targets\n'
             'Single-seed learnability comparison; playing strength requires separate KataGo matches.', ha='center', fontsize=9)
    fig.savefig(args.output,dpi=160);plt.close(fig)
    with args.output.with_suffix('.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=['label','split','turn','metric','value']);writer.writeheader();writer.writerows(rows)
    with args.output.with_suffix('.json').open('x') as stream:
        json.dump(dict(kind='joint19_pilot_plot',inputs=inputs,provisional=provisional,
            validation_population_sha256=next(iter(populations)),output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output),observations=len(rows),provisional=provisional)))


if __name__ == '__main__':
    main()
