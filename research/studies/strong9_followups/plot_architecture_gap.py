"""Render the closed paired architecture gap by move range."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--diagnostic-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.diagnostic.read_bytes()).hexdigest()!=args.diagnostic_sha256:
        raise ValueError('Diagnostic changed')
    report=json.loads(args.diagnostic.read_text())
    if report['status']!='passed' or report['kind']!='closed_larger9_architecture_gap_diagnostic':
        raise ValueError('Expected the audited closed comparison diagnostic')
    rows=[r for r in report['mean_phase_summary'] if r['positions']]
    labels=['Moves 1–16','Moves 17–64','Moves 65–128','Moves 129–256']
    if len(rows)!=len(labels):raise ValueError('Move-range population changed')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.115,1,.82))
    x=np.arange(len(rows));width=.37
    axes[0].bar(x-width/2,[r['mean_cnn_kl'] for r in rows],width,label='CNN',color='#2065b0')
    axes[0].bar(x+width/2,[r['mean_transformer_kl'] for r in rows],width,label='Transformer',color='#bc581e')
    axes[0].set_xticks(x,labels,rotation=16)
    axes[0].set_ylabel('Policy KL (nats)');axes[0].set_title('Mean endpoint KL across two seeds')
    axes[0].legend();axes[0].grid(axis='y',alpha=.18);axes[0].set_axisbelow(True)
    colors=['#7257a3','#ca7c33','#3e9593','#7897c3'];bottom=np.zeros(2)
    for i,label in enumerate(labels):
        y=np.asarray([c['phases']['rows'][i]['contribution_to_overall_gap'] for c in report['cases']])
        axes[1].bar([0,1],y,bottom=bottom,label=label,color=colors[i]);bottom+=y
    axes[1].set_xticks([0,1],['Paired seed 1','Paired seed 2'])
    axes[1].set_ylabel('Contribution to overall transformer − CNN KL')
    axes[1].set_title('Gap weighted by validation positions')
    axes[1].legend(fontsize=9,loc='upper left');axes[1].grid(axis='y',alpha=.18);axes[1].set_axisbelow(True)
    fig.suptitle('Fixed strong-teacher 9×9 corpus · completed 4,096-update comparison',fontsize=13)
    counts=' · '.join(f'{label}: {int(r["positions"]):,} positions' for label,r in zip(labels,rows))
    fig.text(.5,.067,counts,ha='center',fontsize=8.5)
    fig.text(.5,.017,'Same validation population for both seeds. Post-hoc description; opening-family concentration limits generalization.',ha='center',fontsize=9,color='#555555')
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160);plt.close(fig)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['seed','move_range','positions','cnn_kl','transformer_kl','weighted_gap','fraction_of_paired_gap'])
        for c in report['cases']:
            for label,r in zip(labels,c['phases']['rows']):
                writer.writerow([c['seed'],label,r['positions'],r['cnn_kl'],r['transformer_kl'],r['contribution_to_overall_gap'],r['fraction_of_overall_gap']])
    print(json.dumps(dict(output=str(args.output),source_sha256=args.diagnostic_sha256)))


if __name__=='__main__':main()
