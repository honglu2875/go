"""Render actual fixed-probe/validation observations, including partial runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--parent-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',default='Rank 128');p.add_argument('--title',default='Readout rank screen')
    a=p.parse_args()
    parent=json.loads(a.parent_audit.read_text())['validation_curve']
    contents=(a.attempt/'rank-0/artifacts/evaluations.jsonl').read_text()
    lines=contents.splitlines()
    if not contents.endswith('\n'):lines=lines[:-1]
    rows=[json.loads(line) for line in lines]
    validation=[r for r in rows if r['kind']=='visual_heldout']
    training=[r for r in rows if r['kind']=='visual_training_probe']
    series=[('Rank 64 validation',parent,'#626973','--'),
            (a.label+' validation',validation,'#2065b0','-'),
            (a.label+' fixed training probe',training,'#d17b15','-')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for axis in axes:
        for label,curve,color,style in series:
            axis.plot([r['turn'] for r in curve],[r['metrics']['expert_kl'] for r in curve],
                      label=label,color=color,linestyle=style,marker='o',markersize=3,linewidth=1.7)
        axis.set_xlabel('Optimizer updates');axis.set_ylabel('Policy KL (nats per position)')
        axis.grid(alpha=.18);axis.set_xlim(0,1024)
    axes[0].set_title('Full curve');axes[1].set_title('Learning after warmup')
    axes[1].set_ylim(.34,1.04);axes[1].set_xlim(110,1030)
    axes[0].legend(loc='upper right',fontsize=9)
    config=json.loads((a.attempt/'rank-0/artifacts/resolved_config.json').read_text())
    fig.suptitle(a.title+' · same small fixed dataset · seed '+str(config['seed']),fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=160);plt.close(fig)
    with a.output.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['series','update','policy_kl','positions'])
        for label,curve,_,_ in series:
            for r in curve:writer.writerow([label,r['turn'],r['metrics']['expert_kl'],r['metrics']['expert_count']])
    print(json.dumps({'plot':str(a.output),'latest_validation_turn':validation[-1]['turn'],'latest_training_probe_turn':training[-1]['turn']}))


if __name__=='__main__':main()
