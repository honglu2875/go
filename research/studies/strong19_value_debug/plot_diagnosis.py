"""Render the fixed-feature value repair without turning it into a strength claim."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    a=argparse.ArgumentParser();a.add_argument('--analysis',type=Path,required=True)
    a.add_argument('--repair',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    analysis=json.loads(args.analysis.read_text());repair=json.loads(args.repair.read_text())
    if analysis['status']!='passed' or repair['status']!='passed':raise ValueError('Expected passed diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),dpi=160)
    categories=['Teacher value > 0.5','Teacher value < −0.5']
    rows=[analysis['metrics']['evaluation/main/'+x]['probability_mean'] for x in ('positive','negative')]
    bottom=[0.,0.]
    for i,(name,color) in enumerate([('Win','#1b9e77'),('Loss','#d95f02'),('Third outcome','#7570b3')]):
        v=[row[i] for row in rows];axes[0].bar(categories,v,bottom=bottom,label=name,color=color)
        bottom=[b+x for b,x in zip(bottom,v)]
    axes[0].set(title='The trained head suppresses losses',ylabel='Mean predicted probability',ylim=(0,1.03))
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=8)
    for mode,color,label in [('mse','#7570b3','MSE'),('ce','#1b9e77','Cross-entropy')]:
        rows=repair['histories'][mode]
        axes[1].plot([r['turn'] for r in rows],[r['validation']['value_mse'] for r in rows],marker='o',label=label,color=color)
        axes[1].plot([r['turn'] for r in rows],[r['train']['value_mse'] for r in rows],ls='--',alpha=.55,color=color)
    axes[1].set_xscale('symlog',linthresh=1)
    axes[1].set(title='Head-only repair: CE escapes sooner',xlabel='Head updates (fixed backbone)',ylabel='Signed-value MSE')
    axes[1].legend(frameon=False);axes[1].grid(alpha=.15)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Value-head diagnosis · 16 complete games / 6,643 positions',fontsize=13)
    fig.text(.5,.02,'Solid: 8 validation games · Dashed: 8 training games · Diagnostic subset; no playing-strength claim',ha='center',fontsize=8,color='#444444')
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(args.output);plt.close(fig)


if __name__=='__main__':main()
