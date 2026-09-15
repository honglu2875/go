"""Show natural and equal-family validation with fixed training probes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',nargs=2,action='append',metavar=('LABEL','ATTEMPT'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    series=[];common=None;latest={}
    colors=['#2065b0','#bc581e','#4c885b','#805aaa']
    for i,(label,path) in enumerate(a.arm):
        attempt=Path(path);folder=attempt/'rank-0/artifacts'
        config=json.loads((folder/'resolved_config.json').read_text())
        identity=(config['dataset']['manifest_sha256'],config['seed'],config['steps'])
        if common is None:common=identity
        elif identity!=common:raise ValueError('Plot arms use different data, seed, or horizon')
        if identity[0]!='37244b2e743b3b7f80e0b14eccddfc2942aa4d81120c31aab10c4b3b4f733fc2':
            raise ValueError('Not the registered larger 9x9 corpus')
        text=(folder/'evaluations.jsonl').read_text();lines=text.splitlines()
        if not text.endswith('\n'):lines=lines[:-1]
        rows=[json.loads(line) for line in lines]
        for kind,suffix,style in (('visual_heldout','validation','-'),('visual_training_probe','training probe','--')):
            curve=[r for r in rows if r['kind']==kind]
            if not curve:continue
            series.append((label+' '+suffix,curve,colors[i%len(colors)],style))
            latest[label+' '+suffix]=curve[-1]['turn']
    if not series:raise ValueError('No complete evaluations yet')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
    for col,(metric,title) in enumerate((('expert_kl','Position-weighted KL'),('family_kl','Equal-opening-family KL'))):
        for row in range(2):
            axis=axes[row,col]
            for label,curve,color,style in series:
                chosen=curve if row==0 else [r for r in curve if r['turn']>=256]
                axis.plot([r['turn'] for r in chosen],[r['metrics'][metric] for r in chosen],
                          color=color,linestyle=style,label=label,marker='o',markersize=3,linewidth=1.6)
            axis.set_title(title+(' · full curve' if row==0 else ' · after warmup'))
            axis.set_xlabel('Optimizer updates');axis.set_ylabel('KL (nats)');axis.grid(alpha=.18)
            axis.set_xlim(0 if row==0 else 240,common[2]+16)
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'Fixed strong-teacher 9×9 corpus · paired seed {common[1]}',fontsize=13)
    fig.text(.5,.002,'Training probes and validation have different fixed mixtures; compare their trends, not just their absolute gap.',
             ha='center',fontsize=9,color='#555555')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)
    with a.output.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['series','update','position_policy_kl','family_policy_kl','positions','opening_families'])
        for label,curve,_,_ in series:
            for r in curve:
                m=r['metrics'];writer.writerow([label,r['turn'],m['expert_kl'],m['family_kl'],m['expert_count'],m['family_count']])
    print(json.dumps(dict(output=str(a.output),latest=latest)))


if __name__=='__main__':main()
