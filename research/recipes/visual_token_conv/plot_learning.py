#!/usr/bin/env python3
"""Standalone scientific figures from closed, audited sequential learning runs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--audit',nargs=3,action='append',metavar=('LABEL','PATH','SHA256'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE);root=args.workspace_root.resolve()
    if len(args.audit)>4 or len({x[0] for x in args.audit})!=len(args.audit):
        raise ValueError('Use one to four uniquely named audited arms')
    records=[];dataset=None;validation_ids=None
    for label,path,expected in args.audit:
        path=Path(path)
        if sha256(path)!=expected: raise ValueError('Audit differs')
        a=read_json(path)
        if a['status']!='passed': raise ValueError('Run did not pass its audit')
        if dataset is None: dataset=a['dataset_manifest_sha256']
        if dataset!=a['dataset_manifest_sha256']: raise ValueError('Comparison data differ')
        for name,digest in a['input_files'].items():
            if sha256(root/name)!=digest: raise ValueError('Audited input changed')
        metrics_name=next(name for name in a['input_files'] if '/rank-0/' in name and name.endswith('/metrics.jsonl'))
        metrics=[json.loads(x) for x in (root/metrics_name).read_text().splitlines()]
        elapsed={0:0.,**{x['turn']:x['cumulative_learning_seconds'] for x in metrics}}
        for item in a['validation_curve']:
            if validation_ids is None:validation_ids=item['episode_ids_sha256']
            if validation_ids!=item['episode_ids_sha256']:raise ValueError('Validation populations differ')
            records.append({'label':label,'turn':item['turn'],'learning_seconds':elapsed[item['turn']],
                            **item['metrics']})
    args.output.mkdir(parents=True,exist_ok=False)
    columns=['label','turn','learning_seconds',*sorted(set().union(*(set(r)-{'label','turn','learning_seconds'} for r in records)))]
    with (args.output/'learning.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'gozero-sequential-learning-v1','figure.dpi':160})
    colors=['#287a68','#4269af','#b46c27','#8d54a0']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for (label,_,_),color in zip(args.audit,colors):
        rows=[r for r in records if r['label']==label]
        y=[r['expert_kl'] for r in rows]
        axes[0].plot([r['turn'] for r in rows],y,'o-',ms=3,label=label,color=color)
        axes[1].plot([r['learning_seconds']/60 for r in rows],y,'o-',ms=3,label=label,color=color)
    axes[0].set_xlabel('Optimizer updates');axes[1].set_xlabel('Measured training time (minutes)')
    for ax in axes:
        ax.set_ylabel('Validation policy KL (nats)');ax.grid(alpha=.18);ax.set_ylim(bottom=0);ax.legend(frameon=False)
    fig.suptitle('Fixed-data policy learnability')
    fig.savefig(args.output/'learning.png',metadata={'Software':'GoZero immutable plotting recipe'})
    fig.savefig(args.output/'learning.svg',metadata={'Date':None,'Creator':'GoZero immutable plotting recipe'})
    plt.close(fig)
    phases=[('0_16','0–15'),('16_64','16–63'),('64_128','64–127'),('128_256','128–255'),('256_2048','256+')]
    endpoints=[max((r for r in records if r['label']==label),key=lambda r:r['turn']) for label,_,_ in args.audit]
    counts=[endpoints[0]['phase_'+key+'_count'] for key,_ in phases]
    if any([row['phase_'+key+'_count'] for key,_ in phases]!=counts for row in endpoints):raise ValueError('Phase populations differ')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');width=.75/len(endpoints);x=np.arange(len(phases))
    for i,(row,color) in enumerate(zip(endpoints,colors)):
        ax.bar(x+(i-(len(endpoints)-1)/2)*width,[row['phase_'+key+'_kl'] for key,_ in phases],
               width,label=row['label'],color=color)
    ax.set_xticks(x,[label+'\n'+f'{int(n):,} positions' for (_,label),n in zip(phases,counts)])
    ax.set_ylabel('Endpoint validation policy KL (nats)');ax.set_xlabel('Move number (zero-based)')
    ax.set_title('Where the remaining policy error occurs');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(args.output/'phases.png',metadata={'Software':'GoZero immutable plotting recipe'})
    fig.savefig(args.output/'phases.svg',metadata={'Date':None,'Creator':'GoZero immutable plotting recipe'});plt.close(fig)
    manifest={'kind':'audited_learning_figures','operator_snapshot':SOURCE.name,'dataset_manifest_sha256':dataset,
        'audits':[{'label':label,'path':path,'sha256':digest} for label,path,digest in args.audit],
        'matplotlib':matplotlib.__version__,'numpy':np.__version__,'validation_episode_ids_sha256':validation_ids,
        'time_scope':'Update dispatch through completed metric transfer; excludes compile, sampling, evaluation, and checkpoint time.',
        'files':{x.name:{'sha256':sha256(x),'bytes':x.stat().st_size} for x in sorted(args.output.iterdir())}}
    (args.output/'manifest.json').write_bytes(canonical_json(manifest))
    for x in args.output.iterdir():x.chmod(0o444)
    print(json.dumps({'status':'passed','manifest_sha256':sha256(args.output/'manifest.json')}),flush=True)
if __name__=='__main__':main()
