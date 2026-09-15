"""Freeze a selected encoder's full qualification and bounded learning cells."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import re
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import freeze,canonical_json,read_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish
from policy_config import validate

NUMERICAL_FILES=('causal.py','observation_attention.py','policy_config.py','policy_model.py','policy_optimizer.py',
                 'compute_budget.py','profile_causal.py','train_policy.py','audit_learning.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--rates',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.workspace_root.resolve();recipe=Path(__file__).resolve().parent
    if (root!=SOURCE or not re.fullmatch('[a-z][a-z0-9_]*',a.label) or not a.rates
            or len(set(a.rates))!=len(a.rates)):raise ValueError('Invalid working recipe preparation')
    base=read_json(a.candidate);validate(base)
    if base['steps']!=1024 or base['seed']!=91312427:raise ValueError('Expected the fixed 1024-update study')
    base.pop('checkpoint_archive',None);base['checkpoint_temporary']=True
    configurations=[]
    q=read_json(a.candidate);q.pop('checkpoint_archive',None);q['checkpoint_temporary']=True
    q.update(steps=4,checkpoint_every=4,eval_every=4,log_every=1)
    q['evaluation']={**q['evaluation'],'games_per_bucket':4,'run_test':False}
    q['learner']={**q['learner'],'warmup_steps':2,'learning_rate':a.rates[0],'end_learning_rate':float(Decimal(str(a.rates[0]))*Decimal('0.3'))}
    configurations.append(('qualification',recipe/(a.label+'_full_qualification.json'),validate(q)))
    for rate in a.rates:
        c={**base,'learner':{**base['learner'],'learning_rate':rate,'end_learning_rate':float(Decimal(str(rate))*Decimal('0.3'))}}
        label=a.label+'_lr_'+format(rate,'.0e').replace('-','m').replace('+','p')
        configurations.append((label,recipe/(label+'.json'),validate(c)))
    if any(path.exists() for _,path,_ in configurations):raise FileExistsError('Selected variant was already prepared')
    # All recipe configurations are present before freezing any cell, so the
    # source payload is identical and only resolved_config differs.
    for _,path,c in configurations:
        with path.open('xb') as f:f.write(canonical_json(c))
    sources=[]
    for label,path,c in configurations:
        snapshot=freeze(root,recipe,path,root/'.gozero/snapshots')
        sources.append({'label':label,'snapshot':snapshot.name,'config_sha256':sha256(snapshot/'resolved_config.json'),
            'configuration':str(path.relative_to(root)),'peak_learning_rate':c['learner']['learning_rate'],
            'model':c['model'],'numerical_sources':{n:sha256(snapshot/recipe.relative_to(root)/n) for n in NUMERICAL_FILES}})
    if any(x['numerical_sources']!=sources[0]['numerical_sources'] for x in sources):raise ValueError('Numerical sources differ across qualification and learning')
    publish(a.output,{'kind':'frozen_encoder_variant_sources','status':'prepared','created_at':datetime.now(timezone.utc).isoformat(),
        'label':a.label,'base_config_sha256':sha256(a.candidate),'operator_snapshot':sources[1]['snapshot'],
        'qualification':sources[0],'candidates':sources[1:],'scope':'Frozen sources only. Full TPU qualification, a review decision, persistent best-checkpoint allocation and learning registration are required before execution.'})
    print(a.output,sha256(a.output),flush=True)


if __name__=='__main__':main()
