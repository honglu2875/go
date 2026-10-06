"""CPU fixtures for bounded adaptation and protection of prior checkpoints."""
import ast
import copy
from pathlib import Path
import tempfile
import time
from execute_run import ROOT,STUDY,read,sha,publish,require
from decision import gates,select,next_lr,summarize



def row(lr,kl=1.,value=1.):
    return dict(peak_lr=lr,endpoint=dict(expert_kl=kl,family_kl=kl),tail=dict(expert_kl=kl,family_kl=kl,value_mse=value),
                sustained_overfit=False)


def main():
    checks=[]
    base=dict(control=row(.001),lr06=row(.0006,1.04),lr15=row(.0015,1.02))
    require(select(base)=='control','Control should win')
    d=next_lr(base,0);require(d['action']=='run' and .001<d['peak_lr']<.0015,'Interior refinement differs')
    checks.append('interior_refinement')
    high=copy.deepcopy(base);high['lr15']=row(.0015,.97)
    require(select(high)=='lr15' and next_lr(high,0)['peak_lr']==.0024,'Upper extension differs')
    low=copy.deepcopy(base);low['lr06']=row(.0006,.97)
    require(select(low)=='lr06' and next_lr(low,0)['peak_lr']==.000375,'Lower extension differs')
    checks.extend(['upper_boundary_extension','lower_boundary_extension'])
    bad=copy.deepcopy(high);bad['lr15']['tail']['value_mse']=1.06
    require(select(bad)=='control','Value guard failed')
    bad=copy.deepcopy(high);bad['lr15']['sustained_overfit']=True
    require(select(bad)=='control','Overfit guard failed')
    bad=copy.deepcopy(high);bad['lr15']['tail']['expert_kl']=1.001
    require(select(bad)=='control','Tail policy guard failed')
    checks.extend(['value_guard','overfit_guard','tail_guard'])
    require(next_lr(high,1,previous_new='lr06')['action']=='stop','Nonwinning first follow-up must stop')
    require(next_lr(high,2,previous_new='lr15')['action']=='stop','Two-probe bound failed')
    maximum=copy.deepcopy(high);maximum['edge']=row(.004,.9)
    require(next_lr(maximum,0)['action']=='stop','Range bound failed')
    checks.extend(['stop_after_nonwinning_probe','two_probe_limit','range_limit'])
    for name in ('campaign.py','execute_run.py','replicate.py','promote.py','ram_copy.py'):
        text=(STUDY/name).read_text()
        require('retention' not in text and '.unlink(' not in text and 'rmtree' not in text,
                'Non-destructive workflow contains a retirement/deletion path')
    checks.append('checkpoint_deletion_paths_absent')
    reference=ROOT/'research/studies/strong19_recovery/flat-stage-003/audit.json'
    original=read(reference);new=read(STUDY/'auditor-reference-qualification-001.json')
    for key in ('status','parameters','positions','padded_position_slots','checkpoint','validation_history','training_probe_history',
                'overfit_observations','initial_parameters_sha256','all_rank_metrics_and_saved_state_verified'):
        require(original[key]==new[key],'Extended auditor changed reference scientific evidence: '+key)
    require(new['operator_sha256']==sha(STUDY/'audit_run.py'),'Auditor changed after qualification')
    checks.append('extended_auditor_reproduces_full_reference_audit')
    inputs={}
    for path in STUDY.glob('*.py'):
        ast.parse(path.read_text());inputs[str(path.relative_to(ROOT))]=sha(path)
    result=dict(status='passed',created=time.time(),checks=checks,operators=inputs,
                reference_audit_sha256=sha(reference),extended_audit_sha256=sha(STUDY/'auditor-reference-qualification-001.json'),
                scope='Decision/protection fixtures and full reference-checkpoint audit. No synthetic learning metrics retained as experimental results.')
    publish(STUDY/'campaign-cpu-qualification-002.json',result)
    print(__import__('json').dumps(dict(status='passed',checks=checks)))


if __name__=='__main__':main()
