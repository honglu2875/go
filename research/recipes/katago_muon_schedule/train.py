"""Qualify scalar schedule equations; no model, accelerator or learning run."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
import schedule


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(SOURCE);config=read_json(args.config)
    if config['kind']!='katago_muon_schedule_qualification':raise ValueError('Wrong qualification kind')
    reference=args.workspace_root/config['reference']['path']
    if sha(reference)!=config['reference']['sha256']:raise ValueError('Reference changed')
    rows=read_json(reference)['cases'];errors={};checked=0
    for row in rows:
        actual=schedule.settings(**row['inputs'])
        expected=row['expected']
        for key in ('per_sample_lr','source_sum_gradient_clip_cap'):
            error=abs(actual[key]-expected[key])/max(abs(expected[key]),1e-30)
            errors[key]=max(errors.get(key,0.),error)
            if not math.isclose(actual[key],expected[key],rel_tol=2e-14,abs_tol=1e-15):
                raise ValueError('Reference scalar differs: '+key)
            checked+=1
        for role in ('rates','decays'):
            if set(actual[role])!=set(expected[role]):raise ValueError('Parameter-group coverage differs')
            for group,wanted in expected[role].items():
                value=actual[role][group]
                errors[role]=max(errors.get(role,0.),abs(value-wanted)/max(abs(wanted),1e-30))
                if not math.isclose(value,wanted,rel_tol=2e-14,abs_tol=1e-15):
                    raise ValueError('Reference group setting differs: '+role+'.'+group)
                checked+=1
        if actual['mean_to_source_sum_multiplier']!=row['inputs']['global_batch']:
            raise ValueError('Gradient units do not correspond to the global summed loss')
    rejected=[]
    base=rows[0]['inputs']
    for name,change in [('nan_scale',dict(effective_lr_scale=float('nan'))),('negative_samples',dict(samples=-1)),
                        ('empty_batch',dict(global_batch=0)),('missing_factors',dict(factors={})),
                        ('bad_norm_ratio',dict(norm_ratios=dict(input=-1.,normal=1.))),
                        ('zero_alpha',dict(lookahead_alpha=0.)),('ambiguous_warmup',dict(no_lr_warmup=0))]:
        try:schedule.settings(**{**copy.deepcopy(base),**change})
        except ValueError:rejected.append(name)
        else:raise ValueError('Invalid schedule accepted: '+name)
    result=dict(kind='katago_muon_schedule_qualification',status='passed',created=time.time(),snapshot=SOURCE.name,
                reference_sha256=sha(reference),cases=len(rows),scalar_comparisons=checked,
                maximum_relative_error=errors,rejected_invalid_inputs=rejected,
                scope='Scalar equation equivalence only. Does not reproduce neural losses, running-norm observation/update cadence, lookahead state/epoch flushing or published historical hyperparameters.')
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed',cases=len(rows),scalars=checked,sha256=sha(args.output))))


if __name__=='__main__':main()
