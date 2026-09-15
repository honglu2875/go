"""Exercise curve comparison against real audited evidence and corruptions."""
import copy
import json
from pathlib import Path
import time

import lr_compare as compare


def main():
    output=Path(__file__).resolve().parent/'lr-comparison-cpu-001.json'
    if output.exists():raise FileExistsError(output)
    started=time.monotonic()
    source=compare.ROOT/'research/studies/strong9_scaling/transformer-seed1-audit-001.json'
    identity='060c15a033eeec9ffa2169ffc23304fb304d17faef107c7a1cfe551e5f3ac28f'
    arm=compare.audited(compare.ROOT,source,identity)
    curves=arm['audit']['validation_curve']
    paired=compare.pair_curves(curves,curves)
    if any(v['relative_endpoint_gain']!=0 or v['relative_last_three_gain']!=0 for v in compare.gains(paired).values()):
        raise ValueError('Identity comparison differs')
    altered=copy.deepcopy(curves)
    for row in altered:
        for metric in ('expert_kl','family_kl'):row['metrics'][metric]*=.98
    gain=compare.gains(compare.pair_curves(curves,altered))
    if any(abs(v['relative_endpoint_gain']-.02)>1e-12 or abs(v['relative_last_three_gain']-.02)>1e-12 for v in gain.values()):
        raise ValueError('Known two-percent gain differs')
    rejected=[]
    def reject(name,change,*,ratio=False):
        candidate=copy.deepcopy(curves);change(candidate)
        try:
            rows=compare.pair_curves(curves,candidate)
            if ratio:compare.gains(rows)
        except ValueError:rejected.append(name)
        else:raise ValueError('Accepted invalid evidence: '+name)
    reject('missing evaluation',lambda x:x.pop(4))
    reject('duplicate turn',lambda x:x[4].update(turn=x[3]['turn']))
    reject('different games',lambda x:x[4].update(episode_ids_sha256='0'*64))
    reject('different split',lambda x:x[4].update(split=0))
    reject('metric missing',lambda x:x[4]['metrics'].pop('family_kl'))
    reject('position denominator',lambda x:x[4]['metrics'].update(expert_count=x[4]['metrics']['expert_count']+1000))
    reject('family denominator',lambda x:x[4]['metrics'].update(family_count=x[4]['metrics']['family_count']+1))
    reject('target entropy',lambda x:x[4]['metrics'].update(expert_target_entropy=x[4]['metrics']['expert_target_entropy']+.1))
    reject('nonfinite metric',lambda x:x[4]['metrics'].update(expert_kl=float('nan')))
    reject('negative endpoint',lambda x:x[-1]['metrics'].update(expert_kl=-1),ratio=True)
    report=dict(kind='strong9_lr_comparison_cpu_qualification',status='passed',created=time.time(),
        seconds=time.monotonic()-started,parent_audit_sha256=identity,
        source_sha256={p.name:compare.sha(p) for p in (Path(compare.__file__),Path(compare.lr_source.__file__),Path(__file__))},
        actual_audited_rank_draws=sum(map(len,arm['draws'].values())),actual_validation_points=len(curves),
        identity_gains_exact=True,synthetic_two_percent_gains_within_1e_12=True,
        rejected_evidence=rejected,
        scope='CPU comparison arithmetic and evidence validation; the changed curves are synthetic fixtures, not an LR experiment, chosen rate, registration or launch.')
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    output.chmod(0o444)
    print(json.dumps(dict(status='passed',rejected_evidence=len(rejected),sha256=compare.sha(output))))


if __name__=='__main__':main()
