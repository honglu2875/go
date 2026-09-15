"""Deterministic training diagnostics that never consume the minibatch RNG."""
import hashlib
import json
import math


def select(population, count):
    total=sum(len(v) for v in population.values())
    if count < 1 or not total:
        raise ValueError('A training probe needs a nonempty population')
    count=min(count,total)
    quotas={bucket:count*len(entries)/total for bucket,entries in population.items()}
    sizes={bucket:math.floor(value) for bucket,value in quotas.items()}
    remainder=count-sum(sizes.values())
    for bucket in sorted(quotas,key=lambda b:(-(quotas[b]-sizes[b]),b))[:remainder]:
        sizes[bucket]+=1
    def key(entry):
        return hashlib.sha256(json.dumps(['training-probe-v1',entry],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    chosen={bucket:sorted(entries,key=key)[:sizes[bucket]] for bucket,entries in population.items()}
    assert sum(map(len,chosen.values()))==count
    return chosen


def overfit_observation(validation, training, *, patience=3, relative_margin=.005):
    """Diagnostic only: fixed-budget architecture screens never stop adaptively."""
    paired={x['turn']:x for x in training}
    rows=[(v,paired[v['turn']]) for v in validation if v['turn'] in paired]
    if not rows:
        return dict(sustained=False,reason='no paired checkpoints')
    best=min(range(len(rows)),key=lambda i:rows[i][0]['metrics']['expert_kl'])
    v0,t0=(r['metrics']['expert_kl'] for r in rows[best])
    recent=rows[-patience:]
    flagged=len(recent)==patience and all(v['turn']>rows[best][0]['turn']
        and v['metrics']['expert_kl']>v0+relative_margin*max(abs(v0),1e-6)
        and t['metrics']['expert_kl']<t0-relative_margin*max(abs(t0),1e-6) for v,t in recent)
    return dict(sustained=flagged,best_validation_turn=rows[best][0]['turn'],
                patience=patience,relative_margin=relative_margin,action='record only; preserve registered screen horizon')
