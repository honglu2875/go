"""Reconstruct evaluation populations and target entropies with NumPy float64.

This does not reuse the trainer's weighting or metric reduction implementation.
It reads only the explicitly selected train/validation games.
"""
from collections import Counter
import hashlib
import json
import math
import numpy as np


PHASES=((0,16),(16,64),(64,128),(128,256),(256,2048))


def population(data, entries):
    family_positions=Counter()
    for _,shard,episode in entries:
        game=data.shards[shard]['games'][episode]
        if int(game['split']) not in (0,1):
            raise ValueError('Test targets must remain closed')
        family_positions[bytes(game['opening_family'])]+=int(game['length'])
    prefixes=['expert','family',*[f'opponent_{i}' for i in range(8)],
              *[f'phase_{lo}_{hi}' for lo,hi in PHASES]]
    totals={key:0. for prefix in prefixes for key in (prefix+'_count',prefix+'_target_entropy')}
    for role,shard,episode in entries:
        if role!='expert':raise ValueError('Unexpected target population')
        arrays=data.shards[shard];game=arrays['games'][episode]
        lo,hi=map(int,arrays['expert_offsets'][episode:episode+2]);n=hi-lo
        policy=np.asarray(arrays['policies'][lo:hi],dtype=np.float64)
        entropy=-np.sum(policy*np.log(np.maximum(policy,1e-30)),axis=-1)
        sums=float(entropy.sum());opponent=f"opponent_{int(game['opponent'])}"
        for prefix in ('expert',opponent):
            totals[prefix+'_count']+=n;totals[prefix+'_target_entropy']+=sums
        weight=1./family_positions[bytes(game['opening_family'])]
        totals['family_count']+=n*weight;totals['family_target_entropy']+=sums*weight
        for start,end in PHASES:
            prefix=f'phase_{start}_{end}'
            totals[prefix+'_count']+=max(0,min(n,end)-start)
            totals[prefix+'_target_entropy']+=float(entropy[start:end].sum())
    ids=(json.dumps(entries,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return dict(episode_ids_sha256=hashlib.sha256(ids).hexdigest(),games=len(entries),
                opening_families=len(family_positions),totals=totals)


def check(row, expected, *, split):
    if row['split']!=split or row['episode_ids_sha256']!=expected['episode_ids_sha256']:
        raise ValueError('Evaluation game identities or split differ')
    raw=row['raw_totals'];metrics=row['metrics']
    if any(not math.isfinite(float(x)) for x in [*raw.values(),*metrics.values()]):
        raise ValueError('Nonfinite evaluation metric')
    for key,target in expected['totals'].items():
        # Per-device float32 reductions may round before host accumulation.
        tolerance=2e-5*max(1.,abs(target))
        if key.endswith('_count') and not key.startswith('family_'):tolerance=0.
        if abs(raw[key]-target)>tolerance:
            raise ValueError('Evaluation population/target reduction differs: '+key)
    for key in expected['totals']:
        if not key.endswith('_count'):continue
        prefix=key[:-6];n=max(raw[key],1.)
        if not math.isclose(metrics[key],raw[key],rel_tol=2e-6,abs_tol=1e-6):
            raise ValueError('Metric denominator differs: '+key)
        for suffix in ('ce','target_entropy'):
            name=prefix+'_'+suffix
            if not math.isclose(metrics[name],raw[name]/n,rel_tol=2e-6,abs_tol=1e-6):
                raise ValueError('Metric normalization differs: '+name)
        if abs(metrics[prefix+'_kl']-(raw[prefix+'_ce']-raw[prefix+'_target_entropy'])/n)>2e-6:
            raise ValueError('KL differs from cross-entropy minus target entropy')


def audit(data, config, reports):
    buckets=config['dataset']['buckets'];selected=data.bucket_entries(buckets,split=1)
    entries=[entry for bucket in buckets for entry in selected['expert',bucket][:config['evaluation']['games_per_bucket']]]
    validation=population(data,entries)
    turns=[0]+[i for i in range(1,config['steps']+1) if i%config['eval_every']==0 or i==config['steps']]
    curves=[[r['initial_validation'],*r['validation_history']] for r in reports]
    for curve in curves:
        if curve!=curves[0] or [r['turn'] for r in curve]!=turns:
            raise ValueError('Validation schedule or replicated observations differ')
        for row in curve:check(row,validation,split=1)
    probe=None
    if config['evaluation'].get('training_probe_games'):
        import training_probe
        selected=data.bucket_entries(buckets,split=0)
        chosen=training_probe.select({b:selected['expert',b] for b in buckets},config['evaluation']['training_probe_games'])
        probe=population(data,[entry for b in buckets for entry in chosen[b]])
        for report in reports:
            curve=report['training_probe_history']
            if [r['turn'] for r in curve]!=turns:raise ValueError('Probe schedule differs')
            for row in curve:check(row,probe,split=0)
    if config['model'].get('first_pass_aux_weight',0):
        for report in reports:check(report['draft_validation'],validation,split=1)
    return dict(status='passed',validation=validation,training_probe=probe,
                target_semantics='Same fixed raw teacher policy on both colors; NumPy float64 target entropy audit')
