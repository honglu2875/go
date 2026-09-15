"""Paired cluster bootstrap; resample complete games, retaining position weights."""
import numpy as np

def compare(left,right,*,seed=91312731,repetitions=5000):
    if len(left)!=len(right) or len(left)<2:raise ValueError('Need matching complete games')
    counts=np.asarray([x['count'] for x in left],np.float64)
    if not np.array_equal(counts,[x['count'] for x in right]) or (counts<=0).any():raise ValueError('Game counts differ')
    if [x['entry'] for x in left]!=[x['entry'] for x in right]:raise ValueError('Game ordering differs')
    lkl=np.asarray([x['ce']-x['target_entropy'] for x in left]);rkl=np.asarray([x['ce']-x['target_entropy'] for x in right])
    delta=rkl-lkl;top=np.asarray([y['top1']-x['top1'] for x,y in zip(left,right)])
    rng=np.random.Generator(np.random.PCG64(seed));kl_samples=[];top_samples=[]
    # Bound temporary memory while preserving the exact seeded draw sequence.
    for start in range(0,repetitions,100):
        draw=rng.integers(len(left),size=(min(100,repetitions-start),len(left)))
        denominator=counts[draw].sum(-1)
        kl_samples.extend((delta[draw].sum(-1)/denominator).tolist());top_samples.extend((top[draw].sum(-1)/denominator).tolist())
    return {'games':len(left),'positions':int(counts.sum()),'right_minus_left_kl':float(delta.sum()/counts.sum()),
        'right_minus_left_top1':float(top.sum()/counts.sum()),'fraction_games_lower_kl_on_right':float(np.mean(delta<0)),
        'median_per_game_kl_difference':float(np.median(delta/counts)),
        'kl_cluster_bootstrap_percentile_95':np.percentile(kl_samples,[2.5,97.5]).tolist(),
        'top1_cluster_bootstrap_percentile_95':np.percentile(top_samples,[2.5,97.5]).tolist(),
        'bootstrap_seed':seed,'bootstrap_repetitions':repetitions,
        'scope':'Paired descriptive endpoint difference. Whole validation games sampled uniformly with replacement; statistic remains position-weighted. Interval reflects validation-game sampling only, not training seeds, hyperparameter search, teacher population shift or RL strength.'}
