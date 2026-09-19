"""Normalize collected scalar metrics on the host, without device dispatch."""
import math


def averages(raw):
    raw={k:float(v) for k,v in raw.items()}
    if not all(math.isfinite(v) for v in raw.values()):raise ValueError('Nonfinite metric totals')
    if raw['expert_count']!=raw['value_count']:raise ValueError('Policy/value counts differ')
    result={}
    for key,count in raw.items():
        if not key.endswith('_count'):continue
        if count<0:raise ValueError('Negative metric population')
        prefix=key[:-6];denominator=count if count>0 else 1.
        result[key]=count
        if prefix=='value' or prefix.startswith('value_'):
            target_mean=raw[prefix+'_target_sum']/denominator
            result.update({prefix+'_mse':raw[prefix+'_squared_error']/denominator,
                prefix+'_mae':raw[prefix+'_absolute_error']/denominator,
                prefix+'_mean_target':target_mean,prefix+'_mean_prediction':raw[prefix+'_prediction_sum']/denominator,
                prefix+'_zero_predictor_mse':raw[prefix+'_target_squared']/denominator,
                prefix+'_fitted_constant_mse':max(raw[prefix+'_target_squared']/denominator-target_mean**2,0.)})
        else:
            ce=raw[prefix+'_ce']/denominator;entropy=raw[prefix+'_target_entropy']/denominator
            result.update({prefix+'_ce':ce,prefix+'_target_entropy':entropy,prefix+'_kl':ce-entropy})
            for suffix in ('entropy','top1'):
                if prefix+'_'+suffix in raw:result[prefix+'_'+suffix]=raw[prefix+'_'+suffix]/denominator
    for name,value in raw.items():
        if name.startswith('logit_') and name.endswith('_sum'):
            result[name[:-4]]=value/max(raw['value_count'],1.)
    return result
