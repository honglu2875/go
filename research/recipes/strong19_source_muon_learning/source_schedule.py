"""Scalar standard-Muon settings from pinned KataGo's FSON training branch.

These equations do not select a historical checkpoint's LR scale, batch size,
norm-observation cadence or epoch boundaries. Callers supply those explicitly.
"""
import math

MUON_GROUPS = frozenset(('normal','normal_attn','normal_gab','gab_mlp','tab_module'))
ADAM_GROUPS = frozenset(('input','input_noreg','normal_gamma','noreg','output','output_noreg'))
FACTOR_NAMES = frozenset(('head_lr_factor','noreg_lr_factor','muon_adam_lr_factor',
                          'input_wd_factor','normal_wd_factor','normal_attn_wd_factor','gnorm_clip_scale'))


def settings(*, samples, global_batch, effective_lr_scale, factors, norm_ratios,
             lookahead_alpha, no_lr_warmup):
    if type(samples) is not int or samples < 0 or type(global_batch) is not int or global_batch <= 0:
        raise ValueError('Explicit integer sample progress and global position batch required')
    if type(effective_lr_scale) not in (float,int) or not math.isfinite(effective_lr_scale) or effective_lr_scale <= 0:
        raise ValueError('A finite positive effective LR scale is required')
    if set(factors) != FACTOR_NAMES:
        raise ValueError('Every source LR/decay/clipping factor must be explicit')
    if any(type(x) not in (float,int) or not math.isfinite(x) or x < 0 for x in factors.values()):
        raise ValueError('Factors must be finite nonnegative scalars')
    if lookahead_alpha is not None and (type(lookahead_alpha) not in (float,int)
            or not math.isfinite(lookahead_alpha) or not 0 < lookahead_alpha <= 1):
        raise ValueError('Lookahead alpha must be None or in (0,1]')
    if type(no_lr_warmup) is not bool:
        raise ValueError('Warmup choice must be explicit')
    if norm_ratios is not None and (set(norm_ratios) != {'input','normal'} or any(
            type(x) not in (float,int) or not math.isfinite(x) or x < 0 for x in norm_ratios.values())):
        raise ValueError('Supply both source running-norm/baseline ratios or neither')
    warmup = 1.
    if not no_lr_warmup:
        for threshold, divisor in zip(range(250000,2000001,250000),(20.,14.,10.,7.,5.,3.,2.,1.4)):
            if samples < threshold:
                warmup = 1./divisor
                break
    batch_scale = math.sqrt(global_batch/256.)
    base_rate = 1.33*.00003*effective_lr_scale*warmup
    rates, decays = {}, {}
    for group in sorted(MUON_GROUPS | ADAM_GROUPS):
        group_scale = 2. if group in MUON_GROUPS else 1.
        if group in ('input_noreg','noreg'):
            group_scale = factors['noreg_lr_factor']
        elif group == 'output':
            group_scale = factors['head_lr_factor']
        elif group == 'output_noreg':
            group_scale = factors['head_lr_factor']*factors['noreg_lr_factor']
        if group in ADAM_GROUPS:
            group_scale *= factors['muon_adam_lr_factor']
        rates[group] = base_rate*group_scale/(lookahead_alpha or 1.)*batch_scale
        wd_scale = batch_scale*factors.get(group+'_wd_factor',1.)
        if group in MUON_GROUPS | {'input','normal_gamma'}:
            adaptive = 1.
            if norm_ratios is not None:
                ratio = norm_ratios['input' if group == 'input' else 'normal']
                adaptive = math.pow(2.,2.*math.tanh(math.log(ratio+1e-30)*3.))
            group_factor = {'input':2./3.,'normal':1.,'normal_attn':.5,
                            'normal_gab':.3,'gab_mlp':.1,'tab_module':.1,'normal_gamma':.25}[group]
            coefficient = .02 if group in MUON_GROUPS else .009
            decays[group] = coefficient*wd_scale*math.pow(effective_lr_scale*warmup,.70)*adaptive*group_factor
        elif group == 'output':
            decays[group] = .004*wd_scale
        elif group in ('input_noreg','noreg'):
            decays[group] = .000001*wd_scale*math.pow(effective_lr_scale*warmup,.75)
        else:
            decays[group] = .000001*wd_scale
    cap = 11000.*factors['gnorm_clip_scale']*batch_scale/math.sqrt(max(.0000001,effective_lr_scale))
    return dict(rates=rates,decays=decays,warmup_scale=warmup,per_sample_lr=base_rate,
                source_sum_gradient_clip_cap=cap,mean_to_source_sum_multiplier=global_batch)
