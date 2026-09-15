"""Joint-loss bridge to the reference-qualified standard Muon/AuxAdam kernel.

Rates, decays and the summed-gradient clipping cap are explicit fixtures here.
The source schedule, running norms and lookahead need their own integration.
"""
import math

import jax
import jax.numpy as jnp

import katago
import muon
import muon_groups


def validate_config(c):
    if set(c) != {'rates','decays','max_grad_norm','gradient_units','horizon_steps'}:
        raise ValueError('Explicit Muon groups, clipping units and horizon required')
    if c['gradient_units'] != 'global_position_sum':
        raise ValueError('Muon bridge requires explicit global summed-gradient units')
    if type(c['horizon_steps']) is not int or not 1 <= c['horizon_steps'] <= 1000000:
        raise ValueError('Invalid optimizer horizon')
    groups={'input','normal','normal_gamma','noreg','output','output_noreg'}
    for field in ('rates','decays'):
        if set(c[field]) != groups or any(type(x) not in (float,int) or not math.isfinite(x) or x < 0 for x in c[field].values()):
            raise ValueError('Every CNN rate/decay group must be explicit and finite')
    if type(c['max_grad_norm']) not in (float,int) or not math.isfinite(c['max_grad_norm']) or c['max_grad_norm'] <= 0:
        raise ValueError('Positive finite clipping cap required')


def initialize(params):
    return muon.initialize(params,muon_groups.cnn(params))


def apply_gradient(params,state,mean_gradients,loss,positions,config):
    validate_config(config)
    specs=muon_groups.cnn(params)
    raw_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in mean_gradients.values()))
    # The qualified joint objective averages over globally live positions.
    # KataGo multiplies its local summed loss by world size before DDP averages.
    summed=jax.tree.map(lambda g:g*positions,mean_gradients)
    gradient=katago.repvgg_gradient(summed)
    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in gradient.values()))
    scale=jnp.minimum(1.,config['max_grad_norm']/(norm+1e-6))
    gradient=jax.tree.map(lambda g:g*scale,gradient)
    eligible=jnp.isfinite(positions)&(positions>0)&(state['step']<config['horizon_steps'])&jnp.isfinite(norm)
    new_params,new_state,metrics=muon.apply(params,state,gradient,jnp.where(eligible,loss,jnp.nan),
                                          specs,config['rates'],config['decays'])
    return new_params,new_state,dict(**metrics,loss=loss,eligible=eligible,
        raw_mean_gradient_norm=raw_norm,source_sum_gradient_norm=norm,clip_scale=scale,
        normal_learning_rate=jnp.float32(config['rates']['normal']),positions=positions)
