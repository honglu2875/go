"""Shared policy/value backbone prototype; learning settings are not registered.

CNN auxiliary value parameters accompany its existing training-only helper.
The temporal full/first readouts share both policy and value parameters. Neither
readout reruns the encoder. Value outputs follow player-to-move convention.
"""
import math
import jax
import jax.numpy as jnp

import causal
import compact
import heads
import katago
import policy_model


def validate_value_config(v):
    if set(v)!={'hidden','spatial_channels'} or any(type(x) is not int or not 1<=x<=1024 for x in v.values()):
        raise ValueError('Invalid value head configuration')


def initialize(seed,c,v):
    validate_value_config(v);p=policy_model.initialize(seed,c)
    kind='cnn' if c['architecture']=='katago_nested_policy' else 'temporal'
    def add(prefix,head_seed):
        h=heads.initialize(head_seed,kind=kind,width=c['width'],hidden=v['hidden'],spatial_channels=v['spatial_channels'])
        p.update({prefix+'.'+name:value for name,value in h.items()})
    # Keep every existing backbone/policy draw byte-identical.
    add('value_head',seed+234741)
    if kind=='cnn':add('intermediate_value_head',seed+234742)
    return p


def value_params(p,*,auxiliary=False):
    return katago.select(p,'intermediate_value_head' if auxiliary else 'value_head')


def parameter_schema(c,v):
    shape=jax.eval_shape(lambda:initialize(0,c,v))
    return [dict(path=k,shape=list(a.shape),dtype=str(a.dtype),elements=a.size,
                 inference=not k.startswith(('intermediate_','norm_intermediate_')))
            for k,a in sorted(shape.items())]


def temporal_readout(p,out,live):
    logits=heads.temporal(value_params(p),out['latent'])
    result=dict(policy=out['policy'],value_logits=jnp.where(live[...,None],logits,0.),
                value=jnp.where(live,heads.signed_value(logits),0.))
    if 'aux_policy' in out:
        auxiliary=heads.temporal(value_params(p),out['aux_latent'])
        result.update(aux_policy=out['aux_policy'],aux_value_logits=jnp.where(live[...,None],auxiliary,0.),
                      aux_value=jnp.where(live,heads.signed_value(auxiliary),0.))
    return result


def forward(p,b,c,*,training=False,axis_name=None,chunk_frames=32,inner_rematerialize=False):
    live=jnp.arange(b['actions'].shape[1])[None,:]<b['counts'][:,None]
    if c['architecture']=='causal_visual_policy':
        if training:
            out=compact.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,
                chunk_frames=chunk_frames,inner_rematerialize=inner_rematerialize,with_hidden=True)
        else:
            out=causal.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,with_hidden=True)
        return temporal_readout(p,out,live)
    if c['architecture']!='katago_nested_policy':raise ValueError('Unqualified architecture')
    n,t,h,w,ch=b['spatial'].shape
    spatial=b['spatial'].reshape(n*t,h,w,ch)
    out=katago.forward(p,spatial,b['global_features'].reshape(n*t,-1),c,
                       training=training,axis_name=axis_name,with_features=True)
    result={}
    for prefix,features in (('', 'features'),('aux_', 'aux_features')):
        if features not in out:continue
        v=heads.cnn(value_params(p,auxiliary=bool(prefix)),out[features],spatial[...,:1]).reshape(n,t,3)
        result.update({prefix+'policy':out[prefix+'policy'].reshape(n,t,-1),
                       prefix+'value_logits':jnp.where(live[...,None],v,0.),
                       prefix+'value':jnp.where(live,heads.signed_value(v),0.)})
    return result


def losses(p,b,c,*,value_weight,axis_name=None,chunk_frames=32,inner_rematerialize=False):
    # Explicit coefficient required: this preparation does not silently choose
    # the joint study's policy/value tradeoff.
    if type(value_weight) not in (int,float) or not math.isfinite(value_weight) or value_weight<0:
        raise ValueError('An explicit finite nonnegative value coefficient is required')
    out=forward(p,b,c,training=True,axis_name=axis_name,chunk_frames=chunk_frames,
                inner_rematerialize=inner_rematerialize)
    policy=policy_model.averages(policy_model.total_metrics(out['policy'],b,axis_name))
    auxiliary=policy_model.averages(policy_model.total_metrics(out['aux_policy'],b,axis_name))
    value=heads.value_averages(heads.value_totals(out['value'],b,axis_name=axis_name))
    aux_value=heads.value_averages(heads.value_totals(out['aux_value'],b,axis_name=axis_name))
    weight=.8 if c['architecture']=='katago_nested_policy' else c['first_pass_aux_weight']
    policy_loss=(1-weight)*policy['expert_ce']+weight*auxiliary['expert_ce']
    value_loss=(1-weight)*value['value_mse']+weight*aux_value['value_mse']
    return policy_loss+value_weight*value_loss,dict(policy_loss=policy_loss,value_loss=value_loss,
        main_policy_ce=policy['expert_ce'],aux_policy_ce=auxiliary['expert_ce'],
        main_value_mse=value['value_mse'],aux_value_mse=aux_value['value_mse'],positions=value['value_count'])


def first_move(p,spatial,glob,c,*,network_version=0):
    out,cache=causal.first_move(p,spatial,glob,c,network_version=network_version,with_hidden=True)
    return temporal_readout(p,out,jnp.ones(spatial.shape[0],bool)),cache


def prefill(p,b,c,*,network_version=0):
    out,cache=causal.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,
                             with_cache=True,network_version=network_version,with_hidden=True)
    live=jnp.arange(b['actions'].shape[1])[None,:]<b['counts'][:,None]
    return temporal_readout(p,out,live),cache


def append_move(p,cache,previous_action,spatial,glob,c,*,attention_positions,active=None,network_version=0):
    out,updated=causal.append_move(p,cache,previous_action,spatial,glob,c,
        attention_positions=attention_positions,active=active,network_version=network_version,with_hidden=True)
    # A rejected/inactive cache write must not turn the value-head bias into a
    # plausible evaluation. Callers must check cache validity as well.
    live=updated['lengths']>cache['lengths']
    return temporal_readout(p,out,live),updated
