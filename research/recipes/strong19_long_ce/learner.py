"""Functional joint updates for memory/optimizer qualification.

All scientific coefficients are explicit. The AdamW configuration used in
qualification is a fixture, not the registered 19x19 optimizer choice.
"""
import math
import jax
import jax.numpy as jnp

import adamw
import cnn_chunks
import encoder
import compact
import heads
import joint
import policy_model


def validate_optimizer(c):
    required=set('learning_rate end_learning_rate warmup_steps horizon_steps beta1 beta2 epsilon weight_decay max_grad_norm'.split())
    if set(c)!=required:raise ValueError('Optimizer fields differ')
    for key in required-{'warmup_steps','horizon_steps'}:
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]):raise ValueError('Invalid '+key)
    if not 0<c['end_learning_rate']<=c['learning_rate'] or not 0<=c['beta1']<1 or not 0<=c['beta2']<1:
        raise ValueError('Invalid learning rate or momentum')
    if c['epsilon']<=0 or c['weight_decay']<0 or c['max_grad_norm']<=0:raise ValueError('Invalid optimizer scale')
    if (type(c['warmup_steps']) is not int or type(c['horizon_steps']) is not int
            or not 0<=c['warmup_steps']<c['horizon_steps']):raise ValueError('Invalid schedule')


def schedule(step,c):
    validate_optimizer(c)
    progress=jnp.clip((step-c['warmup_steps'])/max(1,c['horizon_steps']-c['warmup_steps']),0.,1.)
    rate=c['end_learning_rate']+.5*(c['learning_rate']-c['end_learning_rate'])*(1+jnp.cos(jnp.pi*progress))
    return rate*jnp.minimum(1.,step/max(1,c['warmup_steps']))


def totals(out,b,c,*,value_weight,axis_name=None):
    if type(value_weight) not in (int,float) or not math.isfinite(value_weight) or value_weight<0:
        raise ValueError('Explicit nonnegative value weight required')
    policy=policy_model.averages(policy_model.total_metrics(out['policy'],b,axis_name))
    auxiliary=policy_model.averages(policy_model.total_metrics(out['aux_policy'],b,axis_name))
    value=heads.value_averages(heads.value_totals(out['value'],b,axis_name=axis_name))
    aux_value=heads.value_averages(heads.value_totals(out['aux_value'],b,axis_name=axis_name))
    weight=.8 if c['architecture']=='katago_nested_policy' else c['first_pass_aux_weight']
    policy_loss=(1-weight)*policy['expert_ce']+weight*auxiliary['expert_ce']
    main_ce=heads.signed_target_cross_entropy(out['value_logits'],b,axis_name=axis_name)
    aux_ce=heads.signed_target_cross_entropy(out['aux_value_logits'],b,axis_name=axis_name)
    value_loss=(1-weight)*main_ce['ce']+weight*aux_ce['ce']
    return policy_loss+value_weight*value_loss,dict(policy_loss=policy_loss,value_loss=value_loss,
        main_policy_ce=policy['expert_ce'],aux_policy_ce=auxiliary['expert_ce'],
        main_value_mse=value['value_mse'],aux_value_mse=aux_value['value_mse'],positions=value['value_count'],
        **{'main_value_'+k:v for k,v in main_ce.items()},**{'aux_value_'+k:v for k,v in aux_ce.items()})


def materialized(p,b,c):
    """Wide encoder reference sharing only the qualified temporal/readout math."""
    if c['architecture']=='katago_nested_policy':return joint.forward(p,b,c,training=True)
    ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
    full,first=encoder.spatial(ep,b['spatial'],c,with_first=True)
    ft=encoder.connect(ep,full,b['global_features'],c)
    dt=encoder.connect(ep,first,b['global_features'],c)
    out=compact.packed_forward(p,ft,dt,b['actions'],b['counts'],b['spatial'].shape[2],c,
        compact.projected(p,full,c),compact.projected(p,first,c),with_hidden=True)
    live=jnp.arange(b['actions'].shape[1])[None,:]<b['counts'][:,None]
    return joint.temporal_readout(p,out,live)


def loss(p,b,c,*,value_weight,path,chunk_frames,axis_name=None):
    if path=='bounded':
        if c['architecture']=='katago_nested_policy':
            return cnn_chunks.losses(p,b,c,value_weight=value_weight,chunk_frames=chunk_frames,
                                     axis_name=axis_name,inner_rematerialize=False)
        return joint.losses(p,b,c,value_weight=value_weight,chunk_frames=chunk_frames,
                            axis_name=axis_name,inner_rematerialize=False)
    if path!='materialized':raise ValueError('Unknown training path')
    # The CNN's global helper moments must include the complete shard population.
    if c['architecture']=='katago_nested_policy':
        return joint.losses(p,b,c,value_weight=value_weight,axis_name=axis_name)
    return totals(materialized(p,b,c),b,c,value_weight=value_weight,axis_name=axis_name)


def apply(p,s,gradient,loss_value,metrics,c,opt):
    rate=schedule(s['step']+1,opt)
    eligible=(metrics['positions']>0)&(s['step']<opt['horizon_steps'])
    gated_loss=jnp.where(eligible,loss_value,jnp.nan)
    updated,state,extra=adamw.apply_gradient(p,s,gradient,gated_loss,learning_rate=rate,
        architecture=c['architecture'],**{k:opt[k] for k in ('beta1','beta2','epsilon','weight_decay','max_grad_norm')})
    return updated,state,{**metrics,**extra,'loss':loss_value,'learning_rate':rate,'eligible':eligible}


def step(c,opt,*,value_weight,path='bounded',chunk_frames=32,mesh=None):
    """Return a pure update; the caller owns compilation, donation and state."""
    validate_optimizer(opt)
    if path not in ('bounded','materialized'):raise ValueError('Unknown training path')
    if type(chunk_frames) is not int or not 1<=chunk_frames<=512:raise ValueError('Invalid chunk')
    if mesh is None:
        objective=lambda p,b:loss(p,b,c,value_weight=value_weight,path=path,chunk_frames=chunk_frames)
    else:
        from jax.sharding import PartitionSpec as P
        objective=jax.shard_map(
            lambda p,b:loss(p,b,c,value_weight=value_weight,path=path,chunk_frames=chunk_frames,axis_name='data'),
            mesh=mesh,in_specs=(P(),P('data')),out_specs=P(),check_vma=False)
    def update(p,s,b):
        (value,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(p,b)
        return apply(p,s,gradient,value,metrics,c,opt)
    return update
