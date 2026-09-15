"""Functional joint updates for memory/optimizer qualification.

All scientific coefficients are explicit. The fixed Muon/AuxAdam configuration
is an execution fixture, not the registered 19x19 training configuration.
"""
import math
import jax
import jax.numpy as jnp

import optimizer
import cnn_chunks
import encoder
import compact
import heads
import joint
import policy_model


def validate_optimizer(c):
    optimizer.validate_config(c)


def totals(out,b,c,*,value_weight,axis_name=None):
    if type(value_weight) not in (int,float) or not math.isfinite(value_weight) or value_weight<0:
        raise ValueError('Explicit nonnegative value weight required')
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
    if c['architecture']!='katago_nested_policy':
        raise ValueError('This source-grouped Muon bridge is for the CNN')
    updated,state,extra=optimizer.apply_gradient(p,s,gradient,loss_value,metrics['positions'],opt)
    return updated,state,{**metrics,**extra}


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
