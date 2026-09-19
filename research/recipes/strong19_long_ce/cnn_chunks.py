"""Bound CNN training activations while retaining global helper statistics.

The first trunk pass merges centered per-chunk moments. The second emits only
policy/value readouts. Both passes remain differentiable and rematerialize whole
chunks. This adds training computation, with no deployed model change. It is a
separate path requiring numerical and full-size TPU qualification before use.
"""
import math
import jax
import jax.numpy as jnp

import heads
import joint
import katago
import policy_model


def merge_moments(left,right):
    n,a,m2=left;k,b,s2=right
    total=n+k;safe=jnp.where(total>0,total,1.)
    delta=b-a
    return total,a+delta*(k/safe),m2+s2+delta**2*(n*k/safe)


def frame_moments(x,mask):
    n=jnp.sum(mask);safe=jnp.where(n>0,n,1.)
    mean=jnp.sum(x*mask,axis=(0,1,2))/safe
    m2=jnp.sum(((x-mean)*mask)**2,axis=(0,1,2))
    return n,mean,m2


def global_moments(local,axis_name=None):
    n,mean,m2=local
    if axis_name is not None:
        total=jax.lax.psum(n,axis_name);safe=jnp.where(total>0,total,1.)
        global_mean=jax.lax.psum(n*mean,axis_name)/safe
        m2=jax.lax.psum(m2+n*(mean-global_mean)**2,axis_name)
        n,mean=total,global_mean
    return dict(count=n,mean=mean,variance=m2/jnp.where(n>0,n,1.))


def inputs(b,c,chunk_frames):
    katago.validate(c)
    if type(chunk_frames) is not int or not 1<=chunk_frames<=512:raise ValueError('Invalid CNN chunk extent')
    n,t,h,w,ch=b['spatial'].shape
    if (h,w,ch)!=(c['max_board_size'],c['max_board_size'],22) or b['global_features'].shape!=(n,t,19):
        raise ValueError('CNN board input contract differs')
    if b['actions'].shape!=(n,t) or b['counts'].shape!=(n,):raise ValueError('CNN history shapes differ')
    frames=n*t;extra=(-frames)%chunk_frames
    s=jnp.pad(b['spatial'].reshape(frames,h,w,ch),((0,extra),(0,0),(0,0),(0,0))).reshape(-1,chunk_frames,h,w,ch)
    g=jnp.pad(b['global_features'].reshape(frames,19),((0,extra),(0,0))).reshape(-1,chunk_frames,19)
    return s,g


def forward(p,b,c,*,chunk_frames,axis_name=None,inner_rematerialize=False,with_statistics=False):
    if type(inner_rematerialize) is not bool:raise ValueError('Inner rematerialization must be boolean')
    s,g=inputs(b,c,chunk_frames);ec={**c,'rematerialize':inner_rematerialize}
    width=c['width'];zero=(jnp.asarray(0.,jnp.float32),jnp.zeros(width,jnp.float32),jnp.zeros(width,jnp.float32))
    def statistic_chunk(carry,sg):
        spatial,glob=sg;x=katago.trunk(p,spatial,glob,ec)
        return merge_moments(carry,frame_moments(x,spatial[...,:1])),None
    local,_=jax.lax.scan(jax.checkpoint(statistic_chunk),zero,(s,g))
    statistics=global_moments(local,axis_name)
    def readout_chunk(sg):
        spatial,glob=sg;x=katago.trunk(p,spatial,glob,ec);mask=spatial[...,:1]
        main=katago.mish((x/math.sqrt(c['layers']+1.)+p['norm_trunkfinal.beta'])*mask)
        helper=(x-statistics['mean'])*jax.lax.rsqrt(statistics['variance']+c['norm_epsilon'])
        helper=katago.mish(katago.norm(helper,mask,p,'norm_intermediate_trunkfinal'))
        output={}
        for prefix,features in (('',main),('aux_',helper)):
            policy_prefix='intermediate_policy_head' if prefix else 'policy_head'
            logits=heads.cnn(joint.value_params(p,auxiliary=bool(prefix)),features,mask)
            output.update({prefix+'policy':katago.policy(features,mask,katago.select(p,policy_prefix),c),
                           prefix+'value_logits':logits,prefix+'value':heads.signed_value(logits)})
        return output
    rows=jax.lax.map(jax.checkpoint(readout_chunk),(s,g))
    n,t=b['actions'].shape;live=jnp.arange(t)[None,:]<b['counts'][:,None]
    def restore(x):return x.reshape(-1,*x.shape[2:])[:n*t].reshape(n,t,*x.shape[2:])
    out=jax.tree.map(restore,rows)
    for name in ('value','aux_value','value_logits','aux_value_logits'):
        out[name]=jnp.where(live[...,None] if name.endswith('logits') else live,out[name],0.)
    return (out,statistics) if with_statistics else out


def losses(p,b,c,*,value_weight,chunk_frames,axis_name=None,inner_rematerialize=False):
    if type(value_weight) not in (int,float) or not math.isfinite(value_weight) or value_weight<0:
        raise ValueError('An explicit finite nonnegative value coefficient is required')
    out=forward(p,b,c,chunk_frames=chunk_frames,axis_name=axis_name,inner_rematerialize=inner_rematerialize)
    policy=policy_model.averages(policy_model.total_metrics(out['policy'],b,axis_name))
    auxiliary=policy_model.averages(policy_model.total_metrics(out['aux_policy'],b,axis_name))
    value=heads.value_averages(heads.value_totals(out['value'],b,axis_name=axis_name))
    aux_value=heads.value_averages(heads.value_totals(out['aux_value'],b,axis_name=axis_name))
    policy_loss=.2*policy['expert_ce']+.8*auxiliary['expert_ce']
    main_ce=heads.signed_target_cross_entropy(out['value_logits'],b,axis_name=axis_name)
    aux_ce=heads.signed_target_cross_entropy(out['aux_value_logits'],b,axis_name=axis_name)
    value_loss=.2*main_ce['ce']+.8*aux_ce['ce']
    return policy_loss+value_weight*value_loss,dict(policy_loss=policy_loss,value_loss=value_loss,
        main_policy_ce=policy['expert_ce'],aux_policy_ce=auxiliary['expert_ce'],
        main_value_mse=value['value_mse'],aux_value_mse=aux_value['value_mse'],positions=value['value_count'],
        **{'main_value_'+k:v for k,v in main_ce.items()},**{'aux_value_'+k:v for k,v in aux_ce.items()})
