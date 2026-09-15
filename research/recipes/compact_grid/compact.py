"""Equivalent bilinear readout with bounded encoder chunks and projected grids.

The reference model's parameter tree, cache/inference code and auxiliary loss
are unchanged. This module supplies a separately qualified training path.
"""
import math
import jax
import jax.numpy as jnp
import causal
import draft_model
import encoder
import policy_model


def validate(c,chunk_frames,inner_rematerialize=None):
    causal.validate(c)
    if (type(chunk_frames) is not int or not 1<=chunk_frames<=512
            or not c['policy_spatial_bias'] or not c.get('policy_context_dim',0)
            or c.get('policy_readout_kind','bilinear')!='bilinear'
            or c.get('policy_refinement_dim',0) or c['encoder_passes']<2
            or not c.get('first_pass_aux_weight',0)):
        raise ValueError('Compact training requires the selected shared-pass bilinear readout')
    if inner_rematerialize is not None and type(inner_rematerialize) is not bool:
        raise ValueError('Inner rematerialization override must be boolean')


def projected(p,features,c):
    fp={**c,'dtype':'float32'}
    return dict(key=causal.linear(features,p['head.context.k.weight'],fp),
                local=causal.linear(features,p['head.local.weight'],fp)[...,0]+p['head.local.bias'])


def policy(p,x,size,c,features):
    h=causal.norm(x,p['head.norm.scale'],c);fp={**c,'dtype':'float32'}
    logits=causal.linear(h,p['head.actions.weight'][causal.action_indices(size,c)].T,fp)
    if (features['local'].shape!=(*x.shape[:-1],size,size)
            or features['key'].shape!=(*x.shape[:-1],size,size,c['policy_context_dim'])):
        raise ValueError('Projected feature shape differs')
    local=features['local'].reshape(*x.shape[:-1],size*size)
    logits=logits+jnp.concatenate((local,jnp.zeros_like(logits[...,:1])),axis=-1)
    q=causal.linear(h,p['head.context.q.weight'],fp)
    correction=jnp.einsum('...d,...ijd->...ij',q,features['key'])/math.sqrt(c['policy_context_dim'])
    correction=correction.reshape(*x.shape[:-1],size*size)
    return logits+jnp.concatenate((correction,jnp.zeros_like(logits[...,:1])),axis=-1)


def encode(p,spatial,glob,c,*,chunk_frames,inner_rematerialize=None):
    validate(c,chunk_frames,inner_rematerialize)
    ec=c if inner_rematerialize is None else {**c,'encoder_rematerialize':inner_rematerialize}
    b,t,h,w,ch=spatial.shape;frames=b*t;extra=(-frames)%chunk_frames
    if (h,w,ch)!=(c['max_board_size'],c['max_board_size'],22) or glob.shape!=(b,t,19):
        raise ValueError('Complete board input contract differs')
    s=jnp.pad(spatial.reshape(frames,h,w,ch),((0,extra),(0,0),(0,0),(0,0))).reshape(-1,chunk_frames,h,w,ch)
    g=jnp.pad(glob.reshape(frames,19),((0,extra),(0,0))).reshape(-1,chunk_frames,19)
    ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
    def chunk(inputs):
        board,globals=inputs
        # The artificial time axis has length one; each encoder frame is
        # independent, including spatial attention and all normalizations.
        full,first=encoder.spatial(ep,board[:,None],ec,with_first=True)
        ft=encoder.connect(ep,full,globals[:,None],c)
        dt=encoder.connect(ep,first,globals[:,None],c)
        result=(ft,dt,projected(p,full,c),projected(p,first,c))
        return jax.tree.map(lambda x:x[:,0],result)
    # Rematerialize the whole chunk: differentiated scan must not retain every
    # frame's internal wide encoder activations across the full history.
    outputs=jax.lax.map(jax.checkpoint(chunk), (s,g))
    def restore(x):
        flat=x.reshape(-1,*x.shape[2:])[:frames]
        return flat.reshape(b,t,*flat.shape[1:])
    return jax.tree.map(restore,outputs)


def forward(p,spatial,glob,actions,counts,c,*,chunk_frames,inner_rematerialize=None):
    full_tokens,draft_tokens,full_features,draft_features=encode(p,spatial,glob,c,chunk_frames=chunk_frames,inner_rematerialize=inner_rematerialize)
    return packed_forward(p,full_tokens,draft_tokens,actions,counts,spatial.shape[2],c,full_features,draft_features)


def packed_forward(p,full_tokens,draft_tokens,actions,counts,size,c,full_features,draft_features):
    b,t,d=full_tokens.shape
    if draft_tokens.shape!=(b,t,d) or actions.shape!=(b,t) or counts.shape!=(b,) or t>c['max_positions']:
        raise ValueError('Invalid complete causal history')
    action=causal.action_tokens(p,actions,size,c)
    x=jnp.stack((draft_tokens,full_tokens,action),axis=2).reshape(b,3*t,d)
    frame=jnp.arange(3*t);positions=jnp.broadcast_to(2*(frame//3)+(frame%3==2),x.shape[:2])
    def layer(x,block):
        q,k,v=causal.project(x,block,positions,c)
        return causal.finish(x,draft_model.attention(q,k,v,c),block,c),None
    x,_=jax.lax.scan(jax.checkpoint(layer) if c['rematerialize'] else layer,x,causal.blocks(p))
    x=x.reshape(b,t,3,d)
    main=policy(p,x[:,:,1],size,c,full_features)
    draft=policy(p,x[:,:,0],size,c,draft_features)
    live=jnp.arange(t)[None,:]<counts[:,None]
    return jnp.where(live[:,:,None],main,0.),jnp.where(live[:,:,None],draft,0.)


def losses(p,b,c,*,chunk_frames,axis_name=None,inner_rematerialize=None):
    main,draft=forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,chunk_frames=chunk_frames,inner_rematerialize=inner_rematerialize)
    m=policy_model.averages(policy_model.total_metrics(main,b,axis_name))
    aux=policy_model.averages(policy_model.total_metrics(draft,b,axis_name))
    weight=c['first_pass_aux_weight']
    return ((1-weight)*m['expert_ce']+weight*aux['expert_ce'],
            {**m,'draft_ce':aux['expert_ce'],'draft_kl':aux['expert_kl'],'draft_top1':aux['expert_top1'],
             'expert_positions':m['expert_count']})


def retained_feature_bytes(*,batch,positions,size,c):
    """Only encoder/readout outputs; excludes optimizer, backward and temporaries."""
    frames=batch*positions;area=size*size
    encoder_element_bytes=2 if c['dtype']=='bfloat16' else 4
    wide=2*frames*area*c['encoder_width']*encoder_element_bytes
    projected_grids=2*frames*area*(c['policy_context_dim']+1)*4
    tokens=2*frames*c['width']*4
    return dict(full_and_first_grid_bytes=wide,projected_grid_bytes=projected_grids,
                tokens_bytes=tokens,reference_outputs_bytes=wide+tokens,
                compact_outputs_bytes=projected_grids+tokens,
                scope='Forward retained outputs only; compiled peak HBM still needs qualification')
