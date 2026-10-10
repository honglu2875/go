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


def encode(p,spatial,glob,c,*,chunk_frames,inner_rematerialize=None,counts=None,router_counts=None):
    validate(c,chunk_frames,inner_rematerialize)
    ec=c if inner_rematerialize is None else {**c,'encoder_rematerialize':inner_rematerialize}
    b,t,h,w,ch=spatial.shape;frames=b*t;extra=(-frames)%chunk_frames
    if (h,w,ch)!=(c['max_board_size'],c['max_board_size'],22) or glob.shape!=(b,t,19):
        raise ValueError('Complete board input contract differs')
    s=jnp.pad(spatial.reshape(frames,h,w,ch),((0,extra),(0,0),(0,0),(0,0))).reshape(-1,chunk_frames,h,w,ch)
    g=jnp.pad(glob.reshape(frames,19),((0,extra),(0,0))).reshape(-1,chunk_frames,19)
    ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
    moe='moe' in c
    if moe and router_counts is None:raise ValueError('MoE training requires live frame counts')
    if moe:
        router_live=(jnp.arange(t)[None,:]<router_counts[:,None]).reshape(frames)
        router_live=jnp.pad(router_live,((0,extra),)).reshape(-1,chunk_frames)
    def chunk(inputs):
        board,globals=inputs[:2]
        # The artificial time axis has length one; each encoder frame is
        # independent, including spatial attention and all normalizations.
        if moe:
            (full,first),stats=encoder.spatial(ep,board[:,None],ec,with_first=True,
                with_router_stats=True,valid_frames=inputs[2][:,None])
        else:
            full,first=encoder.spatial(ep,board[:,None],ec,with_first=True)
        ft=encoder.connect(ep,full,globals[:,None],c)
        dt=encoder.connect(ep,first,globals[:,None],c)
        result=(ft,dt,projected(p,full,c),projected(p,first,c))
        result=jax.tree.map(lambda x:x[:,0],result)
        return (result,stats) if moe else result
    # Rematerialize the whole chunk: differentiated scan must not retain every
    # frame's internal wide encoder activations across the full history.
    if counts is None:
        outputs=jax.lax.map(jax.checkpoint(chunk), (s,g,router_live) if moe else (s,g))
    else:
        if counts.shape!=(b,):raise ValueError('One live frame count per game required')
        live=(jnp.arange(t)[None,:]<counts[:,None]).reshape(frames)
        live=jnp.pad(live,((0,extra),)).reshape(-1,chunk_frames)
        # There are no cross-frame operations in the encoder. Fully padded
        # chunks occur strictly after each game's live causal prefix, so their
        # features cannot affect any supervised token. Keep all original frame
        # indices, chunk boundaries and partial chunks unchanged.
        shape_inputs=(jax.ShapeDtypeStruct(s.shape[1:],s.dtype),jax.ShapeDtypeStruct(g.shape[1:],g.dtype))
        if moe:shape_inputs+= (jax.ShapeDtypeStruct(router_live.shape[1:],router_live.dtype),)
        output_shapes=jax.eval_shape(chunk,shape_inputs)
        def empty(inputs):
            return jax.tree.map(lambda x:jnp.zeros(x.shape,x.dtype),output_shapes)
        def conditional(inputs):
            if moe:
                board,globals,router_active,active=inputs
                return jax.lax.cond(jnp.any(active),chunk,empty,(board,globals,router_active))
            board,globals,active=inputs
            return jax.lax.cond(jnp.any(active),chunk,empty,(board,globals))
        # lax.map is deliberately sequential here. vmap would convert the
        # conditional into a select and evaluate the expensive inactive path.
        outputs=jax.lax.map(jax.checkpoint(conditional),(s,g,router_live,live) if moe else (s,g,live))
    if moe:
        outputs,statistics=outputs
        statistics=jnp.sum(statistics,axis=0)
    def restore(x):
        flat=x.reshape(-1,*x.shape[2:])[:frames]
        return flat.reshape(b,t,*flat.shape[1:])
    restored=jax.tree.map(restore,outputs)
    return (*restored,statistics) if moe else restored


def forward(p,spatial,glob,actions,counts,c,*,chunk_frames,inner_rematerialize=None,with_hidden=False,skip_padding=False,encoder_counts=None):
    encoded=encode(p,spatial,glob,c,chunk_frames=chunk_frames,inner_rematerialize=inner_rematerialize,
        counts=(counts if encoder_counts is None else encoder_counts) if skip_padding else None,router_counts=counts)
    full_tokens,draft_tokens,full_features,draft_features=encoded[:4]
    out=packed_forward(p,full_tokens,draft_tokens,actions,counts,spatial.shape[2],c,full_features,draft_features,with_hidden=with_hidden)
    if 'moe' in c and with_hidden:
        out['router_statistics']=jnp.concatenate((encoded[4],out['router_statistics']),axis=0)
    return out


def packed_forward(p,full_tokens,draft_tokens,actions,counts,size,c,full_features,draft_features,*,with_hidden=False):
    b,t,d=full_tokens.shape
    if draft_tokens.shape!=(b,t,d) or actions.shape!=(b,t) or counts.shape!=(b,) or t>c['max_positions']:
        raise ValueError('Invalid complete causal history')
    action=causal.action_tokens(p,actions,size,c)
    x=jnp.stack((draft_tokens,full_tokens,action),axis=2).reshape(b,3*t,d)
    live=jnp.arange(t)[None,:]<counts[:,None]
    router_live=jnp.repeat(live,3,axis=1)
    frame=jnp.arange(3*t);positions=jnp.broadcast_to(2*(frame//3)+(frame%3==2),x.shape[:2])
    def layer(x,block):
        q,k,v=causal.project(x,block,positions,c)
        if 'fuzzy' in c:
            return causal.finish(x,draft_model.attention(q,k,v,c),block,c,valid=router_live,with_fuzzy_stats=True)
        if 'moe' in c:
            return causal.finish(x,draft_model.attention(q,k,v,c),block,c,valid=router_live,with_router_stats=True)
        return causal.finish(x,draft_model.attention(q,k,v,c),block,c),None
    x,router_stats=jax.lax.scan(jax.checkpoint(layer) if c['rematerialize'] else layer,x,causal.blocks(p))
    x=x.reshape(b,t,3,d)
    main=policy(p,x[:,:,1],size,c,full_features)
    draft=policy(p,x[:,:,0],size,c,draft_features)
    live=jnp.arange(t)[None,:]<counts[:,None]
    main=jnp.where(live[:,:,None],main,0.);draft=jnp.where(live[:,:,None],draft,0.)
    if with_hidden:
        out=dict(policy=main,aux_policy=draft,
                    latent=jnp.where(live[:,:,None],causal.norm(x[:,:,1],p['head.norm.scale'],c),0.),
                    aux_latent=jnp.where(live[:,:,None],causal.norm(x[:,:,0],p['head.norm.scale'],c),0.))
        if 'moe' in c:out['router_statistics']=router_stats
        if 'fuzzy' in c:out['fuzzy_statistics']=router_stats
        return out
    return main,draft


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
