"""Bound main-only policy/value validation without changing model parameters.

CNN frames are read out before the next trunk chunk. The temporal model keeps
one board token and projected spatial features per frame, then performs the
ordinary two-token causal pass. Training-only helper branches are not evaluated.
"""
import jax
import jax.numpy as jnp

import causal
import cnn_chunks
import compact
import encoder
import heads
import joint
import katago
import policy_model


def restore_frames(x,batch,positions):
    flat=x.reshape(-1,*x.shape[2:])[:batch*positions]
    return flat.reshape(batch,positions,*flat.shape[1:])


def cnn_forward(p,b,c,chunk_frames):
    spatial,glob=cnn_chunks.inputs(b,c,chunk_frames)
    def one(inputs):
        s,g=inputs
        out=katago.forward(p,s,g,c,with_features=True)
        logits=heads.cnn(joint.value_params(p),out['features'],s[...,:1])
        return dict(policy=out['policy'],value_logits=logits,value=heads.signed_value(logits))
    out=jax.lax.map(one,(spatial,glob));batch,positions=b['actions'].shape
    out=jax.tree.map(lambda x:restore_frames(x,batch,positions),out)
    live=jnp.arange(positions)[None,:]<b['counts'][:,None]
    out['value_logits']=jnp.where(live[...,None],out['value_logits'],0.)
    out['value']=jnp.where(live,out['value'],0.)
    return out


def temporal_forward(p,b,c,chunk_frames):
    compact.validate(c,chunk_frames)
    spatial,glob=b['spatial'],b['global_features'];batch,positions,size,other,channels=spatial.shape
    if (size,other,channels)!=(c['max_board_size'],c['max_board_size'],22) or glob.shape!=(batch,positions,19):
        raise ValueError('Complete board input contract differs')
    if b['actions'].shape!=(batch,positions) or b['counts'].shape!=(batch,) or positions>c['max_positions']:
        raise ValueError('Invalid complete causal history')
    frames=batch*positions;extra=(-frames)%chunk_frames
    s=jnp.pad(spatial.reshape(frames,size,size,22),((0,extra),(0,0),(0,0),(0,0))).reshape(-1,chunk_frames,size,size,22)
    g=jnp.pad(glob.reshape(frames,19),((0,extra),(0,0))).reshape(-1,chunk_frames,19)
    ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
    def one(inputs):
        s,g=inputs
        features=encoder.spatial(ep,s[:,None],c)
        token=encoder.connect(ep,features,g[:,None],c)
        return jax.tree.map(lambda x:x[:,0],(token,compact.projected(p,features,c)))
    tokens,features=jax.tree.map(lambda x:restore_frames(x,batch,positions),jax.lax.map(one,(s,g)))
    action=causal.action_tokens(p,b['actions'],size,c)
    x=jnp.stack((tokens,action),axis=2).reshape(batch,2*positions,c['width'])
    offsets=jnp.broadcast_to(jnp.arange(2*positions),x.shape[:2])
    def layer(x,block):
        q,k,v=causal.project(x,block,offsets,c)
        return causal.finish(x,causal.full_attention(q,k,v,c),block,c),None
    x,_=jax.lax.scan(layer,x,causal.blocks(p))
    x=x.reshape(batch,positions,2,c['width'])[:,:,0]
    live=jnp.arange(positions)[None,:]<b['counts'][:,None]
    out=dict(policy=jnp.where(live[...,None],compact.policy(p,x,size,c,features),0.),
             latent=jnp.where(live[...,None],causal.norm(x,p['head.norm.scale'],c),0.))
    return {**joint.temporal_readout(p,out,live),'latent':out['latent']}


def forward(p,b,c,*,chunk_frames=32):
    if c['architecture']=='katago_nested_policy':return cnn_forward(p,b,c,chunk_frames)
    if c['architecture']=='causal_visual_policy':return temporal_forward(p,b,c,chunk_frames)
    raise ValueError('Unknown joint model')


def totals(p,b,c,*,chunk_frames=32,axis_name=None):
    out=forward(p,b,c,chunk_frames=chunk_frames)
    return {**policy_model.total_metrics(out['policy'],b,axis_name,stratify=True),
            **heads.value_totals(out['value'],b,axis_name=axis_name,stratify=True)}


def averages(totals):
    policy={k:v for k,v in totals.items() if not k.startswith('value_')}
    result=policy_model.averages(policy)
    # A partial batch may represent less than one unit of family mass. Normalize
    # by its actual positive weight, as for the already qualified value means.
    if 'family_count' in policy:
        count=policy['family_count'];denominator=jnp.where(count>0,count,1.)
        for suffix in ('ce','target_entropy'):result['family_'+suffix]=policy['family_'+suffix]/denominator
        result['family_kl']=result['family_ce']-result['family_target_entropy']
    value={k:v for k,v in totals.items() if k.startswith('value_')}
    return {**result,**heads.value_averages(value)}
