"""Shared first-pass readout with exact full-pass history and no current-full leakage."""
from functools import lru_cache
import math
import jax
import jax.numpy as jnp
import causal
import encoder
from draft_mask import DraftMask,visible


def dense(q,k,v):
    groups=q.shape[2]//k.shape[2];qg=q.reshape(*q.shape[:2],k.shape[2],groups,q.shape[-1])
    scores=jnp.einsum('btkgd,bskd->bkgts',qg,k,preferred_element_type=jnp.float32)/math.sqrt(q.shape[-1])
    positions=jnp.arange(q.shape[1]);mask=visible(positions[:,None],positions[None,:],q.shape[1])
    weights=jax.nn.softmax(jnp.where(mask[None,None,None],scores,-1e30),-1).astype(v.dtype)
    return jnp.einsum('bkgts,bskd->btkgd',weights,v,preferred_element_type=jnp.float32).reshape(q.shape)


@lru_cache(maxsize=16)
def kernel(length,heads,valid_length):
    from jax.experimental.pallas.ops.tpu import splash_attention as splash
    one=DraftMask((length,length),valid_length);mask=splash.MultiHeadMask([one for _ in range(heads)])
    blocks=splash.BlockSizes(block_q=512,block_kv=512,block_kv_compute=512,block_q_dkv=512,block_kv_dkv=512,
        block_kv_dkv_compute=512,block_q_dq=512,block_kv_dq=512)
    with jax.ensure_compile_time_eval():return splash.make_splash_mha(mask,block_sizes=blocks,head_shards=1,q_seq_shards=1)


def attention(q,k,v,c):
    if c['attention_backend']=='xla':return dense(q,k,v)
    valid=q.shape[1];length=math.ceil(valid/512)*512
    def pad(x):return jnp.pad(x.transpose(0,2,1,3),((0,0),(0,0),(0,length-x.shape[1]),(0,0)))
    y=jax.vmap(kernel(length,c['heads'],valid))(pad(q)/math.sqrt(q.shape[-1]),pad(k),pad(v))
    return y.transpose(0,2,1,3)[:,:valid].astype(jnp.float32)


def packed_forward(p,full_tokens,draft_tokens,actions,counts,size,c,full_features,draft_features):
    b,t,d=full_tokens.shape
    if draft_tokens.shape!=(b,t,d) or actions.shape!=(b,t) or t>c['max_positions']:raise ValueError('Invalid draft training sequence')
    action=causal.action_tokens(p,actions,size,c)
    x=jnp.stack((draft_tokens,full_tokens,action),axis=2).reshape(b,3*t,d)
    frame=jnp.arange(3*t);positions=jnp.broadcast_to(2*(frame//3)+(frame%3==2),x.shape[:2])
    def layer(x,block):
        q,k,v=causal.project(x,block,positions,c)
        return causal.finish(x,attention(q,k,v,c),block,c),None
    x,_=jax.lax.scan(jax.checkpoint(layer) if c['rematerialize'] else layer,x,causal.blocks(p))
    x=x.reshape(b,t,3,d)
    main=causal.policy(p,x[:,:,1],size,c,full_features)
    draft=causal.policy(p,x[:,:,0],size,c,draft_features)
    live=jnp.arange(t)[None,:]<counts[:,None]
    return jnp.where(live[:,:,None],main,0.),jnp.where(live[:,:,None],draft,0.)


def forward(p,spatial,glob,actions,counts,c):
    if c['encoder_passes']<2:raise ValueError('Draft/full training requires at least two passes')
    ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
    full,first=encoder.spatial(ep,spatial,c,with_first=True)
    full_token=encoder.connect(ep,full,glob,c);first_token=encoder.connect(ep,first,glob,c)
    return packed_forward(p,full_token,first_token,actions,counts,spatial.shape[2],c,full,first)
