"""Pure JAX causal observation/action transformer with one tied policy head.

RMSNorm is per token. Board encoding uses no batch statistics. The cached
move graph appends [previous action, current board patches, policy readout].
The current action is not an input to the policy it is supposed to predict.
"""
from functools import lru_cache
import math
import numpy as np
import jax
import jax.numpy as jnp
from observation_attention import ObservationMask, visible


def validate(c):
    fields=set('architecture width layers mlp_hidden heads kv_heads max_board_size max_positions dtype rematerialize norm_epsilon rope_theta attention_backend encoder'.split())
    if c.get('encoder')=='overlap_conv':fields.add('encoder_channels')
    if 'encoder_depth' in c:fields.add('encoder_depth')
    if 'observation_attention' in c:fields.add('observation_attention')
    if 'readout_pooling' in c:fields.add('readout_pooling')
    if set(c)!=fields or c['architecture']!='causal_visual_policy' or c['encoder'] not in ('overlap_linear','overlap_conv64','overlap_conv'):
        raise ValueError('Unknown causal policy contract')
    if c['encoder']=='overlap_conv' and (type(c['encoder_channels']) is not int
            or not 8<=c['encoder_channels']<=1024 or c['encoder_channels']%8):
        raise ValueError('Encoder channels must be a multiple of eight in [8, 1024]')
    if 'encoder_depth' in c and (c['encoder']=='overlap_linear' or type(c['encoder_depth']) is not int
            or not 2<=c['encoder_depth']<=8):raise ValueError('Invalid convolutional encoder depth')
    if c.get('observation_attention','causal') not in ('causal','board'):raise ValueError('Invalid observation attention')
    if c.get('readout_pooling','learned') not in ('learned','board_mean'):raise ValueError('Invalid readout pooling')
    for key,lo,hi in [('width',8,4096),('layers',1,96),('mlp_hidden',8,16384),('heads',1,64),('kv_heads',1,64),
                      ('max_board_size',1,52),('max_positions',1,2048)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi: raise ValueError('Invalid '+key)
    if c['width']%c['heads'] or c['heads']%c['kv_heads'] or (c['width']//c['heads'])%2:
        raise ValueError('Incompatible attention heads')
    if c['attention_backend'] not in ('xla','splash') or c['dtype'] not in ('float32','bfloat16') or type(c['rematerialize']) is not bool:
        raise ValueError('Invalid execution contract')
    if any(not math.isfinite(c[k]) or c[k]<=0 for k in ('norm_epsilon','rope_theta')): raise ValueError('Invalid normalization/RoPE')
    if c['attention_backend']=='splash' and (c['width']//c['heads'])%64: raise ValueError('Splash head dimension must divide by 64')
    return c


def encoder_channels(c):
    if c['encoder']=='overlap_linear':return 22
    return 64 if c['encoder']=='overlap_conv64' else c['encoder_channels']


def encoder_depth(c):return 0 if c['encoder']=='overlap_linear' else c.get('encoder_depth',2)


def observation_stride(c,spec):return spec['stride'] if c.get('observation_attention','causal')=='board' else 0


def anchors(size):
    side=min(max(1,size-1),math.ceil(2*size/3))
    # An odd span has no integer midpoint. Use an even number of anchors so
    # the grid is closed under reflection, including 19x19 (14, not 13).
    if max(0,size-2)%2 and side%2:side+=1
    return np.rint(np.linspace(0,max(0,size-2),side)).astype(np.int32)


def layout(size,c):
    if not 1<=size<=c['max_board_size']: raise ValueError('Board exceeds contract')
    n=len(anchors(size));return {'side':n,'patches':n*n,'stride':n*n+2,'capacity':c['max_positions']*(n*n+2)}


def initialize(seed,c):
    validate(c);key=jax.random.key(seed);d=c['width'];L=c['layers'];f=c['mlp_hidden'];kv=c['kv_heads']*(d//c['heads']);p={}
    def weight(name,shape,scale=1.,embedding=False,stack=False):
        nonlocal key
        key,draw=jax.random.split(key)
        std=.02 if embedding else scale/math.sqrt(shape[1 if stack else 0])
        p[name]=jax.random.normal(draw,shape,jnp.float32)*std
    for name,shape,scale in [('q',(d,d),1.),('kv',(d,2*kv),1.),('out',(d,d),1/math.sqrt(2*L)),
                             ('gate',(d,f),1.),('up',(d,f),1.),('down',(f,d),1/math.sqrt(2*L))]:
        weight('blocks.'+name+'.weight',(L,*shape),scale,stack=True)
    for name in ('attn_norm','mlp_norm'):p['blocks.'+name+'.scale']=jnp.ones((L,d),jnp.float32)
    channels=encoder_channels(c)
    weight('encoder.patch.weight',(4*channels,d));p['encoder.patch.bias']=jnp.zeros(d,jnp.float32)
    weight('encoder.global.weight',(19,d))
    for name in ('row','col'):weight('encoder.'+name+'.weight',(c['max_board_size'],d),embedding=True)
    weight('head.actions.weight',(c['max_board_size']**2+1,d),embedding=True)
    weight('token_types.weight',(3,d),embedding=True);weight('readout',(d,),embedding=True)
    p['head.norm.scale']=jnp.ones(d,jnp.float32)
    # Append new random draws after all original parameters. The trunk, head,
    # positions and globals retain the parent's exact initialization.
    if c['encoder']!='overlap_linear':
        for i,inputs in enumerate((22,)+(channels,)*(encoder_depth(c)-1)):
            # weight() divides by sqrt(shape[0]); total fan-in is 9*inputs.
            weight(f'encoder.stem{i}.weight',(3,3,inputs,channels),scale=1/math.sqrt(3*inputs))
            p[f'encoder.stem{i}.scale']=jnp.ones(channels,jnp.float32)
    return p


def parameter_schema(c):
    p=jax.eval_shape(lambda:initialize(0,c))
    return [{'path':k,'shape':list(v.shape),'dtype':str(v.dtype),'elements':math.prod(v.shape)} for k,v in sorted(p.items())]


def linear(x,w,c):
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    return jnp.matmul(x.astype(dtype),w.astype(dtype),preferred_element_type=jnp.float32)


def norm(x,scale,c):
    x=x.astype(jnp.float32)
    return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+c['norm_epsilon'])*scale


def action_indices(size,c):
    a=jnp.arange(size*size)
    return jnp.concatenate([a//size*c['max_board_size']+a%size,jnp.asarray([c['max_board_size']**2])])


def spatial_features(p,spatial,c):
    if c['encoder']=='overlap_linear':return spatial
    b,t,h,w,_=spatial.shape;x=spatial.reshape(b*t,h,w,22)
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    for i in range(encoder_depth(c)):
        x=jax.lax.conv_general_dilated(x.astype(dtype),p[f'encoder.stem{i}.weight'].astype(dtype),
            window_strides=(1,1),padding='SAME',dimension_numbers=('NHWC','HWIO','NHWC'),
            preferred_element_type=jnp.float32)
        x=jax.nn.silu(norm(x,p[f'encoder.stem{i}.scale'],c))
    return x.reshape(b,t,h,w,encoder_channels(c))


def observation_tokens(p,spatial,globals,c):
    b,t,size,width,ch=spatial.shape
    if size!=width or ch!=22 or globals.shape!=(b,t,19): raise ValueError('Expected exact V7 board/global inputs')
    spec=layout(size,c);starts=anchors(size);extent=max(2,size)
    features=spatial_features(p,spatial,c);channels=features.shape[-1]
    s=jnp.pad(features,((0,0),(0,0),(0,extent-size),(0,extent-size),(0,0))).reshape(b,t,extent*extent,channels)
    index=np.asarray([[(y+dy)*extent+x+dx for dy in range(2) for dx in range(2)] for y in starts for x in starts],np.int32)
    patches=jnp.take(s,jnp.asarray(index),axis=2).reshape(b,t,spec['patches'],4*channels)
    pos=(p['encoder.row.weight'][starts,None,:]+p['encoder.col.weight'][None,starts,:]).reshape(spec['patches'],c['width'])
    return (linear(patches,p['encoder.patch.weight'],c)+p['encoder.patch.bias']+pos+p['token_types.weight'][0]
            +linear(globals,p['encoder.global.weight'],c)[:,:,None,:])


def action_tokens(p,actions,size,c):
    return p['head.actions.weight'][action_indices(size,c)[actions]]+p['token_types.weight'][2]


def readout_tokens(p,patches,c):
    read=jnp.broadcast_to(p['readout']+p['token_types.weight'][1],(*patches.shape[:-2],1,c['width']))
    if c.get('readout_pooling','learned')=='board_mean':read=read+jnp.mean(patches,axis=-2,keepdims=True)
    return read


def pack(p,spatial,globals,actions,c):
    patches=observation_tokens(p,spatial,globals,c);b,t,_,d=patches.shape
    if actions.shape!=(b,t) or t>c['max_positions']: raise ValueError('History exceeds contract')
    read=readout_tokens(p,patches,c)
    action=action_tokens(p,actions,spatial.shape[2],c)[:,:,None,:]
    return jnp.concatenate([patches,read,action],2).reshape(b,-1,d)


def rope(x,positions,c):
    frequency=c['rope_theta']**(-jnp.arange(0,x.shape[-1],2,dtype=jnp.float32)/x.shape[-1])
    angle=positions[...,None,None]*frequency
    even,odd=x[...,::2].astype(jnp.float32),x[...,1::2].astype(jnp.float32)
    return jnp.stack([even*jnp.cos(angle)-odd*jnp.sin(angle),even*jnp.sin(angle)+odd*jnp.cos(angle)],-1).reshape(x.shape).astype(x.dtype)


def project(x,p,positions,c):
    b,t,d=x.shape;h,kh=c['heads'],c['kv_heads'];dim=d//h
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    y=norm(x,p['attn_norm.scale'],c)
    q=linear(y,p['q.weight'],c).reshape(b,t,h,dim).astype(dtype)
    kv=linear(y,p['kv.weight'],c).reshape(b,t,2,kh,dim).astype(dtype)
    return rope(q,positions,c),rope(kv[:,:,0],positions,c),kv[:,:,1]


def dense_attention(q,k,v,positions,lengths,observation_stride=0):
    groups=q.shape[2]//k.shape[2];qg=q.reshape(*q.shape[:2],k.shape[2],groups,q.shape[-1])
    scores=jnp.einsum('btkgd,bskd->bkgts',qg,k,preferred_element_type=jnp.float32)/math.sqrt(q.shape[-1])
    keys=jnp.arange(k.shape[1])[None,None,:]
    allowed=visible(positions[:,:,None],keys,observation_stride)&(keys<lengths[:,None,None])
    weights=jax.nn.softmax(jnp.where(allowed[:,None,None],scores,-1e30),-1).astype(v.dtype)
    return jnp.einsum('bkgts,bskd->btkgd',weights,v,preferred_element_type=jnp.float32).reshape(q.shape)


@lru_cache(maxsize=32)
def splash_kernel(length,heads,observation_stride=0,valid_length=None):
    from jax.experimental.pallas.ops.tpu import splash_attention as splash
    one=(ObservationMask((length,length),observation_stride,valid_length or length)
         if observation_stride else splash.CausalMask((length,length)))
    mask=splash.MultiHeadMask([one for _ in range(heads)])
    # Qualified against the transparent forward/backward on this pod before
    # learning. Larger tiles amortize the very frequent long-history blocks.
    blocks=splash.BlockSizes(block_q=512,block_kv=512,block_kv_compute=512,block_q_dkv=512,block_kv_dkv=512,
        block_kv_dkv_compute=512,block_q_dq=512,block_kv_dq=512)
    with jax.ensure_compile_time_eval():
        return splash.make_splash_mha(mask,block_sizes=blocks,head_shards=1,q_seq_shards=1)


def full_attention(q,k,v,c,observation_stride=0):
    if c['attention_backend']=='xla':
        return dense_attention(q,k,v,jnp.broadcast_to(jnp.arange(q.shape[1]),q.shape[:2]),jnp.full((q.shape[0],),q.shape[1]),observation_stride)
    length=math.ceil(q.shape[1]/512)*512
    def pad(x):return jnp.pad(x.transpose(0,2,1,3),((0,0),(0,0),(0,length-x.shape[1]),(0,0)))
    result=jax.vmap(splash_kernel(length,c['heads'],observation_stride,q.shape[1]))(pad(q)/math.sqrt(q.shape[-1]),pad(k),pad(v))
    return result.transpose(0,2,1,3)[:,:q.shape[1]].astype(jnp.float32)


def finish(x,attended,p,c):
    x=x+linear(attended.reshape(x.shape),p['out.weight'],c)
    y=norm(x,p['mlp_norm.scale'],c)
    return x+linear(jax.nn.silu(linear(y,p['gate.weight'],c))*linear(y,p['up.weight'],c),p['down.weight'],c)


def blocks(p):return {k[len('blocks.'):]:v for k,v in p.items() if k.startswith('blocks.')}


def policy(p,x,size,c):
    # Current-policy output embedding is tied to action input embeddings.
    return linear(norm(x,p['head.norm.scale'],c),p['head.actions.weight'][action_indices(size,c)].T,{**c,'dtype':'float32'})


def forward(p,spatial,globals,actions,counts,c,*,with_cache=False,cache_positions=None,network_version=0):
    x=pack(p,spatial,globals,actions,c);spec=layout(spatial.shape[2],c);b,t=actions.shape
    positions=jnp.broadcast_to(jnp.arange(x.shape[1]),x.shape[:2])
    def layer(x,block):
        q,k,v=project(x,block,positions,c)
        y=finish(x,full_attention(q,k,v,c,observation_stride(c,spec)),block,c)
        return y,(k,v) if with_cache else None
    x,cache_values=jax.lax.scan(jax.checkpoint(layer) if c['rematerialize'] else layer,x,blocks(p))
    x=x.reshape(b,t,spec['stride'],c['width'])[:,:,spec['patches']]
    logits=policy(p,x,spatial.shape[2],c)
    live=jnp.arange(t)[None,:]<counts[:,None];logits=jnp.where(live[:,:,None],logits,0.)
    if not with_cache:return logits
    cp=c['max_positions'] if cache_positions is None else cache_positions
    if not t<=cp<=c['max_positions']:raise ValueError('Cache cannot hold prefill')
    capacity=cp*spec['stride'];valid=(counts>0)&(counts<=t);lengths=jnp.where(valid,counts*spec['stride']-1,0)
    def store(x):
        # Layer-major scan output -> batch-major for data sharding.
        x=jnp.swapaxes(x,0,1)
        x=jnp.where((jnp.arange(x.shape[2])[None,:]<lengths[:,None])[:,None,:,None,None],x,0.)
        return jnp.pad(x,((0,0),(0,0),(0,capacity-x.shape[2]),(0,0),(0,0)))
    return logits,{'keys':store(cache_values[0]),'values':store(cache_values[1]),'lengths':lengths,'valid':valid,
                   'network_version':jnp.asarray(network_version,jnp.uint32)}


def first_move(p,spatial,globals,c,*,network_version=0):
    """Opening policy without a fictitious previous action; return a live cache."""
    spec=layout(spatial.shape[1],c);b=spatial.shape[0]
    patches=observation_tokens(p,spatial[:,None],globals[:,None],c)[:,0]
    read=readout_tokens(p,patches,c)
    x=jnp.concatenate([patches,read],1)
    positions=jnp.broadcast_to(jnp.arange(x.shape[1]),x.shape[:2])
    def layer(x,block):
        q,k,v=project(x,block,positions,c)
        # The opening has no historical keys and uses transparent attention.
        y=finish(x,dense_attention(q,k,v,positions,jnp.full((b,),x.shape[1]),observation_stride(c,spec)),block,c)
        return y,(k,v)
    x,(keys,values)=jax.lax.scan(layer,x,blocks(p))
    def store(value):
        value=jnp.swapaxes(value,0,1)
        return jnp.pad(value,((0,0),(0,0),(0,spec['capacity']-value.shape[2]),(0,0),(0,0)))
    return policy(p,x[:,-1],spatial.shape[1],c),{
        'keys':store(keys),'values':store(values),'lengths':jnp.full((b,),spec['stride']-1,jnp.int32),
        'valid':jnp.ones((b,),bool),'network_version':jnp.asarray(network_version,jnp.uint32)}


def append_move(p,cache,previous_action,spatial,globals,c,*,attention_positions,active=None,network_version=0):
    """One graph: previous action + exact new board -> policy and updated cache.

attention_positions bounds EXECUTED attention, independently of allocation.
The cache must end after a policy readout; its pending action is supplied here.
"""
    b,size,_,_=spatial.shape;spec=layout(size,c);stride=spec['stride'];capacity=cache['keys'].shape[2]
    extent=attention_positions*stride-1
    if not 1<=attention_positions<=c['max_positions'] or extent>capacity:raise ValueError('Invalid attention extent')
    active=jnp.ones(b,bool) if active is None else active
    good=(cache['valid']&(cache['lengths']%stride==stride-1)&(previous_action>=0)&(previous_action<=size*size)
          &(cache['lengths']+stride<=extent)&(cache['network_version']==jnp.asarray(network_version,jnp.uint32)))
    write=active&good;positions=cache['lengths'][:,None]+jnp.arange(stride)[None,:]
    lengths=cache['lengths']+jnp.where(write,stride,0)
    patches=observation_tokens(p,spatial[:,None],globals[:,None],c)[:,0]
    action=action_tokens(p,jnp.clip(previous_action,0,size*size),size,c)[:,None,:]
    read=readout_tokens(p,patches,c)
    x=jnp.concatenate([action,patches,read],1)
    rows=jnp.arange(b)[:,None];slots=jnp.clip(positions,0,capacity-1)
    def update(old,new):
        # Select only the new slots. A full-cache where would needlessly touch
        # the allocated context on inactive rows and obstruct buffer aliasing.
        return old.at[rows,slots].set(jnp.where(write[:,None,None,None],new,old[rows,slots]))
    def layer(x,data):
        block,old_k,old_v=data;q,k,v=project(x,block,positions,c)
        k,v=update(old_k,k),update(old_v,v)
        attended=dense_attention(q,k[:,:extent],v[:,:extent],positions,lengths,observation_stride(c,spec))
        return finish(x,attended,block,c),(k,v)
    x,(keys,values)=jax.lax.scan(layer,x,(blocks(p),jnp.swapaxes(cache['keys'],0,1),jnp.swapaxes(cache['values'],0,1)))
    logits=policy(p,x[:,-1],size,c)
    updated={**cache,'keys':jnp.swapaxes(keys,0,1),'values':jnp.swapaxes(values,0,1),'lengths':lengths,
             'valid':cache['valid']&(~active|good)}
    return jnp.where(write[:,None],logits,0.),updated
