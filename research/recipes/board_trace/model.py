"""Canonical history trunk plus independent exact-state policy/value features.

A board observation only affects predictions at its own position. Behavioral
losses cannot update the history or board encoders. Multi-step execution would
need validated per-step board inputs; this model does not invent exact states.
"""
import math
import jax
import jax.numpy as jnp


def initialize(seed,c):
    key=jax.random.key(seed);width=c['width'];actions=c['size']**2+1
    def weight(shape,scale=1.):
        nonlocal key
        key,draw=jax.random.split(key)
        return jax.random.normal(draw,shape,jnp.float32)*(scale/math.sqrt(shape[0]))
    def norm():return {'scale':jnp.ones(width,jnp.float32),'bias':jnp.zeros(width,jnp.float32)}
    blocks=[]
    for _ in range(c['blocks']):
        blocks.append({'n1':norm(),'qkv':weight((width,3*width)),'out':weight((width,width)),
                       'n2':norm(),'mlp_in':weight((width,4*width),math.sqrt(2)),
                       'mlp_out':weight((4*width,width))})
    params={'tokens':weight((actions+2,width),0.2),'positions':weight((c['max_tokens'],width),0.2),
            'player':weight((2,width),0.1),'global':weight((2,width),0.1),'blocks':blocks,'norm':norm(),
            'play':weight((width,actions),0.1),'value':weight((width,1),0.1),
            'behavior_hidden':weight((width,width)),'behavior_style':weight((actions,width)),
            'behavior_bias':jnp.zeros(width,jnp.float32),'behavior_out':weight((width,actions),0.1)}

    board_key=jax.random.fold_in(jax.random.key(seed),0xB04D51);bw=c['board_width']
    def board_weight(shape,scale=1.):
        nonlocal board_key
        board_key,draw=jax.random.split(board_key)
        fanin=math.prod(shape[:-1]) if len(shape)==4 else shape[0]
        return jax.random.normal(draw,shape,jnp.float32)*(scale/math.sqrt(fanin))
    def board_norm():return {'scale':jnp.ones(bw,jnp.float32),'bias':jnp.zeros(bw,jnp.float32)}
    params.update(board_embedding=board_weight((3,bw),.2),board_norm=board_norm(),
        board_blocks=[{'n1':board_norm(),'w1':board_weight((3,3,bw,bw),math.sqrt(2)),
                       'n2':board_norm(),'w2':board_weight((3,3,bw,bw))} for _ in range(c['board_blocks'])],
        board_query=board_weight((width,bw),.1),board_point=board_weight((bw,1),.1),
        board_pass=board_weight((bw,1),.1),board_value=board_weight((bw,1),.1),
        behavior_board_hidden=board_weight((bw,width),.1),behavior_board_query=board_weight((width,bw),.1),
        behavior_board_point=board_weight((bw,1),.1),behavior_board_pass=board_weight((bw,1),.1))
    return params


def norm(x,p):
    centered=x.astype(jnp.float32)-jnp.mean(x.astype(jnp.float32),-1,keepdims=True)
    return centered*jax.lax.rsqrt(jnp.mean(centered**2,-1,keepdims=True)+1e-5)*p['scale']+p['bias']


def linear(x,w,c):
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[c['dtype']]
    return jnp.matmul(x.astype(dtype),w.astype(dtype),precision=jax.lax.Precision.HIGHEST,
                      preferred_element_type=jnp.float32)


def embedding(p,tokens,positions,c):
    constants=jnp.asarray([c['size']/19,c['komi']/(c['size']**2)],jnp.float32)
    return p['tokens'][tokens]+p['positions'][positions]+p['player'][positions%2]+constants@p['global']


def prefill(p,tokens,lengths,c):
    """Return all causal states and per-layer KV. Length is the last valid index."""
    batch,time=tokens.shape;heads=c['heads'];width=c['width'];dim=width//heads
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[c['dtype']]
    x=embedding(p,tokens,jnp.arange(time)[None,:],c);cache=[]
    mask=(jnp.arange(time)[None,:]<=jnp.arange(time)[:,None])[None,None,:,:]
    mask=mask&(jnp.arange(time)[None,None,None,:]<=lengths[:,None,None,None])
    scale=1/math.sqrt(c['blocks'])
    for block in p['blocks']:
        qkv=linear(norm(x,block['n1']),block['qkv'],c).reshape(batch,time,3,heads,dim).astype(dtype)
        q,k,v=(qkv[:,:,i] for i in range(3));cache.append((k,v))
        scores=jnp.einsum('bthd,bshd->bhts',q,k,precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)/math.sqrt(dim)
        weights=jax.nn.softmax(jnp.where(mask,scores,-1e9),-1).astype(dtype)
        y=jnp.einsum('bhts,bshd->bthd',weights,v,precision=jax.lax.Precision.HIGHEST,
                     preferred_element_type=jnp.float32).reshape(batch,time,width)
        x=x+scale*linear(y,block['out'],c)
        x=x+scale*linear(jax.nn.gelu(linear(norm(x,block['n2']),block['mlp_in'],c)),block['mlp_out'],c)
    return norm(x,p['norm']),tuple(cache)


def append(p,cache,tokens,positions,c):
    """Append one actual/hypothetical token per branch without a host exchange."""
    batch=tokens.shape[0];heads=c['heads'];width=c['width'];dim=width//heads
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[c['dtype']]
    x=embedding(p,tokens,positions,c);updated=[];scale=1/math.sqrt(c['blocks'])
    for block,(keys,values) in zip(p['blocks'],cache):
        qkv=linear(norm(x,block['n1']),block['qkv'],c).reshape(batch,3,heads,dim).astype(dtype)
        q,k,v=(qkv[:,i] for i in range(3))
        keys=keys.at[jnp.arange(batch),positions].set(k);values=values.at[jnp.arange(batch),positions].set(v)
        scores=jnp.einsum('bhd,bshd->bhs',q,keys,precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)/math.sqrt(dim)
        mask=jnp.arange(keys.shape[1])[None,None,:]<=positions[:,None,None]
        weights=jax.nn.softmax(jnp.where(mask,scores,-1e9),-1).astype(dtype)
        y=jnp.einsum('bhs,bshd->bhd',weights,values,precision=jax.lax.Precision.HIGHEST,
                     preferred_element_type=jnp.float32).reshape(batch,width)
        x=x+scale*linear(y,block['out'],c)
        x=x+scale*linear(jax.nn.gelu(linear(norm(x,block['n2']),block['mlp_in'],c)),block['mlp_out'],c)
        updated.append((keys,values))
    return norm(x,p['norm']),tuple(updated)


def board_features(p,stones,players,c):
    """Independent pre-action board encodings; players are absolute colors 1/2."""
    bw=c['board_width'];shape=stones.shape[:-1]
    relative=jnp.where(stones==0,0,jnp.where(stones==players[...,None],1,2))
    x=p['board_embedding'][relative].reshape(-1,c['size'],c['size'],bw)
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[c['dtype']]
    def conv(value,kernel):
        return jax.lax.conv_general_dilated(value.astype(dtype),kernel.astype(dtype),(1,1),'SAME',
            dimension_numbers=('NHWC','HWIO','NHWC'),precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32)
    for block in p['board_blocks']:
        y=conv(jax.nn.gelu(norm(x,block['n1'])),block['w1'])
        y=conv(jax.nn.gelu(norm(y,block['n2'])),block['w2'])
        x=x+y/math.sqrt(c['board_blocks'])
    return norm(x,p['board_norm']).reshape(*shape,c['size']**2,bw)


def head_outputs(p,h,observed_counts,board):
    pooled=jnp.mean(board,axis=-2)
    query=h@p['board_query']
    spatial=jnp.sum(board*query[...,None,:],axis=-1)/math.sqrt(board.shape[-1])+(board@p['board_point'])[...,0]
    play=h@p['play']+jnp.concatenate((spatial,pooled@p['board_pass']),axis=-1)
    value=jnp.tanh(h@p['value']+pooled@p['board_value'])[...,0]
    counts=observed_counts/(1+jnp.sum(observed_counts,-1,keepdims=True))
    # Both encoders and all expert heads are protected from behavior gradients.
    observed_board=jax.lax.stop_gradient(board);observed_pool=jax.lax.stop_gradient(pooled)
    behavior=jnp.tanh(jax.lax.stop_gradient(h)@p['behavior_hidden']+counts@p['behavior_style']+
                      p['behavior_bias']+observed_pool@p['behavior_board_hidden'])
    query=behavior@p['behavior_board_query']
    spatial=jnp.sum(observed_board*query[...,None,:],axis=-1)/math.sqrt(board.shape[-1])+(observed_board@p['behavior_board_point'])[...,0]
    behavior=behavior@p['behavior_out']+jnp.concatenate((spatial,observed_pool@p['behavior_board_pass']),axis=-1)
    return play,behavior,value


def historical_counts(tokens,lengths,actions):
    time=tokens.shape[1];positions=jnp.arange(time)
    observed=jax.nn.one_hot(tokens,actions)*(positions[None,:]<=lengths[:,None])[...,None]
    by_color=[jnp.cumsum(observed*(((positions-1)%2)==color)[None,:,None],axis=1) for color in (0,1)]
    return jnp.stack(by_color,axis=2)  # [game, prefix-position, observed-player, action]


def predictions(p,batch,c):
    h,_=prefill(p,batch['tokens'],batch['lengths'],c);actions=c['size']**2+1
    counts=historical_counts(batch['tokens'],batch['lengths'],actions)
    context=jnp.take_along_axis(counts,(jnp.arange(batch['tokens'].shape[1])%2)[None,:,None,None],axis=2)[:,:,0]
    players=jnp.broadcast_to(1+jnp.arange(batch['tokens'].shape[1])%2,batch['tokens'].shape)
    board=board_features(p,batch['stones'],players,c)
    return head_outputs(p,h,context,board)


def losses(p,batch,c):
    play,behavior,value=predictions(p,batch,c)
    expert_mask=batch['expert_mask'];behavior_mask=batch['behavior_mask'];value_mask=batch['value_mask']
    expert_count=expert_mask.sum();behavior_count=behavior_mask.sum();value_count=value_mask.sum()
    expert=-jnp.sum(expert_mask*jnp.sum(batch['policies']*jax.nn.log_softmax(play),-1))/jnp.maximum(1.,expert_count)
    observed_nll=-jnp.take_along_axis(jax.nn.log_softmax(behavior),batch['observed'][...,None],axis=-1)[...,0]
    observed=jnp.sum(behavior_mask*observed_nll)/jnp.maximum(1.,behavior_count)
    value_loss=jnp.sum(value_mask*(value-batch['values'])**2)/jnp.maximum(1.,value_count)
    entropy=-jnp.sum(batch['policies']*jnp.log(jnp.maximum(batch['policies'],1e-30)),axis=-1)
    metrics={'play_loss':expert,'behavior_loss':observed,'value_loss':value_loss,
             'expert_tokens':expert_count,'behavior_tokens':behavior_count,'value_tokens':value_count,
             'target_entropy':jnp.sum(expert_mask*entropy)/jnp.maximum(1.,expert_count),
             'play_top1':jnp.sum(expert_mask*(jnp.argmax(play,-1)==jnp.argmax(batch['policies'],-1)))/jnp.maximum(1.,expert_count),
             'behavior_top1':jnp.sum(behavior_mask*(jnp.argmax(behavior,-1)==batch['observed']))/jnp.maximum(1.,behavior_count),
             'illegal_probability':jnp.sum(expert_mask*jnp.sum(jax.nn.softmax(play)*(~batch['legal']),-1))/jnp.maximum(1.,expert_count)}
    return expert+observed+value_loss,metrics


