"""Plain-JAX causal transformer with separate play and observed-behavior heads.

The expert head distills saved MCTS distributions; the behavior head predicts
observations from a separately sampled population. Rust remains authoritative
for legality and game termination during decoding.
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
    return {'tokens':weight((actions+2,width),0.2),'positions':weight((c['max_tokens'],width),0.2),
            'player':weight((2,width),0.1),'global':weight((2,width),0.1),'blocks':blocks,'norm':norm(),
            'play':weight((width,actions),0.1),'value':weight((width,1),0.1),
            'behavior_hidden':weight((width,width)),'behavior_style':weight((actions,width)),
            'behavior_bias':jnp.zeros(width,jnp.float32),'behavior_out':weight((width,actions),0.1)}


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


def head_outputs(p,h,observed_counts):
    counts=observed_counts/(1+jnp.sum(observed_counts,-1,keepdims=True))
    # Behavioral gradients cannot alter the expert trunk in this control.
    behavior=jnp.tanh(jax.lax.stop_gradient(h)@p['behavior_hidden']+counts@p['behavior_style']+p['behavior_bias'])
    return h@p['play'],behavior@p['behavior_out'],jnp.tanh(h@p['value'])[...,0]


def historical_counts(tokens,lengths,actions):
    time=tokens.shape[1];positions=jnp.arange(time)
    observed=jax.nn.one_hot(tokens,actions)*(positions[None,:]<=lengths[:,None])[...,None]
    by_color=[jnp.cumsum(observed*(((positions-1)%2)==color)[None,:,None],axis=1) for color in (0,1)]
    return jnp.stack(by_color,axis=2)  # [game, prefix-position, observed-player, action]


def losses(p,batch,c):
    h,_=prefill(p,batch['tokens'],batch['lengths'],c);actions=c['size']**2+1
    counts=historical_counts(batch['tokens'],batch['lengths'],actions)
    context=jnp.take_along_axis(counts,(jnp.arange(batch['tokens'].shape[1])%2)[None,:,None,None],axis=2)[:,:,0]
    play,behavior,value=head_outputs(p,h,context)
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


def decode(p,tokens,lengths,episodes,key,c,*,horizon,samples,coupling='independent',oracle_behavior=False,root_legal=None):
    """Both roles and k samples run inside one scan/HLO. No exact device rules.

    Counter-based own Gumbels depend on game, episode and absolute real ply,
    not packet boundaries or selected samples. Rust applies exact legal masks
    to own logits and these same draws, then filters continuations by the past.
    """
    games=tokens.shape[0];actions=c['size']**2+1
    h,cache=prefill(p,tokens,lengths,c)
    counts=historical_counts(tokens,lengths,actions)[jnp.arange(games),lengths]
    h=h[jnp.arange(games),lengths]
    def expand(a):return jnp.broadcast_to(a[:,None,None],(games,2,samples,*a.shape[1:])).reshape(games*2*samples,*a.shape[1:])
    h=expand(h);counts=expand(counts);cache=jax.tree.map(expand,cache)
    positions=expand(lengths);views=jnp.broadcast_to(jnp.arange(2)[None,:,None],(games,2,samples)).reshape(-1)
    legal=None if root_legal is None else expand(root_legal)
    game_keys=jax.vmap(lambda game,episode:jax.random.fold_in(jax.random.fold_in(key,game),episode))(jnp.arange(games,dtype=jnp.uint32),episodes)
    plies=lengths[:,None]+jnp.arange(horizon)[None,:]
    keys=jax.vmap(lambda k,ps:jax.vmap(lambda t:jax.random.fold_in(k,t))(ps))(game_keys,plies)
    base_noise=jax.vmap(jax.vmap(lambda k:jax.random.gumbel(k,(actions,),jnp.float32)))(keys)
    own_noise=base_noise*c['expert_temperature']
    behavior_temperature=c['expert_temperature'] if oracle_behavior else c['behavior_temperature']
    if coupling=='shared':
        opponent_noise=jnp.broadcast_to(base_noise[:,None,None],(games,2,samples,horizon,actions))*behavior_temperature
    elif coupling=='independent':
        branch_ids=jnp.arange(2*samples,dtype=jnp.uint32)
        def branch_noise(game_key,ps):
            return jax.vmap(lambda branch:jax.vmap(lambda ply:jax.random.gumbel(
                jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(game_key,0xBEEF17),branch),ply),
                (actions,),jnp.float32))(ps))(branch_ids)
        opponent_noise=jax.vmap(branch_noise)(game_keys,plies).reshape(games,2,samples,horizon,actions)*behavior_temperature
    else:raise ValueError('Unknown sampling coupling')
    own_expanded=jnp.broadcast_to(own_noise[:,None,None],(games,2,samples,horizon,actions)).reshape(-1,horizon,actions)
    opponent_noise=opponent_noise.reshape(-1,horizon,actions)
    def step(carry,depth):
        h,cache,counts=carry;player=(positions+depth)%2
        context=counts[jnp.arange(h.shape[0]),player]
        play,behavior,_=head_outputs(p,h,context)
        if oracle_behavior:behavior=play
        logits=jnp.where((player==views)[:,None],play,behavior)
        if legal is not None:logits=jnp.where((depth!=0)|legal,logits,-jnp.inf)
        noise=jnp.where((player==views)[:,None],own_expanded[:,depth],opponent_noise[:,depth])
        chosen=jnp.argmax(logits+noise,-1).astype(jnp.int32)
        counts=counts.at[jnp.arange(h.shape[0]),player,chosen].add(1.)
        h,cache=jax.lax.cond(depth+1<horizon,lambda pair:append(p,pair[1],chosen,positions+depth+1,c),lambda pair:pair,(h,cache))
        return (h,cache,counts),(chosen,play)
    _,(chosen,play)=jax.lax.scan(step,(h,cache,counts),jnp.arange(horizon))
    return (chosen.T.reshape(games,2,samples,horizon),
            jnp.transpose(play,(1,0,2)).reshape(games,2,samples,horizon,actions),own_noise)
