"""Persistent device KV with native-selected branches and actual-move commits."""
import jax
import jax.numpy as jnp
from model import prefill,append,historical_counts,head_outputs,board_features


def empty_cache(c,games,samples):
    shape=(games,2*samples,c['max_tokens'],c['heads'],c['width']//c['heads'])
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[c['dtype']]
    return {'kv':tuple((jnp.zeros(shape,dtype),jnp.zeros(shape,dtype))for _ in range(c['blocks'])),
            'counts':jnp.zeros((games,2,c['size']**2+1),jnp.float32)}


def rebuild_cache(p,tokens,lengths,c,*,samples):
    """Cold reconstruction from canonical histories; no speculative provenance."""
    _,kv=prefill(p,tokens,lengths,c)
    expand=lambda x:jnp.broadcast_to(x[:,None],(x.shape[0],2*samples,*x.shape[1:]))
    counts=historical_counts(tokens,lengths,c['size']**2+1)[jnp.arange(tokens.shape[0]),lengths]
    return {'kv':jax.tree.map(expand,kv),'counts':counts}


def decode_cached(p,retained,controls,lengths,episodes,key,root_stones,c,*,horizon,samples,
                  coupling='independent',oracle_behavior=False,root_legal=None):
    """One graph gathers validated KV, appends the actual root token and decodes.

    Only the four policy/resolver arrays are fetched by the host. The returned
    carry stays on device. Entries at or beyond the actual root token are
    erased before append; inactive game carries remain bitwise unchanged.
    """
    games=lengths.shape[0];actions=c['size']**2+1
    active=controls[:,0].astype(jnp.bool_);reset=controls[:,1].astype(jnp.bool_)
    positions=jnp.where(active,lengths,0);selector=controls[:,2];last_tokens=controls[:,3]
    past=(jnp.arange(c['max_tokens'])[None,:]<positions[:,None])&(~reset[:,None])
    select=lambda x:jnp.where(past[:,:,None,None],x[jnp.arange(games),selector],0)
    root_h,root_kv=append(p,jax.tree.map(select,retained['kv']),last_tokens,positions,c)
    committed=controls[:,5:5+horizon];count=controls[:,4]
    valid=jnp.arange(horizon)[None,:]<count[:,None]
    player=(positions[:,None]-count[:,None]+jnp.arange(horizon)[None,:])%2
    increments=jnp.einsum('gdc,gda->gca',jax.nn.one_hot(player,2),
        jax.nn.one_hot(committed,actions)*valid[...,None])
    root_counts=jnp.where(reset[:,None,None],0,retained['counts']+increments)
    def expand(x):return jnp.broadcast_to(x[:,None,None],(games,2,samples,*x.shape[1:])).reshape(games*2*samples,*x.shape[1:])
    h=expand(root_h);counts=expand(root_counts);cache=jax.tree.map(expand,root_kv);stones=expand(root_stones)
    positions=expand(positions);views=jnp.broadcast_to(jnp.arange(2)[None,:,None],(games,2,samples)).reshape(-1)
    legal=None if root_legal is None else expand(root_legal)
    game_keys=jax.vmap(lambda game,episode:jax.random.fold_in(jax.random.fold_in(key,game),episode))(jnp.arange(games,dtype=jnp.uint32),episodes)
    plies=lengths[:,None]+jnp.arange(horizon)[None,:]
    keys=jax.vmap(lambda k,ps:jax.vmap(lambda t:jax.random.fold_in(k,t))(ps))(game_keys,plies)
    base_noise=jax.vmap(jax.vmap(lambda k:jax.random.gumbel(k,(actions,),jnp.float32)))(keys)
    own_noise=base_noise*c['expert_temperature']
    behavior_temperature=c['expert_temperature']if oracle_behavior else c['behavior_temperature']
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
        h,cache,counts,stones=carry;player=(positions+depth)%2
        context=counts[jnp.arange(h.shape[0]),player]
        board=board_features(p,stones,player+1,c)
        play,behavior,_=head_outputs(p,h,context,board)
        if oracle_behavior:behavior=play
        logits=jnp.where((player==views)[:,None],play,behavior)
        if legal is not None:logits=jnp.where((depth!=0)|legal,logits,-jnp.inf)
        noise=jnp.where((player==views)[:,None],own_expanded[:,depth],opponent_noise[:,depth])
        chosen=jnp.argmax(logits+noise,-1).astype(jnp.int32)
        counts=counts.at[jnp.arange(h.shape[0]),player,chosen].add(1.)
        h,cache=jax.lax.cond(depth+1<horizon,lambda pair:append(p,pair[1],chosen,positions+depth+1,c),lambda pair:pair,(h,cache))
        used_stones=stones;index=jnp.minimum(chosen,actions-2);old=stones[jnp.arange(stones.shape[0]),index]
        color=jnp.where(chosen<actions-1,player+1,old).astype(jnp.uint8)
        stones=stones.at[jnp.arange(stones.shape[0]),index].set(color)
        return (h,cache,counts,stones),(chosen,play,used_stones)
    (_,cache,_,_),(chosen,play,used_stones)=jax.lax.scan(step,(h,cache,counts,stones),jnp.arange(horizon))
    def retain(new,old):
        new=new.reshape(games,2*samples,*new.shape[1:])
        return jnp.where(active.reshape(games,*([1]*(new.ndim-1))),new,old)
    retained={'kv':jax.tree.map(retain,cache,retained['kv']),
        'counts':jnp.where(active[:,None,None],root_counts,retained['counts'])}
    return (chosen.T.reshape(games,2,samples,horizon),
        jnp.transpose(play,(1,0,2)).reshape(games,2,samples,horizon,actions),own_noise,
        jnp.transpose(used_stones,(1,0,2)).reshape(games,2,samples,horizon,actions-1),retained)
