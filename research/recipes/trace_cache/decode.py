"""One compiled paired trace with advisory place-only board dynamics."""
import jax
import jax.numpy as jnp
from model import prefill,append,historical_counts,head_outputs,board_features


def decode(p,tokens,lengths,episodes,key,root_stones,c,*,horizon,samples,coupling='independent',oracle_behavior=False,root_legal=None,board_update="place"):
    """Both roles and k samples run inside one scan/HLO. No exact device rules.

    Counter-based own Gumbels depend on game, episode and absolute real ply,
    not packet boundaries or selected samples. Rust applies exact legal masks
    to own logits and these same draws, then filters continuations by the past.
    """
    if board_update not in ('place','hold'):raise ValueError('Unknown advisory state update')
    games=tokens.shape[0];actions=c['size']**2+1
    h,cache=prefill(p,tokens,lengths,c)
    counts=historical_counts(tokens,lengths,actions)[jnp.arange(games),lengths]
    h=h[jnp.arange(games),lengths]
    def expand(a):return jnp.broadcast_to(a[:,None,None],(games,2,samples,*a.shape[1:])).reshape(games*2*samples,*a.shape[1:])
    h=expand(h);counts=expand(counts);cache=jax.tree.map(expand,cache);stones=expand(root_stones)
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
        used_stones=stones
        if board_update=="place":
            index=jnp.minimum(chosen,actions-2)
            old=stones[jnp.arange(stones.shape[0]),index]
            color=jnp.where(chosen<actions-1,player+1,old).astype(jnp.uint8)
            stones=stones.at[jnp.arange(stones.shape[0]),index].set(color)
        return (h,cache,counts,stones),(chosen,play,used_stones)
    _,(chosen,play,used_stones)=jax.lax.scan(step,(h,cache,counts,stones),jnp.arange(horizon))
    return (chosen.T.reshape(games,2,samples,horizon),
            jnp.transpose(play,(1,0,2)).reshape(games,2,samples,horizon,actions),own_noise,
            jnp.transpose(used_stones,(1,0,2)).reshape(games,2,samples,horizon,actions-1))
