"""Per-game sufficient statistics for paired fixed-validation comparisons."""
import jax
import jax.numpy as jnp
from policy_model import PHASES

def totals(logits,batch):
    live=jnp.arange(batch['actions'].shape[1])[None,:]<batch['counts'][:,None]
    logp=jax.nn.log_softmax(jnp.where(batch['legal'],logits,-1e9),-1)
    ce=-jnp.sum(batch['policies']*logp,-1)
    entropy=-jnp.sum(batch['policies']*jnp.log(jnp.maximum(batch['policies'],1e-30)),-1)
    result={'count':jnp.sum(live,-1),'ce':jnp.sum(ce*live,-1),'target_entropy':jnp.sum(entropy*live,-1),
            'top1':jnp.sum((jnp.argmax(logp,-1)==jnp.argmax(batch['policies'],-1))*live,-1)}
    turn=jnp.arange(batch['actions'].shape[1])[None,:]
    for lo,hi in PHASES:
        mask=live&(turn>=lo)&(turn<hi);prefix=f'phase_{lo}_{hi}'
        result[prefix+'_count']=jnp.sum(mask,-1)
        result[prefix+'_ce']=jnp.sum(ce*mask,-1)
        result[prefix+'_target_entropy']=jnp.sum(entropy*mask,-1)
    return result
