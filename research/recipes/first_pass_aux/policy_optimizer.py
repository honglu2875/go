"""Functional AdamW with semantic decay and recorded parameter-group changes."""
import jax
import jax.numpy as jnp
import katago

def initialize(params):
    return {'first':jax.tree.map(jnp.zeros_like,params),'second':jax.tree.map(jnp.zeros_like,params),
            'step':jnp.asarray(0,jnp.int32)}

def group(name):
    if name.startswith('encoder.') or name.startswith('token_types.') or name=='readout':return 'input'
    if name.startswith('head.'):return 'head'
    if name.startswith(('intermediate_','norm_intermediate_')): return 'helper'
    if name.startswith(('policy_head.','norm_trunkfinal.')): return 'head'
    if name.startswith(('conv_spatial.','linear_global.')): return 'input'
    return 'trunk'

def decay(name):
    # Stacking adds a dimension to normalization vectors; shape is NOT the rule.
    return name.endswith('.weight')

def apply_gradient(params,state,grads,loss,*,learning_rate,beta1=.9,beta2=.95,epsilon=1e-8,
                   weight_decay=.01,max_grad_norm=1.,architecture='katago_nested_policy'):
    raw_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in grads.values()))
    if architecture=='katago_nested_policy': grads=katago.repvgg_gradient(grads)
    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in grads.values()))
    scale=jnp.minimum(1.,max_grad_norm/jnp.maximum(norm,1e-12)); groups=sorted(set(map(group,params)))
    metrics={'grad_norm':norm,'raw_grad_norm':raw_norm,'clip_scale':scale}
    for label in groups:
        metrics['gradient_norm_'+label]=jnp.sqrt(sum(jnp.sum(g*g) for k,g in grads.items() if group(k)==label))
        metrics['parameter_norm_'+label]=jnp.sqrt(sum(jnp.sum(p*p) for k,p in params.items() if group(k)==label))
    grads=jax.tree.map(lambda g:g*scale,grads)
    first=jax.tree.map(lambda a,g:beta1*a+(1-beta1)*g,state['first'],grads)
    second=jax.tree.map(lambda a,g:beta2*a+(1-beta2)*g*g,state['second'],grads)
    step=state['step']+1
    deltas={k:learning_rate*(first[k]/(1-beta1**step)/(jnp.sqrt(second[k]/(1-beta2**step))+epsilon)
             +(weight_decay*p if decay(k) else 0.)) for k,p in params.items()}
    candidate={k:p-deltas[k] for k,p in params.items()}
    for label in groups:
        metrics['update_norm_'+label]=jnp.sqrt(sum(jnp.sum(d*d) for k,d in deltas.items() if group(k)==label))
    finite=jnp.isfinite(loss)&jnp.isfinite(norm)
    finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((candidate,first,second))]))
    params=jax.tree.map(lambda new,old:jnp.where(finite,new,old),candidate,params)
    state=jax.tree.map(lambda new,old:jnp.where(finite,new,old),{'first':first,'second':second,'step':step},state)
    return params,state,{**metrics,'accepted':finite,'loss':loss}
