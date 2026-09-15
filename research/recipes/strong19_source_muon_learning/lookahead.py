"""Functional slow weights and clocks matching the pinned KataGo trainer."""
import math

import jax
import jax.numpy as jnp


def validate(config):
    if set(config)!={'k','alpha'}:raise ValueError('Explicit Lookahead configuration required')
    k,alpha=config['k'],config['alpha']
    if k is None and alpha is None:return
    if (type(k) is not int or not 1<=k<=1000000 or type(alpha) not in (int,float)
            or not math.isfinite(alpha) or not 0<alpha<1):
        raise ValueError('Use a positive period and alpha in (0,1), or disable both')


def initialize(params,config):
    validate(config)
    if not params or any(p.dtype!=jnp.float32 for p in params.values()):
        raise ValueError('Expected float32 master parameters')
    return dict(slow=jax.tree.map(jnp.copy,params) if config['k'] is not None else {},counter=jnp.int32(0))


def validate_state(params,state,config):
    validate(config)
    if (set(state)!={'slow','counter'} or state['counter'].shape!=() or state['counter'].dtype!=jnp.int32
            or set(state['slow'])!=(set(params) if config['k'] is not None else set())):
        raise ValueError('Lookahead state coverage differs')
    for key,value in state['slow'].items():
        if value.shape!=params[key].shape or value.dtype!=jnp.float32:
            raise ValueError('Slow parameter schema differs')


def after_step(params,state,config,*,accepted):
    """Apply after the fast optimizer; rejected updates must leave its p unchanged."""
    validate_state(params,state,config)
    if config['k'] is None:
        return params,state,dict(synchronized=jnp.bool_(False))
    counter=state['counter']+jnp.asarray(accepted,jnp.int32)
    sync=jnp.asarray(accepted,jnp.bool_)&(counter>=config['k'])
    slow=jax.tree.map(lambda s,p:jnp.where(sync,s+config['alpha']*(p-s),s),state['slow'],params)
    fast=jax.tree.map(lambda s,p:jnp.where(sync,s,p),slow,params)
    return fast,dict(slow=slow,counter=jnp.where(sync,jnp.int32(0),counter)),dict(synchronized=sync)


def begin_subepoch(state):
    """Source resets the clock here without discarding unsynchronized fast p."""
    return dict(slow=state['slow'],counter=jnp.int32(0))


def end_epoch(params,state,config):
    """Source flushes after the entire subepoch loop; moments remain untouched."""
    validate_state(params,state,config)
    # Call at the host epoch boundary, outside the donated training executable.
    # The source copies into distinct fast storage. Returning the slow tree
    # directly aliases both arguments and makes the next donation invalid.
    return (jax.tree.map(jnp.copy,state['slow']) if config['k'] is not None else params),state
