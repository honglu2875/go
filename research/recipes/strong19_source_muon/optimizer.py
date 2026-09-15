"""Source-grouped Muon/AuxAdam, dynamic schedule scalars and Lookahead."""
import jax
import jax.numpy as jnp
import katago
import lookahead
import muon
import muon_groups
import source_runtime


def validate_config(config):
    if set(config)!={'source_runtime','gradient_units','horizon_steps'} or config['gradient_units']!='actual_global_position_sum':
        raise ValueError('Explicit source runtime and actual gradient position units required')
    source_runtime.validate(config['source_runtime'],config['horizon_steps'])


def model_norms(params):
    specs=muon_groups.cnn(params)
    return {group:jnp.sqrt(sum(jnp.sum(p*p) for key,p in params.items() if specs[key]['group']==group)) for group in ('input','normal')}


def initialize(params,config,hyperparameters):
    # Host boundary: slow storage must remain distinct from fast p at donation.
    validate_config(config)
    return dict(**muon.initialize(params,muon_groups.cnn(params)),
        lookahead=lookahead.initialize(params,config['source_runtime']['lookahead']),hyperparameters=hyperparameters)


def apply_gradient(params,state,mean_gradients,loss,positions,config):
    validate_config(config);specs=muon_groups.cnn(params)
    if set(state)!={'first','second','step','lookahead','hyperparameters'}:raise ValueError('Incomplete source optimizer state')
    h=state['hyperparameters']
    if h.shape!=(13,) or h.dtype!=jnp.float32:raise ValueError('Source scalar schema differs')
    rates={k:h[i] for i,k in enumerate(source_runtime.GROUPS)}
    decays={k:h[6+i] for i,k in enumerate(source_runtime.GROUPS)}
    raw_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in mean_gradients.values()))
    summed=jax.tree.map(lambda g:g*positions,mean_gradients);gradient=katago.repvgg_gradient(summed)
    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in gradient.values()))
    scale=jnp.minimum(1.,h[-1]/(norm+1e-6));gradient=jax.tree.map(lambda g:g*scale,gradient)
    eligible=jnp.isfinite(positions)&(positions>0)&(state['step']<config['horizon_steps'])&jnp.isfinite(norm)&jnp.all(jnp.isfinite(h))&(h[-1]>0)
    fast_state={k:state[k] for k in ('first','second','step')}
    p,s,metrics=muon.apply(params,fast_state,gradient,jnp.where(eligible,loss,jnp.nan),specs,rates,decays)
    p,slow,extra=lookahead.after_step(p,state['lookahead'],config['source_runtime']['lookahead'],accepted=metrics['accepted'])
    state=dict(**s,lookahead=slow,hyperparameters=h)
    return p,state,dict(**metrics,loss=loss,eligible=eligible,raw_mean_gradient_norm=raw_norm,
        source_sum_gradient_norm=norm,clip_scale=scale,normal_learning_rate=rates['normal'],
        normal_weight_decay=decays['normal'],source_sum_gradient_clip_cap=h[-1],positions=positions,
        schedule_reference_batch_positions=jnp.float32(config['source_runtime']['reference_batch_positions']),
        lookahead_synchronized=extra['synchronized'],lookahead_counter=slow['counter'])
