"""Pure JAX value readouts; the CNN path follows pinned KataGo value arithmetic.

Only the three-logit value branch is represented here. Score/ownership and other
auxiliary branches have no stored teacher labels in this corpus and are separate
tasks. Head integration and the joint training objective remain unregistered.
"""
import math

import jax
import jax.numpy as jnp


def mish(x):return x*jax.nn.tanh(jax.nn.softplus(x))


def initialize(seed,*,kind,width=768,hidden=256,spatial_channels=256):
    if kind not in ('cnn','temporal') or any(type(x) is not int or x<1 for x in (width,hidden,spatial_channels)):
        raise ValueError('Invalid value-head dimensions')
    key=jax.random.key(seed);p={}
    def weight(name,shape,*,gain=math.sqrt(2.210277),scale=1.,fan=None):
        nonlocal key
        key,k=jax.random.split(key)
        fan=math.prod(shape[:-1]) if fan is None else fan
        p[name]=jax.random.truncated_normal(k,-2.,2.,shape,jnp.float32)*scale*gain/math.sqrt(fan)/.87962566103423978
    if kind=='cnn':
        weight('conv1.weight',(1,1,width,spatial_channels))
        p['bias1.beta']=jnp.zeros(spatial_channels,jnp.float32)
        input_width=3*spatial_channels
    else:input_width=width
    weight('linear2.weight',(input_width,hidden))
    weight('linear2.bias',(hidden,),scale=.2,fan=input_width)
    weight('linear_valuehead.weight',(hidden,3),gain=1.)
    weight('linear_valuehead.bias',(3,),gain=1.,scale=.2,fan=hidden)
    return p


def pool(x,mask):
    count=jnp.sum(mask,axis=(1,2));safe=jnp.maximum(count,1.)
    mean=jnp.sum(x,axis=(1,2))/safe
    offset=jnp.sqrt(safe)-14.
    return jnp.concatenate((mean,mean*offset/10.,mean*(offset**2/100.-.1)),axis=-1)


def finish(p,pooled):
    y=mish(jnp.matmul(pooled,p['linear2.weight'])+p['linear2.bias'])
    return jnp.matmul(y,p['linear_valuehead.weight'])+p['linear_valuehead.bias']


def cnn(p,x,mask):
    if x.ndim!=4 or mask.shape!=(*x.shape[:-1],1):raise ValueError('Expected NHWC trunk and mask')
    x=x.astype(jnp.float32);mask=mask.astype(jnp.float32)
    y=jax.lax.conv_general_dilated(x,p['conv1.weight'],(1,1),'SAME',
                                  dimension_numbers=('NHWC','HWIO','NHWC'),preferred_element_type=jnp.float32)
    y=mish((y+p['bias1.beta'])*mask)
    logits=finish(p,pool(y,mask))
    # Entirely padded game frames do not exist in official board-only batches.
    # Define their logits as finite zeros, with no gradient contribution.
    live=jnp.sum(mask,axis=(1,2,3))>0
    return jnp.where(live[:,None],logits,0.)


def temporal(p,normalized_latent):
    """Read the same normalized causal latent used by the policy projection."""
    return finish(p,normalized_latent.astype(jnp.float32))


def signed_value(logits):
    if logits.shape[-1]!=3:raise ValueError('Expected the three value logits')
    probability=jax.nn.softmax(logits.astype(jnp.float32),axis=-1)
    return probability[...,0]-probability[...,1]


PHASES=((0,16),(16,64),(64,128),(128,256),(256,2048))


def value_totals(prediction,b,*,axis_name=None,stratify=False):
    if prediction.shape!=b['values'].shape or prediction.ndim!=2 or b['counts'].shape!=(prediction.shape[0],):
        raise ValueError('Value/history shapes differ')
    live=jnp.arange(prediction.shape[1])[None,:]<b['counts'][:,None]
    # Mask before subtraction; padded target bytes must not affect loss/gradient.
    target=jnp.where(live,b['values'],0.);pred=jnp.where(live,prediction,0.)
    error=pred-target
    def totals(weight,prefix):
        return {prefix+'_count':jnp.sum(weight),prefix+'_squared_error':jnp.sum(error**2*weight),
                prefix+'_absolute_error':jnp.sum(jnp.abs(error)*weight),
                prefix+'_target_sum':jnp.sum(target*weight),prefix+'_target_squared':jnp.sum(target**2*weight),
                prefix+'_prediction_sum':jnp.sum(pred*weight)}
    out=totals(live,'value')
    if stratify:
        out.update(totals(jnp.where(live,b['family_weights'][:,None],0.),'value_family'))
        for opponent in range(8):
            out.update(totals(live&(b['opponent'][:,None]==opponent),f'value_opponent_{opponent}'))
        time=jnp.arange(prediction.shape[1])[None,:]
        for lo,hi in PHASES:out.update(totals(live&(time>=lo)&(time<hi),f'value_phase_{lo}_{hi}'))
    return jax.tree.map(lambda x:jax.lax.psum(x,axis_name),out) if axis_name is not None else out


def value_averages(totals):
    """Normalize raw population totals, including fractional family mass.

    The fitted-constant error uses this population's own target mean: it is an
    in-population diagnostic lower bound for a constant predictor, not a model
    fitted on the training split or a held-out generalization result.
    """
    out={}
    for name,n in totals.items():
        if not name.endswith('_count'):continue
        prefix=name[:-6];denom=jnp.where(n>0,n,1.)
        mean=totals[prefix+'_target_sum']/denom
        out.update({name:n,prefix+'_mse':totals[prefix+'_squared_error']/denom,
                    prefix+'_mae':totals[prefix+'_absolute_error']/denom,
                    prefix+'_mean_target':mean,prefix+'_mean_prediction':totals[prefix+'_prediction_sum']/denom,
                    prefix+'_zero_predictor_mse':totals[prefix+'_target_squared']/denom,
                    prefix+'_fitted_constant_mse':jnp.maximum(totals[prefix+'_target_squared']/denom-mean**2,0.)})
    return out
