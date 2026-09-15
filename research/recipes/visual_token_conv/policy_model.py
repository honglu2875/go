"""Single policy objective; architecture adapters share exactly the same targets."""
import jax
import jax.numpy as jnp
import katago
import causal


def implementation(c):
    if c['architecture']=='katago_nested_policy': return katago
    if c['architecture']=='causal_visual_policy': return causal
    raise ValueError('Architecture has not yet been qualified')


def initialize(seed,c): return implementation(c).initialize(seed,c)
def parameter_schema(c): return implementation(c).parameter_schema(c)


def logits(p,b,c,*,training=False,axis_name=None):
    if c['architecture']=='causal_visual_policy':
        return causal.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c)
    n,t,h,w,ch=b['spatial'].shape
    out=implementation(c).forward(p,b['spatial'].reshape(n*t,h,w,ch),b['global_features'].reshape(n*t,-1),c,
        training=training,axis_name=axis_name)
    return jax.tree.map(lambda a:a.reshape(n,t,-1),out)


PHASES=((0,16),(16,64),(64,128),(128,256),(256,2048))


def total_metrics(prediction,b,axis_name=None,*,stratify=False):
    mask=jnp.arange(b['actions'].shape[1])[None,:]<b['counts'][:,None]
    logp=jax.nn.log_softmax(jnp.where(b['legal'],prediction,-1e9),-1)
    ce=-jnp.sum(b['policies']*logp,-1)
    entropy=-jnp.sum(b['policies']*jnp.log(jnp.maximum(b['policies'],1e-30)),-1)
    totals={'expert_count':jnp.sum(mask),'expert_ce':jnp.sum(ce*mask),
            'expert_target_entropy':jnp.sum(entropy*mask),
            'expert_entropy':jnp.sum(-jnp.sum(jnp.exp(logp)*logp,-1)*mask),
            'expert_top1':jnp.sum((jnp.argmax(logp,-1)==jnp.argmax(b['policies'],-1))*mask)}
    if stratify:
        time=jnp.arange(b['actions'].shape[1])[None,:]
        for lo,hi in PHASES:
            part=mask&(time>=lo)&(time<hi); prefix=f'phase_{lo}_{hi}'
            totals[prefix+'_count']=jnp.sum(part)
            totals[prefix+'_ce']=jnp.sum(ce*part)
            totals[prefix+'_target_entropy']=jnp.sum(entropy*part)
    if axis_name is not None: totals=jax.tree.map(lambda x:jax.lax.psum(x,axis_name),totals)
    return totals


def averages(totals):
    n=jnp.maximum(totals['expert_count'],1.)
    result={k:v/n if k!='expert_count' else v for k,v in totals.items() if k.startswith('expert_')}
    result['expert_kl']=result['expert_ce']-result['expert_target_entropy']
    for lo,hi in PHASES:
        prefix=f'phase_{lo}_{hi}'
        if prefix+'_count' in totals:
            count=totals[prefix+'_count'];denom=jnp.maximum(count,1.)
            result[prefix+'_count']=count
            result[prefix+'_ce']=totals[prefix+'_ce']/denom
            result[prefix+'_target_entropy']=totals[prefix+'_target_entropy']/denom
            result[prefix+'_kl']=result[prefix+'_ce']-result[prefix+'_target_entropy']
    return result


def losses(p,b,c,*,axis_name=None):
    if c['architecture']=='causal_visual_policy':
        prediction=logits(p,b,c,training=True,axis_name=axis_name)
        m=averages(total_metrics(prediction,b,axis_name))
        return m['expert_ce'],{**m,'expert_positions':m['expert_count']}
    main,helper=logits(p,b,c,training=True,axis_name=axis_name)
    m=averages(total_metrics(main,b,axis_name)); aux=averages(total_metrics(helper,b,axis_name))
    loss=.2*m['expert_ce']+.8*aux['expert_ce']
    return loss,{**m,'helper_ce':aux['expert_ce'],'helper_kl':aux['expert_kl'],
                 'expert_positions':m['expert_count']}
