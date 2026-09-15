"""Recipe-owned functional CNN. No framework modules or mutable model state."""
import math
import jax
import jax.numpy as jnp


def initialize(seed, input_channels, config):
    width=config['width'];groups=config['groups']
    if width<=0 or groups<=0 or width%groups or config['blocks']<0 or config['value_hidden']<=0:
        raise ValueError('Invalid model dimensions')
    key=jax.random.key(seed)
    def weight(shape,scale=1.0):
        nonlocal key
        key,draw=jax.random.split(key)
        return jax.random.normal(draw,shape,jnp.float32)*(scale/math.sqrt(math.prod(shape[:-1])))
    def norm():
        return {'scale':jnp.ones(width,jnp.float32),'bias':jnp.zeros(width,jnp.float32)}
    blocks=[]
    for _ in range(config['blocks']):
        blocks.append({'w1':weight((3,3,width,width),math.sqrt(2)), 'n1':norm(),
                       'w2':weight((3,3,width,width),math.sqrt(2)), 'n2':norm()})
    return {'stem':weight((3,3,input_channels,width),math.sqrt(2)), 'stem_norm':norm(), 'blocks':blocks,
            'policy':weight((1,1,width,1),0.1),'policy_bias':jnp.zeros((),jnp.float32),
            'pass':weight((width,1),0.1),'pass_bias':jnp.zeros((1,),jnp.float32),
            'value_hidden':weight((width,config['value_hidden']),math.sqrt(2)),
            'value_bias':jnp.zeros(config['value_hidden'],jnp.float32),
            'value_out':weight((config['value_hidden'],1),0.1),'value_out_bias':jnp.zeros(1,jnp.float32)}


def apply(params,features,config):
    dtype={'float32':jnp.float32,'bfloat16':jnp.bfloat16}[config['dtype']]
    def conv(x,w):
        return jax.lax.conv_general_dilated(x,w.astype(dtype),(1,1),'SAME',
                 dimension_numbers=('NHWC','HWIO','NHWC'),precision=jax.lax.Precision.HIGHEST)
    def norm(x,p):
        shape=x.shape;grouped=x.astype(jnp.float32).reshape((*shape[:-1],config['groups'],shape[-1]//config['groups']))
        mean=jnp.mean(grouped,axis=(1,2,4),keepdims=True)
        variance=jnp.mean(jnp.square(grouped-mean),axis=(1,2,4),keepdims=True)
        normalized=((grouped-mean)*jax.lax.rsqrt(variance+1e-5)).reshape(shape)
        return (normalized*p['scale']+p['bias']).astype(dtype)
    x=jax.nn.relu(norm(conv(features.astype(dtype),params['stem']),params['stem_norm']))
    for block in params['blocks']:
        y=jax.nn.relu(norm(conv(x,block['w1']),block['n1']))
        x=jax.nn.relu(x+norm(conv(y,block['w2']),block['n2']))
    spatial=conv(x,params['policy']).astype(jnp.float32)[...,0]+params['policy_bias']
    pooled=jnp.mean(x.astype(jnp.float32),axis=(1,2))
    pass_logit=pooled@params['pass']+params['pass_bias']
    logits=jnp.concatenate([spatial.reshape((features.shape[0],-1)),pass_logit],axis=1)
    hidden=jax.nn.relu(pooled@params['value_hidden']+params['value_bias'])
    value=jnp.tanh(hidden@params['value_out']+params['value_out_bias'])[:,0]
    return logits,value


def augment(features,policies,symmetries):
    size=features.shape[1]
    def single(x,pi,symmetry):
        grid=pi[:-1].reshape((size,size))
        branches=[]
        for symmetry_id in range(8):
            def transform(args,s=symmetry_id):
                f,p=args
                f=jnp.rot90(f,s%4,axes=(0,1));p=jnp.rot90(p,s%4,axes=(0,1))
                if s>=4:f=jnp.flip(f,axis=1);p=jnp.flip(p,axis=1)
                return f,p
            branches.append(transform)
        x,grid=jax.lax.switch(symmetry,branches,(x,grid))
        return x,jnp.concatenate([grid.reshape(-1),pi[-1:]])
    return jax.vmap(single)(features,policies,symmetries)


def losses(params,features,policies,outcomes,model_config,learner_config):
    logits,values=apply(params,features,model_config)
    legal=jnp.concatenate([features[...,-1].reshape((features.shape[0],-1))>0.5,
                           jnp.ones((features.shape[0],1),dtype=bool)],axis=1)
    log_probs=jax.nn.log_softmax(jnp.where(legal,logits,-1e9),axis=-1)
    policy_loss=-jnp.mean(jnp.sum(policies*log_probs,axis=-1))
    value_loss=jnp.mean(jnp.square(values-outcomes))
    penalty=learner_config['l2']*sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))
    metrics={'policy_loss':policy_loss,'value_loss':value_loss,'l2_penalty':penalty,
             'policy_entropy':-jnp.mean(jnp.sum(jnp.exp(log_probs)*log_probs,axis=-1))}
    return policy_loss+value_loss+penalty,metrics


def update(params,velocity,features,policies,outcomes,key,model_config,learner_config):
    if learner_config['augment_symmetries']:
        symmetries=jax.random.randint(key,(features.shape[0],),0,8)
        features,policies=augment(features,policies,symmetries)
    (loss,metrics),grads=jax.value_and_grad(losses,has_aux=True)(params,features,policies,outcomes,model_config,learner_config)
    norm=jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    scale=jnp.minimum(1.0,learner_config['max_grad_norm']/(norm+1e-8))
    velocity=jax.tree.map(lambda m,g:learner_config['momentum']*m+scale*g,velocity,grads)
    params=jax.tree.map(lambda p,m:p-learner_config['learning_rate']*m,params,velocity)
    return params,velocity,{'loss':loss,'grad_norm':norm,**metrics}
