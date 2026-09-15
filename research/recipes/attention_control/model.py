"""Functional CNN/attention controls with common heads, losses and optimizer."""
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
        if config['architecture']=='cnn':
            blocks.append({'w1':weight((3,3,width,width),math.sqrt(2)), 'n1':norm(),
                           'w2':weight((3,3,width,width),math.sqrt(2)), 'n2':norm()})
        else:
            hidden=width*config['mlp_ratio']
            blocks.append({'qkv':weight((width,3*width)), 'projection':weight((width,width)),
                           'n1':norm(), 'n2':norm(), 'mlp_in':weight((width,hidden),math.sqrt(2)),
                           'mlp_in_bias':jnp.zeros(hidden,jnp.float32),
                           'mlp_out':weight((hidden,width)), 'mlp_out_bias':jnp.zeros(width,jnp.float32)})
    params={'stem':weight((3,3,input_channels,width),math.sqrt(2)), 'stem_norm':norm(), 'blocks':blocks,
            'policy':weight((1,1,width,1),0.1),'policy_bias':jnp.zeros((),jnp.float32),
            'pass':weight((width,1),0.1),'pass_bias':jnp.zeros((1,),jnp.float32),
            'value_hidden':weight((width,config['value_hidden']),math.sqrt(2)),
            'value_bias':jnp.zeros(config['value_hidden'],jnp.float32),
            'value_out':weight((config['value_hidden'],1),0.1),'value_out_bias':jnp.zeros(1,jnp.float32)}
    # Append the draw after all shared parameters so the paired control has the
    # same trunk/policy/value initialization and exactly the same parameter tree.
    params['ownership']=weight((1,1,width,1),0.1)
    params['ownership_bias']=jnp.zeros((),jnp.float32)
    params['score']=weight((width,1),0.1)
    params['score_bias']=jnp.zeros(1,jnp.float32)
    if config['architecture']=='attention':
        # One shared table ties all eight rotations/reflections of each offset.
        # Zero initialization makes no assumption about a preferred distance.
        size=config['max_board_size']
        params['relative_bias']=jnp.zeros((size*(size+1)//2,config['heads']),jnp.float32)
        params['final_norm']=norm()
    return params


def relative_indices(size):
    points=jnp.arange(size*size)
    dy=jnp.abs(points[:,None]//size-points[None,:]//size)
    dx=jnp.abs(points[:,None]%size-points[None,:]%size)
    high=jnp.maximum(dy,dx);low=jnp.minimum(dy,dx)
    return high*(high+1)//2+low


def layer_norm(x,p):
    x=x.astype(jnp.float32)
    centered=x-jnp.mean(x,axis=-1,keepdims=True)
    return centered*jax.lax.rsqrt(jnp.mean(jnp.square(centered),axis=-1,keepdims=True)+1e-5)*p['scale']+p['bias']


def linear(x,w,dtype):
    return jnp.matmul(x.astype(dtype),w.astype(dtype),precision=jax.lax.Precision.HIGHEST,
                      preferred_element_type=jnp.float32)


def attention(x,block,bias,heads,dtype):
    batch,tokens,width=x.shape
    qkv=linear(x,block['qkv'],dtype).reshape(batch,tokens,3,heads,width//heads)
    q,k,v=(qkv[:,:,i].astype(dtype) for i in range(3))
    scores=jnp.einsum('bthd,bshd->bhts',q,k,precision=jax.lax.Precision.HIGHEST,
                      preferred_element_type=jnp.float32)/math.sqrt(width//heads)
    weights=jax.nn.softmax(scores+bias[None,...],axis=-1).astype(dtype)
    values=jnp.einsum('bhts,bshd->bthd',weights,v,precision=jax.lax.Precision.HIGHEST,
                      preferred_element_type=jnp.float32).reshape(batch,tokens,width)
    return linear(values,block['projection'],dtype)


def apply(params,features,config,*,with_ownership=False,with_targets=False):
    if features.ndim!=4 or features.shape[1]!=features.shape[2] or features.shape[1]>config['max_board_size']:
        raise ValueError('Expected square NHWC board within configured size limit')
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
    if config['architecture']=='cnn':
        for block in params['blocks']:
            y=jax.nn.relu(norm(conv(x,block['w1']),block['n1']))
            x=jax.nn.relu(x+norm(conv(y,block['w2']),block['n2']))
    else:
        shape=x.shape
        x=x.reshape(shape[0],-1,shape[-1]).astype(jnp.float32)
        bias=jnp.moveaxis(params['relative_bias'][relative_indices(shape[1])],-1,0)
        residual_scale=1/math.sqrt(max(1,len(params['blocks'])))
        for block in params['blocks']:
            x=x+residual_scale*attention(layer_norm(x,block['n1']),block,bias,config['heads'],dtype)
            y=linear(layer_norm(x,block['n2']),block['mlp_in'],dtype)+block['mlp_in_bias']
            y=linear(jax.nn.gelu(y,approximate=True),block['mlp_out'],dtype)+block['mlp_out_bias']
            x=x+residual_scale*y
        x=layer_norm(x,params['final_norm']).reshape(shape).astype(dtype)
    spatial=conv(x,params['policy']).astype(jnp.float32)[...,0]+params['policy_bias']
    pooled=jnp.mean(x.astype(jnp.float32),axis=(1,2))
    pass_logit=pooled@params['pass']+params['pass_bias']
    logits=jnp.concatenate([spatial.reshape((features.shape[0],-1)),pass_logit],axis=1)
    hidden=jax.nn.relu(pooled@params['value_hidden']+params['value_bias'])
    value=jnp.tanh(hidden@params['value_out']+params['value_out_bias'])[:,0]
    score_value=jnp.tanh(pooled@params['score']+params['score_bias'])[:,0]
    factor=config['score_utility_factor']
    utility=(value+factor*score_value)/(1+factor)
    if with_targets or with_ownership:
        ownership=conv(x,params['ownership']).astype(jnp.float32)[...,0]+params['ownership_bias']
        if with_targets:return logits,value,ownership,score_value
        return logits,utility,ownership
    return logits,utility


def score_targets(features,ownership,scale):
    # Both ownership and the signed komi plane use the current player's
    # perspective. Labels are terminal; features can be from any earlier ply.
    area=features.shape[1]*features.shape[2]
    margin=jnp.sum(ownership,axis=(1,2))+features[:,0,0,-3]*area
    return jnp.arctan(margin/(scale*math.sqrt(area)))*(2/math.pi)


def augment(features,policies,ownership,symmetries):
    size=features.shape[1]
    def single(x,pi,owner,symmetry):
        grid=pi[:-1].reshape((size,size))
        branches=[]
        for symmetry_id in range(8):
            def transform(args,s=symmetry_id):
                f,p,o=args
                f=jnp.rot90(f,s%4,axes=(0,1));p=jnp.rot90(p,s%4,axes=(0,1));o=jnp.rot90(o,s%4)
                if s>=4:f=jnp.flip(f,axis=1);p=jnp.flip(p,axis=1);o=jnp.flip(o,axis=1)
                return f,p,o
            branches.append(transform)
        x,grid,owner=jax.lax.switch(symmetry,branches,(x,grid,owner))
        return x,jnp.concatenate([grid.reshape(-1),pi[-1:]]),owner
    return jax.vmap(single)(features,policies,ownership,symmetries)


def losses(params,features,policies,outcomes,ownership,model_config,learner_config):
    logits,values,owner_logits,score_value=apply(params,features,model_config,with_targets=True)
    legal=jnp.concatenate([features[...,-1].reshape((features.shape[0],-1))>0.5,
                           jnp.ones((features.shape[0],1),dtype=bool)],axis=1)
    log_probs=jax.nn.log_softmax(jnp.where(legal,logits,-1e9),axis=-1)
    policy_loss=-jnp.mean(jnp.sum(policies*log_probs,axis=-1))
    value_loss=jnp.mean(jnp.square(values-outcomes))
    # Soft binary labels allow neutral intersections. Mean over board and batch
    # keeps this coefficient independent of board area. Logits are pretanh:
    # sigmoid(2*logit) gives (tanh(logit)+1)/2, as in KataGo's ownership loss.
    owner_targets=(ownership+1)*0.5
    ownership_loss=jnp.mean(jax.nn.softplus(2*owner_logits)-owner_targets*(2*owner_logits))
    score_loss=jnp.mean(jnp.square(score_value-score_targets(features,ownership,model_config['score_scale'])))
    penalty=learner_config['l2']*sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))
    metrics={'policy_loss':policy_loss,'value_loss':value_loss,'l2_penalty':penalty,
             'policy_entropy':-jnp.mean(jnp.sum(jnp.exp(log_probs)*log_probs,axis=-1)),
             'ownership_loss':ownership_loss,'ownership_mse':jnp.mean(jnp.square(jnp.tanh(owner_logits)-ownership)),
             'score_loss':score_loss}
    return policy_loss+value_loss+penalty+learner_config['ownership_weight']*ownership_loss+learner_config['score_weight']*score_loss,metrics


def update(params,velocity,features,policies,outcomes,ownership,key,model_config,learner_config):
    if learner_config['augment_symmetries']:
        symmetries=jax.random.randint(key,(features.shape[0],),0,8)
        features,policies,ownership=augment(features,policies,ownership,symmetries)
    (loss,metrics),grads=jax.value_and_grad(losses,has_aux=True)(params,features,policies,outcomes,ownership,model_config,learner_config)
    norm=jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    scale=jnp.minimum(1.0,learner_config['max_grad_norm']/(norm+1e-8))
    velocity=jax.tree.map(lambda m,g:learner_config['momentum']*m+scale*g,velocity,grads)
    params=jax.tree.map(lambda p,m:p-learner_config['learning_rate']*m,params,velocity)
    return params,velocity,{'loss':loss,'grad_norm':norm,**metrics}
