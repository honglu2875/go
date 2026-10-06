"""One transient query reads the current grid; temporal caches stay unchanged."""
import math
import jax
import jax.numpy as jnp


def initialize(seed,c):
    width=c['width'];spatial=c['encoder_width'];dim=c['policy_refinement_dim']
    key=jax.random.fold_in(jax.random.key(seed),637927)
    result={}
    for name,shape in [('q',(width,dim)),('k',(spatial,dim)),('v',(spatial,dim)),
                       ('up',(dim,2*dim)),('down',(2*dim,dim))]:
        key,draw=jax.random.split(key)
        result['head.refine.'+name+'.weight']=jax.random.normal(draw,shape,jnp.float32)/math.sqrt(shape[0])
    # The new residual path preserves the exact initial policy.
    result['head.refine.out.weight']=jnp.zeros((dim,width),jnp.float32)
    result['head.refine.norm.scale']=jnp.ones(dim,jnp.float32)
    return result


def refine(p,h,features,c):
    if features is None:raise ValueError('Refinement requires the current spatial grid')
    dim=c['policy_refinement_dim']
    def project(x,name):return x.astype(jnp.float32)@p['head.refine.'+name+'.weight']
    grid=features.reshape(*h.shape[:-1],-1,c['encoder_width']).astype(jnp.float32)
    grid=grid*jax.lax.rsqrt(jnp.mean(grid*grid,-1,keepdims=True)+c['norm_epsilon'])
    query=project(h,'q');keys=project(grid,'k');values=project(grid,'v')
    scores=jnp.einsum('...d,...nd->...n',query,keys)/math.sqrt(dim)
    weights=jax.nn.softmax(scores,-1)
    query=query+jnp.einsum('...n,...nd->...d',weights,values)
    normalized=query*jax.lax.rsqrt(jnp.mean(query*query,-1,keepdims=True)+c['norm_epsilon'])*p['head.refine.norm.scale']
    query=query+project(jax.nn.silu(project(normalized,'up')),'down')
    return h+project(query,'out')


def flops(c,points):
    dim=c.get('policy_refinement_dim',0)
    return 2*(c['width']*dim+2*points*c['encoder_width']*dim+
              2*points*dim+4*dim*dim+dim*c['width']) if dim else 0
