"""Plain-JAX residual Go CNN with explicitly bounded causal board history.

Each observation is a separate training example. Eight recent six-plane boards,
history-presence flags, and the actions strictly preceding those boards form
spatial channels. There is no access to the action being predicted. Shared
spatial convolutions and global pooling support any configured square size.
This is a baseline architecture, not a port of KataGo's nested bottleneck model.
"""
import math
import jax
import jax.numpy as jnp
if __package__:
    from .config import validate_model
else:
    from config import validate_model


def initialize(seed,c):
    validate_model(c); key=jax.random.key(seed); d=c['width']; layers=c['layers']
    def weight(shape,scale=1.):
        nonlocal key
        key,draw=jax.random.split(key)
        fan=math.prod(shape[:-1])
        return jax.random.normal(draw,shape,jnp.float32)*(scale/math.sqrt(fan))
    blocks=[]
    for _ in range(layers):
        blocks.append({'norm1':jnp.ones(d,jnp.float32),'conv1':weight((3,3,d,d)),
                       'norm2':jnp.ones(d,jnp.float32),'conv2':weight((3,3,d,d),1/math.sqrt(2*layers))})
    return {'stem':weight((3,3,c['history']*9+1,d)),
            'blocks':jax.tree.map(lambda *x:jnp.stack(x),*blocks),
            'norm':jnp.ones(d,jnp.float32),'global':weight((2*d,d)),
            'expert_point':weight((d,1),.1),'expert_pass':weight((d,1),.1),
            'behavior_norm':jnp.ones(d,jnp.float32),'behavior_adapter':weight((d,d)),
            'behavior_point':weight((d,1),.1),'behavior_pass':weight((d,1),.1),
            'value':weight((d,3),.1),'score':weight((d,1),.1),'ownership':weight((d,1),.1)}


def parameter_schema(c):
    shape=jax.eval_shape(lambda:initialize(0,c))
    return [{'path':jax.tree_util.keystr(path),'shape':list(a.shape),'dtype':str(a.dtype),'elements':math.prod(a.shape)}
            for path,a in jax.tree_util.tree_flatten_with_path(shape)[0]]


def history_features(observations,actions,c):
    observations,actions=jnp.asarray(observations),jnp.asarray(actions)
    b,t,size,width,channels=observations.shape
    if size!=width or channels!=6 or actions.shape!=(b,t) or size>c['max_board_size'] or t>c['max_positions']:
        raise ValueError('CNN observation/action shapes exceed contract')
    planes=[]; times=jnp.arange(t); points=jnp.arange(size*size).reshape(size,size)
    for lag in range(c['history']):
        source=times-lag; present=source>=0
        obs=observations[:,jnp.maximum(source,0)]*present[None,:,None,None,None]
        prior=source-1; valid=prior>=0
        act=actions[:,jnp.maximum(prior,0)]
        move=(act[:,:,None,None]==points[None,None])*valid[None,:,None,None]
        passed=(act==size*size)*valid[None,:]
        shape=(b,t,size,size,1)
        planes.extend([obs,jnp.broadcast_to(present[None,:,None,None,None],shape),move[...,None],
                       jnp.broadcast_to(passed[:,:,None,None,None],shape)])
    planes.append(jnp.ones((b,t,size,size,1),observations.dtype))
    return jnp.concatenate(planes,-1).astype(jnp.float32)


def norm(x,scale,c):
    x=x.astype(jnp.float32)
    return x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+c['norm_epsilon'])*scale


def linear(x,w,c):
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    return jnp.matmul(x.astype(dtype),w.astype(dtype),precision=jax.lax.Precision.HIGHEST,
                      preferred_element_type=jnp.float32)


def conv(x,w,c):
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    return jax.lax.conv_general_dilated(x.astype(dtype),w.astype(dtype),(1,1),'SAME',
        dimension_numbers=('NHWC','HWIO','NHWC'),precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32)


def frames(p,x,c,exit_depth=None):
    depth=c['layers'] if exit_depth is None else exit_depth
    if not 1<=depth<=c['layers']: raise ValueError('Invalid CNN depth')
    x=conv(x,p['stem'],c)
    def layer(x,block):
        y=conv(jax.nn.silu(norm(x,block['norm1'],c)),block['conv1'],c)
        y=conv(jax.nn.silu(norm(y,block['norm2'],c)),block['conv2'],c)
        return x+y,None
    apply=jax.checkpoint(layer) if c['rematerialize'] else layer
    x,_=jax.lax.scan(apply,x,jax.tree.map(lambda a:a[:depth],p['blocks']))
    x=norm(x,p['norm'],c)
    global_state=jax.nn.silu(linear(jnp.concatenate([jnp.mean(x,(1,2)),jnp.max(x,(1,2))],-1),p['global'],c))
    local=x+global_state[:,None,None]
    expert=jnp.concatenate([linear(local,p['expert_point'],c)[...,0].reshape(x.shape[0],-1),
                            linear(global_state,p['expert_pass'],c)],-1)
    behavior=local if c['behavior_updates_trunk'] else jax.lax.stop_gradient(local)
    behavior=behavior+jax.nn.silu(linear(norm(behavior,p['behavior_norm'],c),p['behavior_adapter'],c))
    bg=global_state if c['behavior_updates_trunk'] else jax.lax.stop_gradient(global_state)
    behavior=jnp.concatenate([linear(behavior,p['behavior_point'],c)[...,0].reshape(x.shape[0],-1),
                              linear(bg,p['behavior_pass'],c)],-1)
    logits=linear(global_state,p['value'],c); probability=jax.nn.softmax(logits,-1)
    return {'expert_logits':expert,'behavior_logits':behavior,'value_logits':logits,
            'value':probability[:,2]-probability[:,0],
            'score':jnp.tanh(linear(global_state,p['score'],c)[:,0]),
            'ownership_logits':linear(local,p['ownership'],c)[...,0]}


def forward(p,observations,actions,counts,c,*,exit_depth=None,return_exits=()):
    if return_exits: raise ValueError('CNN baseline does not train auxiliary exits')
    x=history_features(observations,actions,c); b,t,size,_,channels=x.shape
    chunk=min(c['microbatch'],b*t); total=math.ceil(b*t/chunk)*chunk
    x=jnp.pad(x.reshape(b*t,size,size,channels),((0,total-b*t),(0,0),(0,0),(0,0)))
    def run(y): return frames(p,y,c,exit_depth)
    run=jax.checkpoint(run) if c['rematerialize'] else run
    result=jax.lax.map(run,x.reshape(-1,chunk,size,size,channels))
    live=jnp.arange(t)[None,:]<counts[:,None]
    def restore(a):
        a=a.reshape(total,*a.shape[2:])[:b*t].reshape(b,t,*a.shape[2:])
        return jnp.where(live.reshape(b,t,*([1]*(a.ndim-2))),a,0.)
    return jax.tree.map(restore,result)
