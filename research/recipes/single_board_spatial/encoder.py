"""One board embedding from a residual CNN with weights shared across passes.

ConvNeXt-inspired channel mixing and affine LayerNorm; 3x3 depthwise kernels.
The residual stream has the declared execution dtype at every block boundary.
Checkpointing recomputes activations and never detaches a refinement pass.
"""
import math
import jax
import jax.numpy as jnp


def dtype(c): return jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32


def initialize(seed,c):
    key=jax.random.key(seed);p={};w=c['encoder_width'];f=w*c['encoder_expansion'];k=c['encoder_blocks'];d=c['width'];b=c['connector_channels']
    def weight(name,shape,fan):
        nonlocal key
        key,draw=jax.random.split(key);p[name]=jax.random.normal(draw,shape,jnp.float32)/math.sqrt(fan)
    weight('stem.weight',(3,3,22,w),9*22)
    p['stem.bias']=jnp.zeros(w);p['stem.norm.scale']=jnp.ones(w);p['stem.norm.bias']=jnp.zeros(w)
    weight('blocks.depthwise.weight',(k,3,3,1,w),9)
    weight('blocks.up.weight',(k,w,f),w);weight('blocks.down.weight',(k,f,w),f)
    for name,n in [('depthwise.bias',w),('norm.bias',w),('up.bias',f),('down.bias',w)]:p['blocks.'+name]=jnp.zeros((k,n))
    p['blocks.norm.scale']=jnp.ones((k,w));p['blocks.gamma']=jnp.full((k,w),c['encoder_layer_scale'],jnp.float32)
    weight('compress.weight',(w,b),w);p['compress.bias']=jnp.zeros(b)
    weight('flat.weight',(c['max_board_size']**2*b,d),c['max_board_size']**2*b);p['flat.bias']=jnp.zeros(d)
    p['token.norm.scale']=jnp.ones(d);p['token.norm.bias']=jnp.zeros(d)
    weight('global.weight',(19,d),19);p['type']=jax.random.normal(jax.random.fold_in(key,1),(d,),jnp.float32)*.02
    return p


def norm(x,scale,bias,c):
    x=x.astype(jnp.float32);x=x-jnp.mean(x,-1,keepdims=True)
    return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+c['norm_epsilon'])*scale+bias


def linear(x,w,c):
    return jnp.matmul(x.astype(dtype(c)),w.astype(dtype(c)),preferred_element_type=jnp.float32)


def convolution(x,w,c,groups=1):
    return jax.lax.conv_general_dilated(x.astype(dtype(c)),w.astype(dtype(c)),
        window_strides=(1,1),padding='SAME',dimension_numbers=('NHWC','HWIO','NHWC'),
        feature_group_count=groups,preferred_element_type=jnp.float32)


def block(x,p,c):
    y=convolution(x,p['depthwise.weight'],c,c['encoder_width'])+p['depthwise.bias']
    y=norm(y,p['norm.scale'],p['norm.bias'],c)
    y=jax.nn.gelu(linear(y,p['up.weight'],c)+p['up.bias'])
    y=linear(y,p['down.weight'],c)+p['down.bias']
    return (x.astype(jnp.float32)+p['gamma']*y).astype(dtype(c))


def spatial(p,s,c):
    batch,time,h,w,ch=s.shape
    x=s.reshape(batch*time,h,w,ch)
    x=convolution(x,p['stem.weight'],c)+p['stem.bias']
    x=jax.nn.gelu(norm(x,p['stem.norm.scale'],p['stem.norm.bias'],c)).astype(dtype(c))
    blocks={k[len('blocks.'):]:v for k,v in p.items() if k.startswith('blocks.')}
    fn=(jax.checkpoint(lambda x,p:block(x,p,c)) if c['encoder_rematerialize'] else lambda x,p:block(x,p,c))
    def scan(x,p):return fn(x,p),None
    for _ in range(c['encoder_passes']):x,_=jax.lax.scan(scan,x,blocks)
    return x.reshape(batch,time,h,w,c['encoder_width'])


def encode(p,s,g,c,*,with_features=False):
    if s.shape[-3:]!=(c['max_board_size'],c['max_board_size'],22) or g.shape!=(*s.shape[:2],19):
        raise ValueError('Fixed-board connector input differs from the model contract')
    features=spatial(p,s,c)
    x=features
    x=linear(x,p['compress.weight'],c)+p['compress.bias']
    x=linear(x.reshape(*s.shape[:2],-1),p['flat.weight'],c)+p['flat.bias']
    x=norm(x,p['token.norm.scale'],p['token.norm.bias'],c)
    token=x+linear(g,p['global.weight'],c)+p['type']
    return (token,features) if with_features else token


def flops(c,size):
    if size!=c['max_board_size']:raise ValueError('Fixed-board connector')
    w=c['encoder_width'];f=w*c['encoder_expansion'];b=c['connector_channels'];d=c['width']
    block_weights=9*w+2*w*f
    return 2*size**2*(9*22*w+c['encoder_blocks']*c['encoder_passes']*block_weights+w*b)+2*size**2*b*d+2*19*d
