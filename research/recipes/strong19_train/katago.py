"""Pure JAX KataGo nested-bottleneck current-policy architecture.

Arithmetic port of pinned KataGo model_pytorch.py (see KATAGO_LICENSE.md).
Training is deliberately policy-only. The helper uses the same policy target
and global masked batch statistics; it is never called during inference.
"""
import math
import jax
import jax.numpy as jnp


def validate(c):
    fields=set('architecture width mid_width gpool_width policy_width layers dtype rematerialize microbatch max_board_size max_positions norm_epsilon'.split())
    if set(c) != fields or c['architecture'] != 'katago_nested_policy':
        raise ValueError('Unknown KataGo architecture contract')
    for key,lo,hi in [('width',8,2048),('mid_width',4,1024),('gpool_width',1,512),('policy_width',4,512),
                      ('layers',1,80),('microbatch',1,512),('max_board_size',1,52),('max_positions',1,2048)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi: raise ValueError('Invalid '+key)
    if c['gpool_width'] >= c['mid_width'] or c['dtype'] not in ('float32','bfloat16'):
        raise ValueError('Invalid KataGo channel/dtype configuration')
    if type(c['rematerialize']) is not bool or c['norm_epsilon'] != 1e-4:
        raise ValueError('KataGo normalization/rematerialization contract differs')
    return c


def initialize(seed,c):
    validate(c); key=jax.random.key(seed); result={}
    gain=math.sqrt(2.210277)
    def add(name,shape,*,scale=None,identity=False,fanin=None,stack=0,repvgg=False):
        nonlocal key
        if scale is None: value=jnp.zeros(shape,jnp.float32)
        else:
            key,k=jax.random.split(key)
            fan=math.prod(shape[stack:-1]) if fanin is None else fanin
            std=scale*(1 if identity else gain)/math.sqrt(fan)/.87962566103423978
            value=jax.random.truncated_normal(k,-2.,2.,shape,jnp.float32)*std*(.8 if repvgg else 1.)
            if repvgg:
                key,k=jax.random.split(key)
                center_shape=shape[:stack]+shape[-2:]
                bonus=jax.random.truncated_normal(k,-2.,2.,center_shape,jnp.float32)
                bonus*=.6*gain/math.sqrt(shape[-2])/.87962566103423978
                value=value.at[(slice(None),)*stack+(1,1)].add(bonus)
        result[name]=value
    def norm(name,channels,lead=()):
        for part in ('gamma','beta'): add(name+'.'+part,(*lead,channels))
    def convunit(name,ci,co,k,lead=(),pool=0):
        norm(name+'.norm',ci,lead)
        if pool:
            add(name+'.convpool.conv1r.weight',(*lead,3,3,ci,co),scale=.8,stack=len(lead))
            add(name+'.convpool.conv1g.weight',(*lead,3,3,ci,pool),scale=math.sqrt(.6),stack=len(lead))
            norm(name+'.convpool.normg',pool,lead)
            add(name+'.convpool.linear_g.weight',(*lead,3*pool,co),scale=math.sqrt(.6),stack=len(lead))
        else:
            add(name+'.conv.weight',(*lead,k,k,ci,co),scale=1.,stack=len(lead),repvgg=k==3)
    def block(name,pool,lead=()):
        w,m,g=c['width'],c['mid_width'],c['gpool_width'] if pool else 0
        convunit(name+'.normactconvp',w,m,1,lead)
        for i in range(2):
            ng=g if i==0 else 0
            convunit(name+f'.blockstack.{i}.normactconv1',m,m-ng,3,lead,ng)
            convunit(name+f'.blockstack.{i}.normactconv2',m-ng,m,3,lead)
        convunit(name+'.normactconvq',m,w,1,lead)
    w=c['width']; h=c['policy_width']; cycles=c['layers']//3
    add('conv_spatial.weight',(3,3,22,w),scale=.8)
    add('linear_global.weight',(19,w),scale=.6)
    if cycles:
        for group in ('a','b','g'): block('cycles.'+group,group=='g',(cycles,))
    for i in range(c['layers']%3): block('tail.'+str(i),False)
    add('norm_trunkfinal.beta',(w,))
    norm('norm_intermediate_trunkfinal',w)
    for name in ('policy_head','intermediate_policy_head'):
        add(name+'.conv1p.weight',(1,1,w,h),scale=.8)
        add(name+'.conv1g.weight',(1,1,w,h),scale=1.)
        add(name+'.biasg.beta',(h,)); add(name+'.bias2.beta',(h,))
        add(name+'.linear_g.weight',(3*h,h),scale=.6)
        add(name+'.linear_pass.weight',(3*h,h),scale=1.)
        add(name+'.linear_pass.bias',(h,),scale=.2,fanin=3*h)
        add(name+'.linear_pass2.weight',(h,1),scale=.3,identity=True)
        add(name+'.conv2p.weight',(1,1,h,1),scale=.3,identity=True)
    return result


def parameter_schema(c):
    shape=jax.eval_shape(lambda:initialize(0,c))
    return [{'path':k,'shape':list(v.shape),'dtype':str(v.dtype),'elements':math.prod(v.shape)} for k,v in sorted(shape.items())]


def inference_parameter_count(c):
    return sum(x['elements'] for x in parameter_schema(c) if not x['path'].startswith(('intermediate_','norm_intermediate_')))


def select(p,prefix):
    return {k[len(prefix)+1:]:v for k,v in p.items() if k.startswith(prefix+'.')}


def mish(x): return x*jnp.tanh(jax.nn.softplus(x))


def linear(x,w,c):
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    return jnp.matmul(x.astype(dtype),w.astype(dtype),preferred_element_type=jnp.float32)


def conv(x,w,c):
    dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32
    return jax.lax.conv_general_dilated(x.astype(dtype),w.astype(dtype),(1,1),'SAME',
        dimension_numbers=('NHWC','HWIO','NHWC'),preferred_element_type=jnp.float32)


def norm(x,mask,p,prefix,scale=1.):
    return (x*((p[prefix+'.gamma']+1.)*scale)+p[prefix+'.beta'])*mask


def gpool(x,mask):
    n=jnp.maximum(jnp.sum(mask,axis=(1,2)),1.)
    mean=jnp.sum(x,axis=(1,2))/n
    maximum=jnp.max(x+mask-1.,axis=(1,2))
    return jnp.concatenate([mean,mean*(jnp.sqrt(n)-14.)/10.,maximum],axis=-1)


def unit(x,mask,p,name,c,scale=1.):
    y=mish(norm(x,mask,p,name+'.norm',scale))
    if name+'.conv.weight' in p: return conv(y,p[name+'.conv.weight'],c)
    prefix=name+'.convpool'
    r=conv(y,p[prefix+'.conv1r.weight'],c)
    g=conv(y,p[prefix+'.conv1g.weight'],c)
    g=mish(norm(g,mask,p,prefix+'.normg'))
    g=linear(gpool(g,mask),p[prefix+'.linear_g.weight'],c)
    return r+g[:,None,None,:]


def block(x,mask,p,c,index):
    y=unit(x,mask,p,'normactconvp',c,jax.lax.rsqrt(index+1.))
    for i in range(2):
        z=unit(y,mask,p,f'blockstack.{i}.normactconv1',c,1./math.sqrt(i+1))
        z=unit(z,mask,p,f'blockstack.{i}.normactconv2',c)
        y=y+z
    return x+unit(y,mask,p,'normactconvq',c,1./math.sqrt(3.))


def trunk(p,spatial,globals,c):
    mask=spatial[...,:1]
    x=conv(spatial,p['conv_spatial.weight'],c)+linear(globals,p['linear_global.weight'],c)[:,None,None,:]
    cycles=c['layers']//3
    if cycles:
        def step(x,data):
            i,blocks=data
            for j,group in enumerate(('a','b','g')):
                x=block(x,mask,select(blocks,group),c,3*i+j)
            return x,None
        fn=jax.checkpoint(step) if c['rematerialize'] else step
        x,_=jax.lax.scan(fn,x,(jnp.arange(cycles,dtype=jnp.float32),select(p,'cycles')))
    for i in range(c['layers']%3):
        fn=lambda x: block(x,mask,select(p,'tail.'+str(i)),c,jnp.asarray(3*cycles+i,jnp.float32))
        x=(jax.checkpoint(fn) if c['rematerialize'] else fn)(x)
    return x


def trunk_batched(p,spatial,globals,c):
    """Bound trunk activation storage without changing helper BN populations."""
    n=spatial.shape[0]; mb=min(n,c['microbatch']); extra=(-n)%mb
    if n <= mb: return trunk(p,spatial,globals,c)
    s=jnp.pad(spatial,((0,extra),(0,0),(0,0),(0,0))).reshape(-1,mb,*spatial.shape[1:])
    g=jnp.pad(globals,((0,extra),(0,0))).reshape(-1,mb,globals.shape[-1])
    def fn(sg): return trunk(p,sg[0],sg[1],c)
    out=jax.lax.map(jax.checkpoint(fn) if c['rematerialize'] else fn,(s,g))
    return out.reshape(-1,*out.shape[2:])[:n]


def policy(x,mask,p,c):
    # KataGo disables AMP for output heads. Keep this true even in BF16 trunk runs.
    fp={**c,'dtype':'float32'}
    xp=conv(x,p['conv1p.weight'],fp)
    xg=conv(x,p['conv1g.weight'],fp)
    xg=mish((xg+p['biasg.beta'])*mask)
    pooled=gpool(xg,mask)
    passed=linear(mish(linear(pooled,p['linear_pass.weight'],fp)+p['linear_pass.bias']),p['linear_pass2.weight'],fp)
    xp=xp+linear(pooled,p['linear_g.weight'],fp)[:,None,None,:]
    xp=mish((xp+p['bias2.beta'])*mask)
    logits=conv(xp,p['conv2p.weight'],fp)-(1.-mask)*5000.
    return jnp.concatenate([logits.reshape(x.shape[0],-1),passed],-1)


def forward(p,spatial,globals,c,*,training=False,axis_name=None,with_features=False):
    """Inputs [N,H,W,22], [N,19]; no future action or target is an input."""
    x=trunk_batched(p,spatial,globals,c); mask=spatial[...,:1]
    y=mish((x/math.sqrt(c['layers']+1.)+p['norm_trunkfinal.beta'])*mask)
    main=policy(y,mask,select(p,'policy_head'),c)
    if not training: return dict(policy=main,features=y) if with_features else main
    total=lambda x: jax.lax.psum(x,axis_name) if axis_name is not None else x
    n=jnp.maximum(total(jnp.sum(mask)),1.)
    mean=total(jnp.sum(x*mask,axis=(0,1,2)))/n
    variance=total(jnp.sum(((x-mean)*mask)**2,axis=(0,1,2)))/n
    z=(x-mean)*jax.lax.rsqrt(variance+c['norm_epsilon'])
    z=mish(norm(z,mask,p,'norm_intermediate_trunkfinal'))
    helper=policy(z,mask,select(p,'intermediate_policy_head'),c)
    return dict(policy=main,aux_policy=helper,features=y,aux_features=z) if with_features else (main,helper)


def repvgg_gradient(gradient):
    """Official rvglr center multiplier before clipping and optimizer moments."""
    def apply(name,g):
        if '.normactconv' in name and name.endswith('.conv.weight') and g.shape[-4:-2]==(3,3):
            return g.at[(slice(None),)*(g.ndim-4)+(1,1)].multiply(2.)
        return g
    return {k:apply(k,g) for k,g in gradient.items()}


def reference_names(p,c):
    """Map stacked JAX leaves to complete official state_dict names."""
    output={}
    for key,value in p.items():
        if key.startswith('cycles.'):
            group=key.split('.')[1]; rest=key.split('.',2)[2]; j={'a':0,'b':1,'g':2}[group]
            output[key]=[(f'blocks.{3*i+j}.{rest}',i) for i in range(c['layers']//3)]
        elif key.startswith('tail.'):
            _,i,rest=key.split('.',2); output[key]=[(f'blocks.{3*(c["layers"]//3)+int(i)}.{rest}',None)]
        else: output[key]=[(key,None)]
    return output
