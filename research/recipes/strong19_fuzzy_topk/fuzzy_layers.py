"""Temporal-only adapter. Encoder and all non-FFN initial draws are unchanged."""
import math
import jax
import jax.numpy as jnp
from gozero.fuzzy_topk import choicewise, diagnostics


def validate(c):
    if 'fuzzy' not in c:return
    f=c['fuzzy']
    if 'moe' in c or set(f)!={'groups','selected','implementation','bias'}:
        raise ValueError('Fuzzy and routed MoE are separate ablations')
    if any(type(f[k]) is not int or f[k]<1 for k in ('groups','selected')) or f['groups']>32 or f['groups']*f['selected']>32768:
        raise ValueError('Invalid grouped feature dictionary')
    if f['implementation']!='choicewise' or f['bias'] is not True:
        raise ValueError('Only the explicit biased choicewise contract is qualified')


def replace(p,seed,c):
    validate(c);f=c['fuzzy'];d=c['width'];h=f['groups']*f['selected'];layers=c['layers']
    for name in ('gate','up','down'):del p['blocks.'+name+'.weight']
    key=jax.random.fold_in(jax.random.key(seed),721893)
    up,down=jax.random.split(key)
    p['blocks.fuzzy.up.weight']=jax.random.normal(up,(layers,d,h),jnp.float32)/math.sqrt(d)
    p['blocks.fuzzy.down.weight']=jax.random.normal(down,(layers,h,d),jnp.float32)/math.sqrt(2*layers*h)
    p['blocks.fuzzy.up.bias']=jnp.zeros((layers,h),jnp.float32)
    p['blocks.fuzzy.down.bias']=jnp.zeros((layers,d),jnp.float32)


def forward(p,x,c,valid=None):
    f=c['fuzzy']
    return choicewise(x,p['fuzzy.up.weight'],p['fuzzy.up.bias'],p['fuzzy.down.weight'],p['fuzzy.down.bias'],
        groups=f['groups'],dtype=jnp.bfloat16 if c['dtype']=='bfloat16' else jnp.float32,valid=valid)


def metrics(stats,c,axis_name=None):
    return diagnostics(stats,groups=c['fuzzy']['groups'],selected=c['fuzzy']['selected'],axis_name=axis_name)
