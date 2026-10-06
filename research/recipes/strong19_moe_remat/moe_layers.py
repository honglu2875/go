"""Expert adapters for the cloned spatial/temporal model, without a framework."""
import math
import jax
import jax.numpy as jnp
from gozero import moe


def validate(c):
    if 'moe' not in c:
        return
    m = moe.validate(c['moe'])
    if m.get('scope', 'all') != 'all' or m.get('permutation_vjp', False):
        raise ValueError('This recipe uses all-layer MoE and the original routing transpose')
    widths = [c['encoder_width'] * c['encoder_expansion'], c['mlp_hidden']]
    if c.get('encoder_attention_blocks', 0):
        widths.append(c['encoder_attention_mlp_hidden'])
    if any(w % m['top_k'] for w in widths):
        raise ValueError('Expert widths must exactly divide the dense active width')


def replace(p, prefix, *, seed, layers, width, dense_hidden, gated, bias, c, down_scale=1.):
    """Preserve all non-FFN initial draws; allocate only real expert parameters."""
    if 'moe' not in c:
        return p
    m = c['moe']
    key = jax.random.fold_in(jax.random.key(seed), 802341)
    fn = lambda k: moe.initialize(k, width=width, hidden=dense_hidden // m['top_k'],
                                 experts=m['experts'], gated=gated, bias=bias, down_scale=down_scale)
    weights = jax.vmap(fn)(jax.random.split(key, layers))
    for name in ['up.weight', 'down.weight'] + (['gate.weight'] if gated else []) + (['up.bias', 'down.bias'] if bias else []):
        del p[prefix + name]
    p.update({prefix + 'moe.' + k: v for k, v in weights.items()})
    return p


def forward(p, x, c, *, valid=None):
    weights = {k[4:]: v for k, v in p.items() if k.startswith('moe.')}
    return moe.feed_forward(weights, x, top_k=c['moe']['top_k'],
                            dtype=jnp.bfloat16 if c['dtype'] == 'bfloat16' else jnp.float32,
                            backend=c['moe']['backend'], valid=valid,
                            tiling=tuple(c['moe'].get('tiling', (128,128,128))),
                            remat_activations=c['moe'].get('remat_activations',False))


def loss(statistics, c, *, axis_name=None):
    return moe.router_metrics(statistics, c['moe'], axis_name=axis_name)
