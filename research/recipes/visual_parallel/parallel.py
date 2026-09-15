"""Local-host tensor parallelism for the unchanged plain-JAX decoder.

The residual stream is replicated within a model group. Attention Q and MLP
expansions are column-sharded; their output projections are row-sharded. K/V
projection weights stay replicated because the trained packed [K,V] matrix does
not have a contiguous per-head sharding. Cached K/V heads are model-sharded.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
if __package__:
    from . import model
else:
    import model


class Layout:
    def __init__(self, params, c, tensor_parallel):
        devices = jax.local_devices()
        if (type(tensor_parallel) is not int or tensor_parallel < 1 or len(devices) % tensor_parallel
                or any(c[k] % tensor_parallel for k in ('heads', 'kv_heads', 'mlp_hidden'))):
            raise ValueError('Tensor group must divide local devices, attention heads and MLP width')
        self.c = {**c, 'attention_backend': 'xla'}
        self.mesh = Mesh(np.asarray(devices).reshape(-1, tensor_parallel), ('data', 'model'))
        self.tensor_parallel = tensor_parallel
        self.batched = NamedSharding(self.mesh, P('data'))
        self.replicated = NamedSharding(self.mesh, P())
        specs = jax.tree.map(lambda _: P(), params)
        for block in specs['blocks']:
            for key in ('q', 'gate', 'up'):
                block[key] = P(None, 'model')
            for key in ('out', 'down'):
                block[key] = P('model', None)
        self.param_shardings = jax.tree.map(lambda s: NamedSharding(self.mesh, s), specs,
                                           is_leaf=lambda x: isinstance(x, P))
        self.params = jax.device_put(params, self.param_shardings)
        self.cache_shardings = {'keys': (NamedSharding(self.mesh, P('data', None, 'model')),) * c['layers'],
            'values': (NamedSharding(self.mesh, P('data', None, 'model')),) * c['layers'],
            'lengths': self.batched, 'valid': self.batched, 'network_version': self.replicated}

    def put(self, x):
        return jax.device_put(x, self.batched)

    def prefill(self, observations, actions, counts, *, capacity, version):
        def fn(p, o, a, n):
            return model.forward(p, o, a, n, self.c, with_cache=True, cache_positions=capacity, network_version=version)
        return jax.jit(fn, in_shardings=(self.param_shardings, self.batched, self.batched, self.batched),
            out_shardings=(self.batched, self.cache_shardings))(
                self.params, self.put(observations), self.put(actions), self.put(counts))

    def continuation(self, *, version, donate=True):
        def fn(p, cache, o, a, n, prefix_lengths):
            cache = {**cache, 'lengths': prefix_lengths}
            return model.score_continuation(p, cache, o, a, n, self.c, network_version=version)
        return jax.jit(fn, in_shardings=(self.param_shardings, self.cache_shardings,
            self.batched, self.batched, self.batched, self.batched),
            out_shardings=(self.batched, self.cache_shardings), donate_argnums=(1,) if donate else ())
