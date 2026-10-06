"""Exact CPU check of the dynamic-mask diagnostic, including empty shards."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
import adamw
import learner
from qualify_padding import fixture, compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == 'cpu' and jax.device_count() == 4
    started = time.time()
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    opt = dict(learning_rate=.001, end_learning_rate=.0003, warmup_steps=2, horizon_steps=8,
               beta1=.9, beta2=.95, epsilon=1e-8, weight_decay=.01, max_grad_norm=1.)
    checks = []
    for dtype in ('float32', 'bfloat16'):
        c, p, batch = fixture(dtype)
        dense_batch = {**batch, 'encoder_counts': jnp.full_like(batch['counts'], batch['actions'].shape[1])}
        skip_batch = {**batch, 'encoder_counts': batch['counts']}
        original = jax.jit(learner.step(c, opt, value_weight=.7, chunk_frames=2, mesh=mesh))
        conditional = jax.jit(learner.step(c, opt, value_weight=.7, chunk_frames=2, mesh=mesh, skip_padding=True))
        executable = conditional.lower(p, adamw.initialize(p), dense_batch).compile()
        states = [(p, adamw.initialize(p)) for _ in range(3)]
        for update in (1, 2):
            values = [original(*states[0], dense_batch), executable(*states[1], dense_batch),
                      executable(*states[2], skip_batch)]
            checks.append(dict(dtype=dtype, update=update,
                               conditional_vs_original=compare(values[0], values[1], rtol=0., atol=0.),
                               skipping_vs_dense=compare(values[1], values[2], rtol=0., atol=0.)))
            states = [v[:2] for v in values]
            print(json.dumps(checks[-1]), flush=True)
        jax.clear_caches()
    result = dict(status='passed', kind='dynamic_padding_control_cpu', seconds=time.time() - started,
                  checks=checks, sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in Path(__file__).parent.glob('*.py')})
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
