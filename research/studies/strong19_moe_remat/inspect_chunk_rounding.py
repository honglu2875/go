"""Measure gradient-only and optimizer drift when increasing encoder chunks.

Unlike the initial equivalence report, gradient norms here exclude metrics
and token counters, which would otherwise dominate a mixed-tree relative norm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'research/recipes/strong19_moe_remat'), str(ROOT / 'packages/gozero/src')]
import numpy as np
import jax
import joint, learner, adamw
from diagnose import compare
from qualify_remat import gate
from gozero.snapshots import canonical_json


def error(a, b):
    squared = reference = dot = actual = 0.
    maximum = 0.
    different = elements = 0
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        squared += float(np.sum((x - y) ** 2)); reference += float(np.sum(y * y))
        actual += float(np.sum(x * x)); dot += float(np.sum(x * y))
        maximum = max(maximum, float(np.max(np.abs(x - y))))
        different += int(np.count_nonzero(x != y)); elements += x.size
    return dict(relative_l2=(squared / max(reference, 1e-30)) ** .5,
                cosine=dot / max((reference * actual) ** .5, 1e-30),
                max_abs=maximum, different=different, elements=elements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    plan = json.loads((STUDY / 'system-proposal-config-001.json').read_text())
    base = plan['reference_config']
    c = dict(base['model'], width=16, encoder_width=16, layers=2, mlp_hidden=32,
             heads=2, kv_heads=1, max_board_size=3, max_positions=8, encoder_blocks=2,
             encoder_attention_blocks=1, encoder_attention_heads=2, encoder_attention_mlp_hidden=16,
             connector_channels=2, policy_context_dim=4, encoder_expansion=2, attention_backend='xla')
    c['moe'] = {**c['moe'], 'backend': 'ragged'}
    rng = np.random.default_rng(48)
    batch = dict(spatial=rng.normal(size=(4, 6, 3, 3, 22)).astype('float32'),
                 global_features=rng.normal(size=(4, 6, 19)).astype('float32'),
                 actions=rng.integers(0, 10, (4, 6), dtype=np.int32), counts=np.array([6, 3, 2, 0], np.int32),
                 policies=np.full((4, 6, 10), .1, np.float32), legal=np.ones((4, 6, 10), bool),
                 values=rng.uniform(-1, 1, (4, 6)).astype('float32'))
    opt = {k: v for k, v in base['learner'].items() if k not in ('games_per_host', 'augmentation')}
    opt['horizon_steps'] = 512
    rows = []
    for precision in ('float32', 'bfloat16'):
        configs = [{**c, 'dtype': precision, 'moe': {**c['moe'], 'remat_activations': flag}} for flag in (False, True)]
        params = joint.initialize(44, configs[0], dict(hidden=8, spatial_channels=8))
        state = adamw.initialize(params)
        observations = []
        for model, chunk in ((configs[0], 2), (configs[1], 2), (configs[1], 4)):
            fn = jax.jit(jax.value_and_grad(lambda p: learner.loss(p, batch, model,
                         value_weight=.7, path='bounded', chunk_frames=chunk, skip_padding=True), has_aux=True))
            observations.append(fn(params)); jax.block_until_ready(observations[-1])
        for index, chunk in ((1, 2), (2, 4)):
            a, b = observations[index], observations[0]
            groups = {}
            for group in ('input', 'trunk', 'head', 'value'):
                groups[group] = error({k: v for k, v in a[1].items() if adamw.group(k) == group},
                                      {k: v for k, v in b[1].items() if adamw.group(k) == group})
            row = dict(precision=precision, chunk=chunk, loss=error(a[0][0], b[0][0]),
                       metrics=error(a[0][1], b[0][1]), gradient=error(a[1], b[1]), gradient_groups=groups)
            if chunk == 2:
                assert all(row[k]['different'] == 0 for k in ('loss', 'metrics', 'gradient'))
            rows.append(row)
        updates = [jax.jit(learner.step(model, opt, value_weight=.7, chunk_frames=chunk, skip_padding=True))
                   for model, chunk in ((configs[0], 2), (configs[1], 4))]
        states = [(params, state), (params, state)]
        trajectories = []
        for step in range(1, 4):
            results = [fn(*value, batch) for fn, value in zip(updates, states)]
            states = [(r[0], r[1]) for r in results]
            reference, actual = [jax.tree.map(np.asarray, value) for value in states]
            comparison = compare(reference, actual, plan)
            scalars = [{k: float(v) for k, v in result[2].items()} for result in results]
            accepted = gate(comparison, scalars[1], scalars[0], 'remat16', plan)
            trajectories.append(dict(step=step, groups=comparison['groups'], gate=accepted))
        rows.append(dict(precision=precision, larger_chunk_optimizer_trajectory=trajectories))
        print(json.dumps(dict(precision=precision, gradient_relative_l2=rows[-2]['gradient']['relative_l2'],
                   optimizer_gates=[r['gate']['status'] for r in trajectories])), flush=True)
        jax.clear_caches()
    report = dict(status='measured', rows=rows, seconds=time.time() - started,
                  operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='CPU small-model numerical diagnostics, no full-size or TPU qualification implied.')
    with args.output.open('xb') as stream:
        stream.write(canonical_json(report))


if __name__ == '__main__':
    main()
