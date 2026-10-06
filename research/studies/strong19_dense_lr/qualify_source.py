"""Bind unchanged model code and qualify the LR-only intervention on CPU."""
import json
from pathlib import Path
import sys
import time

from execute_run import ROOT, STUDY, read, require, sha, publish


def main():
    recipe = ROOT / 'research/recipes/strong19_dense_lr'
    parent = ROOT / 'research/recipes/strong19_moe_comparison'
    snapshot = ROOT / '.gozero/snapshots/27df66a36b2a477c676cc4071c791a20465677c6634b5369fdad22c4eeb225c3'
    from gozero.snapshots import verify
    manifest = verify(snapshot); sources = {}
    for path in parent.glob('*.py'):
        name = str(path.relative_to(ROOT)); digest = manifest['files'][name]['sha256']
        require(sha(path) == sha(recipe / path.name) == digest, 'Numerical code changed: ' + path.name)
        sources[str((recipe / path.name).relative_to(ROOT))] = digest
    for name, record in manifest['files'].items():
        if name.startswith('packages/') and name.endswith('.py'):
            require(sha(ROOT / name) == record['sha256'], 'Library changed: ' + name)
            sources[name] = record['sha256']
    sys.path.insert(0, str(recipe))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import adamw, learner, train_config
    require(all(d.platform == 'cpu' for d in jax.devices()), 'CPU qualification only')
    base = read(ROOT / 'research/studies/strong19_recovery/flat-config-001.json')
    options = {name: read(STUDY / (name + '-config-001.json')) for name in ('lr06', 'lr15')}
    settings = {}
    for name, c in options.items():
        train_config.validate(c)
        expected = json.loads(json.dumps(base))
        factor = .6 if name == 'lr06' else 1.5
        expected['learner'].update(learning_rate=base['learner']['learning_rate'] * factor,
                                  end_learning_rate=base['learner']['learning_rate'] * factor * .3)
        expected['checkpoint_every'] = 128; expected['checkpoint_disk']['peer'] = 2
        require(c == expected, 'Non-LR scientific change: ' + name)
        settings[name] = {**{k: v for k, v in c['learner'].items() if k not in ('games_per_host', 'augmentation')},
                          'horizon_steps': c['steps']}
    baseline = {**{k: v for k, v in base['learner'].items() if k not in ('games_per_host', 'augmentation')},
                'horizon_steps': 512}
    steps = jnp.arange(1, 513, dtype=jnp.int32)
    baseline_rates = np.asarray(learner.schedule(steps, baseline))
    schedules = {}
    for name, opt in settings.items():
        rates = np.asarray(learner.schedule(steps, opt)); factor = opt['learning_rate'] / .001
        np.testing.assert_allclose(rates, baseline_rates * factor, rtol=3e-7, atol=1e-10)
        np.testing.assert_allclose(rates[[39, 511]], [opt['learning_rate'], opt['end_learning_rate']], rtol=2e-7)
        schedules[name] = dict(first=float(rates[0]), peak=float(rates[39]), at_128=float(rates[127]),
                               at_512=float(rates[511]), ratio_to_control=factor)
    # The same analytic gradient/state checks task-step scaling and semantic
    # decay together; norm/bias parameters must remain exempt from decay.
    params = {'blocks.up.weight': jnp.array([[.7, -.3], [.2, .9]], jnp.float32),
              'blocks.mlp_norm.scale': jnp.array([1., .8], jnp.float32)}
    grads = {'blocks.up.weight': jnp.array([[.2, -.5], [.7, -.1]], jnp.float32),
             'blocks.mlp_norm.scale': jnp.array([.3, -.9], jnp.float32)}
    state = adamw.initialize(params)
    calls = {}
    for name, rate in [('control', .001), ('lr06', .0006), ('lr15', .0015)]:
        calls[name] = adamw.apply_gradient(params, state, grads, jnp.asarray(1.), learning_rate=rate,
                                           architecture='causal_visual_policy')
    control, control_state, control_metrics = calls['control']
    for name, factor in [('lr06', .6), ('lr15', 1.5)]:
        actual, saved, metrics = calls[name]
        for a, b in zip(jax.tree.leaves(saved), jax.tree.leaves(control_state), strict=True):
            np.testing.assert_array_equal(a, b)
        for key in params:
            np.testing.assert_allclose(params[key] - actual[key], factor * (params[key] - control[key]),
                                       rtol=3e-4, atol=1e-7)
        for key in ('grad_norm', 'raw_grad_norm', 'clip_scale', 'accepted'):
            np.testing.assert_array_equal(metrics[key], control_metrics[key])
        for key in metrics:
            if key.startswith('update_norm_'):
                np.testing.assert_allclose(metrics[key], factor * control_metrics[key], rtol=3e-7)
    result = dict(status='passed', created=time.time(), sources=sources, schedules=schedules,
        operator_sha256=sha(Path(__file__)), configs={name: sha(STUDY / (name + '-config-001.json')) for name in options},
        scope='Exact numerical source lineage; only peak/end LR differ scientifically. All 512 LR values retain '
              'the same normalized schedule; analytic AdamW moments and gradient diagnostics are unchanged and deltas scale with LR.')
    publish(STUDY / 'source-qualification-001.json', result)
    print(json.dumps(dict(status='passed', schedules=schedules, unchanged_source_files=len(sources))))


if __name__ == '__main__': main()
