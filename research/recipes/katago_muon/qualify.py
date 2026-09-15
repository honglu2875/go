"""Reference equations, semantic groups, failure handling and fresh continuation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
import groups
import heads
import katago
import muon
import state_io


def fixture(root, c):
    ref = root / c['reference']['path']
    if checkpoints.sha256(ref / 'manifest.json') != c['reference']['sha256']:
        raise ValueError('Reference manifest changed')
    info = read_json(ref / 'manifest.json')
    if checkpoints.sha256(ref / 'arrays.npz') != info['arrays_sha256']:
        raise ValueError('Reference arrays changed')
    with np.load(ref / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    p = {r['name']: jnp.asarray(arrays[f'p0.{i}']) for i, r in enumerate(info['leaves'])}
    specs = {r['name']: dict(group=r['group'], layout=r['layout'] if r['group'] in muon.MUON_GROUPS else None)
             for r in info['leaves']}
    gradients = [{r['name']: jnp.asarray(arrays[f'g{step}.{i}']) for i, r in enumerate(info['leaves'])}
                 for step in range(1, 5)]
    return info, arrays, p, specs, gradients


def save(path, p, state, specs, config_sha):
    meta, arrays = state_io.flatten(jax.device_get(p), jax.device_get(state), specs,
                                   configuration_sha256=config_sha, source_sha256=SOURCE.name)
    return checkpoints.write(path, state=meta, arrays=arrays, actors='[]')


def resume(args, c):
    verify(SOURCE)
    info, arrays, p, specs, gradients = fixture(args.workspace_root, c)
    config_sha = checkpoints.sha256(SOURCE / 'resolved_config.json')
    meta, flat, actors = checkpoints.read(args.resume, expected_manifest_sha256=args.manifest_sha256)
    if actors != '[]' or meta['step'] != 2:
        raise ValueError('Expected the two-update optimizer checkpoint')
    p, state = state_io.restore(meta, flat, expected_schema=state_io.schema(p, specs),
                               configuration_sha256=config_sha, source_sha256=SOURCE.name)
    update = jax.jit(lambda p, s, g: muon.apply(p, s, g, jnp.float32(1), specs, info['rates'], info['decays']))
    p, state = jax.tree.map(jnp.asarray, (p, state))
    for gradient in gradients[2:]:
        p, state, metrics = update(p, state, gradient)
        if not bool(metrics['accepted']):
            raise ValueError('Continuation update rejected')
    print(json.dumps(dict(status='passed', manifest_sha256=save(args.output, p, state, specs, config_sha))), flush=True)


def run(args, c):
    started = time.monotonic()
    verify(SOURCE)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='katago_muon_cpu_qualification', status='running', snapshot=SOURCE.name, created=time.time())
    try:
        if jax.default_backend() != 'cpu' or jax.device_count() != 4 or jax.process_count() != 1:
            raise ValueError('Expected one process and four virtual CPU devices')
        info, arrays, initial, specs, gradients = fixture(args.workspace_root, c)
        config_sha = checkpoints.sha256(SOURCE / 'resolved_config.json')
        p, state = initial, muon.initialize(initial, specs)
        update = jax.jit(lambda p, s, g, loss: muon.apply(p, s, g, loss, specs, info['rates'], info['decays'], return_updates=True))
        comparisons = []
        report['comparisons'] = comparisons
        for step, gradient in enumerate(gradients, 1):
            p, state, metrics = update(p, state, gradient, jnp.float32(1))
            if not bool(metrics['accepted']):
                raise ValueError('Reference update rejected')
            for i, leaf in enumerate(info['leaves']):
                key = leaf['name']
                actual = np.asarray(metrics['updates'][key], np.float64)
                expected = arrays[f'u{step}.{i}'].astype(np.float64)
                denom = max(float(np.linalg.norm(expected)), 1e-30)
                relative = float(np.linalg.norm(actual - expected) / denom)
                ratio = float(np.linalg.norm(actual) / denom)
                cosine = float(np.vdot(actual.ravel(), expected.ravel()) / max(float(np.linalg.norm(actual)) * denom, 1e-30))
                comparisons.append(dict(step=step, name=key, group=leaf['group'], relative_rms=relative,
                                        cosine=cosine, norm_ratio=ratio,
                                        maximum_update_error=float(np.max(np.abs(actual - expected))),
                                        maximum_parameter_error=float(np.max(np.abs(np.asarray(p[key]) - arrays[f'p{step}.{i}'])))))
                if leaf['group'] in muon.MUON_GROUPS:
                    if relative >= .03 or cosine <= .999 or abs(ratio - 1) >= .03:
                        raise ValueError(f'Muon reference direction differs at step {step}, {key}: RMS={relative}, cosine={cosine}, ratio={ratio}')
                    np.testing.assert_allclose(p[key], arrays[f'p{step}.{i}'], atol=1e-5, rtol=2e-5)
                else:
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                    np.testing.assert_allclose(p[key], arrays[f'p{step}.{i}'], atol=2e-6, rtol=2e-5)
                    np.testing.assert_allclose(state['second'][key], arrays[f'v{step}.{i}'], atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(state['first'][key], arrays[f'm{step}.{i}'], atol=2e-6, rtol=2e-5)
            if step == 2:
                prefix_sha = save(args.output / 'prefix', p, state, specs, config_sha)
        full_sha = save(args.output / 'full', p, state, specs, config_sha)

        # Confirm every actual JAX CNN policy/value parameter receives the same
        # semantic group as its official model counterpart, including stacks.
        config = dict(architecture='katago_nested_policy', width=16, mid_width=8, gpool_width=2,
                      policy_width=4, layers=4, dtype='float32', rematerialize=True, microbatch=2,
                      max_board_size=19, max_positions=1536, norm_epsilon=1e-4)
        parameters = jax.jit(lambda: katago.initialize(97, config))()
        for prefix in ('value_head', 'intermediate_value_head'):
            value = jax.jit(lambda: heads.initialize(98, kind='cnn', width=16, hidden=7, spatial_channels=5))()
            parameters.update({prefix + '.' + name: a for name, a in value.items()})
        classifications = groups.cnn(parameters)
        official_count = 0
        for key, references in katago.reference_names(parameters, config).items():
            for name, index in references:
                if classifications[key]['group'] != info['official_parameter_groups'][name]['group']:
                    raise ValueError('Official CNN parameter group differs: ' + key)
                official_count += 1
        # Explicit coefficient checks cover the observed Torch/JAX BF16 scalar
        # promotion difference, independently of the complete optimizer test.
        x = jnp.asarray(arrays['bf16_input'], jnp.bfloat16)
        for name in arrays:
            if name.startswith('bf16_scalar_'):
                scalar = float(name.removeprefix('bf16_scalar_'))
                actual = jax.jit(lambda v: muon.scalar_bf16(v, scalar))(x)
                np.testing.assert_array_equal(np.asarray(actual, np.float32), arrays[name])

        zero = jax.tree.map(jnp.zeros_like, initial)
        zp, zs, zm = update(initial, muon.initialize(initial, specs), zero, jnp.float32(1))
        if not bool(zm['accepted']) or int(zs['step']) != 1:
            raise ValueError('Zero-gradient update rejected')
        for key in initial:
            np.testing.assert_array_equal(zm['updates'][key], np.zeros_like(initial[key]))
            wanted = np.asarray(initial[key]) * (1 - info['rates'][specs[key]['group']] * info['decays'][specs[key]['group']])
            np.testing.assert_allclose(zp[key], wanted, atol=1e-8, rtol=2e-7)
        bad = dict(gradients[0])
        key = next(iter(bad))
        bad[key] = jnp.full_like(bad[key], jnp.nan)
        for gradient, loss in [(bad, jnp.float32(1)), (gradients[0], jnp.float32('nan'))]:
            rejected, rs, rm = update(p, state, gradient, loss)
            if bool(rm['accepted']):
                raise ValueError('Nonfinite update accepted')
            for actual, expected in zip(jax.tree.leaves((rejected, rs)), jax.tree.leaves((p, state))):
                np.testing.assert_array_equal(actual, expected)
        meta, flat, _ = checkpoints.read(args.output / 'full', expected_manifest_sha256=full_sha)
        expected_schema = state_io.schema(initial, specs)
        rejected_cases = []
        for label, modified, changed_arrays in [
            ('foreign_source', {**meta, 'source_sha256': '0' * 64}, flat),
            ('missing_momentum', meta, {k: v for k, v in flat.items() if k != 'm_0000'}),
            ('extra_second_moment', meta, {**flat, 'v_extra': np.zeros(1, np.float32)}),
            ('nonfinite_master', meta, {**flat, 'p_0000': np.full_like(flat['p_0000'], np.nan)}),
        ]:
            try:
                state_io.restore(modified, changed_arrays, expected_schema=expected_schema,
                                 configuration_sha256=config_sha, source_sha256=SOURCE.name)
            except ValueError:
                rejected_cases.append(label)
            else:
                raise ValueError('Invalid checkpoint accepted: ' + label)
        command = [sys.executable, '-B', str(Path(__file__)), '--workspace-root', str(args.workspace_root),
                   '--config', str(SOURCE / 'resolved_config.json'), '--resume', str(args.output / 'prefix'),
                   '--manifest-sha256', prefix_sha, '--output', str(args.output / 'resumed')]
        with (args.output / 'resume.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ), check=True, timeout=240)
        resumed_sha = checkpoints.sha256(args.output / 'resumed/manifest.json')
        resumed_meta, resumed_arrays, _ = checkpoints.read(args.output / 'resumed', expected_manifest_sha256=resumed_sha)
        if meta != resumed_meta or set(flat) != set(resumed_arrays):
            raise ValueError('Fresh continuation metadata/coverage differs')
        for name in flat:
            np.testing.assert_array_equal(flat[name], resumed_arrays[name])
        if full_sha != resumed_sha:
            raise ValueError('Deterministic complete optimizer checkpoint differs')
        report.update(status='passed', reference_sha256=c['reference']['sha256'], comparisons=comparisons,
                      cnn_stacked_leaves=len(parameters), cnn_official_parameters_checked=official_count,
                      cnn_group_counts={g: sum(s['group'] == g for s in classifications.values())
                                        for g in sorted({s['group'] for s in classifications.values()})},
                      zero_gradients='passed', nonfinite_rejection='passed', checkpoint_rejections=rejected_cases,
                      fresh_continuation=dict(arrays=len(flat), full_manifest_sha256=full_sha, resumed_manifest_sha256=resumed_sha),
                      scope='Small synthetic-gradient optimizer updates against the actual pinned scalar implementation, complete small CNN group coverage and exact optimizer-only fresh continuation. No neural-loss learning, TPU throughput or historical schedule reproduction.')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        path = args.output / 'result.json'
        with path.open('xb') as f:
            f.write(canonical_json(report))
        path.chmod(0o444)
        print(json.dumps(dict(status=report['status'], seconds=report['seconds'], sha256=checkpoints.sha256(path))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('workspace-root', 'config', 'resume', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    a = p.parse_args()
    resume(a, read_json(a.config))
