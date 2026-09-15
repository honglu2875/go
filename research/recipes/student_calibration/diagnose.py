#!/usr/bin/env python3
"""Retain fixed-student predictions for every selected validation position."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.board_sequence_batches import Dataset
from gozero.causal_artifacts import validate as validate_candidate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(c):
    fields = 'schema_version kind workspace_root dataset candidates split phase_edges calibration_edges pass_target_edges saturation_absolute_value aggregate_reference_tolerance maximum_seconds maximum_output_bytes_per_arm_per_rank platform expected_processes expected_devices batch_games maximum_games_per_role'
    require(set(c) == set(fields.split()) and c['schema_version'] == 1 and c['kind'] == 'student_calibration', 'Diagnostic configuration differs')
    require(c['platform'] in ('cpu', 'tpu') and c['split'] == 1 and c['maximum_seconds'] == 600
            and c['maximum_output_bytes_per_arm_per_rank'] == 128 * 2**20, 'Unexpected diagnostic scope')
    require((c['expected_processes'], c['expected_devices'], c['batch_games'], c['maximum_games_per_role'])
            == ((1, 1, 4, 1) if c['platform'] == 'cpu' else (4, 16, 32, None)), 'Unqualified execution dimensions')
    require(c['phase_edges'] == [0, 20, 60, 120, 324]
            and c['calibration_edges'] == [i / 10 for i in range(11)]
            and c['pass_target_edges'] == [0., .01, .1, .5, 1.000001]
            and c['saturation_absolute_value'] == .95 and c['aggregate_reference_tolerance'] == .0002,
            'Analysis strata or tolerance changed')
    require(set(c['candidates']) == {'empty', 'exact'}, 'Both fixed models are required')
    return c


def selection(data, c):
    entries = []
    for role in ('expert', 'behavior'):
        values = data.indices[role, c['split']]
        if c['maximum_games_per_role'] is not None:
            values = values[:c['maximum_games_per_role']]
        entries.extend((role, *value) for value in values)
    return entries


def run(args, c, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import model

    rank, world = jax.process_index(), jax.process_count(); root = Path(c['workspace_root']).resolve()
    require(world == c['expected_processes'] and jax.device_count() == c['expected_devices']
            and jax.default_backend() == c['platform'], 'Runtime topology differs')
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'], rank=rank, world=world, board_mode='exact')
    entries = selection(data, c)
    global_shards = [r['id'] for r in data.manifest['shards'] if r['id'] % world == rank]
    public_entries = [(role, global_shards[shard], episode) for role, shard, episode in entries]
    mesh = Mesh(np.asarray(jax.local_devices()), ('games',)); rep = NamedSharding(mesh, P()); batched = NamedSharding(mesh, P('games'))
    report.update(host_rank=int(os.environ.get('GOZERO_HOST_RANK', '0')), jax_rank=rank, world_size=world,
                  jax_version=jax.__version__, devices=[str(d) for d in jax.devices()], local_shard_ids=global_shards,
                  dataset_manifest_sha256=c['dataset']['manifest_sha256'], parent_manifest_sha256=data.manifest['parent_dataset']['manifest_sha256'],
                  selected_entries=public_entries, selected_entries_sha256=hashlib.sha256(canonical_json(public_entries)).hexdigest(), arms={})
    start = time.monotonic()
    for arm in ('empty', 'exact'):
        descriptor_path = SOURCE / c['candidates'][arm]['path']
        require(sha256(descriptor_path) == c['candidates'][arm]['sha256'], 'Fixed candidate descriptor changed')
        descriptor = read_json(descriptor_path); trained = validate_candidate(root, descriptor); net = trained['config']['model']
        require(net['board_mode'] == arm and net['dtype'] == 'bfloat16'
                and trained['model_code_sha256'] == sha256(Path(__file__).with_name('model.py')), 'Model arithmetic differs from training')
        template = jax.eval_shape(lambda: model.initialize(trained['config']['seed'], net))
        leaves, tree = jax.tree.flatten(template)
        schema = [{'path': jax.tree_util.keystr(p), 'shape': list(v.shape), 'dtype': str(v.dtype)}
                  for p, v in jax.tree_util.tree_flatten_with_path(template)[0]]
        require(schema == trained['model_schema'], 'Parameter tree differs')
        parameters = tree.unflatten([trained['arrays'][f'p_{i:04d}'] for i in range(len(leaves))])
        params = jax.device_put(parameters, rep)
        elements_sha = hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in jax.tree.leaves(parameters))).hexdigest()
        dummy = data.batch([None] * c['batch_games'])
        before = time.perf_counter()
        lowered = jax.jit(lambda p, b: model.predictions(p, b, net), in_shardings=(rep, batched),
                          out_shardings=(batched, batched, batched)).lower(params, jax.device_put(dummy, batched))
        hlo = args.output / (arm + '.hlo.txt'); hlo.write_text(lowered.compiler_ir('hlo').as_hlo_text())
        forward = lowered.compile(); compilation = time.perf_counter() - before
        rows = {'ids': [], 'logits': [], 'values': []}; calls = 0; seconds = 0.; cpu_seconds = 0.
        for begin in range(0, len(entries), c['batch_games']):
            require(time.monotonic() - start < c['maximum_seconds'], 'Diagnostic execution deadline reached')
            selected = entries[begin:begin + c['batch_games']]
            batch = data.batch(selected + [None] * (c['batch_games'] - len(selected)))
            if arm == 'empty':
                batch['stones'].fill(0)
            before = time.perf_counter(); cpu_before = time.process_time()
            play, behavior, value = jax.device_get(forward(params, jax.device_put(batch, batched)))
            seconds += time.perf_counter() - before; cpu_seconds += time.process_time() - cpu_before; calls += 1
            require(all(np.isfinite(v).all() for v in (play, behavior, value)), 'Nonfinite model output')
            for row, (role, shard, episode) in enumerate(selected):
                count = int(batch['lengths'][row]) + 1
                ids = np.stack((np.full(count, int(role == 'behavior')), np.full(count, global_shards[shard]),
                                np.full(count, episode), np.arange(count)), axis=-1).astype(np.int32)
                rows['ids'].append(ids)
                rows['logits'].append(np.asarray((play if role == 'expert' else behavior)[row, :count], np.float32))
                rows['values'].append(np.asarray(value[row, :count], np.float32))
        arrays = {k: np.concatenate(v) for k, v in rows.items()}
        require(sum(v.nbytes for v in arrays.values()) <= c['maximum_output_bytes_per_arm_per_rank'], 'Raw prediction archive exceeded budget')
        path = args.output / (arm + '.npz'); np.savez_compressed(path, **arrays)
        report['arms'][arm] = {'descriptor': descriptor, 'descriptor_sha256': sha256(descriptor_path),
                               'model_code_sha256': trained['model_code_sha256'], 'parameter_elements_sha256': elements_sha,
                               'parameter_count': sum(v.size for v in trained['arrays'].values()), 'execution_dtype': net['dtype'],
                               'archive': path.name, 'archive_sha256': sha256(path), 'uncompressed_bytes': sum(v.nbytes for v in arrays.values()),
                               'hlo_sha256': sha256(hlo), 'compile_seconds': compilation, 'forward_seconds': seconds,
                               'forward_process_cpu_seconds': cpu_seconds, 'forward_calls': calls,
                               'positions': len(arrays['ids']), 'expert_positions': int(np.sum(arrays['ids'][:, 0] == 0)),
                               'behavior_positions': int(np.sum(arrays['ids'][:, 0] == 1))}
        print(json.dumps({'kind': 'student_calibration_arm_complete', 'arm': arm, **{k: report['arms'][arm][k]
                           for k in ('positions', 'expert_positions', 'behavior_positions', 'compile_seconds', 'forward_seconds')}}), flush=True)
    require(sha256(Path(c['dataset']['path']) / 'manifest.json') == c['dataset']['manifest_sha256'], 'Dataset changed during evaluation')
    verify(SOURCE); mh.sync_global_devices('student-calibration-complete')
    report.update(status='passed', elapsed_seconds=time.monotonic() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); c = validate(read_json(a.config))
    require(canonical_json(c) == canonical_json(read_json(SOURCE / 'resolved_config.json')), 'Configuration is not frozen')
    require(os.environ.get('JAX_PLATFORMS', c['platform']) == c['platform'], 'Requested platform differs')
    os.environ['JAX_PLATFORMS'] = c['platform']; a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'resolved_config.json').write_bytes(canonical_json(c))
    report = {'schema_version': 1, 'kind': 'fixed_student_calibration', 'status': 'running', 'snapshot_id': SOURCE.name,
              'config_sha256': sha256(a.config), 'training_updates': 0, 'new_games': 0,
              'claims_sample_efficiency': False, 'claims_strength': False, 'claims_mfu': False, 'started_unix': time.time()}
    distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(a, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (a.output / 'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k not in ('devices', 'selected_entries', 'arms')}), flush=True)
        if distributed:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
