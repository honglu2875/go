"""Read-only per-expert optimizer diagnostics from a durable MoE checkpoint.

Moment energy is a gradient-history diagnostic, not an observed routing load.
No dataset, labels, TPU device or inference process is opened by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
import numpy as np
from gozero.snapshots import canonical_json, read_json
from gozero.checkpoints import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    path = args.checkpoint.resolve()
    if args.output.exists() or path.is_symlink() or path.name.startswith('.'):
        raise ValueError('Require a complete checkpoint and a new report path')
    group_path, disk_path = path.with_suffix('.group.json'), path.with_suffix('.disk.json')
    group, disk = read_json(group_path), read_json(disk_path)
    if (group['kind'] != 'visual_replicated_checkpoint_group'
            or group['snapshot_id'] != args.snapshot or set(group['host_manifests']) != {'0', '1', '2', '3'}
            or Path(group['owner_checkpoint_path']) != path or disk['status'] != 'passed'
            or disk['group_sha256'] != sha256(group_path)):
        raise ValueError('The complete disk-replicated owner checkpoint is required')
    identities = {name: sha256(file) for name, file in [('group', group_path), ('disk', disk_path), ('manifest', path / 'manifest.json')]}
    if identities['manifest'] != group['host_manifests']['0']:
        raise ValueError('Owner manifest identity differs')
    manifest = read_json(path / 'manifest.json')
    if set(manifest['files']) != {'state.json', 'actors.json', 'arrays.npz'}:
        raise ValueError('Unexpected checkpoint file set')
    for name, expected in manifest['files'].items():
        file = path / name
        if file.is_symlink() or file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']:
            raise ValueError('Checkpoint integrity failure: ' + name)
        identities[name] = expected['sha256']
    state = read_json(path / 'state.json')
    metadata = state['optimizer_metadata']
    if (state['snapshot_id'] != args.snapshot or state['host_rank'] != 0 or not state['owns_replicated_arrays']
            or state['turn'] != group['turn'] or metadata['step'] != state['turn']):
        raise ValueError('Checkpoint state identity differs')
    layers = {}
    with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)):
            raise ValueError('Duplicate NPZ keys')
        schema = metadata['parameter_schema']
        expected = {'step'} | {f'{prefix}_{i:04d}' for i in range(len(schema)) for prefix in ('p', 'm', 'v')}
        if set(arrays.files) != expected or int(arrays['step']) != state['turn']:
            raise ValueError('Optimizer array coverage differs')
        for index, entry in enumerate(schema):
            name = entry['path']
            if '.moe.' not in name:
                continue
            trunk, parameter = name.split('.moe.')
            is_router = parameter == 'router'
            for prefix in ('p', 'm', 'v') if is_router else ('m', 'v'):
                array = arrays[f'{prefix}_{index:04d}']
                if list(array.shape) != entry['shape'] or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError('Invalid expert tensor')
                if prefix == 'v' and np.any(array < 0):
                    raise ValueError('Negative AdamW second moment')
                # FFNs have [layers, experts, ...]; routers are [layers, width, experts].
                for layer_index, value in enumerate(array):
                    experts = value.T if is_router else value
                    row = layers.setdefault(f'{trunk}/{layer_index:02d}', dict(experts=[dict(first_squared=0., second_mass=0., elements=0, ever_nonzero_second=False) for _ in range(experts.shape[0])]))
                    if is_router:
                        row['router_' + prefix + '_norm'] = [float(np.linalg.norm(e.astype(np.float64))) for e in experts]
                    else:
                        for expert, value in zip(row['experts'], experts, strict=True):
                            v = value.astype(np.float64)
                            if prefix == 'm':
                                expert['first_squared'] += float(np.sum(v * v))
                                expert['elements'] += value.size
                            else:
                                expert['second_mass'] += float(np.sum(v))
                                expert['ever_nonzero_second'] |= bool(np.any(value != 0))
                del array
    for row in layers.values():
        first = sum(e['first_squared'] for e in row['experts'])
        second = sum(e['second_mass'] for e in row['experts'])
        for expert in row['experts']:
            expert.update(first_norm=expert['first_squared'] ** .5,
                          first_energy_fraction=expert['first_squared'] / first if first else 0.,
                          second_mass_fraction=expert['second_mass'] / second if second else 0.)
        row['experts_with_no_recorded_task_gradient'] = sum(not e['ever_nonzero_second'] for e in row['experts'])
    if len(layers) != 42:
        raise ValueError('Expected 24 encoder and 18 temporal expert layers')
    report = dict(status='passed', snapshot=args.snapshot, turn=state['turn'], checkpoint=str(path.relative_to(ROOT)),
                  checkpoint_identities=identities, layers=layers, seconds=time.time() - started,
                  operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Per-expert AdamW moment energy. Zero second moment means no representable nonzero FFN task gradient accumulated; moment share is not token-load share. No model execution or held-out target access.')
    with args.output.open('xb') as stream:
        stream.write(canonical_json(report)); stream.flush()
    summaries = {}
    for name, row in layers.items():
        trunk = name.split('/')[0]
        target = summaries.setdefault(trunk, dict(layers=0, experts_without_gradient=0, worst_max_second_mass_fraction=0.))
        target['layers'] += 1; target['experts_without_gradient'] += row['experts_with_no_recorded_task_gradient']
        target['worst_max_second_mass_fraction'] = max(target['worst_max_second_mass_fraction'], max(e['second_mass_fraction'] for e in row['experts']))
    print(json.dumps(dict(status='passed', turn=state['turn'], summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()
