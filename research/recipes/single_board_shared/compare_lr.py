"""Extract a learning-rate intervention from a verified fixed-data comparison."""
import argparse
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
from audit_qualification import NUMERICAL_FILES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--comparison-sha256', required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--multiplier', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(SOURCE)
    if sha256(a.comparison) != a.comparison_sha256:
        raise ValueError('Comparison identity changed')
    r = read_json(a.comparison)
    if (r['status'] != 'passed' or r['operator_snapshot'] != SOURCE.name
            or not r['all_game_and_augmentation_draws_equal']):
        raise ValueError('Expected the same frozen operator and verified draws')
    for name, h in r['input_files'].items():
        if sha256(a.workspace_root / name) != h:
            raise ValueError('Comparison input changed: ' + name)
    rows = {x['label']: x for x in r['arms']}
    if len(rows) != len(r['arms']) or a.control == a.candidate:
        raise ValueError('Arm labels must be distinct')
    left, right = rows[a.control], rows[a.candidate]
    for key in ('model', 'parameters', 'initial_parameter_elements_sha256', 'position_exposures'):
        if left[key] != right[key]:
            raise ValueError('Not a learning-rate-only contrast: ' + key)
    numerical_sources = []
    for row in (left, right):
        source = a.workspace_root / '.gozero/snapshots' / row['snapshot']
        manifest = verify(source)
        hashes = {name: item['sha256'] for name, item in manifest['files'].items()
                  if name.startswith('packages/') or name in ('uv.lock', 'pyproject.toml', 'Cargo.lock')}
        hashes.update({'recipe/' + name: sha256(source / manifest['recipe'] / name)
                       for name in NUMERICAL_FILES})
        numerical_sources.append(hashes)
    if numerical_sources[0] != numerical_sources[1]:
        raise ValueError('Numerical source, library or dependency locks differ')
    if a.multiplier <= 0 or right['peak_lr'] != left['peak_lr'] * a.multiplier:
        raise ValueError('Learning-rate multiplier differs')
    paired = []
    if len(left['curve']) != len(right['curve']):
        raise ValueError('Evaluation cadence differs')
    for x, y in zip(left['curve'], right['curve']):
        for key in ('turn', 'validation_ids_sha256', 'expert_count'):
            if x[key] != y[key]:
                raise ValueError('Validation population differs')
        paired.append({'turn': x['turn'], 'candidate_minus_control_kl': y['expert_kl'] - x['expert_kl'],
                       'candidate_minus_control_top1': y['expert_top1'] - x['expert_top1']})
    out = {'status': 'passed', 'kind': 'shared_encoder_lr_contrast',
           'operator_snapshot': SOURCE.name, 'comparison': str(a.comparison),
           'comparison_sha256': a.comparison_sha256, 'control': a.control, 'candidate': a.candidate,
           'peak_lr_multiplier': a.multiplier,
           'initial_parameter_elements_sha256': left['initial_parameter_elements_sha256'],
           'numerical_sources': numerical_sources[0],
           'position_exposures': left['position_exposures'], 'paired_curve': paired,
           'scope': ('Same model, initialization, data draws and non-LR settings verified. '
                     'One-seed descriptive evidence; no strength or convergence claim.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(canonical_json(out))
    a.output.chmod(0o444)
    print(canonical_json({'status': 'passed', 'endpoint': paired[-1], 'sha256': sha256(a.output)}).decode())


if __name__ == '__main__':
    main()
