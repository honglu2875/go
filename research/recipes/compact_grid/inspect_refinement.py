"""Read learned residual scales from an independently audited checkpoint."""
import argparse
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def statistics(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        'mean': float(x.mean()),
        'rms': float(np.sqrt(np.mean(x * x))),
        'absolute_quantiles': dict(zip(
            ['min', 'p10', 'median', 'p90', 'p99', 'max'],
            np.quantile(np.abs(x), [0, .1, .5, .9, .99, 1]).tolist())),
        'fraction_absolute_above_1e-3': float(np.mean(np.abs(x) > 1e-3)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(SOURCE)
    root = a.workspace_root.resolve()
    if checkpoints.sha256(a.audit) != a.audit_sha256:
        raise ValueError('Audit identity changed')
    audit = read_json(a.audit)
    if audit['status'] != 'passed':
        raise ValueError('Expected a successful independent audit')
    attempt = root / 'runs' / audit['attempt']
    if checkpoints.sha256(attempt / 'result.json') != audit['closed_result_sha256']:
        raise ValueError('Attempt closure changed')
    report_path = attempt / 'rank-0/artifacts/result.json'
    if checkpoints.sha256(report_path) != audit['input_files'][str(report_path.relative_to(root))]:
        raise ValueError('Audited learner report changed')
    report = read_json(report_path)
    source = root / '.gozero/snapshots' / audit['training_snapshot']
    verify(source)
    c = read_json(source / 'resolved_config.json')['model']
    index = next(i for i, row in enumerate(report['model_schema'])
                 if row['path'] == 'encoder.blocks.gamma')
    saved = report['latest_checkpoint']
    state, arrays, _ = checkpoints.read(
        Path(saved['path']), expected_manifest_sha256=saved['manifest_sha256'],
        array_prefix=f'p_{index:04d}')
    if state['snapshot_id'] != source.name or state['model_schema'] != report['model_schema']:
        raise ValueError('Checkpoint source/schema differs')
    gamma = arrays[f'p_{index:04d}']
    if gamma.shape != (c['encoder_blocks'], c['encoder_width']) or not np.isfinite(gamma).all():
        raise ValueError('Invalid learned residual scales')
    result = {
        'status': 'passed', 'operator_snapshot': SOURCE.name,
        'training_snapshot': source.name, 'attempt': attempt.name,
        'audit_path': str(a.audit), 'audit_sha256': a.audit_sha256,
        'checkpoint_manifest_sha256': saved['manifest_sha256'],
        'initial_gamma': c['encoder_layer_scale'], 'passes': c['encoder_passes'],
        'all_blocks': statistics(gamma),
        'per_unique_block': [statistics(x) for x in gamma],
        'scope': ('Learned parameter statistics only. Nonzero residual scales do not establish '
                  'that repeated passes improve validation or quantify their activation contribution.'),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(canonical_json(result))
    a.output.chmod(0o444)
    print(canonical_json({'status': 'passed', 'output': str(a.output),
                          'all_blocks': result['all_blocks']}).decode())


if __name__ == '__main__':
    main()
