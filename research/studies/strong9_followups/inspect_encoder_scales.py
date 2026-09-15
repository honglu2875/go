"""Read encoder residual scales from the two audited larger9 checkpoints."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def statistics(value):
    x = np.asarray(value, np.float64)
    return dict(mean=float(x.mean()), rms=float(np.sqrt(np.mean(x*x))),
                absolute_quantiles=dict(zip(('min', 'p10', 'median', 'p90', 'p99', 'max'),
                    np.quantile(np.abs(x), [0, .1, .5, .9, .99, 1]).tolist())),
                fraction_negative=float(np.mean(x < 0)),
                fraction_absolute_above_1e_3=float(np.mean(np.abs(x) > 1e-3)))


def main():
    output = STUDY / 'encoder-scales-001.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    pinned = [('seed1', '060c15a033eeec9ffa2169ffc23304fb304d17faef107c7a1cfe551e5f3ac28f'),
              ('seed2', '34971ce106609c2ad9e8b7155689cdaea5bababd381a5aa1badd9e8ce27822d2')]
    results = {}
    for seed, identity in pinned:
        audit_path = ROOT / 'research/studies/strong9_scaling' / ('transformer-' + seed + '-audit-001.json')
        if checkpoints.sha256(audit_path) != identity:
            raise ValueError('Independent run audit changed')
        audit = read_json(audit_path)
        if audit['status'] != 'passed':
            raise ValueError('Run was not independently audited')
        attempt = ROOT / 'runs' / audit['attempt']
        if checkpoints.sha256(attempt / 'result.json') != audit['closed_result_sha256']:
            raise ValueError('Run closure changed')
        report_path = attempt / 'rank-0/artifacts/result.json'
        if checkpoints.sha256(report_path) != audit['input_files'][str(report_path.relative_to(ROOT))]:
            raise ValueError('Audited learner report changed')
        report = read_json(report_path)
        source = ROOT / '.gozero/snapshots' / audit['training_snapshot']
        verify(source)
        config = read_json(source / 'resolved_config.json')['model']
        saved = report['latest_checkpoint']
        state, arrays, _ = checkpoints.read(Path(saved['owner_checkpoint_path']),
            expected_manifest_sha256=saved['manifest_sha256'], array_prefix='p_')
        if state['snapshot_id'] != source.name or state['model_schema'] != report['model_schema']:
            raise ValueError('Checkpoint source or parameter schema changed')
        counts = {'encoder.blocks.gamma': config['encoder_blocks'] - config['encoder_attention_blocks'],
                  'encoder.attention.attn.gamma': config['encoder_attention_blocks'],
                  'encoder.attention.mlp.gamma': config['encoder_attention_blocks']}
        groups = {}
        for name, count in counts.items():
            index = next(i for i, row in enumerate(report['model_schema']) if row['path'] == name)
            value = arrays[f'p_{index:04d}']
            if value.shape != (count, config['encoder_width']) or not np.isfinite(value).all():
                raise ValueError('Invalid residual scale tensor')
            groups[name] = dict(all_blocks=statistics(value),
                               per_unique_block=[statistics(row) for row in value])
        results[seed] = dict(audit_sha256=identity, attempt=attempt.name, training_snapshot=source.name,
            checkpoint_manifest_sha256=saved['manifest_sha256'], initial_scale=config['encoder_layer_scale'],
            shared_passes=config['encoder_passes'], groups=groups)
        del arrays, value
    result = dict(kind='audited_encoder_residual_scale_observation', status='passed', created=time.time(),
        operator_sha256=checkpoints.sha256(Path(__file__)),
        checkpoint_reader_sha256=checkpoints.sha256(Path(checkpoints.__file__)), seeds=results,
        seconds=time.monotonic()-started,
        scope='Endpoint parameter distributions from both completed transformer seeds, with complete checkpoint file hashes verified. These do not measure residual activation contribution, identify an optimal initialization, or establish the benefit of shared passes. No intervention selected or launched.')
    with output.open('xb') as stream:
        stream.write(canonical_json(result))
    output.chmod(0o444)
    print(json.dumps(dict(status='passed', sha256=checkpoints.sha256(output), seconds=result['seconds'],
        median_absolute_scales={seed:{name:g['all_blocks']['absolute_quantiles']['median']
            for name,g in item['groups'].items()} for seed,item in results.items()})))


if __name__ == '__main__':
    main()
