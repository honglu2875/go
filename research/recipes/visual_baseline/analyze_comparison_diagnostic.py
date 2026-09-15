"""Describe saved comparison curves and optimizer logs; never launch training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    root = SOURCE.parents[2]
    study = root / 'research/studies/visual_causal'
    learning_path = study / 'cnn_learning_result.json'
    if sha256(learning_path) != args.learning_sha256:
        raise ValueError('Learning audit identity changed')
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Diagnostic outputs already exist')
    learning = read_json(learning_path)
    if learning['status'] != 'passed' or not learning['all_1024_rank_update_draws_exact']:
        raise ValueError('Comparison is not audited')
    evidence = {str(learning_path.relative_to(root)): sha256(learning_path)}
    arms = {}
    for name, arm in learning['arms'].items():
        rank_rows = []
        for rank in range(4):
            relative = f"runs/{arm['attempt']}/rank-{rank}/artifacts/metrics.jsonl"
            path = root / relative
            if sha256(path) != learning['evidence'][relative]:
                raise ValueError('Raw training metrics changed')
            evidence[relative] = sha256(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if [row['turn'] for row in rows] != list(range(1, 129)):
                raise ValueError('Incomplete update sequence')
            rank_rows.append(rows)
        rows = rank_rows[0]
        global_fields = ('loss', 'grad_norm', 'accepted', 'learning_rate',
                         'expert_loss', 'behavior_loss', 'value_loss')
        for other in rank_rows[1:]:
            if any(a[key] != b[key] for a, b in zip(rows, other) for key in global_fields):
                raise ValueError('Global optimizer metrics differ between ranks')
        curves = []
        for row in [arm['initial_validation'], *arm['validation_history']]:
            curves.append({'update': row['turn'], **{
                key: row['metrics'][key]
                for key in ('expert_kl', 'behavior_ce', 'value_mse')}})
        norms = [row['grad_norm'] for row in rows]
        if not all(math.isfinite(x) for x in norms):
            raise ValueError('Nonfinite gradient diagnostic')
        arms[name] = {
            'validation': curves,
            'accepted_updates': sum(row['accepted'] == 1 for row in rows),
            'updates_with_norm_above_clip_threshold': sum(x > 1 for x in norms),
            'gradient_norm': {'minimum': min(norms), 'median': statistics.median(norms),
                              'maximum': max(norms)},
            'decay_only_multiplicative_factor': math.prod(
                1 - .01 * row['learning_rate'] for row in rows),
            'critical_learning_seconds': arm['critical_learning_seconds'],
        }
    result = {
        'schema_version': 1, 'kind': 'posthoc_saved_comparison_diagnostic',
        'status': 'passed', 'operator_snapshot': SOURCE.name, 'evidence': evidence,
        'arms': arms, 'new_tpu_attempts': 0,
        'limitations': [
            'Descriptive analysis of the existing one-seed 128-update runs.',
            'Only three validation measurements; no convergence claim.',
            'Per-checkpoint learning time was not saved; update 64 is not an equal-compute endpoint.',
            'No per-head gradients or parameter-update norms were saved.',
            'Global gradient clipping does not map directly to an Adam effective learning rate.',
            'The decay-only factor excludes gradients and their interaction with parameter trajectories.',
        ],
    }
    lines = ['# Understanding the CNN–transformer comparison', '',
             'The immediate research window is closed. Training, inference qualification, '
             'KataGo transcript/board/score audits and the final deferred-repair probe are '
             'complete. The collection-holdout diagnostic and second fixed-block refill '
             'timing pair are deferred. Capped evaluation games retain their unresolved '
             'outcomes. This document analyzes existing artifacts and launches no TPU work.', '',
             '## What the saved learning curves show', '',
             '| Update | CNN expert KL | Transformer expert KL | CNN behavior CE | Transformer behavior CE | CNN value MSE | Transformer value MSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cnn, transformer in zip(arms['cnn']['validation'], arms['transformer']['validation']):
        if cnn['update'] != transformer['update']:
            raise ValueError('Validation endpoints differ')
        values = [entry[key] for key in ('expert_kl', 'behavior_ce', 'value_mse')
                  for entry in (cnn, transformer)]
        lines.append(f"| {cnn['update']} | " + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['',
        'The CNN already leads all three validation metrics at update 64, including '
        'when compared with the transformer at update 128. The transformer improves '
        'between 64 and 128, so these measurements do not establish a plateau. Its '
        'value MSE changes little from initialization, while the CNN learns much more '
        'of the terminal-outcome target. Both the representation and optimization '
        'deserve investigation before attributing this to architecture alone.', '',
        'Both arms accepted all 128 updates. CNN gradients exceeded the global clip '
        f"threshold on {arms['cnn']['updates_with_norm_above_clip_threshold']}/128 updates; "
        f"transformer gradients did so on {arms['transformer']['updates_with_norm_above_clip_threshold']}/128. "
        'These are finite, accepted updates, and Adam prevents a direct interpretation '
        'of the clipping factor as an effective learning-rate reduction. Per-head '
        'gradients and update-to-parameter norms were not saved.', '',
        'The normalization decay-mask discrepancy is real. Under this short schedule, '
        f"decay alone would shrink a decayed parameter by {100 * (1 - arms['cnn']['decay_only_multiplicative_factor']):.5f}%. "
        'This arithmetic gives its direct scale, not a bound on its indirect effect '
        'through learning. Fix the semantic mask in a newly frozen control.', '',
        '## Next comparison, one question at a time', '',
        '1. Walk through the exact inputs, token ordering, current-board visibility, '
        'policy readouts and gradient paths. Keep the expert, behavior and value roles '
        'explicit. Test small-fixture overfitting and one-device/distributed gradient '
        'agreement if the code review exposes an unresolved concern.',
        '2. Measure a longer common schedule with predefined update and learning-time '
        'checkpoints, matched data draws and additional seeds. Record per-head gradients, '
        'logit/value distributions and update-to-parameter norms. Use validation for '
        'diagnosis; retain fresh games for the next external comparison.',
        '3. Isolate representation choices: first match the eight-board history; '
        'separately test a spatial policy readout; separately test a small CNN observation '
        'encoder feeding the causal transformer. Combining these immediately would '
        'obscure which intervention helped.', '',
        'Before another substantial training run, reserve persistent checkpoint space '
        'and use /dev/shm for disposable staging or restored working copies. The current '
        'persistent filesystem has less than 1 GB free. Preserved checkpoints and '
        'archives must remain available; RAM scratch is not the durable copy.', '',
        '[Audited eight-hour report](REPORT.md), [raw diagnostic](comparison_diagnostic.json), '
        '[recipe and limitations](../../recipes/visual_baseline/README.md).', '']
    report = '\n'.join(lines).encode()
    result['report_sha256'] = hashlib.sha256(report).hexdigest()
    verify(SOURCE)
    for path, payload in [(args.report, report), (args.output, canonical_json(result))]:
        with path.open('xb') as stream:
            stream.write(payload)
        path.chmod(0o444)
    print(json.dumps({'output': str(args.output), 'sha256': sha256(args.output),
                      'report': str(args.report), 'report_sha256': sha256(args.report),
                      'new_tpu_attempts': 0}))


if __name__ == '__main__':
    main()
