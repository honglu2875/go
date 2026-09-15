"""Apply the registered two-seed decision after rechecking both paired audits.

This operator reports evidence; it never schedules training or chooses a
checkpoint from an unregistered validation turn.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
REGISTRATION_SHA = '600b818e0ffc6649351e3fcaa50756c7c80c8f6bfed9b91c720c3beda908e96e'
PROTOCOL_SHA = 'dfe09a30164a4106603ca649bf9dab26b6db39bff3fcbbd0c6016f6fab237dde'
COMPARE_SHA = 'e72466eb00549394a046ba956b46d9b3502ed16381572eda29980b22da3ef75e'
METRICS = ('expert_kl', 'family_kl')
SEEDS = (91312427, 91312428)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def decision(contrasts):
    if len(contrasts) != 2 or tuple(c['seed'] for c in contrasts) != SEEDS:
        raise ValueError('Require both registered paired seeds in order')
    metric_rows = {}
    for metric in METRICS:
        per_seed = []
        for contrast in contrasts:
            curve = contrast['paired_validation']
            if [row['turn'] for row in curve] != list(range(0, 4097, 256)):
                raise ValueError('Incomplete validation schedule')
            for row in curve:
                for arm in ('cnn', 'transformer'):
                    value = row[arm][metric]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('Invalid KL evidence')
            cnn, transformer = (curve[-1][arm][metric] for arm in ('cnn', 'transformer'))
            cnn_tail, transformer_tail = (statistics.mean(row[arm][metric] for row in curve[-3:])
                                          for arm in ('cnn', 'transformer'))
            if cnn <= 0 or cnn_tail <= 0:
                raise ValueError('Relative improvement requires a positive reference')
            per_seed.append(dict(seed=contrast['seed'], cnn_endpoint=cnn, transformer_endpoint=transformer,
                                 relative_endpoint_gain=1-transformer/cnn,
                                 cnn_last_three_mean=cnn_tail, transformer_last_three_mean=transformer_tail,
                                 relative_last_three_gain=1-transformer_tail/cnn_tail))
        mean_gain = statistics.mean(row['relative_endpoint_gain'] for row in per_seed)
        gates = dict(both_seed_endpoints_improved=all(row['relative_endpoint_gain'] > 0 for row in per_seed),
                     mean_relative_gain_at_least_half_percent=mean_gain >= .005,
                     neither_seed_last_three_mean_regressed=all(row['relative_last_three_gain'] >= 0 for row in per_seed))
        metric_rows[metric] = dict(per_seed=per_seed, mean_relative_endpoint_gain=mean_gain,
                                   gates=gates, passed=all(gates.values()))
    improved = all(row['passed'] for row in metric_rows.values())
    return dict(outcome='registered_transformer_improvement' if improved else 'criterion_not_met',
                transformer_improvement_established=improved, metrics=metric_rows,
                failed_gates=[metric+'.'+gate for metric, row in metric_rows.items()
                              for gate, passed in row['gates'].items() if not passed],
                scope='Two paired fixed-data supervised seeds. Failure to meet this criterion does not establish CNN superiority. No test-set, playing-strength or RL-efficiency conclusion.')


def inspect(plan):
    expected = dict(registration_sha256=REGISTRATION_SHA, protocol_sha256=PROTOCOL_SHA,
                    compare_operator_sha256=COMPARE_SHA, seeds=list(SEEDS), threshold=.005,
                    relative_gain_aggregation='arithmetic mean of per-seed relative endpoint gains',
                    tail_gate='each seed and each primary metric must have nonnegative last-three-mean gain')
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('Conclusion plan differs: '+key)
    if sha(Path(__file__)) != plan['operator_sha256']:
        raise ValueError('Conclusion operator changed')
    for path, digest in ((STUDY/'registration-001.json', REGISTRATION_SHA),
                         (STUDY/'comparison-protocol-001.json', PROTOCOL_SHA),
                         (STUDY/'compare.py', COMPARE_SHA)):
        if sha(path) != digest:
            raise ValueError('Pinned comparison source changed: '+str(path))
    required = [STUDY/f'seed{seed}-contrast-001.json' for seed in (1, 2)]
    required += [STUDY/f'{arm}-seed{seed}-audit-001.json' for seed in (1, 2) for arm in ('cnn', 'transformer')]
    return dict(status='ready_for_reaudit' if all(p.exists() for p in required) else 'awaiting_registered_runs',
                missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()], mutations=False)


def reaudit(seed_index, folder):
    original_path = STUDY/f'seed{seed_index}-contrast-001.json'
    original = read(original_path)
    if original['status'] != 'passed' or original['kind'] != 'strong9_paired_seed_contrast':
        raise ValueError('Not a passed paired contrast')
    if original['operator_sha256'] != COMPARE_SHA or original['registration_sha256'] != REGISTRATION_SHA:
        raise ValueError('Paired contrast has different provenance')
    audits = [STUDY/f'{arm}-seed{seed_index}-audit-001.json' for arm in ('cnn', 'transformer')]
    out = folder/f'seed{seed_index}.json'
    command = [sys.executable, '-B', str(STUDY/'compare.py'), '--workspace-root', str(ROOT),
               '--registration', str(STUDY/'registration-001.json'), '--registration-sha256', REGISTRATION_SHA,
               '--cnn-audit', str(audits[0]), '--cnn-audit-sha256', sha(audits[0]),
               '--transformer-audit', str(audits[1]), '--transformer-audit-sha256', sha(audits[1]),
               '--output', str(out)]
    subprocess.run(command, check=True, timeout=180, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    regenerated = read(out)
    # The timestamp is the only legitimate difference in a re-derived contrast.
    if {k:v for k,v in original.items() if k!='created'} != {k:v for k,v in regenerated.items() if k!='created'}:
        raise ValueError('Paired contrast does not reproduce from its audited inputs')
    return original, dict(path=str(original_path.relative_to(ROOT)), sha256=sha(original_path),
                          reaudit_sha256=sha(out), audits={str(p.relative_to(ROOT)):sha(p) for p in audits})


def markdown(report):
    d = report['decision']
    lines = ['The transformer '+('meets' if d['transformer_improvement_established'] else 'does not meet')+
             ' the registered two-seed improvement criterion on the larger fixed 9×9 corpus.', '',
             'Positive gains mean lower transformer KL. Endpoint is update 4,096; both primary metrics '
             'must improve in each seed, with at least 0.5% mean relative gain and no last-three-mean '
             'regression in either seed.', '',
             '| Metric | Seed | CNN endpoint | Transformer endpoint | Relative gain | Last-three gain |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for metric, row in d['metrics'].items():
        for seed in row['per_seed']:
            lines.append(f"| {metric} | {seed['seed']} | {seed['cnn_endpoint']:.8f} | "
                         f"{seed['transformer_endpoint']:.8f} | {100*seed['relative_endpoint_gain']:.3f}% | "
                         f"{100*seed['relative_last_three_gain']:.3f}% |")
    lines.append('')
    for metric, row in d['metrics'].items():
        lines += [f"{metric}: mean relative endpoint gain {100*row['mean_relative_endpoint_gain']:.3f}%.", '']
    if d['failed_gates']:
        lines += ['Unmet criteria: '+', '.join(d['failed_gates'])+'.', '']
    lines += ['Overfitting diagnostic: '+('a sustained flag is present; a separate paired horizon or data-view study needs registration.'
              if report['sustained_overfit'] else 'no audited sustained flag at the fixed endpoint.'), '',
              '| Seed | Arm | Parameters | Training positions | Reserved chip-hours | Trained decode median ms |',
              '| ---: | --- | ---: | ---: | ---: | ---: |']
    for contrast in report['contrasts']:
        for arm in contrast['arms']:
            lines.append(f"| {contrast['seed']} | {arm['label']} | {arm['parameter_count']} | {arm['expert_positions']} | "
                         f"{arm['reserved_chip_hours']:.4f} | {arm['trained_decode_median_ms']:.3f} |")
    lines += ['', 'Reserved chip-hours are observed attempt reservations, not a billing total. Matched deployed '
              'FLOPs do not imply equal training work. Both paired contrasts were reproduced from the pinned '
              'per-run audits, including exact draws and evaluation populations.', '',
              'Every registered validation and fixed training-probe point follows. Best observed turns remain '
              'diagnostics and do not replace the registered endpoint.', '']
    for contrast in report['contrasts']:
        lines += [f"Seed {contrast['seed']}", '',
                  '| Update | CNN val position KL | T val position KL | CNN val family KL | T val family KL | CNN train-probe KL | T train-probe KL |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for val, train in zip(contrast['paired_validation'], contrast['paired_training_probe']):
            values = [val['cnn']['expert_kl'], val['transformer']['expert_kl'], val['cnn']['family_kl'],
                      val['transformer']['family_kl'], train['cnn']['expert_kl'], train['transformer']['expert_kl']]
            lines.append('| '+str(val['turn'])+' | '+' | '.join(f'{v:.8f}' for v in values)+' |')
        lines.append('')
    lines += [d['scope'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Conclusion plan changed')
    readiness = inspect(read(args.plan))
    if args.inspect:
        print(json.dumps(readiness));return
    if readiness['missing']:
        raise ValueError('Registered evidence is incomplete: '+', '.join(readiness['missing']))
    if args.output is None:
        raise ValueError('A new output path is required')
    output = args.output.resolve()
    if output.parent != STUDY or output.exists() or output.with_suffix('.md').exists():
        raise ValueError('Use new conclusion paths in this study')
    with tempfile.TemporaryDirectory(prefix='strong9-conclusion-', dir='/tmp') as folder:
        pairs = [reaudit(seed, Path(folder)) for seed in (1, 2)]
    contrasts = [pair[0] for pair in pairs]
    observations = [dict(seed=c['seed'], arm=arm['label'], observations=arm['overfit_observations'])
                    for c in contrasts for arm in c['arms']]
    report = dict(kind='strong9_two_seed_conclusion', status='passed', created=time.time(),
                  plan_sha256=sha(args.plan), operator_sha256=sha(Path(__file__)),
                  evidence=[pair[1] for pair in pairs], decision=decision(contrasts),
                  overfit_observations=observations,
                  sustained_overfit=any(o['sustained'] for row in observations for o in row['observations']),
                  contrasts=contrasts, test_targets_decoded=False)
    text = markdown(report)
    report['markdown_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    with output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False);f.write('\n')
    with output.with_suffix('.md').open('x') as f:
        f.write(text)
    output.chmod(0o444);output.with_suffix('.md').chmod(0o444)
    print(json.dumps(dict(status='passed', outcome=report['decision']['outcome'], receipt_sha256=sha(output))))


if __name__ == '__main__':
    main()
