"""Describe where the closed paired architecture KL gap occurs.

This is post-hoc descriptive analysis of existing validation aggregates.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from lr_compare import audited, pair_curves, sha

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    'cnn-seed1': '226cf79338c1f6fcaa363a38708fd88bb70d727b4b2900168b67470f1e9daaae',
    'transformer-seed1': '060c15a033eeec9ffa2169ffc23304fb304d17faef107c7a1cfe551e5f3ac28f',
    'cnn-seed2': '8003ebb2cccd58761ace388c5fb97fd575da786fe8088d0950057eb05dd74e1f',
    'transformer-seed2': '34971ce106609c2ad9e8b7155689cdaea5bababd381a5aa1badd9e8ce27822d2',
}
PHASES = ('phase_0_16', 'phase_16_64', 'phase_64_128', 'phase_128_256', 'phase_256_2048')


def decomposition(left, right, groups):
    total = left['expert_count']
    delta = right['expert_kl'] - left['expert_kl']
    result = []
    for group in groups:
        count = left[group + '_count']
        if count != right[group + '_count']:
            raise ValueError('Different stratum populations')
        a, b = left[group + '_kl'], right[group + '_kl']
        contribution = (b - a) * count / total
        result.append(dict(group=group, positions=count, fraction_positions=count/total,
            cnn_kl=a, transformer_kl=b, kl_difference=b-a,
            contribution_to_overall_gap=contribution,
            fraction_of_overall_gap=contribution/delta if delta else None))
    if sum(r['positions'] for r in result) != total:
        raise ValueError('Strata do not partition the evaluation population')
    reconstructed = sum(r['contribution_to_overall_gap'] for r in result)
    # Aggregates are logged after FP32 reduction; independently rounded KLs
    # differ slightly from subtracting the overall two CE totals.
    if not math.isclose(reconstructed, delta, rel_tol=0, abs_tol=2e-6):
        raise ValueError('KL gap decomposition does not close')
    return dict(rows=result, overall_gap=delta,
        reconstructed_gap=reconstructed, rounding_residual=delta-reconstructed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = []
    for seed in ('seed1', 'seed2'):
        arms = [audited(ROOT, ROOT/'research/studies/strong9_scaling'/(name+'-'+seed+'-audit-001.json'),
            PINS[name+'-'+seed]) for name in ('cnn', 'transformer')]
        if arms[0]['draws'] != arms[1]['draws']:
            raise ValueError('Training draws differ')
        pairs = pair_curves(*(a['audit']['validation_curve'] for a in arms))
        for a, b in zip((r['parent'] for r in pairs), (r['candidate'] for r in pairs)):
            if a['expert_count'] != b['expert_count']:
                raise ValueError('Validation population differs')
        left, right = pairs[-1]['parent'], pairs[-1]['candidate']
        cases.append(dict(seed=arms[0]['config']['seed'],
            audit_sha256={n:arms[i]['audit_sha256'] for i,n in enumerate(('cnn','transformer'))},
            phases=decomposition(left,right,PHASES),
            opponents=decomposition(left,right,tuple('opponent_'+str(i) for i in range(8))),
            endpoint=dict(cnn_top1=left['expert_top1'], transformer_top1=right['expert_top1'],
                cnn_predicted_entropy=left['expert_entropy'], transformer_predicted_entropy=right['expert_entropy'],
                target_entropy=left['expert_target_entropy']),
            phase_curves=[dict(turn=r['turn'],**decomposition(r['parent'],r['candidate'],PHASES)) for r in pairs[1:]]))
    summary = []
    for i, name in enumerate(PHASES):
        rows = [c['phases']['rows'][i] for c in cases]
        summary.append(dict(group=name, positions=rows[0]['positions'],
            mean_cnn_kl=statistics.mean(r['cnn_kl'] for r in rows),
            mean_transformer_kl=statistics.mean(r['transformer_kl'] for r in rows),
            mean_kl_difference=statistics.mean(r['kl_difference'] for r in rows),
            mean_contribution=statistics.mean(r['contribution_to_overall_gap'] for r in rows),
            mean_fraction_of_paired_gap=statistics.mean(r['fraction_of_overall_gap'] for r in rows)))
    result = dict(kind='closed_larger9_architecture_gap_diagnostic', status='passed',
        created=time.time(), operator_sha256=sha(Path(__file__)),
        paired_reader_sha256=sha(Path(__file__).with_name('lr_compare.py')),
        cases=cases, mean_phase_summary=summary,
        scope='Post-hoc descriptive decomposition of two paired seeds on the '
            'same fixed validation population. Phase and opponent partitions '
            'are overlapping descriptions and must not be added together. '
            'Predicted entropy alone is not a calibration test. The concentrated '
            'opening-family mixture limits generalization. No cause, significance, '
            'new selection criterion, held-out test or strength claim.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed',sha256=sha(args.output),phases=summary,
        endpoints=[c['endpoint'] for c in cases])))


if __name__ == '__main__':
    main()
