"""Compare one scalar-only intervention using the qualified paired audit math."""
import argparse
import json
from pathlib import Path
import statistics
import time

import lr_compare as paired
import scalar_source
from lr_compare import read, sha

ROOT = Path(__file__).resolve().parents[3]


def initialization_identical(parent_reports, candidate_reports, mechanism):
    if len(parent_reports) != 4 or len(candidate_reports) != 4:
        raise ValueError('Expected all four rank reports')
    initialized = lambda reports:{r['initial_parameter_elements_sha256'] for r in reports}
    initial_parent, initial_candidate = initialized(parent_reports), initialized(candidate_reports)
    if len(initial_parent) != 1 or len(initial_candidate) != 1:
        raise ValueError('Initialization differs across ranks')
    identical = initial_parent == initial_candidate
    if mechanism not in scalar_source.FIELDS or identical != (mechanism == 'lr_floor'):
        raise ValueError('Observed initialization is inconsistent with the intervention')
    return identical


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--stage', choices=('seed1', 'seed2'), required=True)
    p.add_argument('--candidate-audit', type=Path, required=True)
    p.add_argument('--candidate-audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.plan) != a.plan_sha256:
        raise ValueError('Scientific registration changed')
    plan = read(a.plan)
    if plan['kind'] != 'strong9_scalar_intervention' or plan['comparison_operator_sha256'] != sha(Path(__file__)):
        raise ValueError('Wrong comparison plan or operator')
    if (sha(Path(scalar_source.__file__)) != plan['source_helper_sha256']
            or sha(Path(paired.__file__)) != plan['paired_helper_sha256']):
        raise ValueError('Comparison dependencies changed')
    stage = plan['stages'][a.stage]
    parent = paired.audited(ROOT, ROOT/stage['parent_audit']['path'], stage['parent_audit']['sha256'])
    candidate = paired.audited(ROOT, a.candidate_audit.resolve(), a.candidate_audit_sha256)
    if parent['snapshot'].name != stage['parent_snapshot'] or candidate['snapshot'].name != stage['snapshot']:
        raise ValueError('Unregistered source pair')
    isolation = scalar_source.identical_source(parent['snapshot'], candidate['snapshot'], plan['mechanism'])
    if isolation['change']['candidate_value'] != plan['candidate_value']:
        raise ValueError('Wrong scalar value')
    if parent['draws'] != candidate['draws'] or parent['audit']['expert_positions'] != candidate['audit']['expert_positions']:
        raise ValueError('Training draws or position exposures differ')
    identical = initialization_identical(parent['reports'], candidate['reports'], plan['mechanism'])
    if parent['reports'][0]['model_schema'] != candidate['reports'][0]['model_schema']:
        raise ValueError('Parameter shapes changed')
    validation = paired.pair_curves(parent['audit']['validation_curve'], candidate['audit']['validation_curve'])
    probe = paired.pair_curves(parent['audit']['training_probe_curve'], candidate['audit']['training_probe_curve'])
    gains = paired.gains(validation)
    threshold = plan['screen_min_relative_gain']
    if type(threshold) not in (int, float) or not 0 < threshold < 1:
        raise ValueError('Explicit screen threshold required')
    overfit = any(x['sustained'] for x in candidate['audit']['overfit_observations'])
    screen = all(g['relative_endpoint_gain'] >= threshold and g['relative_last_three_gain'] >= 0
                 for g in gains.values()) and not overfit
    def describe(arm):
        audit, report = arm['audit'], arm['reports'][0]
        return dict(audit_path=arm['audit_path'], audit_sha256=arm['audit_sha256'],
            snapshot=arm['snapshot'].name, attempt=audit['attempt'],
            parameter_count=report['parameter_count'], expert_positions=audit['expert_positions'],
            timing=audit['timing'], reserved_chip_hours=audit['reserved_chip_hours'],
            clipped_updates=audit['clipped_updates'], overfit_observations=audit['overfit_observations'],
            trained_decode_median_ms=1000*statistics.median(report['trained_decode_profile']['host_dispatch_latency_seconds']))
    result = dict(kind='strong9_scalar_paired_contrast', status='passed', created=time.time(),
        plan_sha256=a.plan_sha256, operator_sha256=sha(Path(__file__)), stage=a.stage,
        seed=parent['config']['seed'], parent=describe(parent), candidate=describe(candidate),
        isolation=isolation, initial_parameters_identical=identical,
        initialization_scope=('Identical observed full initialization digests.' if identical else
            'Only encoder-scale initialization differs by source/config construction; preparation checks unchanged other arrays on CPU. Full TPU initial digests differ as expected; no claim of a separately measured non-scale TPU digest.'),
        all_training_draws_identical=True, all_evaluation_populations_identical=True,
        endpoint_and_tail_gains=gains, paired_validation=validation, paired_training_probe=probe,
        screen_min_relative_gain=threshold, screen_passed=screen, sustained_overfit=overfit,
        scope='One paired fixed-data scalar intervention. Selection requires paired second-seed confirmation. No global optimizer/architecture optimality, RL efficiency or playing-strength claim.')
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed', screen_passed=screen, gains=gains, sha256=sha(a.output))))


if __name__ == '__main__':
    main()
