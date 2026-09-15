"""Describe audited validation against measured learning time, without fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
REGISTRATION = '600b818e0ffc6649351e3fcaa50756c7c80c8f6bfed9b91c720c3beda908e96e'
METRICS = ('expert_kl', 'family_kl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrast', type=Path, required=True)
    parser.add_argument('--contrast-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.contrast) != args.contrast_sha256:
        raise ValueError('Expected unchanged contrast and a new output directory')
    contrast = read(args.contrast)
    if (contrast['status'] != 'passed' or contrast['registration_sha256'] != REGISTRATION
            or not contrast['all_training_draws_identical']
            or not contrast['all_evaluation_populations_identical']):
        raise ValueError('Require the audited registered paired comparison')
    registration_path = ROOT / 'research/studies/strong9_scaling/registration-001.json'
    if sha(registration_path) != REGISTRATION:
        raise ValueError('Registration changed')

    arms = {}; inputs = {str(args.contrast.relative_to(ROOT)): args.contrast_sha256}
    for arm in contrast['arms']:
        label = arm['label']; audit_path = Path(arm['audit_path'])
        if label not in ('cnn', 'transformer') or label in arms or not audit_path.is_relative_to(ROOT):
            raise ValueError('Invalid or duplicate arm')
        if sha(audit_path) != arm['audit_sha256']:
            raise ValueError('Audit changed')
        audit = read(audit_path)
        if (audit['status'] != 'passed' or audit['attempt'] != arm['attempt']
                or audit['training_snapshot'] != arm['snapshot']
                or audit['validation_curve'] != arm['validation_curve']):
            raise ValueError('Contrast differs from its audit')
        folder = ROOT / 'runs' / arm['attempt']
        metric_path = folder / 'rank-0/artifacts/metrics.jsonl'
        expected = audit['input_files'][str(metric_path.relative_to(ROOT))]
        if sha(metric_path) != expected or sha(folder / 'result.json') != audit['closed_result_sha256']:
            raise ValueError('Closed learning record changed')
        inputs[str(audit_path.relative_to(ROOT))] = arm['audit_sha256']
        inputs[str(metric_path.relative_to(ROOT))] = expected
        rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
        if [row['turn'] for row in rows] != list(range(1, 4097)):
            raise ValueError('Incomplete update history')
        clocks = [0.] + [row['cumulative_learning_seconds'] for row in rows]
        if any(not math.isfinite(b) or b <= a for a, b in zip(clocks, clocks[1:])):
            raise ValueError('Invalid cumulative learning clock')
        if clocks[-1] != arm['timing']['learning_seconds']:
            raise ValueError('Endpoint learning clock differs')
        curve = []
        for row in arm['validation_curve']:
            values = {metric: row['metrics'][metric] for metric in METRICS}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError('Invalid validation metric')
            curve.append(dict(turn=row['turn'], learning_seconds=clocks[row['turn']], **values))
        if [row['turn'] for row in curve] != list(range(0, 4097, 256)):
            raise ValueError('Incomplete validation curve')
        arms[label] = dict(attempt=arm['attempt'], snapshot=arm['snapshot'], curve=curve,
                           timing=arm['timing'], complete_attempt_seconds=read(folder / 'result.json')['elapsed_seconds'])
    if set(arms) != {'cnn', 'transformer'}:
        raise ValueError('Missing paired arm')

    common_budget = min(arm['curve'][-1]['learning_seconds'] for arm in arms.values())
    brackets = {}
    for label, arm in arms.items():
        before = [point for point in arm['curve'] if point['learning_seconds'] <= common_budget]
        after = [point for point in arm['curve'] if point['learning_seconds'] >= common_budget]
        brackets[label] = dict(last_observed_at_or_before_budget=before[-1],
                               first_observed_at_or_after_budget=after[0])

    # These endpoint-derived targets are exploratory, not prospective success gates.
    reference = arms['transformer']['curve'][-1]
    attainment = {}
    for metric in METRICS:
        target = reference[metric]; attainment[metric] = dict(target=target, arms={})
        for label, arm in arms.items():
            first = next((i for i, point in enumerate(arm['curve']) if point[metric] <= target), None)
            attainment[metric]['arms'][label] = None if first is None else dict(
                preceding_observation=arm['curve'][first-1] if first else None,
                first_observation_at_or_below=arm['curve'][first])

    args.output.mkdir(parents=True, exist_ok=False)
    table = args.output / 'curves.csv'
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'turn', 'learning_seconds', *METRICS])
        writer.writeheader()
        for label, arm in arms.items():
            writer.writerows(dict(arm=label, **row) for row in arm['curve'])
    report = dict(kind='audited_learning_time_diagnostic', status='passed', created=time.time(),
                  operator_sha256=sha(Path(__file__)), input_sha256=inputs, seed=contrast['seed'],
                  arms=arms, common_learning_seconds=common_budget, observed_brackets=brackets,
                  endpoint_derived_first_attainment=attainment,
                  csv_sha256=sha(table),
                  scope='Exploratory secondary diagnostic of one paired seed. Learning clocks exclude compilation, data sampling, validation, checkpointing and dispatch setup; individual validation wall timestamps were not recorded. Brackets are observations, not interpolated losses or certified crossing times: unobserved updates can be nonmonotone. Endpoint-derived thresholds are post hoc. No equal-exposure, fixed-wall-budget, MFU, RL-efficiency or strength claim; the registered endpoint decision is unchanged.')
    result = args.output / 'result.json'
    result.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    result.chmod(0o444); table.chmod(0o444)
    print(json.dumps(dict(status='passed', common_learning_seconds=common_budget,
                         observed_brackets=brackets, result_sha256=sha(result))))


if __name__ == '__main__':
    main()
