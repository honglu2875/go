"""Describe closed validation curves against measured learning-update time.

This does not change the registered same-update intervention gate. Comparisons
use observed checkpoints, without interpolating an unobserved validation loss.
All-rank timing intervals make the nearby checkpoint comparison conservative.
"""
import argparse
import csv
import json
from pathlib import Path
import time

from lr_compare import audited, read, sha

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def curve(arm):
    attempt = ROOT / 'runs' / arm['audit']['attempt']
    logs = [[json.loads(line) for line in (attempt / f'rank-{host}/artifacts/metrics.jsonl').read_text().splitlines()]
            for host in range(4)]
    if any([row['turn'] for row in log] != list(range(1, 4097)) for log in logs):
        raise ValueError('Incomplete learning-clock observations')
    for log in logs:
        clocks = [row['cumulative_learning_seconds'] for row in log]
        if clocks != sorted(clocks) or any(row['accepted'] != 1 for row in log):
            raise ValueError('Learning clock regressed or an update was rejected')
    result = []
    for evaluation in arm['audit']['validation_curve']:
        turn = evaluation['turn']
        clocks = [log[turn - 1]['cumulative_learning_seconds'] if turn else 0. for log in logs]
        result.append(dict(turn=turn, minimum_rank_learning_seconds=min(clocks),
            maximum_rank_learning_seconds=max(clocks), rank_learning_seconds=clocks,
            policy_kl=evaluation['metrics']['expert_kl'], family_kl=evaluation['metrics']['family_kl'],
            position_exposures=sum(row['expert_positions'] for row in logs[0][:turn]),
            validation_population_sha256=evaluation['episode_ids_sha256']))
    return result


def compare(cnn, transformer):
    end = transformer[-1]
    before = [row for row in cnn if row['maximum_rank_learning_seconds'] <= end['minimum_rank_learning_seconds']]
    after = [row for row in cnn if row['minimum_rank_learning_seconds'] >= end['maximum_rank_learning_seconds']]
    if not before or not after:
        raise ValueError('CNN observations do not bracket the transformer final learning time')
    left, right = before[-1], after[0]
    if any(row['validation_population_sha256'] != end['validation_population_sha256'] for row in (left, right)):
        raise ValueError('Validation populations differ')
    metrics = {key: dict(transformer=end[key], cnn_before=left[key], cnn_after=right[key],
                        relative_gain_against_cnn_after=1 - end[key] / right[key])
               for key in ('policy_kl', 'family_kl')}
    return dict(transformer_final=end, cnn_before=left, cnn_after=right, metrics=metrics,
        transformer_uses_less_learning_time_and_has_lower_both_kls_than_cnn_after=
            end['maximum_rank_learning_seconds'] < right['minimum_rank_learning_seconds']
            and all(end[key] < right[key] for key in metrics),
        equal_update_gate_changed=False,
        scope='Observed checkpoint comparison. The CNN after point has already spent more learning time on every rank. '
              'No interpolation or claim about the unobserved exact-budget checkpoint. Transformer uses more training positions. '
              'Learning-only clocks exclude data preparation, compilation, evaluation, checkpointing and generation cost.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    registration_path = STUDY / 'lr15-registration-001.json'
    registration = read(registration_path)
    arms = {}; contrasts = {}; inputs = {str(registration_path.relative_to(ROOT)): sha(registration_path)}
    for seed in ('seed1', 'seed2'):
        for label, record in [('CNN', registration['cnn_controls'][seed]),
                              ('Transformer', registration['stages'][seed]['parent_audit'])]:
            path = ROOT / record['path']; data = audited(ROOT, path, record['sha256'])
            inputs[record['path']] = record['sha256']
            arms[seed + '/' + label] = dict(snapshot=data['snapshot'].name, attempt=data['audit']['attempt'],
                                           seed=data['config']['seed'], curve=curve(data))
        contrasts[seed] = compare(arms[seed + '/CNN']['curve'], arms[seed + '/Transformer']['curve'])
    optional = STUDY / 'scale1e-2/seed1-contrast-001.json'
    if optional.exists():
        result = read(optional)
        if result['status'] != 'passed' or not result['screen_passed']:
            raise ValueError('The optional encoder first seed has not passed its audit/screen')
        item = result['candidate']; data = audited(ROOT, ROOT / item['audit_path'], item['audit_sha256'])
        inputs[str(optional.relative_to(ROOT))] = sha(optional)
        inputs[item['audit_path']] = item['audit_sha256']
        arms['seed1/Transformer scale0.01'] = dict(snapshot=data['snapshot'].name, attempt=data['audit']['attempt'],
                                                seed=data['config']['seed'], curve=curve(data),
                                                scope='First-seed screen only; second-seed confirmation pending')
        contrasts['seed1_scale0.01'] = compare(arms['seed1/CNN']['curve'], arms['seed1/Transformer scale0.01']['curve'])
    output = dict(kind='closed9_learning_time_observation', status='passed', created=time.time(),
        operator_sha256=sha(Path(__file__)), inputs=inputs, arms=arms, contrasts=contrasts,
        clock='Cumulative measured learning-update seconds; minimum/maximum intervals over all four ranks.',
        scope='Post-hoc descriptive throughput/learnability tradeoff. No intervention selection criterion changed; '
              'not equal position exposure, full job time, RL efficiency or playing strength.')
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    with args.output.with_suffix('.csv').open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'arm', 'update', 'minimum_learning_seconds', 'maximum_learning_seconds',
                         'position_policy_kl', 'family_policy_kl', 'training_position_exposures'])
        for name, arm in arms.items():
            for row in arm['curve']:
                writer.writerow([arm['seed'], name.split('/', 1)[1], row['turn'], row['minimum_rank_learning_seconds'],
                    row['maximum_rank_learning_seconds'], row['policy_kl'], row['family_kl'], row['position_exposures']])
    print(json.dumps(dict(status='passed', sha256=sha(args.output), comparisons={key:dict(
        cnn_after_turn=value['cnn_after']['turn'], metrics=value['metrics'],
        observed_dominance=value['transformer_uses_less_learning_time_and_has_lower_both_kls_than_cnn_after'])
        for key, value in contrasts.items()})))


if __name__ == '__main__':
    main()
