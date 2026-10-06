"""Report preregistered common endpoints without extrapolating shorter arms."""
import csv
import json
from pathlib import Path

from execute_run import ROOT, STUDY, publish, read, require, sha

METRICS = ('expert_kl', 'family_kl', 'expert_top1', 'value_mse', 'value_family_mse')


def rows(attempt, host=0):
    path = ROOT / 'runs' / attempt / f'rank-{host}/artifacts/metrics.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare_draws(attempt, reference, steps):
    for host in range(4):
        artifact = ROOT / 'runs' / attempt / f'rank-{host}/artifacts'
        rank = read(artifact / 'result.json')['jax_rank']
        actual = rows(attempt, host)
        require(len(actual) >= steps, 'Missing registered update')
        for got, expected in zip(actual[:steps], reference['draws'][:steps]):
            for key in ('turn', 'bucket', 'positions'):
                require(got[key] == expected[key], 'Draw clock differs: ' + key)
            for key in ('local_entries_sha256', 'local_symmetries'):
                require(got[key] == expected['ranks'][rank][key], 'Logical-rank sampler differs: ' + key)


def report(registration, outcomes, folder):
    records = dict(registration['references'])
    records.update({arm: dict(attempt=outcome['attempt'],
        audit=str((STUDY / registration['arms'][arm]['output_directory'] / 'audit.json').relative_to(ROOT)),
        audit_sha256=outcome['audit_sha256']) for arm, outcome in outcomes.items()})
    replay = read(ROOT / registration['draw_reference'])
    audits, histories, clocks, logs = {}, {}, {}, {}
    for arm, item in records.items():
        path = ROOT / item['audit']
        require(sha(path) == item['audit_sha256'], 'Audit changed: ' + arm)
        audit = read(path); audits[arm] = audit
        require(audit['status'] == 'passed' and audit['all_rank_metrics_and_saved_state_verified'],
                'Incomplete audit: ' + arm)
        require(audit['dataset_manifest_sha256'] == registration['dataset_manifest_sha256'],
                'Different corpus: ' + arm)
        limit = 64 if arm == 'all_experts' else 128
        require(audit['steps'] >= limit, 'Insufficient updates: ' + arm)
        compare_draws(item['attempt'], replay, limit)
        logs[arm] = rows(item['attempt'])
        clocks[arm] = {0: 0., **{r['turn']: r['cumulative_learning_seconds'] for r in logs[arm]}}
        histories[arm] = {}
        for field, population in (('validation_history', registration['validation_population_sha256']),
                                  ('training_probe_history', registration['probe_population_sha256'])):
            require(all(r['episode_ids_sha256'] == population for r in audit[field]),
                    'Evaluation population differs: ' + arm)
            histories[arm][field] = {r['turn']: r['metrics'] for r in audit[field]}
    for arm, log in logs.items():
        for got, control in zip(log, logs['dense']):
            for key in ('turn', 'bucket', 'positions', 'learning_rate'):
                require(got[key] == control[key], 'Reference clock differs: ' + arm + ':' + key)

    def point(arm, turn):
        metrics = histories[arm]['validation_history'][turn]
        probe = histories[arm]['training_probe_history'][turn]
        tail = [turn - 32, turn - 16, turn]
        return dict(parameters=audits[arm]['parameters'],
            validation={k: metrics[k] for k in METRICS},
            training_probe={k: probe[k] for k in METRICS},
            validation_minus_probe={k: metrics[k] - probe[k] for k in METRICS},
            last_three_validation_mean={k: sum(histories[arm]['validation_history'][t][k] for t in tail) / 3
                                        for k in METRICS},
            learning_seconds=clocks[arm][turn],
            positions=replay['draws'][turn - 1]['cumulative_positions'])

    tables = {str(t): {a: point(a, t) for a in arms} for t, arms in (
        (64, ('cnn', 'dense', 'temporal', 'all_experts')),
        (128, ('cnn', 'dense', 'temporal')))}
    for table in tables.values():
        baseline = table['dense']
        for row in table.values():
            row['relative_change_vs_dense'] = {k: row['validation'][k] / baseline['validation'][k] - 1
                                              for k in METRICS}
            row['learning_time_ratio_vs_dense'] = row['learning_seconds'] / baseline['learning_seconds']
    # Discrete observed validation points only, with the common time bounded by
    # each registered endpoint. Historical measurements are explicitly labeled.
    timing = {}
    for endpoint, table in tables.items():
        budget = min(r['learning_seconds'] for r in table.values())
        timing[endpoint] = dict(common_learning_seconds=budget, arms={})
        for arm in table:
            turn = max(t for t in histories[arm]['validation_history']
                       if t <= int(endpoint) and clocks[arm][t] <= budget)
            timing[endpoint]['arms'][arm] = dict(turn=turn, learning_seconds=clocks[arm][turn],
                metrics={k: histories[arm]['validation_history'][turn][k] for k in METRICS})
    router = {}
    for arm in ('temporal', 'all_experts'):
        log = logs[arm]; names = [k for k in log[0] if k.startswith('moe_')]
        require(all(r['moe_dropped_tokens'] == 0 for r in log), 'Dropped expert tokens')
        router[arm] = dict(endpoint={k: log[-1][k] for k in names}, all_tokens_retained=True,
            maximum_dead_expert_fraction=max(r['moe_dead_expert_fraction'] for r in log),
            maximum_load_cv2=max(r['moe_load_cv2'] for r in log))
    result = dict(status='passed', kind='matched_cnn_dense_moe_comparison',
        dataset_manifest_sha256=registration['dataset_manifest_sha256'], common_endpoints=tables,
        secondary_attention={str(t): point('attention', t) for t in (64, 128)},
        historical_equal_learning_time=timing, router=router,
        overfit_observations={a: [x for x in audit['overfit_observations'] if x['turn'] <= (64 if a == 'all_experts' else 128)]
                              for a, audit in audits.items()},
        budgets=registration['budgets'], inputs=records,
        scope='One seed on fixed teacher data. Common 64/128-update endpoints of an unchanged 512-update schedule; '
              'architecture-specific helper losses preserved. Historical timing is not simultaneous wall time. '
              'No extrapolated all-expert 128 result, test access, playing-strength or MFU claim.')
    publish(folder / 'comparison.json', result)
    with (folder / 'curves.csv').open('x') as stream:
        writer = csv.writer(stream); writer.writerow(['arm', 'split', 'turn', 'learning_seconds', *METRICS])
        for arm in records:
            for field, split in (('validation_history', 'validation'), ('training_probe_history', 'train_probe')):
                for turn, metrics in sorted(histories[arm][field].items()):
                    if turn <= (64 if arm == 'all_experts' else 128):
                        writer.writerow([arm, split, turn, clocks[arm][turn], *[metrics[k] for k in METRICS]])
    text = ['# Matched CNN, dense-transformer and MoE comparison', '']
    for endpoint, table in tables.items():
        text += [f'At {endpoint} updates:', '',
            '| Architecture | Parameters | Policy KL | Value MSE | Top-1 | Learning hours |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for arm, row in table.items():
            m = row['validation']
            text.append(f"| {arm} | {row['parameters']:,} | {m['expert_kl']:.6f} | {m['value_mse']:.6f} | "
                        f"{m['expert_top1']:.4f} | {row['learning_seconds']/3600:.3f} |")
        text.append('')
    text += [result['scope'], '', 'All-rank state, exact logical-rank game/D4 draws, evaluation populations and '
             'checkpoint replicas were verified. JSON includes train/validation separation, last-three means, '
             'router diagnostics, overfit flags and the secondary attention-pooling reference.']
    with (folder / 'RESULTS.md').open('x') as stream: stream.write('\n'.join(text) + '\n')
    return result
