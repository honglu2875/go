"""Compare the fixed dense LR grid and apply its preregistered screen rule."""
import csv
import json
import math
import statistics
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
    control = registration['reference']; records = {'control': control}
    records.update({arm: dict(attempt=outcome['attempt'], peak_learning_rate=registration['arms'][arm]['peak_learning_rate'],
        audit=str((STUDY / registration['arms'][arm]['output_directory'] / 'audit.json').relative_to(ROOT)),
        audit_sha256=outcome['audit_sha256']) for arm, outcome in outcomes.items()})
    replay = read(ROOT / registration['draw_reference']); audits = {}; logs = {}; histories = {}; clocks = {}
    for arm, item in records.items():
        path = ROOT / item['audit']; require(sha(path) == item['audit_sha256'], 'Audit changed: ' + arm)
        audit = read(path); audits[arm] = audit
        require(audit['status'] == 'passed' and audit['all_rank_metrics_and_saved_state_verified']
                and audit['parameters'] == 232011540 and audit['steps'] >= 128, 'Wrong audited model: ' + arm)
        require(audit['dataset_manifest_sha256'] == registration['dataset_manifest_sha256'], 'Different corpus')
        compare_draws(item['attempt'], replay, 128); logs[arm] = rows(item['attempt'])[:128]
        require(audit['initial_parameters_sha256'] == audits['control']['initial_parameters_sha256'],
                'Initial weights differ: ' + arm)
        histories[arm] = {}; clocks[arm] = {0: 0., **{r['turn']: r['cumulative_learning_seconds'] for r in logs[arm]}}
        for field, population in (('validation_history', registration['validation_population_sha256']),
                                  ('training_probe_history', registration['probe_population_sha256'])):
            history = [r for r in audit[field] if r['turn'] <= 128]
            require([r['turn'] for r in history] == list(range(0, 129, 16))
                    and all(r['episode_ids_sha256'] == population for r in history), 'Evaluation differs')
            histories[arm][field] = {r['turn']: r['metrics'] for r in history}
        factor = item['peak_learning_rate'] / .001
        for actual, base in zip(logs[arm], logs['control'], strict=True):
            require(all(actual[k] == base[k] for k in ('turn', 'bucket', 'positions')), 'Exposure schedule differs')
            require(math.isclose(actual['learning_rate'], base['learning_rate'] * factor,
                                 rel_tol=3e-7, abs_tol=1e-10), 'LR shape or scale differs')
        # Rates cannot change initial evaluation before update one.
        for field in histories[arm]:
            for metric in METRICS:
                require(math.isclose(histories[arm][field][0][metric], histories['control'][field][0][metric],
                                     rel_tol=2e-6, abs_tol=2e-6), 'Fresh evaluation differs')
    table = {}
    for arm, item in records.items():
        val = histories[arm]['validation_history']; probe = histories[arm]['training_probe_history']
        flags = [x for x in audits[arm]['overfit_observations'] if x['turn'] <= 128]
        table[arm] = dict(peak_learning_rate=item['peak_learning_rate'],
            endpoint={k: val[128][k] for k in METRICS}, training_probe={k: probe[128][k] for k in METRICS},
            last_three={k: sum(val[t][k] for t in (96, 112, 128))/3 for k in METRICS},
            validation_minus_probe={k: val[128][k]-probe[128][k] for k in METRICS},
            learning_seconds=clocks[arm][128], sustained_overfit_flags=sum(x['sustained'] for x in flags),
            clipped_updates=sum(x['clip_scale'] < 1 for x in logs[arm]),
            median_raw_gradient_norm=statistics.median(x['raw_grad_norm'] for x in logs[arm]),
            maximum_raw_gradient_norm=max(x['raw_grad_norm'] for x in logs[arm]))
    base = table['control']; candidates = []; decisions = {}
    for arm in registration['order']:
        row = table[arm]
        gates = dict(endpoint_policy=all(row['endpoint'][k] <= .995*base['endpoint'][k] for k in ('expert_kl','family_kl')),
            tail_policy=all(row['last_three'][k] <= base['last_three'][k] for k in ('expert_kl','family_kl')),
            tail_value=row['last_three']['value_mse'] <= 1.05*base['last_three']['value_mse'],
            no_sustained_overfit=row['sustained_overfit_flags'] == 0)
        decisions[arm] = dict(passed=all(gates.values()), gates=gates)
        if all(gates.values()): candidates.append(arm)
    chosen = min(candidates, key=lambda arm: (table[arm]['endpoint']['expert_kl'], table[arm]['endpoint']['family_kl'],
                                             table[arm]['peak_learning_rate'])) if candidates else 'control'
    result = dict(status='passed', kind='dense_learning_rate_grid_comparison', steps=128, schedule_steps=512,
        positions=replay['draws'][127]['cumulative_positions'], table=table, decisions=decisions,
        selected_for_possible_longer_confirmation=chosen, production_settings_changed=False,
        inputs=records, dataset_manifest_sha256=registration['dataset_manifest_sha256'],
        overfit_observations={a:[x for x in q['overfit_observations'] if x['turn'] <= 128] for a,q in audits.items()},
        scope='LR scale only, one matched seed and a 128-update prefix. Historical control timing. '
              'Selection is for possible longer confirmation, not convergence, production promotion or tuned MoE comparison.')
    publish(folder / 'comparison.json', result)
    with (folder / 'curves.csv').open('x') as stream:
        writer=csv.writer(stream); writer.writerow(['arm','peak_lr','split','turn','learning_seconds',*METRICS])
        for arm in records:
            for field, split in (('validation_history','validation'),('training_probe_history','train_probe')):
                for turn, metrics in sorted(histories[arm][field].items()):
                    writer.writerow([arm,records[arm]['peak_learning_rate'],split,turn,clocks[arm][turn],*[metrics[k] for k in METRICS]])
    text=['# Dense learning-rate screen', '', 'All arms share 128 updates and 7,001,181 position exposures.', '',
          '| Peak LR | Policy KL | Family KL | Value MSE | Top-1 | Learning hours |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('lr06','control','lr15'):
        row=table[arm];m=row['endpoint']
        text.append(f"| {row['peak_learning_rate']:.1e} | {m['expert_kl']:.6f} | {m['family_kl']:.6f} | {m['value_mse']:.6f} | {m['expert_top1']:.4f} | {row['learning_seconds']/3600:.3f} |")
    text += ['', f"Candidate for possible longer confirmation: {chosen}. Production settings are unchanged.", '', result['scope'], '',
             'See comparison.json for every selection gate, last-three means, clipping diagnostics and overfit observations. '
             'All initial weights, logical-rank draws, LR schedule ratios, saved states and disk peer copies were audited.']
    with (folder/'RESULTS.md').open('x') as stream:stream.write('\n'.join(text)+'\n')
    return result
