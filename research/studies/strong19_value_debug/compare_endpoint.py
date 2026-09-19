"""Review the completed objective-only repair against the frozen pilot controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
METRICS = ('expert_kl', 'family_kl', 'value_mse', 'value_family_mse')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def read(path, expected=None):
        path = ROOT / path
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed input: {path.name}')
        inputs[str(path.relative_to(ROOT))] = digest
        return json.loads(path.read_text())

    pilot = read('research/studies/strong19_scaling/PILOT_RESULTS_001.json')
    audits = {}
    attempts = {}
    for name, directory in [('source_cnn', 'pilot-source-cnn-001'),
                            ('cnn_adamw', 'pilot-cnn-adamw-001'),
                            ('transformer_mse', 'pilot-transformer-001')]:
        old = pilot['arms']['transformer' if name == 'transformer_mse' else name]
        audits[name] = read(f'research/studies/strong19_scaling/{directory}/audit.json',
                            old['audit_sha256'])
        attempts[name] = [old['attempt']]

    ce_closures = []
    for directory, plan_name in [('ce-prefix-001', 'ce-prefix-plan-002.json'),
                                 ('ce-continuation-001', 'ce-continuation-plan-001.json')]:
        closure = read(STUDY.relative_to(ROOT) / directory / 'result.json')
        plan = read(STUDY.relative_to(ROOT) / plan_name, closure['plan_sha256'])
        audit = read(STUDY.relative_to(ROOT) / directory / 'audit.json', closure['audit_sha256'])
        replica = read(STUDY.relative_to(ROOT) / directory / 'replica.json', closure['replica_sha256'])
        require(closure['status'] == audit['status'] == replica['status'] == 'passed', 'Repair closure failed')
        require(closure['initial_parameters_and_validation_matched'], 'Initial comparison failed')
        require(audit['steps'] == plan['stop_turn'] and audit['positions'] == plan['expected_positions'],
                'Repair horizon differs')
        require(len(replica['copies']) >= 2 and replica['attempt'] == closure['attempt'], 'Missing retained copy')
        ce_closures.append(closure)
    audits['transformer_ce'] = audit
    attempts['transformer_ce'] = [row['attempt'] for row in ce_closures]
    require(audits['transformer_ce']['initial_parameters_sha256'] ==
            audits['transformer_mse']['initial_parameters_sha256'], 'Repair initialization differs')

    # Complete audit already replays sampling. Independently compare the actual
    # all-rank update logs, including D4 and the complete resumed sequence here.
    logs = {}
    learning_seconds = {}
    wall_seconds = {}
    for name, segments in attempts.items():
        all_ranks = []
        learning_seconds[name] = 0.0
        wall_seconds[name] = 0.0
        for rank in range(4):
            rows = []
            offset = 0.0
            for attempt in segments:
                path = ROOT / 'runs' / attempt / f'rank-{rank}/artifacts/metrics.jsonl'
                inputs[str(path.relative_to(ROOT))] = sha(path)
                segment = [json.loads(line) for line in path.read_text().splitlines()]
                rows.extend([dict(row, total_learning_seconds=offset + row['cumulative_learning_seconds'])
                             for row in segment])
                offset += segment[-1]['cumulative_learning_seconds']
            require([r['turn'] for r in rows] == list(range(1, 109)), 'Noncontiguous accepted updates')
            require(all(r['accepted'] for r in rows), 'Rejected learning update')
            all_ranks.append(rows)
            if rank == 0:
                learning_seconds[name] = offset
        logs[name] = all_ranks
        for attempt in segments:
            closure = read(f'runs/{attempt}/result.json')
            require(closure['status'] == 'passed', 'Attempt did not close successfully')
            wall_seconds[name] += closure['elapsed_seconds']
        a = audits[name]
        require(a['status'] == 'passed' and a['steps'] == 108 and a['positions'] == 5787025,
                'Control horizon or complete audit differs')
        require(a['all_rank_metrics_and_saved_state_verified'], 'Saved state not verified')
        for rank, rows in enumerate(all_ranks):
            for old, new in zip(logs['source_cnn'][rank], rows):
                for field in ('turn', 'bucket', 'positions', 'local_entries_sha256', 'local_symmetries'):
                    require(old[field] == new[field], f'Unmatched {field}: {name}, rank {rank}')
    for rank in range(4):
        for old, new in zip(logs['transformer_mse'][rank], logs['transformer_ce'][rank]):
            require(old['learning_rate'] == new['learning_rate'], 'Repair LR clock changed')

    arms = {}
    curves = []
    for name, a in audits.items():
        summaries = {}
        for split in ('validation_history', 'training_probe_history'):
            history = a[split]
            require([r['turn'] for r in history] == list(range(0, 109, 9)), 'Evaluation cadence changed')
            for row, control in zip(history, audits['source_cnn'][split]):
                require(row['episode_ids_sha256'] == control['episode_ids_sha256'] and
                        row['metrics']['expert_count'] == control['metrics']['expert_count'],
                        'Evaluation population changed')
                turn = row['turn']
                curves.append(dict(arm=name, split=split, turn=turn,
                    learning_seconds=logs[name][0][turn - 1]['total_learning_seconds'] if turn else 0.0,
                    **{k: row['metrics'][k] for k in METRICS}))
            last = history[-1]['metrics']
            summaries[split] = dict(endpoint={k: last[k] for k in METRICS},
                previous_observation={k: history[-2]['metrics'][k] for k in METRICS},
                tail_three_mean={k: sum(r['metrics'][k] for r in history[-3:]) / 3 for k in METRICS},
                positions=last['expert_count'], episode_ids_sha256=history[-1]['episode_ids_sha256'])
        arms[name] = dict(attempts=attempts[name], parameters=a['parameters'], positions=a['positions'],
            learning_seconds=learning_seconds[name], attempt_wall_seconds=wall_seconds[name],
            sustained_overfit_flags=sum(r['sustained'] for r in a['overfit_observations']), **summaries)
    ce = arms['transformer_ce']
    comparisons = {}
    for name, arm in arms.items():
        if name == 'transformer_ce':
            continue
        comparisons[name] = {
            split: {k: ce[split]['endpoint'][k] / arm[split]['endpoint'][k] - 1 for k in METRICS}
            for split in ('validation_history', 'training_probe_history')}
        comparisons[name]['relative_learning_time_change'] = ce['learning_seconds'] / arm['learning_seconds'] - 1
    metrics = audits['transformer_ce']['validation_history'][-1]['metrics']
    result = dict(kind='completed_joint19_objective_repair_review', status='passed', created=time.time(),
        finished_utc=datetime.fromtimestamp(ce_closures[-1]['finished'], timezone.utc).isoformat(),
        input_sha256=inputs, operator_sha256=sha(Path(__file__)), steps=108, arms=arms,
        ce_relative_changes=comparisons, curves=curves,
        final_value_diagnostics={k: v for k, v in metrics.items()
            if k.startswith('logit_') or k in ('value_mean_prediction', 'value_mean_target')},
        conclusions=[
            'The objective-only repair removes the measured saturation and lowers final value error.',
            'Policy improves modestly; the source-optimizer CNN remains the strongest policy control.',
            'CNNs used signed MSE. A common-CE comparison is required before attributing gains to architecture.',
            'Value error rises at the last observation in both train probe and validation; no sustained overfit flag.',
            'Learning time sums both CE segments; attempt wall time also includes duplicated startup/compilation.',
        ],
        scope='Single-seed, equal-data supervised pilot. No uncertainty estimate, MFU, playing-strength or RL-efficiency claim.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='passed', output=str(args.output),
        endpoints={k: v['validation_history']['endpoint'] for k, v in arms.items()},
        learning_minutes={k: v['learning_seconds'] / 60 for k, v in arms.items()},
        ce_relative_changes=comparisons)))


if __name__ == '__main__':
    main()
