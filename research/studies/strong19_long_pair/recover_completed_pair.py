"""Reconcile the completed pair after the two recorded disk-full monitor errors.

No training is launched and no original result is changed. All registered
scientific comparison gates are rerun against the completed artifacts.
"""
import json
from pathlib import Path
import time

from execute_pair_v2 import compare_draws, inspect
from execute_run import ROOT, STUDY, publish, read, require, sha
from observation import METRICS, observe


def main():
    registration = STUDY / 'registration-002.json'
    digest = 'd343a8960ad09bca2ca58205d3d53b5fec1efeee0f53bb0cdb29e72b65254d08'
    original_path = STUDY / 'sequence-002/result.json'
    original_sha = sha(original_path)
    require(original_sha == '659d7df93d8badb940eb09753934947290d38eed9248af36784fe500505022c6',
            'Unexpected original sequence result')
    original = read(original_path)
    require(original['status'] == 'failed' and original['error'] ==
            "ValueError('Live observation failed; review before starting another arm')",
            'This recovery is limited to the recorded monitor failure')
    events_path = STUDY / 'sequence-002/events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    errors = [row for row in events if row['kind'] == 'monitor_error']
    cleanup = ROOT / 'research/studies/storage_cleanup_20260920/result-001.json'
    cleanup_result = read(cleanup)
    require(cleanup_result['status'] == 'passed' and len(errors) == 2 and all(
        row['stage'] == 'transformer' and row['error'] == "OSError(28, 'No space left on device')"
        and row['time'] < cleanup_result['completed'] for row in errors),
        'Unexpected monitoring failure or unresolved cleanup')
    r = inspect(registration, digest)
    reference = read(ROOT / r['draw_reference'])
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (registration, original_path, events_path, cleanup)}
    outcomes = {'qualification': original['arms']['qualification']}
    logs = {}
    source_inputs_checked = {}
    for arm in r['order']:
        folder = STUDY / (arm + '-long-002')
        result_path = folder / 'result.json'
        result = read(result_path)
        audit_path = folder / 'audit.json'
        replica_path = folder / 'replica.json'
        plan_path = STUDY / (arm + '-long-plan-002.json')
        for p in (result_path, audit_path, replica_path, plan_path):
            inputs[str(p.relative_to(ROOT))] = sha(p)
        require(result['status'] == 'passed' and result['plan_sha256'] == sha(plan_path)
                and result['audit_sha256'] == sha(audit_path)
                and result['replica_sha256'] == sha(replica_path), 'Arm closure differs')
        a = read(audit_path)
        replica = read(replica_path)
        item = r['arms'][arm]
        require(a['status'] == replica['status'] == 'passed' and a['steps'] == 512
                and a['positions'] == reference['total_positions'] == 27217367
                and a['parameters'] == item['parameters']
                and a['snapshot'] == replica['snapshot'] == item['snapshot']
                and a['configuration_sha256'] == item['config_sha256']
                and a['all_rank_metrics_and_saved_state_verified'], 'Audit scope differs')
        for name, expected in a['inputs'].items():
            require(sha(Path(name)) == expected, 'Audited input changed: ' + name)
            source_inputs_checked[name] = expected
        require(replica['attempt'] == result['attempt'] and len(replica['copies']) == 2
                and len({copy['host'] for copy in replica['copies']}) == 2
                and replica['copies'][0]['files'] == replica['copies'][1]['files'],
                'Peer retention receipt differs')
        attempt = result['attempt']
        closed = read(ROOT / 'runs' / attempt / 'result.json')
        require(closed['status'] == 'passed' and closed['snapshot_id'] == item['snapshot'],
                'Attempt closure differs')
        compare_draws(attempt, reference)
        for key, population in [('validation_history', r['validation_population_sha256']),
                                ('training_probe_history', r['probe_population_sha256'])]:
            require([row['turn'] for row in a[key]] == list(range(0, 513, 16))
                    and all(row['episode_ids_sha256'] == population for row in a[key]),
                    'Evaluation population/cadence differs')
        parent = read(ROOT / r['initialization_controls'][arm])
        require(a['initial_parameters_sha256'] == parent['initial_parameters_sha256'],
                'Initialization differs')
        observation = observe(ROOT, attempt, item['snapshot'], validation_positions=60284)
        require(observation['closure'] == 'passed' and observation['completed_updates'] == 512,
                'Final read-only observation did not pass')
        logs[arm] = []
        for rank in range(4):
            artifact = ROOT / 'runs' / attempt / f'rank-{rank}/artifacts'
            report = read(artifact / 'result.json')
            process = read(artifact.parent / 'result.json')
            require(report['status'] == process['status'] == 'passed' and report['turn'] == 512
                    and process['source_integrity'] and process['returncode'] == 0
                    and not process['timed_out'] and not process['cancelled'], 'Rank failed')
            rows = [json.loads(line) for line in (artifact / 'metrics.jsonl').read_text().splitlines()]
            require(all(row['accepted'] == 1 for row in rows), 'Rejected training update')
            logs[arm].append(rows)
        outcomes[arm] = dict(status='passed', attempt=attempt, steps=512,
            parameters=a['parameters'], positions=a['positions'], result_sha256=sha(result_path),
            audit_sha256=sha(audit_path), replica_sha256=sha(replica_path),
            endpoint={k: a['validation_history'][-1]['metrics'][k] for k in METRICS},
            probe_endpoint={k: a['training_probe_history'][-1]['metrics'][k] for k in METRICS},
            learning_seconds=logs[arm][0][-1]['cumulative_learning_seconds'],
            sustained_overfit_flags=sum(row['sustained'] for row in a['overfit_observations']))
        if arm in original['arms']:
            require(outcomes[arm] == original['arms'][arm], 'Prior completed-arm review changed')
    for cnn_rows, transformer_rows in zip(logs['cnn'], logs['transformer']):
        require(len(cnn_rows) == len(transformer_rows) == 512, 'Paired clock differs')
        require(all(a['learning_rate'] == b['learning_rate'] for a, b in zip(cnn_rows, transformer_rows)),
                'Paired learning rates differ')
    require(sha(original_path) == original_sha, 'Original sequence record changed')
    comparison = {k: dict(cnn=outcomes['cnn']['endpoint'][k], transformer=outcomes['transformer']['endpoint'][k],
        relative_transformer_gain=1-outcomes['transformer']['endpoint'][k]/outcomes['cnn']['endpoint'][k])
        for k in METRICS}
    result = dict(kind=r['kind'], status='passed', registration_sha256=digest,
        started=original['started'], finished=max(read(STUDY / (arm+'-long-002/result.json'))['finished']
            for arm in r['order']), reviewed=time.time(), arms=outcomes, comparisons=comparison,
        operator_sha256=sha(Path(__file__)), input_sha256=inputs,
        rechecked_audit_inputs=source_inputs_checked, recovered_monitor_errors=errors,
        original_sequence_status='failed', original_sequence_result_sha256=original_sha,
        scope='Post-completion reconciliation of two disk-full monitoring errors. Original records retained. '
              'Both training arms, audits and recorded peer copies passed; registered scientific gates rerun. '
              'No training restarted, settings changed, checkpoint selected or test targets inspected.')
    output = STUDY / 'sequence-recovery-001'
    output.mkdir(exist_ok=False)
    for arm in r['order']:
        publish(output / (arm+'-review.json'), outcomes[arm])
    publish(output / 'result.json', result)
    print(json.dumps(dict(status='passed', result=str(output / 'result.json'), comparisons=comparison)), flush=True)


if __name__ == '__main__':
    main()
