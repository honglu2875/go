"""Read live pilot diagnostics without executing a model or accessing test data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

METRICS = ('expert_kl', 'family_kl', 'value_mse', 'value_family_mse')


def lines(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # The learner may still be appending its last record.
    return [json.loads(x) for x in data[:data.rfind(b'\n')+1].splitlines()]


def observe(root, attempt, expected_snapshot, *, validation_positions):
    folder = root / 'runs' / attempt
    launch = json.loads((folder / 'launch.json').read_text())
    if launch['snapshot_id'] != expected_snapshot:
        raise ValueError('Unexpected live attempt')
    config = json.loads((root / '.gozero/snapshots' / expected_snapshot / 'resolved_config.json').read_text())
    if config['training']['purpose'] != 'learning' or config['evaluation']['run_test']:
        raise ValueError('Only registered scientific train/validation observations are supported')
    artifact = folder / 'rank-0/artifacts'
    updates = lines(artifact / 'metrics.jsonl')
    evaluations = lines(artifact / 'evaluations.jsonl')
    if [x['turn'] for x in updates] != list(range(1, len(updates)+1)):
        raise ValueError('Missing or repeated update')
    if any(x['accepted'] != 1 for x in updates):
        raise ValueError('An update was rejected')
    histories = {'validation': [], 'training_probe': []}
    for row in evaluations:
        key = {'joint_heldout': 'validation', 'joint_training_probe': 'training_probe'}[row['kind']]
        if key == 'validation' and row['raw_totals']['expert_count'] != validation_positions:
            raise ValueError('Validation population differs')
        histories[key].append({k: row[k] for k in ('turn', 'episode_ids_sha256', 'metrics')})
    for rows in histories.values():
        if rows and (len({x['episode_ids_sha256'] for x in rows}) != 1
                     or [x['turn'] for x in rows] != sorted({x['turn'] for x in rows})):
            raise ValueError('Diagnostic population or chronology changed')
    snapshot = root / '.gozero/snapshots' / expected_snapshot
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    name = manifest['recipe'] + '/training_probe.py'; helper = snapshot / name
    if hashlib.sha256(helper.read_bytes()).hexdigest() != manifest['files'][name]['sha256']:
        raise ValueError('Registered overfit diagnostic changed')
    spec = importlib.util.spec_from_file_location('registered_probe', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    flags = [module.overfit_observation(histories['validation'], histories['training_probe'], metric=m)
             for m in METRICS]
    closed = folder / 'result.json'
    return dict(kind='live_joint19_pilot_observation', attempt=attempt, snapshot=expected_snapshot,
        observed=time.time(), closure=json.loads(closed.read_text())['status'] if closed.exists() else 'running',
        completed_updates=len(updates), position_exposures=sum(x['positions'] for x in updates),
        rank0_learning_seconds=updates[-1]['cumulative_learning_seconds'] if updates else 0.,
        last_update=updates[-1] if updates else None, histories=histories,
        provisional_overfit_flags=flags,
        scope='Live rank-0 observations. Final all-rank audit, checkpoint verification and paired contrasts are separate. '
              'Learning time excludes compilation, evaluation and data work. Diagnostic flags do not choose a checkpoint.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--attempt', required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--validation-positions', required=True, type=int)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    result = observe(a.root, a.attempt, a.snapshot, validation_positions=a.validation_positions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n').encode()
    temp = a.output.with_suffix('.tmp'); temp.write_bytes(data); temp.replace(a.output)
    print(json.dumps(dict(attempt=a.attempt, updates=result['completed_updates'],
                         sha256=hashlib.sha256(data).hexdigest(), closure=result['closure'])))


if __name__ == '__main__':
    main()
