"""Describe a fixed, matched log prefix without selecting or launching a run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[3]
REGISTRATION = ROOT / 'research/studies/strong9_scaling/registration-001.json'
REGISTRATION_SHA = '600b818e0ffc6649351e3fcaa50756c7c80c8f6bfed9b91c720c3beda908e96e'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_rows(path):
    raw = path.read_bytes()
    # An append in progress is not a complete record.
    end = raw.rfind(b'\n') + 1
    return [json.loads(line) for line in raw[:end].splitlines() if line.strip()]


def summary(values):
    values = sorted(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Missing/nonfinite numerical diagnostic')
    def quantile(q):
        position = q * (len(values) - 1)
        i = int(position)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (position - i)
    return dict(count=len(values), mean=statistics.fmean(values), min=values[0],
                p10=quantile(.1), median=quantile(.5), p90=quantile(.9), max=values[-1])


def optimizer_windows(rows):
    windows = [(1, 64), (65, min(512, len(rows))), (max(65, len(rows) - 511), len(rows))]
    result = []
    groups = sorted(key[len('parameter_norm_'):] for key in rows[-1] if key.startswith('parameter_norm_'))
    for first, last in windows:
        if first > last:
            continue
        used = rows[first-1:last]
        record = dict(first_turn=first, last_turn=last,
                      clipped_updates=sum(row['clip_scale'] < 1. for row in used),
                      statistics={key: summary([row[key] for row in used]) for key in
                                  ('learning_rate', 'clip_scale', 'grad_norm', 'raw_grad_norm')}, groups={})
        for group in groups:
            record['groups'][group] = {
                kind: summary([row[kind + '_norm_' + group] for row in used])
                for kind in ('parameter', 'gradient', 'update')}
            record['groups'][group]['update_to_parameter'] = summary([
                row['update_norm_' + group] / max(row['parameter_norm_' + group], 1e-30) for row in used])
        result.append(record)
    return result


def cohorts(left, right):
    if left['episode_ids_sha256'] != right['episode_ids_sha256'] or left['split'] != right['split']:
        raise ValueError('Different evaluation population')
    a, b = left['metrics'], right['metrics']
    keys = sorted(key[:-3] for key in a if key.endswith('_kl'))
    result = {}
    for key in keys:
        n, m = a[key + '_count'], b[key + '_count']
        if not math.isclose(n, m, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError('Different cohort mass: ' + key)
        if not math.isclose(a[key + '_target_entropy'], b[key + '_target_entropy'], rel_tol=1e-6, abs_tol=2e-6):
            raise ValueError('Different teacher entropy: ' + key)
        x, y = a[key + '_kl'], b[key + '_kl']
        result[key] = dict(count=n, cnn_kl=x, transformer_kl=y, delta=y-x,
                           relative_delta=(y / x - 1) if n > 0 and x > 0 else None)
    total = result['expert']['count']
    for prefix in ('opponent_', 'phase_'):
        selected = [value for key, value in result.items() if key.startswith(prefix)]
        if sum(value['count'] for value in selected) != total:
            raise ValueError('Cohorts do not partition positions')
        for value in selected:
            value['contribution_to_position_kl_delta'] = value['delta'] * value['count'] / total
        difference = sum(value['contribution_to_position_kl_delta'] for value in selected)
        if abs(difference-result['expert']['delta']) > 3e-6:
            raise ValueError('Weighted cohort deltas do not recover aggregate delta')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnn-attempt', required=True)
    p.add_argument('--transformer-attempt', required=True)
    p.add_argument('--seed-index', type=int, choices=(1, 2), required=True)
    p.add_argument('--through-turn', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not 256 <= args.through_turn <= 4096 or args.through_turn % 256:
        raise ValueError('New output and registered evaluation boundary required')
    raw = REGISTRATION.read_bytes()
    if sha(raw) != REGISTRATION_SHA:
        raise ValueError('Registration changed')
    registration = json.loads(raw)
    models = {}
    for name, attempt in [('cnn', args.cnn_attempt), ('transformer', args.transformer_attempt)]:
        if Path(attempt).name != attempt:
            raise ValueError('Expected an attempt name')
        folder = ROOT / 'runs' / attempt
        launch = json.loads((folder / 'launch.json').read_text())
        expected = registration['snapshots'][name]['seed' + str(args.seed_index)]
        if launch['snapshot_id'] != expected:
            raise ValueError('Unregistered source for this arm/seed')
        base = folder / 'rank-0/artifacts'
        config = json.loads((base / 'resolved_config.json').read_text())
        frozen = json.loads((ROOT / '.gozero/snapshots' / expected / 'resolved_config.json').read_text())
        if config != frozen or config['model'] != registration['models'][name]:
            raise ValueError('Running configuration differs')
        rows = [row for row in read_rows(base / 'metrics.jsonl') if row['turn'] <= args.through_turn]
        if [row['turn'] for row in rows] != list(range(1, args.through_turn + 1)):
            raise ValueError('Missing/duplicate training updates')
        if any(not row['accepted'] for row in rows):
            raise ValueError('Rejected update in diagnostic prefix')
        complete = [row for row in read_rows(base / 'evaluations.jsonl') if row['turn'] <= args.through_turn]
        allowed = {'visual_heldout', 'visual_training_probe', 'draft_heldout'}
        if any(row['kind'] not in allowed for row in complete):
            raise ValueError('Unknown evaluation kind')
        additional = [row for row in complete if row['kind'] == 'draft_heldout']
        if additional and (name != 'transformer' or len(additional) != 1 or
                           additional[0]['turn'] != config['steps'] or additional[0]['split'] != 1):
            raise ValueError('Unexpected auxiliary diagnostic')
        evaluations = [row for row in complete if row['kind'] != 'draft_heldout']
        by_key = {(row['kind'], row['turn']): row for row in evaluations}
        if len(by_key) != len(evaluations):
            raise ValueError('Duplicate evaluation')
        required = {(kind, turn) for kind in ('visual_heldout', 'visual_training_probe')
                    for turn in range(0, args.through_turn + 1, 256)}
        if set(by_key) != required:
            raise ValueError('Incomplete paired diagnostic history')
        models[name] = dict(attempt=attempt, snapshot=expected, config=config, rows=rows,
                            evaluations=evaluations, by_key=by_key, auxiliary_evaluations=additional)
    a, b = models['cnn'], models['transformer']
    for x, y in zip(a['rows'], b['rows']):
        for key in ('turn', 'bucket', 'local_entries_sha256', 'local_symmetries',
                    'expert_positions', 'learning_rate'):
            if x[key] != y[key]:
                raise ValueError('Draw/exposure/schedule mismatch: ' + key)
    curves = []
    for turn in range(0, args.through_turn + 1, 256):
        curves.append(dict(turn=turn, populations={
            kind: cohorts(a['by_key'][kind, turn], b['by_key'][kind, turn])
            for kind in ('visual_heldout', 'visual_training_probe')}))
    evidence = {}
    for name, model in models.items():
        last = model['rows'][-1]
        evidence[name] = dict(attempt=model['attempt'], snapshot=model['snapshot'],
            config_sha256=sha(canonical(model['config'])),
            used_update_prefix_sha256=sha(canonical(model['rows'])),
            used_evaluation_prefix_sha256=sha(canonical(model['evaluations'])),
            position_exposures=sum(row['expert_positions'] for row in model['rows']),
            learning_seconds=last['cumulative_learning_seconds'],
            sampling_seconds=last['cumulative_sampling_seconds'],
            auxiliary_evaluations_excluded_from_main_comparison=dict(rows=len(model['auxiliary_evaluations']),
                sha256=sha(canonical(model['auxiliary_evaluations']))),
            optimizer_windows=optimizer_windows(model['rows']))
    result = dict(kind='matched_learning_prefix_diagnostic', status='passed', created=time.time(),
        operator_sha256=sha(Path(__file__).read_bytes()), registration_sha256=REGISTRATION_SHA,
        seed_index=args.seed_index, through_turn=args.through_turn, evidence=evidence, curves=curves,
        scope='Read-only diagnostic of a fixed owner-host log prefix. Whole comparison audits and two-seed decisions remain separate. Cohort summaries are descriptive, not independent significance tests. No job launched or schedule changed.',
        group_semantics='Transformer input includes encoder/connector and token types; trunk includes temporal blocks and remaining embeddings. CNN input is its spatial/global stem; trunk is nested CNN. Norms are aggregate group norms, not layerwise measurements. Update norms include AdamW weight decay; small raw gradients alone do not imply a frozen block.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed',through_turn=args.through_turn,
        output_sha256=sha(args.output.read_bytes()),latest=curves[-1],
        learning_time_ratio=evidence['transformer']['learning_seconds']/evidence['cnn']['learning_seconds'])))


if __name__ == '__main__':
    main()
