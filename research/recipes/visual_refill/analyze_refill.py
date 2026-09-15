"""Compare complete frozen self-play jobs and independently replay their targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocked', required=True); p.add_argument('--refill', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--registration', default='research/studies/visual_causal/refill_registration.json')
    p.add_argument('--registration-sha256')
    a = p.parse_args()
    verify(SOURCE); root = SOURCE.parents[2]
    registration = root / a.registration; reg = read_json(registration)
    if a.registration_sha256 is not None and checkpoints.sha256(registration) != a.registration_sha256:
        raise ValueError('Registration identity differs')
    evidence = {}; reports = {}; packets = {}; configurations = {}; attempts = {}
    for arm, name in [('blocked', a.blocked), ('refill', a.refill)]:
        stage = next(s for s in reg['stages'] if s['stage'] == arm + '_233m')
        source = root / '.gozero/snapshots' / stage['snapshot_id']; verify(source)
        if checkpoints.sha256(source / 'resolved_config.json') != stage['config_sha256']: raise ValueError('Registered configuration changed')
        c = read_json(source / 'resolved_config.json'); configurations[arm] = c
        if c['fixture_pass_after'] is not None or c['fixture_pass_spread'] != 0: raise ValueError('Fixture data cannot establish throughput')
        receipt_path = artifact(root, c['native_receipt']); receipt = read_json(receipt_path)
        native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
        path = root / 'runs' / name; closed = read_json(path / 'result.json'); attempts[arm] = closed
        if closed['status'] != 'passed' or closed['snapshot_id'] != source.name: raise ValueError('Incomplete generation attempt')
        evidence[str((path / 'result.json').relative_to(root))] = checkpoints.sha256(path / 'result.json')
        reports[arm] = []; packets[arm] = []
        for host in range(4):
            base = path / f'rank-{host}/artifacts'; r = read_json(base / 'result.json'); reports[arm].append(r)
            if (r['status'] != 'passed' or not r['generation_complete'] or r['refill'] != (arm == 'refill')
                    or r['counters']['games'] != 256 or r['counters']['active_slot_steps'] != r['counters']['moves']):
                raise ValueError('Wrong scheduled game count or slot accounting')
            evidence[str((base / 'result.json').relative_to(root))] = checkpoints.sha256(base / 'result.json')
            cp = base / 'checkpoints/turn-000000001'; group = read_json(cp.with_suffix('.group.json'))
            state, arrays, actors = checkpoints.read(cp, expected_manifest_sha256=group['host_manifests'][str(host)])
            actors = json.loads(actors)
            if len(actors) != 256 or state['counters'] != r['counters']: raise ValueError('Generation records differ')
            _, legal, endings = native.replay_observations(json.dumps(c['rules']), arrays['actions'], arrays['offsets'])
            legal = legal.reshape(arrays['legal'].shape); endings = json.loads(endings)
            if not np.array_equal(legal, arrays['legal']): raise ValueError('Generated legality differs on independent native replay')
            if (not np.isfinite(arrays['policies']).all() or np.any(arrays['policies'] < 0)
                    or np.any(arrays['policies'][~legal] != 0)
                    or not np.allclose(arrays['policies'].sum(-1), 1., atol=2e-5, rtol=0)):
                raise ValueError('Invalid MCTS target distribution')
            for index, actor in enumerate(actors):
                begin, end = map(int, arrays['offsets'][index:index + 2]); ending = endings[index]
                expected_id = f'{source.name}:host{host}:round{1 + index // 32}:slot{index % 32}'
                if (actor['game_id'] != expected_id or actor['actions'] != arrays['actions'][begin:end].tolist()
                        or actor['terminal'] != ending['terminal'] or actor['white_score'] != ending['white_score']
                        or bool(arrays['terminal'][index]) != ending['terminal']): raise ValueError('Game order, seed identity or ending differs')
                outcome = np.zeros(end - begin, np.float32) if not ending['terminal'] else np.where(
                    np.arange(end - begin) % 2 == 0, -np.sign(ending['white_score']), np.sign(ending['white_score'])).astype(np.float32)
                if not np.array_equal(outcome, arrays['outcomes'][begin:end]): raise ValueError('Outcome target differs')
                actor['game_id'] = actor['game_id'].split(':', 1)[1]
            packets[arm].append((arrays, actors))
    left, right = (dict(configurations[k]) for k in ('blocked', 'refill'))
    left.pop('refill'); right.pop('refill')
    if left != right: raise ValueError('Paired workloads differ beyond the registered refill switch')
    comparison = []; exact = True
    for host in range(4):
        (before, games0), (after, games1) = packets['blocked'][host], packets['refill'][host]
        matching = {k: before[k].dtype == after[k].dtype and np.array_equal(before[k], after[k]) for k in before}
        same_records = games0 == games1
        exact &= all(matching.values()) and same_records
        comparison.append({'host': host, 'bitwise_arrays': matching, 'exact_game_records': same_records,
            'matching_action_tapes': sum(x['actions'] == y['actions'] for x, y in zip(games0, games1)),
            'matching_endings': sum((x['terminal'], x['white_score']) == (y['terminal'], y['white_score']) for x, y in zip(games0, games1))})
    critical = {arm: max(r['segment_seconds'] for r in rows) for arm, rows in reports.items()}
    speed = critical['blocked'] / critical['refill']
    summary = {}
    for arm, rows in reports.items():
        moves = sum(r['counters']['moves'] for r in rows)
        summary[arm] = {'attempt': attempts[arm]['attempt_id'], 'attempt_seconds': attempts[arm]['elapsed_seconds'],
            'attempt_chip_hours': attempts[arm]['reserved_chip_hours'], 'critical_segment_seconds': critical[arm],
            'games': sum(r['counters']['games'] for r in rows), 'moves': moves,
            'terminal_games': sum(r['counters']['terminal_games'] for r in rows),
            'capped_games': sum(r['counters']['capped_games'] for r in rows),
            'moves_per_second': moves / critical[arm], 'host_active_slot_fraction': [r['active_slot_fraction'] for r in rows],
            'host_requests_per_dispatch': [r['cache_stats']['requests'] / r['cache_stats']['dispatches'] for r in rows],
            'aggregate_cache_stats': {k: sum(r['cache_stats'][k] for r in rows) for k in rows[0]['cache_stats']},
            'host_timings': [r['timings'] for r in rows]}
    result = {'schema_version': 1, 'kind': 'visual_frozen_refill_generation_audit', 'status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': checkpoints.sha256(registration), 'evidence': evidence,
        'comparison': comparison, 'all_targets_and_records_exact': bool(exact), 'critical_segment_speedup': speed,
        'passed_registered_gate': bool(exact and speed >= reg['primary_gate']['minimum_complete_generation_segment_speedup']),
        'arms': summary, 'scope': 'Actual complete native self-play generation with the same fixed1024 episode seeds,32slots perhost,weights,search and one final checkpoint group. Every generated board legality and terminal target rechecked. Any array/trace difference fails the exact-systems gate even if throughput improves. No Go strength or MFU claim.'}
    with a.output.open('xb') as stream: stream.write(canonical_json(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence', 'arms')}))


if __name__ == '__main__': main()
