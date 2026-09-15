"""Reproduce descriptive episode/position reuse in the fixed online pilot."""
from collections import Counter
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    verify(SOURCE); root = SOURCE.parents[2]
    panel = read_json(root / 'research/studies/visual_causal/online_katago_registration.json')
    audited_path = root / 'research/studies/visual_causal/online_learning_result.json'
    if sha256(audited_path) != panel['learning_audit_sha256'] or a.output.exists(): raise ValueError('Original learning audit differs or output exists')
    audit = read_json(audited_path); evidence = {str(audited_path.relative_to(root)): sha256(audited_path)}
    descriptor = read_json(root / 'eval/visual_causal/augmented.json'); owner = root / descriptor['checkpoint']['path']
    group = read_json(owner.with_suffix('.group.json')); initial = []
    for host in range(4):
        cp = owner.parents[3] / f'rank-{host}/artifacts/checkpoints' / owner.name; manifest = read_json(cp / 'manifest.json')
        if sha256(cp / 'manifest.json') != group['host_manifests'][str(host)] or sha256(cp / 'state.json') != manifest['files']['state.json']['sha256']:
            raise ValueError('Parent sampler state changed')
        initial.append(read_json(cp / 'state.json')); evidence[str((cp / 'state.json').relative_to(root))] = sha256(cp / 'state.json')
    results = {}
    for arm, info in audit['arms'].items():
        source = root / '.gozero/snapshots' / info['snapshot_id']; verify(source); c = read_json(source / 'resolved_config.json')
        data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256']); population = data.bucket_entries(c['dataset']['buckets'])
        draws = {role: Counter() for role in ['expert', 'behavior']}; position_exposures = Counter()
        for host in range(4):
            rng = np.random.Generator(np.random.PCG64()); rng.bit_generator.state = initial[host]['numpy_rng']
            bucket_rng = np.random.Generator(np.random.PCG64()); bucket_rng.bit_generator.state = initial[host]['bucket_rng']
            path = root / 'runs' / info['attempt'] / f'rank-{host}/artifacts/metrics.jsonl'
            if sha256(path) != audit['evidence'][str(path.relative_to(root))]: raise ValueError('Audited update records changed')
            records = [__import__('json').loads(line) for line in path.read_text().splitlines()]
            for index, record in enumerate(records):
                warm = c['dataset']['warmup_buckets']; bucket = warm[index] if index < len(warm) else int(bucket_rng.choice(c['dataset']['buckets'], p=c['dataset']['bucket_probabilities']))
                selected = [population[role, bucket][int(i)] for role in ['expert', 'behavior']
                            for i in rng.integers(len(population[role, bucket]), size=c['learner']['games_per_role'])]
                if hashlib.sha256(canonical_json(selected)).hexdigest() != record['local_entries_sha256']:
                    raise ValueError('Episode draw reconstruction differs')
                for role, shard, episode in selected:
                    draws[role][shard, episode] += 1
                    position_exposures[role] += int(np.diff(data.shards[shard][role + '_offsets'][episode:episode + 2])[0])
        summary = {}
        for role in draws:
            eligible = [entry for bucket in c['dataset']['buckets'] for entry in population[role, bucket]]
            rows = []; available_positions = 0
            for _, shard, episode in eligible:
                arrays = data.shards[shard]; length = int(np.diff(arrays[role + '_offsets'][episode:episode + 2])[0]); available_positions += length
                rows.append({'game_id': int(arrays[role + '_game_ids'][episode]), 'positions': length, 'draws': draws[role][shard, episode]})
            if position_exposures[role] != info['added_exposures'][role]: raise ValueError('Position reuse differs from original audit')
            values = np.asarray([r['draws'] for r in rows])
            summary[role] = {'eligible_games': len(rows), 'distinct_games_drawn': int((values > 0).sum()), 'total_episode_draws': int(values.sum()),
                'available_positions': available_positions, 'position_exposures': position_exposures[role],
                'exposures_per_available_position': position_exposures[role] / available_positions,
                'draws_per_eligible_game_quantiles_0_25_50_75_100': np.quantile(values, [0, .25, .5, .75, 1]).tolist(), 'games': rows}
        results[arm] = summary
    result = {'schema_version': 1, 'kind': 'visual_online_episode_reuse_diagnostic', 'status': 'passed', 'operator_snapshot': SOURCE.name,
        'evidence': evidence, 'arms': results, 'scope': 'Descriptive post-hoc reuse of fixed dataset entries and correlated position labels. Every episode draw is reconstructed from the pinned parent RNG and matched to the original update log. Distinct game IDs need not imply independent positions. No causal attribution of the negative strength result or new hypothesis test.'}
    with a.output.open('xb') as stream: stream.write(canonical_json(result))
    print(canonical_json({arm: {role: {k: v for k, v in r.items() if k != 'games'} for role, r in rows.items()} for arm, rows in results.items()}).decode().strip())


if __name__ == '__main__': main()
