"""Audit paired fixed Adam continuations and reconstruct every sampler draw."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset


def elements(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        v = arrays[key]; h.update(canonical_json([key, list(v.shape), str(v.dtype)])); h.update(v.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', required=True); p.add_argument('--selfplay', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    registration = root / 'research/studies/visual_causal/online_execution_registration.json'; reg = read_json(registration)
    candidate = read_json(root / 'eval/visual_causal/augmented.json'); owner = artifact(root, candidate['checkpoint']['path'])
    parent_group = read_json(owner.with_suffix('.group.json'))
    _, initial_arrays, _ = checkpoints.read(owner, expected_manifest_sha256=parent_group['host_manifests']['0'])
    initial = elements({k: v for k, v in initial_arrays.items() if k.startswith('p_')}); del initial_arrays
    parent_states = []
    for host in range(4):
        path = owner.parents[3] / f'rank-{host}/artifacts/checkpoints' / owner.name
        manifest = read_json(path / 'manifest.json')
        if checkpoints.sha256(path / 'manifest.json') != parent_group['host_manifests'][str(host)]: raise ValueError('Parent rank manifest changed')
        if checkpoints.sha256(path / 'state.json') != manifest['files']['state.json']['sha256']: raise ValueError('Parent rank state changed')
        parent_states.append(read_json(path / 'state.json'))
    result = {}; evidence = {}; behavior_draws = {}; augmentation_draws = {}; cost = 0.
    for arm, attempt, declaration in zip(('control', 'selfplay'), (a.control, a.selfplay), reg['arms']):
        source = root / '.gozero/snapshots' / declaration['snapshot_id']; verify(source)
        if declaration['arm'] != arm or checkpoints.sha256(source / 'resolved_config.json') != declaration['config_sha256']:
            raise ValueError('Registered continuation changed')
        c = read_json(source / 'resolved_config.json'); data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
        by_bucket = data.bucket_entries(c['dataset']['buckets'])
        path = root / 'runs' / attempt; closed = read_json(path / 'result.json'); cost += closed['reserved_chip_hours']
        if closed['status'] != 'passed' or closed['snapshot_id'] != source.name: raise ValueError('Incomplete paired continuation')
        evidence[str((path / 'result.json').relative_to(root))] = checkpoints.sha256(path / 'result.json')
        reports = []; logs = []; local_counts = []; behavior_draws[arm] = []; augmentation_draws[arm] = []
        for host in range(4):
            base = path / f'rank-{host}/artifacts'; report = read_json(base / 'result.json'); reports.append(report)
            evidence[str((base / 'result.json').relative_to(root))] = checkpoints.sha256(base / 'result.json')
            if (report['status'] != 'passed' or not report['training_complete'] or report['initial_turn'] != 1024
                    or report['turn'] != 1280 or report['initial_parameter_elements_sha256'] != initial
                    or report['fork_lineage']['parent_candidate_sha256'] != c['fork']['candidate_sha256']):
                raise ValueError('Wrong initial or final learner state')
            final = base / 'checkpoints/turn-000001280'; group = read_json(final.with_suffix('.group.json'))
            saved, arrays, actors = checkpoints.read(final, expected_manifest_sha256=group['host_manifests'][str(host)])
            if (actors != '{}' or group['host_jax_mapping'] != parent_group['host_jax_mapping']
                    or saved['counters'] != report['counters']): raise ValueError('Final checkpoint state differs')
            if host == 0:
                if elements(arrays) != group['replicated_arrays_elements_sha256'] or not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError('Replicated final arrays differ or contain nonfinite values')
            elif arrays: raise ValueError('Non-owner contains replicated arrays')
            del arrays
            records = [json.loads(s) for s in (base / 'metrics.jsonl').read_text().splitlines()]; logs.append(records)
            evidence[str((base / 'metrics.jsonl').relative_to(root))] = checkpoints.sha256(base / 'metrics.jsonl')
            if [x['turn'] for x in records] != list(range(1025, 1281)): raise ValueError('Missing fixed-endpoint updates')
            streams = []
            for key in ('numpy_rng', 'bucket_rng', 'augmentation_rng'):
                rng = np.random.Generator(np.random.PCG64()); rng.bit_generator.state = parent_states[host][key]; streams.append(rng)
            sample, buckets, d4 = streams; counts = []; behavior = []; symmetries = []
            for index, record in enumerate(records):
                warmup = c['dataset']['warmup_buckets']
                bucket = warmup[index] if index < len(warmup) else int(buckets.choice(c['dataset']['buckets'], p=c['dataset']['bucket_probabilities']))
                entries = [by_bucket[role, bucket][int(i)] for role in ('expert', 'behavior')
                           for i in sample.integers(len(by_bucket[role, bucket]), size=c['learner']['games_per_role'])]
                codes = d4.integers(0, 8, len(entries)).tolist()
                if (record['bucket'] != bucket or record['local_entries_sha256'] != hashlib.sha256(canonical_json(entries)).hexdigest()
                        or record['local_symmetries'] != codes or record['accepted'] != 1.
                        or not np.isclose(record['learning_rate'], 1e-5, rtol=0, atol=1e-12)):
                    raise ValueError('Sampler, D4 or optimizer schedule does not replay')
                role_counts = {role: sum(int(np.diff(data.shards[shard][role + '_offsets'][game:game + 2])[0])
                    for role_, shard, game in entries if role_ == role) for role in ('expert', 'behavior')}
                counts.append(role_counts); behavior.append([e for e in entries if e[0] == 'behavior']); symmetries.append(codes)
            for key, rng in zip(('numpy_rng', 'bucket_rng', 'augmentation_rng'), streams):
                if saved[key] != rng.bit_generator.state: raise ValueError('Final stream state differs from independent replay')
            local_counts.append(counts); behavior_draws[arm].append(behavior); augmentation_draws[arm].append(symmetries)
        for index in range(256):
            for role in ('expert', 'behavior'):
                total = sum(v[index][role] for v in local_counts)
                if any(log[index][role + '_positions'] != total for log in logs): raise ValueError('Global exposure count differs')
        exposure = {role: sum(v[role] for rows in local_counts for v in rows) for role in ('expert', 'behavior')}
        for role in exposure:
            if reports[0]['counters'][role + '_positions'] - parent_states[0]['counters'][role + '_positions'] != exposure[role]:
                raise ValueError('Cumulative exposure counter differs')
        if any(r['initial_validation'] != reports[0]['initial_validation'] or r['validation_history'] != reports[0]['validation_history']
               or r['test'] != reports[0]['test'] for r in reports): raise ValueError('Global held-out metrics differ between ranks')
        result[arm] = {'attempt': attempt, 'snapshot_id': source.name, 'initial_validation': reports[0]['initial_validation'],
            'final_validation': reports[0]['validation_history'][-1], 'test': reports[0]['test'], 'added_exposures': exposure,
            'host_timings': [r['segment_timing'] for r in reports], 'training_population': reports[0]['training_population'],
            'checkpoint': reports[0]['latest_checkpoint']}
    if result['control']['initial_validation'] != result['selfplay']['initial_validation']: raise ValueError('Paired held-out initialization differs')
    if augmentation_draws['control'] != augmentation_draws['selfplay']: raise ValueError('Paired D4 draws differ')
    matched_behavior = behavior_draws['control'] == behavior_draws['selfplay']
    if not matched_behavior: raise ValueError('Shared sampler rejection changed paired behavior draws; this comparison needs explicit review')
    output = {'schema_version': 1, 'kind': 'visual_online_fixed_endpoint_learning_audit', 'status': 'passed',
              'operator_snapshot': SOURCE.name, 'registration_sha256': checkpoints.sha256(registration), 'evidence': evidence,
              'parent_parameter_elements_sha256': initial, 'matched_behavior_draws': matched_behavior,
              'matched_d4_draws': True, 'audited_rank_updates': 2048, 'attempt_chip_hours': cost, 'arms': result,
              'validation_ratios_selfplay_over_control': {k: result['selfplay']['final_validation']['metrics'][k] / result['control']['final_validation']['metrics'][k]
                  for k in ('expert_kl', 'behavior_ce', 'value_mse')},
              'scope': 'One fixed256-update continuation per arm from identical complete D4 Adam state. Fresh self-play targets can differ from historical targets; historical prediction losses alone cannot establish Go strength or RL efficiency. External fixed-endpoint KataGo comparison is required.'}
    with a.output.open('xb') as f: f.write(canonical_json(output))
    print(json.dumps({k: v for k, v in output.items() if k not in ('evidence', 'arms')}))


if __name__ == '__main__': main()
