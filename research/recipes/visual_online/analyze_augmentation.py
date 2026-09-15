"""Audit the fixed endpoint and matched exposure of whole-history D4 learning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--reference-attempt', required=True)
    p.add_argument('--augmented-attempt', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if a.output.exists() or sha256(a.registration) != a.registration_sha256:
        raise ValueError('Output exists or registration differs')
    registration = read_json(a.registration); evidence = {}; arms = {}; cost = 0.
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path)
        return read_json(path)
    prior = root / 'research/studies/visual_causal/learning_pilot_result.json'
    if sha256(prior) != registration['reference_learning_result_sha256']:
        raise ValueError('Reference learning audit differs')
    remember(prior)
    stage = next(s for s in registration['stages'] if s['stage'] == '233m_learning')
    for name, attempt in [('reference', a.reference_attempt), ('augmented', a.augmented_attempt)]:
        directory = root / 'runs' / attempt; closed = remember(directory / 'result.json')
        if closed['status'] != 'passed':
            raise ValueError('Learning attempt did not close successfully')
        snapshot = root / '.gozero/snapshots' / closed['snapshot_id']; verify(snapshot)
        config = remember(snapshot / 'resolved_config.json')
        if name == 'augmented' and (snapshot.name != stage['snapshot_id'] or
                sha256(snapshot / 'resolved_config.json') != stage['config_sha256']):
            raise ValueError('Augmented arm differs from its registration')
        ranks = []; cost += closed['reserved_chip_hours']
        for host in range(4):
            report = remember(directory / f'rank-{host}/artifacts/result.json')
            path = directory / f'rank-{host}/artifacts/metrics.jsonl'
            evidence[str(path.relative_to(root))] = sha256(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if (report['status'] != 'passed' or report['host_rank'] != host or
                    report['snapshot_id'] != snapshot.name or not report['training_complete'] or
                    report['turn'] != 1024 or config['steps'] != 1024 or
                    [r['turn'] for r in rows] != list(range(1, 1025)) or
                    any(r['accepted'] != 1 for r in rows)):
                raise ValueError('Missing or rejected scheduled update')
            checkpoint = Path(report['latest_checkpoint']['path'])
            group = remember(checkpoint.with_suffix('.group.json'))
            state = remember(checkpoint / 'state.json')
            if (sha256(checkpoint.with_suffix('.group.json')) != report['latest_checkpoint']['group_sha256'] or
                    sha256(checkpoint / 'manifest.json') != group['host_manifests'][str(host)]):
                raise ValueError('Distributed checkpoint commitment differs')
            ranks.append({'report': report, 'rows': rows, 'state': state})
        for rank in ranks[1:]:
            for field in ('counters', 'validation_history', 'test', 'initial_parameter_elements_sha256'):
                if rank['report'][field] != ranks[0]['report'][field]:
                    raise ValueError('Replicated scientific reports differ: ' + field)
        arms[name] = {'config': config, 'ranks': ranks, 'snapshot': snapshot.name}
    reference, augmented = arms['reference'], arms['augmented']
    config = json.loads(json.dumps(augmented['config']))
    if config['learner'].pop('augmentation') != 'd4' or config != reference['config']:
        raise ValueError('Learning configurations differ beyond D4 augmentation')
    counts = Counter()
    fields = ('turn', 'bucket', 'local_entries_sha256', 'expert_positions', 'behavior_positions', 'learning_rate')
    for host in range(4):
        x, y = reference['ranks'][host], augmented['ranks'][host]
        random = np.random.Generator(np.random.PCG64(config['seed'] + 400003 + 104729 * y['report']['jax_rank']))
        for r, s in zip(x['rows'], y['rows']):
            if any(r[k] != s[k] for k in fields):
                raise ValueError('Episode exposure or scheduled update differs')
            expected = random.integers(0, 8, 2 * config['learner']['games_per_role']).tolist()
            if expected != s['local_symmetries']:
                raise ValueError('Recorded D4 transforms differ from independent RNG replay')
            counts.update(expected)
        if random.bit_generator.state != y['state']['augmentation_rng']:
            raise ValueError('Final augmentation RNG differs from independently replayed stream')
        for field in ('numpy_rng', 'bucket_rng', 'counters'):
            if x['state'][field] != y['state'][field]:
                raise ValueError('Final sampling state or exposure differs')
    final = {name: arm['ranks'][0]['report'] for name, arm in arms.items()}
    for key in ('initial_parameter_elements_sha256', 'initial_validation', 'counters', 'model_schema'):
        if final['reference'][key] != final['augmented'][key]:
            raise ValueError('Initial model, evaluation population or exposure differs: ' + key)
    endpoint = {name: report['validation_history'][-1] for name, report in final.items()}
    if (any(r['turn'] != 1024 or r['split'] != 1 for r in endpoint.values()) or
            endpoint['reference']['episode_ids_sha256'] != endpoint['augmented']['episode_ids_sha256']):
        raise ValueError('Wrong registered held-out endpoint')
    ratios = {k: endpoint['augmented']['metrics'][k] / endpoint['reference']['metrics'][k]
              for k in ('expert_kl', 'behavior_ce', 'value_mse')}
    gate = registration['primary_screen']
    gates = {'expert_kl': ratios['expert_kl'] <= gate['maximum_deep_expert_kl_ratio'],
             'behavior_ce': ratios['behavior_ce'] <= gate['maximum_deep_behavior_ce_ratio'],
             'value_mse': ratios['value_mse'] <= gate['maximum_value_mse_ratio']}
    result = {'schema_version': 1, 'kind': 'visual_233m_d4_learning_audit', 'audit_status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256,
        'evidence': evidence, 'matched_update_records': 4096, 'matched_exposure': final['reference']['counters'],
        'independently_replayed_d4_transforms': sum(counts.values()), 'symmetry_counts': dict(counts),
        'validation': {name: r['metrics'] for name, r in endpoint.items()},
        'test': {name: r['test'] for name, r in final.items()}, 'validation_ratios': ratios,
        'component_gates': gates, 'component_screen_passed': all(gates.values()),
        'learning_attempt_chip_hours': cost,
        'segment_timings': {name: r['segment_timing'] for name, r in final.items()},
        'limitations': 'One seed and a pre-existing weak-teacher dataset. D4 is an established Go baseline. No playing-strength, architecture superiority, speculative throughput or online RL improvement follows.',
        'next_gate': registration['next_gate']}
    a.output.write_bytes(canonical_json(result))
    print(canonical_json({k: result[k] for k in ('audit_status', 'component_screen_passed', 'validation_ratios',
        'matched_update_records', 'independently_replayed_d4_transforms')}).decode().strip())


if __name__ == '__main__':
    main()
