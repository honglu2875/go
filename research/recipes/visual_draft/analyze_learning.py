#!/usr/bin/env python3
"""Audit the prespecified 233M early-exit pilot and its complete matched exposure."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-attempts', nargs=2, required=True); p.add_argument('--early-exit-attempt', required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if sha256(a.registration) != a.registration_sha256 or a.output.exists():
        raise ValueError('Registration identity differs or result already exists')
    registration = read_json(a.registration); arms = {}; evidence = {}; cost = 0.
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path)
        return read_json(path)
    for declared, attempts in zip(registration['arms'], [a.control_attempts, [a.early_exit_attempt]]):
        snapshot = root / '.gozero/snapshots' / declared['snapshot_id']; manifest = verify(snapshot)
        config = read_json(snapshot / 'resolved_config.json')
        if sha256(snapshot / 'resolved_config.json') != declared['config_sha256']:
            raise ValueError('Learning configuration changed')
        ranks = []
        for host in range(4):
            rows = []; reports = []
            for attempt in attempts:
                directory = root / 'runs' / attempt
                closed = remember(directory / 'result.json')
                if closed['status'] != 'passed' or closed['snapshot_id'] != snapshot.name:
                    raise ValueError('Pilot attempt did not close successfully')
                if host == 0:
                    cost += closed['reserved_chip_hours']
                report = remember(directory / f'rank-{host}/artifacts/result.json'); reports.append(report)
                if report['status'] != 'passed' or report['host_rank'] != host or report['snapshot_id'] != snapshot.name:
                    raise ValueError('Host scientific result differs')
                path = directory / f'rank-{host}/artifacts/metrics.jsonl'; evidence[str(path.relative_to(root))] = sha256(path)
                rows.extend(json.loads(line) for line in path.read_text().splitlines())
            final = reports[-1]
            if (not final['training_complete'] or final['turn'] != config['steps']
                    or [r['turn'] for r in rows] != list(range(1, config['steps'] + 1))
                    or any(r['accepted'] != 1 for r in rows)):
                raise ValueError('Run missed a scheduled update or rejected a gradient')
            group_path = Path(final['latest_checkpoint']['path']).with_suffix('.group.json')
            group = remember(group_path)
            if (sha256(group_path) != final['latest_checkpoint']['group_sha256']
                    or sha256(Path(final['latest_checkpoint']['path']) / 'manifest.json') != group['host_manifests'][str(host)]):
                raise ValueError('Committed distributed checkpoint identity differs')
            ranks.append({'reports': reports, 'rows': rows})
        primary = ranks[0]['reports'][-1]
        for rank in ranks[1:]:
            r = rank['reports'][-1]
            for field in ['counters', 'validation_history', 'test', 'initial_parameter_elements_sha256']:
                if r[field] != primary[field]:
                    raise ValueError('Replicated scientific reports diverged: ' + field)
        arms[declared['arm']] = {'config': config, 'ranks': ranks, 'source_manifest': manifest}
    control, early = arms['control'], arms['early_exit']
    c0, c1 = json.loads(json.dumps(control['config'])), json.loads(json.dumps(early['config']))
    c0['exits']['loss_weight'] = c1['exits']['loss_weight']
    if c0 != c1:
        raise ValueError('Learning arms differ beyond registered auxiliary weight')
    for path, entry in control['source_manifest']['files'].items():
        if path.endswith('.py') and entry != early['source_manifest']['files'].get(path):
            raise ValueError('Matched arms used different implementation code')
    fields = ['turn', 'bucket', 'local_entries_sha256', 'expert_positions', 'behavior_positions', 'learning_rate']
    for host in range(4):
        for x, y in zip(control['ranks'][host]['rows'], early['ranks'][host]['rows']):
            if any(x[k] != y[k] for k in fields):
                raise ValueError('Arms did not receive matched inputs and scheduled updates')
    first_control = control['ranks'][0]['reports'][0]
    first_early = early['ranks'][0]['reports'][0]
    if first_control['initial_validation'] != first_early['initial_validation']:
        raise ValueError('Initialized model predictions or held-out population differ')
    final = {name: value['ranks'][0]['reports'][-1] for name, value in arms.items()}
    if (final['control']['initial_parameter_elements_sha256'] != final['early_exit']['initial_parameter_elements_sha256']
            or final['control']['counters'] != final['early_exit']['counters']):
        raise ValueError('Initial weights or total exposure differ')
    endpoint = {name: value['validation_history'][-1] for name, value in final.items()}
    if any(x['turn'] != 1024 or x['split'] != 1 for x in endpoint.values()):
        raise ValueError('Wrong registered validation endpoint')
    if endpoint['control']['episode_ids_sha256'] != endpoint['early_exit']['episode_ids_sha256']:
        raise ValueError('Held-out endpoint populations differ')
    tv = {name: statistics.mean(1 - result['metrics'][role + '_exit_3_distribution_overlap'] for role in ('expert', 'behavior'))
          for name, result in endpoint.items()}
    ratios = {key: endpoint['early_exit']['metrics'][key] / endpoint['control']['metrics'][key]
              for key in ['expert_kl', 'behavior_ce', 'value_mse']}
    screen = registration['primary_component_screen']
    gates = {'early_exit_tv': tv['early_exit'] / tv['control'] <= screen['maximum_average_total_variation_ratio_vs_control'],
        'expert_kl': ratios['expert_kl'] <= screen['maximum_deep_expert_kl_ratio'],
        'behavior_ce': ratios['behavior_ce'] <= screen['maximum_deep_behavior_ce_ratio'],
        'value_mse': ratios['value_mse'] <= screen['maximum_deep_value_mse_ratio']}
    result = {'schema_version': 1, 'kind': 'visual_233m_early_exit_learning_audit', 'audit_status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256, 'evidence': evidence,
        'matched_update_records': 4 * 1024, 'matched_exposure': final['control']['counters'],
        'initial_parameter_elements_sha256': final['control']['initial_parameter_elements_sha256'],
        'validation': {k: v['metrics'] for k, v in endpoint.items()},
        'test': {k: v['test'] for k, v in final.items()},
        'average_shallow_tv': tv, 'shallow_tv_ratio': tv['early_exit'] / tv['control'],
        'deep_validation_ratios': ratios, 'component_gates': gates, 'component_screen_passed': all(gates.values()),
        'attempt_chip_hours_including_initial_checkpoint_qualification': cost,
        'segment_timings': {k: [r['segment_timing'] for r in v['ranks'][0]['reports']] for k, v in arms.items()},
        'limitations': 'One seed, small pre-existing weak-teacher dataset, substantial overfitting in both arms. Probability overlap is measured on identical exact held-out histories; it is not actual speculative acceptance. No architecture superiority, real-game improvement or faster RL claim.',
        'next_gate': registration['next_gate']}
    a.output.write_bytes(canonical_json(result))
    print(canonical_json({k: result[k] for k in ['audit_status', 'component_screen_passed', 'shallow_tv_ratio', 'deep_validation_ratios', 'matched_exposure']}).decode().strip())


if __name__ == '__main__':
    main()
