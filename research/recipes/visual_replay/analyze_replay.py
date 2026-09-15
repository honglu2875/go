"""Recompute the registered critical-path native suffix scorer comparison."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    study = root / 'research/studies/visual_causal'; registration = read_json(study / 'native_suffix_registration.json')
    path = root / 'runs' / a.attempt; closed = read_json(path / 'result.json')
    stage = next(s for s in registration['stages'] if s['stage'] == '233m_scorer_screen')
    if closed['status'] != 'passed' or closed['snapshot_id'] != stage['snapshot_id']: raise ValueError('Wrong completed profile')
    source = root / '.gozero/snapshots' / stage['snapshot_id']; verify(source)
    if sha256(source / 'resolved_config.json') != stage['config_sha256']: raise ValueError('Profile conditions changed')
    config = read_json(source / 'resolved_config.json'); reports = []
    evidence = {str((path / 'result.json').relative_to(root)): sha256(path / 'result.json')}
    keys = ('requests', 'prefix_hits', 'dispatches', 'appended_positions', 'padded_position_slots', 'native_validated_history_moves')
    for host in range(config['expected_processes']):
        rp = path / f'rank-{host}/artifacts/result.json'; r = read_json(rp); reports.append(r)
        evidence[str(rp.relative_to(root))] = sha256(rp)
        if (r['status'] != 'passed' or r['snapshot_id'] != source.name or r['host_rank'] != host
                or r['parameter_count'] != 233137152 or r['native_receipt_sha256'] != registration['native_receipt_sha256']
                or len(r['repetitions']) != config['repetitions']): raise ValueError('Profile rank identity differs')
        leaves = sum(v['exact_leaf_predictions'] for v in r['repetitions'])
        if r['bitwise_head_comparisons'] != leaves * 6: raise ValueError('Not every head was compared')
        for repetition, v in enumerate(r['repetitions']):
            if v['repetition'] != repetition or len(v['roots']) > config['slots'] * config['plies']: raise ValueError('Profile work coverage differs')
            for key in keys:
                if v['stats']['full'][key] != v['stats']['suffix'][key]: raise ValueError('Search work differs')
            for arm in ('full', 'suffix'):
                if v['stats'][arm]['requests'] != v['exact_leaf_predictions'] or v['scorer_seconds'][arm] <= 0:
                    raise ValueError('Profile request count or timing differs')
    if len({r['candidate_sha256'] for r in reports}) != 1: raise ValueError('Model weights differ')
    critical = {arm: [max(r['repetitions'][i]['scorer_seconds'][arm] for r in reports) for i in range(config['repetitions'])]
                for arm in ('full', 'suffix')}
    medians = {arm: statistics.median(v) for arm, v in critical.items()}
    speed = medians['full'] / medians['suffix']
    sums = {arm: {key: sum(v['stats'][arm][key] for r in reports for v in r['repetitions'])
                  for key in reports[0]['repetitions'][0]['stats'][arm]} for arm in ('full', 'suffix')}
    result = {'schema_version': 1, 'kind': 'visual_native_suffix_profile_audit', 'status': 'passed',
              'operator_snapshot': SOURCE.name, 'registration_sha256': sha256(study / 'native_suffix_registration.json'),
              'attempt': a.attempt, 'source_snapshot': source.name, 'evidence': evidence,
              'candidate_sha256': reports[0]['candidate_sha256'], 'critical_seconds': critical,
              'critical_median_seconds': medians, 'critical_median_scorer_speedup': speed,
              'bitwise_head_comparisons': sum(r['bitwise_head_comparisons'] for r in reports),
              'exact_leaf_predictions': sum(v['exact_leaf_predictions'] for r in reports for v in r['repetitions']),
              'aggregate_counters_and_seconds': sums,
              'passed_registered_scorer_gate': speed >= registration['primary_screen']['minimum_critical_median_scorer_speedup'],
              'registered_gate': registration['primary_screen'],
              'scope': 'Paired actual native MCTS leaf scorer, same trained model and workload; all six returned heads bitwise equal. Critical host maximum per repetition, then median. Excludes shared search advancement, initial prefill/compilation and external queues. No end-to-end MFU or Go strength claim.',
              'next_gate': registration['next_gate']}
    with a.output.open('xb') as f: f.write(canonical_json(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence', 'aggregate_counters_and_seconds')}))


if __name__ == '__main__': main()
