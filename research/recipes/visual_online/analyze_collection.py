"""Aggregate all three fixed collection diagnostics with complete game coverage."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset
from evaluate_collection import entries


def averages(raw):
    result = {k: v for k, v in raw.items() if k.endswith('_count')}
    for k, v in raw.items():
        if not k.endswith('_count'): result[k] = v / max(1., raw[k.split('_')[0] + '_count'])
    result['expert_kl'] = result['expert_ce'] - result['expert_target_entropy']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--attempts', nargs=3, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    if sha256(a.registration) != a.registration_sha256 or a.output.exists(): raise ValueError('Registration differs or output exists')
    reg = read_json(a.registration); evidence = {}; results = {}; costs = 0.; previous_ids = None
    def remember(path): evidence[str(path.relative_to(root))] = sha256(path); return read_json(path)
    for arm, attempt in zip(reg['arms'], a.attempts):
        source = root / '.gozero/snapshots' / arm['snapshot_id']; verify(source)
        if sha256(source / 'resolved_config.json') != arm['config_sha256']: raise ValueError('Configuration differs')
        c = read_json(source / 'resolved_config.json'); base = root / 'runs' / attempt; closed = remember(base / 'result.json')
        if closed['status'] != 'passed' or closed['snapshot_id'] != source.name: raise ValueError('Incomplete registered diagnostic')
        costs += closed['reserved_chip_hours']; raw = {}; all_ids = []
        for host in range(4):
            r = remember(base / f'rank-{host}/artifacts/result.json')
            if (r['status'] != 'passed' or r['snapshot_id'] != source.name or r['host_rank'] != host
                    or r['candidate_sha256'] != arm['candidate_sha256'] or r['parameter_count'] != 233137152):
                raise ValueError('Rank or trained model identity differs')
            data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'], rank=host, world=4)
            groups, selected, total = entries(data, root / c['collection_receipt'], c['collection_receipt_sha256'], c['buckets'])
            if selected != r['selected_games'] or total != reg['games']: raise ValueError('Held-out selection differs')
            all_ids.extend(selected); per_rank = {}; expected_batches = []
            for b, rows in groups.items():
                for index in range(0, len(rows), c['sequences_per_host']): expected_batches.append((b, index))
            if [(b['bucket'], b['index']) for b in r['batches']] != expected_batches: raise ValueError('Batch coverage differs')
            for batch in r['batches']:
                totals = batch['totals']
                if not np.isfinite(list(totals.values())).all(): raise ValueError('Nonfinite metrics')
                for k, v in totals.items(): per_rank[k] = per_rank.get(k, 0.) + v
            if per_rank != r['raw_totals'] or averages(per_rank) != r['averages']: raise ValueError('Rank metric arithmetic differs')
            if per_rank['expert_count'] != sum(x['moves'] for x in selected) or per_rank['behavior_count'] != 0:
                raise ValueError('Position or role coverage differs')
            for k, v in per_rank.items(): raw[k] = raw.get(k, 0.) + v
        if (len(all_ids) != reg['games'] or len({x['id'] for x in all_ids}) != len(all_ids)
                or sum(x['moves'] for x in all_ids) != reg['positions']): raise ValueError('Incomplete global holdout population')
        if previous_ids is not None and all_ids != previous_ids: raise ValueError('Arms used different complete games')
        previous_ids = all_ids
        results[arm['arm']] = {'attempt': attempt, 'candidate_sha256': arm['candidate_sha256'], 'raw_totals': raw,
                              'averages': averages(raw), 'attempt_chip_hours': closed['reserved_chip_hours']}
    ratios = {key: results['selfplay']['averages'][key] / results['control']['averages'][key]
              for key in ['expert_kl', 'expert_ce', 'value_mse']}
    report = {'schema_version': 1, 'kind': 'visual_collection_posthoc_audit', 'status': 'passed', 'operator_snapshot': SOURCE.name,
        'registration_sha256': a.registration_sha256, 'evidence': evidence, 'games_per_arm': reg['games'], 'positions_per_arm': reg['positions'],
        'arms': results, 'selfplay_over_control_loss_ratios': ratios, 'attempt_chip_hours': costs,
        'scope': 'Complete213 game/19557 position collection-held-out diagnostic after observing the fixed training and external-game endpoints. These are correlated position losses, not213independent position samples or a new model-selection criterion. Metric arithmetic and eligibility/coverage are independently replayed; individual logits were not exported for an independent model recomputation. No strength or RL speedup claim.'}
    with a.output.open('xb') as stream: stream.write(canonical_json(report))
    print(canonical_json({k: v for k, v in report.items() if k != 'evidence'}).decode().strip())


if __name__ == '__main__': main()
