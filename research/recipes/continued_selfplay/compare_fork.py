#!/usr/bin/env python3
"""Compare a complete-state fork with the original learner's continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoint_forks as forks, checkpoints
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def load(root, directory, final_turn):
    result = read_json(directory / 'result.json'); source = root / '.gozero/snapshots' / result['snapshot_id']; verify(source)
    path = directory / 'checkpoints' / f'turn-{final_turn:09d}'; group_path = path.with_suffix('.group.json'); group = read_json(group_path)
    state, arrays, actors = checkpoints.read(path, expected_manifest_sha256=result['latest_checkpoint']['manifest_sha256'])
    require(result['status'] == 'passed' and result['turn'] == final_turn == state['turn'] == group['turn']
            and state['snapshot_id'] == group['snapshot_id'] == source.name
            and state['config_sha256'] == group['config_sha256'] == result['config_sha256'] == checkpoints.sha256(source / 'resolved_config.json')
            and state['jax_rank'] == result['jax_rank'] and state['world_size'] == group['world_size'] == result['world_size']
            and group['rank_manifests'][state['jax_rank']] == result['latest_checkpoint']['manifest_sha256']
            and checkpoints.sha256(group_path) == result['latest_checkpoint']['group_sha256']
            and state['counters']['updates'] == group['updates']
            and state['native_sha256'] == result['native']['binary_sha256']
            and forks.replicated_digest(arrays) == group['replicated_state_sha256'], 'Checkpoint/result/group identity differs')
    require(checkpoints.sha256(directory / 'model_export.npz') == result['model_export_sha256'], 'Export changed')
    with np.load(directory / 'model_export.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {k for k in arrays if k.startswith('p_')}
                and all(np.array_equal(saved[k], arrays[k]) for k in saved.files), 'Export is not the checkpoint model')
    return result, source, state, arrays, actors, group


def compare(root, reference, child, final_turn, host_rank):
    ra, sa, a, aa, ga, group_a = load(root, reference, final_turn)
    rb, sb, b, ab, gb, group_b = load(root, child, final_turn)
    config = read_json(sb / 'resolved_config.json'); origin, spec, _, _ = forks.contract(root, sb, config)
    require(origin['kind'] == 'fork' and ra['snapshot_id'] == origin['parent_snapshot']
            and rb['kind'] == 'continued_selfplay_training' and rb['initialization'] == b['initialization'] == origin
            and spec['learning_rate_mode'] == 'inherit_constant', 'This comparison requires an unchanged original-parent fork')
    require(hashlib.sha256(canonical_json(origin)).hexdigest() == group_b['initialization_sha256'], 'Child group lost initialization provenance')
    rank = b['jax_rank']; entry = next(e for e in spec['rank_checkpoints'] if e['jax_rank'] == rank)
    require(entry['host_rank'] == host_rank and rb['host_rank'] == host_rank
            and b['phase_start_turn'] == rb['phase_start_turn'] == spec['parent_group']['turn']
            and b['phase_start_counters'] == entry['counters'] and rb['phase_turns'] == final_turn - b['phase_start_turn']
            and rb['forked_from']['parent_group_sha256'] == spec['parent_group_sha256'], 'Fork phase or rank identity differs')
    expected_delta = {k: v - entry['counters'][k] for k, v in forks.work_counters(b['counters']).items()}
    require(rb['phase_counters'] == expected_delta, 'Phase counters reset or double-counted parent work')
    require(set(aa) == set(ab), 'Checkpoint array trees differ')
    for name in aa:
        require(aa[name].dtype == ab[name].dtype and np.array_equal(aa[name], ab[name]), 'Saved array differs: ' + name)
    require(json.loads(ga) == json.loads(gb), 'Complete actor state differs')
    require(ra['model_export_sha256'] == rb['model_export_sha256'], 'Exported parameter bytes differ')
    # Strip only provenance already explicitly checked above. Native identity,
    # replay metadata, sampler state, last metrics and all work counters remain.
    for state in (a, b):
        state.pop('snapshot_id'); state.pop('config_sha256')
        state['counters'] = forks.work_counters(state['counters'])
    for name in ('initialization', 'phase_start_turn', 'phase_start_counters'):
        b.pop(name)
    require(a == b, 'Scientific state differs beyond verified fork provenance')
    games = sorted((child / 'games').glob('*.json'))
    expected_games = expected_delta['completed_games'] + expected_delta['truncated_games']
    require(len(games) == expected_games, 'Fork game record count differs')
    for game in games:
        for suffix in ('.json', '.sgf'):
            path = game.with_suffix(suffix)
            require(path.read_bytes() == (reference / 'games' / path.name).read_bytes(), 'Subsequent complete game differs')
    return {'host_rank': host_rank, 'jax_rank': rank, 'arrays_exact': len(aa), 'complete_actors_exact': True,
            'scientific_state_exact_except_verified_provenance': True, 'subsequent_games_exact': len(games),
            'phase_counters': expected_delta, 'parent_turn': spec['parent_group']['turn'], 'final_turn': final_turn,
            'parent_snapshot': sa.name, 'child_snapshot': sb.name, 'model_export_sha256': rb['model_export_sha256'],
            'reference_result_sha256': checkpoints.sha256(reference / 'result.json'),
            'fork_result_sha256': checkpoints.sha256(child / 'result.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--fork', type=Path, required=True); p.add_argument('--hosts', type=int, choices=(1, 4), required=True)
    p.add_argument('--final-turn', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    report = {'schema_version': 1, 'kind': 'complete_selfplay_fork_comparison', 'status': 'failed', 'analysis_snapshot': SOURCE.name,
              'reference': str(a.reference), 'fork': str(a.fork), 'hosts': [],
              'scope': 'Exact fresh-process continuation with unchanged scientific settings; same topology, disks and native binary. No injected failure or strength claim.'}
    try:
        for host in range(a.hosts):
            relative = Path('.') if a.hosts == 1 else Path(f'rank-{host}/artifacts')
            report['hosts'].append(compare(root, a.reference / relative, a.fork / relative, a.final_turn, host))
        require({r['jax_rank'] for r in report['hosts']} == set(range(a.hosts)), 'Rank coverage differs')
        if a.hosts == 4:
            require(all(read_json(path / 'result.json')['status'] == 'passed' for path in (a.reference, a.fork)), 'Pod controller failed')
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        with a.output.open('xb') as stream:
            stream.write(canonical_json(report))
        a.output.chmod(0o444); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
