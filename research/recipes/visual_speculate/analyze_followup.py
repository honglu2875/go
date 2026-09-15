"""Audit a registered checkpoint followup and reproduce every acceptance draw."""
import argparse
import hashlib
from pathlib import Path
import statistics
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_history import Replay
from gozero.visual_sequence_batches import Dataset
from speculate import resolve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--repair', type=Path)
    p.add_argument('--repair-sha256')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if sha256(a.registration) != a.registration_sha256 or a.output.exists():
        raise ValueError('Registration changed or output exists')
    reg = read_json(a.registration); repair = None
    source_id = reg['snapshot_id']
    if a.repair is not None:
        if sha256(a.repair) != a.repair_sha256: raise ValueError('Repair registration changed')
        repair = read_json(a.repair)
        if repair['parent_registration_sha256'] != a.registration_sha256: raise ValueError('Wrong repair lineage')
        source_id = repair['replacement_snapshot']
    elif a.repair_sha256 is not None:
        raise ValueError('Missing repair registration')
    source = root / '.gozero/snapshots' / source_id; verify(source)
    if sha256(source / 'resolved_config.json') != reg['config_sha256']:
        raise ValueError('Registered configuration differs')
    c = read_json(source / 'resolved_config.json')
    if sha256(artifact(source, c['candidate'])) != reg['candidate_sha256']:
        raise ValueError('Registered checkpoint differs')
    evidence = {}
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path)
        return read_json(path)
    directory = root / 'runs' / a.attempt; closed = remember(directory / 'result.json')
    if closed['status'] != 'passed' or closed['snapshot_id'] != source.name:
        raise ValueError('Attempt did not finish the registered execution')
    attempts = [closed]
    if repair is not None:
        failed = remember(root / 'runs' / repair['retained_failed_attempt'] / 'result.json')
        if failed['status'] != 'failed' or failed['snapshot_id'] != reg['snapshot_id']:
            raise ValueError('Repair failure history differs')
        attempts.append(failed)
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    rules = data.manifest['rules']; size = rules['size']; positions = c['root_positions']; batch = c['sequences_per_host']
    receipt_path = artifact(root, c['native_receipt']); receipt = remember(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    replay = Replay(native, rules, c['cache_positions'])
    eligible = []
    for shard, episode in data.indices['expert', 1]:
        arrays = data.shards[shard]; begin, end = map(int, arrays['expert_offsets'][episode:episode + 2])
        if end - begin >= positions: eligible.append(['expert', shard, episode])
    reports = []; decisions_checked = 0; future_positions_checked = 0
    expected_names = [f'h{h}-{role}' for h in c['horizons'] for role in c['draft_roles']]
    for host in range(4):
        base = directory / f'rank-{host}/artifacts'; r = remember(base / 'result.json'); reports.append(r)
        if (r['status'] != 'passed' or r['snapshot_id'] != source.name or r['host_rank'] != host
                or r['parameter_count'] != 233137152 or r['candidate_sha256'] != reg['candidate_sha256']
                or r['global_sequences'] != 4 * batch or r['native_receipt_sha256'] != sha256(receipt_path)
                or [x['name'] for x in r['conditions']] != expected_names):
            raise ValueError('Rank identity or scientific contract differs')
        rng = np.random.default_rng(c['seed'] + host)
        selected = [eligible[int(i)] for i in rng.choice(len(eligible), size=batch, replace=len(eligible) < batch)]
        histories = []
        for _, shard, episode in selected:
            arrays = data.shards[shard]; begin = int(arrays['expert_offsets'][episode])
            histories.append(arrays['expert_actions'][begin:begin + positions - 1].tolist())
        if selected != r['selected_episode_ids'] or histories != r['exact_root_histories']:
            raise ValueError('Held-out root draw differs')
        obs = np.stack(replay(histories)); actions = np.asarray([h + [size * size] for h in histories], np.int32)
        if hashlib.sha256(obs.tobytes() + actions.tobytes()).hexdigest() != r['exact_root_input_sha256']:
            raise ValueError('Complete-history root input differs')
        for record in r['conditions']:
            path = base / (record['name'] + '.npz')
            if sha256(path) != record['packet_arrays_sha256']: raise ValueError('Saved packet changed')
            evidence[str(path.relative_to(root))] = sha256(path)
            with np.load(path, allow_pickle=False) as archive: rows = dict(archive)
            horizon = record['horizon']
            if rows['actions'].shape != (batch, horizon): raise ValueError('Packet shape differs')
            if record['maximum_target_vs_full_policy_tv'] > c['maximum_policy_tv']:
                raise ValueError('Recorded full-reference numerical gate failed')
            for game, history in enumerate(histories):
                tape = list(history)
                for t in range(horizon):
                    if not rows['active'][game, t]:
                        if rows['active'][game, t:].any(): raise ValueError('Noncontiguous active proposals')
                        break
                    exact = replay([tape])[0][-1]
                    legal = np.r_[exact[..., 5].reshape(-1) > .5, True]
                    if not np.array_equal(legal, rows['legal'][game, t]): raise ValueError('Device legality differs from Rust')
                    action = int(rows['actions'][game, t])
                    if not legal[action]: raise ValueError('Illegal proposal')
                    tape.append(action); terminal = tape[-2:] == [size * size, size * size]
                    if bool(rows['appended'][game, t]) == terminal: raise ValueError('Terminal append mask differs')
                    if not terminal and not np.array_equal(replay([tape])[0][-1], rows['future_observations'][game, t]):
                        raise ValueError('Device future observation differs from Rust')
                    future_positions_checked += 1
            for repetition in range(c['repetitions']):
                decisions = resolve(rows, rng.random((batch, horizon)), rng.random((batch, horizon)))
                if decisions != record['resolutions'][repetition]: raise ValueError('Acceptance RNG or maximal coupling differs')
                tapes = []
                for history, decision in zip(histories, decisions):
                    tape = history + decision['actions']
                    tapes.append(tape[:-1] if tape[-2:] == [size * size, size * size] else tape)
                replay(tapes); decisions_checked += len(decisions)
    repetitions = c['repetitions']
    baseline = [max(r['one_full_step_seconds'][i] for r in reports) for i in range(repetitions)]
    reference = statistics.median(baseline); conditions = []
    for index, name in enumerate(expected_names):
        records = [r['conditions'][index] for r in reports]
        times = [max(r['packet_seconds'][i] for r in records) for i in range(repetitions)]
        audited = [max(r['packet_seconds'][i] + r['coupling_and_native_audit_seconds'][i] for r in records) for i in range(repetitions)]
        accepted = []; resolved = []
        for i in range(repetitions):
            decisions = [d for r in records for d in r['resolutions'][i]]
            accepted.append(statistics.mean(d['accepted_draft_moves'] for d in decisions))
            resolved.append(statistics.mean(len(d['actions']) for d in decisions))
        device_speed = statistics.median(reference * n / t for n, t in zip(resolved, times))
        with_audit = statistics.median(reference * n / t for n, t in zip(resolved, audited))
        conditions.append({'name': name, 'maximum_full_reference_policy_tv': max(r['maximum_target_vs_full_policy_tv'] for r in records),
            'critical_packet_seconds': times, 'critical_packet_and_audit_seconds': audited,
            'resolved_moves_per_root': resolved, 'accepted_draft_moves_per_root': accepted,
            'same_path_mean_overlap': statistics.mean(r['mean_same_path_overlap'] for r in records),
            'median_optimistic_amortized_speedup_excluding_all_cpu_audit': device_speed,
            'median_amortized_speedup_including_candidate_audit': with_audit,
            'screen_passed': with_audit >= reg['screen']['minimum_amortized_independent_root_speedup']})
    passed = any(x['screen_passed'] for x in conditions)
    result = {'schema_version': 1, 'kind': 'visual_speculation_checkpoint_followup_audit', 'audit_status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256, 'evidence': evidence,
        'candidate_sha256': reg['candidate_sha256'], 'critical_one_full_step_seconds': baseline, 'conditions': conditions,
        'acceptance_decisions_reproduced': decisions_checked, 'exact_future_positions_replayed': future_positions_checked,
        'screen_passed': passed, 'attempt_chip_hours': closed['reserved_chip_hours'],
        'repair_sha256': a.repair_sha256, 'attempt_chip_hours_including_failure': sum(x['reserved_chip_hours'] for x in attempts),
        'decision': ('Candidate merits a materialized-cache closed-loop qualification; this screen cannot promote a rollout algorithm.' if passed
                     else 'Do not promote this packet configuration: no condition passes the registered independent-root amortization screen.'),
        'limitations': 'Independent roots; one proposal draw per root reused for timing and fresh acceptance draws. CPU replay is charged only to the speculative arm; device-only ratios are also reported. Root prefill, final cache output materialization and future cache repair are excluded. The profile returns proposal rows only, allowing dead-code elimination of unused terminal predictions and caches. Thus even a positive result would require a materialized-cache continuous-loop comparison. Block BF16 target probabilities can differ from serial evaluation. This verifies policy coupling to the recorded target, not preservation of MCTS decisions. No completed-game throughput, hardware MFU or strength claim.'}
    with a.output.open('xb') as stream: stream.write(canonical_json(result))
    print(canonical_json({k: result[k] for k in ['audit_status', 'conditions', 'screen_passed', 'acceptance_decisions_reproduced', 'exact_future_positions_replayed']}).decode().strip())


if __name__ == '__main__': main()
