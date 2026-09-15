"""Audit bounded policy-loop work, exact final boards and retained cache lengths."""
import argparse
import hashlib
import json
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
from gozero.visual_sequence_batches import Dataset
from gozero.visual_history import Replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--stage', choices=['tpu_small', 'tpu_233m'], required=True)
    p.add_argument('--attempt', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    if sha256(a.registration) != a.registration_sha256 or a.output.exists(): raise ValueError('Registration differs or output exists')
    reg = read_json(a.registration); stage = next(x for x in reg['stages'] if x['stage'] == a.stage)
    source = root / '.gozero/snapshots' / stage['snapshot_id']; verify(source)
    if sha256(source / 'resolved_config.json') != stage['config_sha256']: raise ValueError('Scientific configuration changed')
    c = read_json(source / 'resolved_config.json'); base = root / 'runs' / a.attempt; evidence = {}
    def remember(path):
        evidence[str(path.relative_to(root))] = sha256(path); return read_json(path)
    closed = remember(base / 'result.json')
    if closed['status'] != 'passed' or closed['snapshot_id'] != source.name: raise ValueError('Incomplete or different attempt')
    receipt_path = artifact(root, c['native_receipt']); receipt = remember(receipt_path)
    native = load_library(receipt_path.parent / receipt['filename'], receipt['binary_sha256'])
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256']); rules = data.manifest['rules']; size = rules['size']
    replay = Replay(native, rules, c['cache_positions']); batch = c['sequences_per_host']; positions = c['root_positions']
    eligible = []
    for shard, episode in data.indices['expert', 1]:
        begin, end = map(int, data.shards[shard]['expert_offsets'][episode:episode + 2])
        if end - begin >= positions: eligible.append(['expert', shard, episode])
    reports = []; checked_moves = 0; completed_paths = 0; maxima = []
    for host in range(4):
        path = base / f'rank-{host}/artifacts'; r = remember(path / 'result.json'); reports.append(r)
        if (r['status'] != 'passed' or r['snapshot_id'] != source.name or r['host_rank'] != host
                or r['global_sequences'] != batch * 4 or r['native_receipt_sha256'] != sha256(receipt_path)):
            raise ValueError('Rank identity or topology differs')
        if a.stage == 'tpu_233m' and (r['parameter_count'] != 233137152 or r['candidate_sha256'] != reg['candidate_sha256']):
            raise ValueError('Full-model checkpoint differs')
        rng = np.random.default_rng(c['seed'] + host)
        selected = [eligible[int(i)] for i in rng.choice(len(eligible), size=batch, replace=len(eligible) < batch)]
        histories = []
        for _, shard, episode in selected:
            arrays = data.shards[shard]; begin = int(arrays['expert_offsets'][episode])
            histories.append(arrays['expert_actions'][begin:begin + positions - 1].tolist())
        if histories != r['exact_root_histories'] or selected != r['selected_episode_ids']: raise ValueError('Root selection differs')
        obs = np.stack(replay(histories)); actions = np.asarray([h + [size * size] for h in histories], np.int32)
        if hashlib.sha256(obs.tobytes() + actions.tobytes()).hexdigest() != r['exact_root_input_sha256']:
            raise ValueError('Exact root observations differ')
        if [x['horizon'] for x in r['conditions']] != [0, *c['horizons']]: raise ValueError('Condition coverage differs')
        net = c['model'] if c['model'] is not None else read_json(root / '.gozero/snapshots' / read_json(artifact(source, c['candidate']))['training_snapshot'] / 'resolved_config.json')['model']
        stride = ((size + net['patch_size'] - 1) // net['patch_size']) ** 2 + 2
        for condition in r['conditions']:
            if len(condition['repetitions']) != c['repetitions']: raise ValueError('Missing repetition')
            for rep, record in enumerate(condition['repetitions']):
                if record['repetition'] != rep or record['horizon'] != condition['horizon']: raise ValueError('Wrong repetition identity')
                payload_path = path / record['records_path']; payload = remember(payload_path)
                if evidence[str(payload_path.relative_to(root))] != record['records_sha256']: raise ValueError('Loop transcript changed')
                tapes = payload['histories']; moves = 0; terminals = 0
                if len(tapes) != batch: raise ValueError('Wrong path coverage')
                for game, tape in enumerate(tapes):
                    added = len(tape) - len(histories[game]); moves += added
                    if tape[:len(histories[game])] != histories[game] or not 0 <= added <= c['moves_per_root']:
                        raise ValueError('Loop exceeded fixed work or changed root history')
                    terminal = tape[-2:] == [size * size, size * size]; terminals += int(terminal)
                    if not terminal and added != c['moves_per_root']: raise ValueError('Live root stopped early')
                    exact = replay([tape[:-1] if terminal else tape])[0][-1]
                    stones = np.argmax(np.stack([1 - exact[..., 0] - exact[..., 1], exact[..., 0], exact[..., 1]], -1), -1).reshape(-1).tolist()
                    if (stones != payload['final_stones'][game] or (payload['final_passes'][game] >= 2) != terminal
                            or payload['remaining'][game] != 0 or payload['cache_lengths'][game] != (len(tape) + 1 - int(terminal)) * stride - 1):
                        raise ValueError('Final native board, ending, work or retained cache length differs')
                if record['committed_moves'] != moves or record['native_positions_checked'] != moves:
                    raise ValueError('Committed-work accounting differs')
                if not 0 < record['seconds'] or not 0 <= record['maximum_full_policy_tv'] <= c['maximum_policy_tv']:
                    raise ValueError('Timing or full-reference numerical gate failed')
                if condition['horizon']:
                    accepted = sum(sum(x['accepted']) for x in payload['acceptance_records'])
                    replacements = sum(sum(v >= 0 for v in x['replacement']) for x in payload['acceptance_records'])
                    if (accepted != record['accepted_draft_moves'] or replacements != record['replacement_moves']
                            or moves != accepted + replacements or record['dispatches'] != 2 * record['iterations'] + 1
                            or len(payload['acceptance_records']) != record['iterations'] or record['neural_repairs'] > replacements):
                        raise ValueError('Acceptance and repair accounting differs')
                elif payload['acceptance_records'] or record['dispatches'] != record['iterations']:
                    raise ValueError('Serial execution accounting differs')
                record['terminal_paths'] = terminals; checked_moves += moves; completed_paths += terminals
                maxima.append(record['maximum_full_policy_tv'])
    conditions = []
    for index, horizon in enumerate([0, *c['horizons']]):
        repetitions = []
        for rep in range(c['repetitions']):
            rows = [r['conditions'][index]['repetitions'][rep] for r in reports]
            baseline = [r['conditions'][0]['repetitions'][rep] for r in reports]
            seconds = max(x['seconds'] for x in rows); moves = sum(x['committed_moves'] for x in rows)
            base_rate = sum(x['committed_moves'] for x in baseline) / max(x['seconds'] for x in baseline)
            repetitions.append({'critical_seconds': seconds, 'committed_moves': moves, 'moves_per_second': moves / seconds,
                'speedup_over_same_repetition_serial': moves / seconds / base_rate,
                'terminal_paths': sum(x['terminal_paths'] for x in rows), 'host_dispatches': [x['dispatches'] for x in rows],
                'host_neural_repairs': [x['neural_repairs'] for x in rows], 'accepted_draft_moves': sum(x['accepted_draft_moves'] for x in rows),
                'replacement_moves': sum(x['replacement_moves'] for x in rows),
                'critical_seconds_including_final_audit': max(x['seconds'] + x['audit_seconds'] for x in rows)})
        ratio = statistics.median(x['speedup_over_same_repetition_serial'] for x in repetitions)
        conditions.append({'horizon': horizon, 'repetitions': repetitions, 'median_speedup': ratio,
            'passes_timing_screen': bool(horizon and ratio >= reg['gate']['minimum_median_committed_moves_per_second_speedup'])})
    result = {'schema_version': 1, 'kind': 'visual_continuous_policy_loop_audit', 'status': 'passed',
        'operator_snapshot': SOURCE.name, 'registration_sha256': a.registration_sha256, 'stage': a.stage,
        'evidence': evidence, 'attempt_chip_hours': closed['reserved_chip_hours'], 'conditions': conditions,
        'native_committed_moves_checked': checked_moves, 'terminal_paths': completed_paths, 'maximum_reported_full_policy_tv': max(maxima),
        'passes_timing_screen': any(x['passes_timing_screen'] for x in conditions),
        'limitations': 'A bounded expert-policy recurrence with deep replacement verification deferred into the next packet. All rollback, deferred work and final deep drain are timed; initial prefill and compilation are excluded. Different sampled paths can end at different lengths, and BF16 targets have measured differences from full-prefix evaluation. Few repetitions are a screen, not a robust performance interval. Raw tapes/final states are independently audited; numerical comparisons are retained as per-rank metrics. Individual probability packets and acceptance uniforms were not saved for independent draw replay. CPU tests check the pending-root target against full-prefix predictions and native boards. No MCTS, hardware MFU, strength or RL sample-efficiency claim.'}
    with a.output.open('xb') as stream: stream.write(canonical_json(result))
    print(canonical_json({k: v for k, v in result.items() if k != 'evidence'}).decode().strip())


if __name__ == '__main__': main()
