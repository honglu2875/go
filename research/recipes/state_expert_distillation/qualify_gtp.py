#!/usr/bin/env python3
"""Qualify both expert adapters against actual KataGo boards and terminal scoring."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src')); sys.path.insert(0, str(SOURCE / 'eval'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient, GTPCommandError
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.scoring import score_string
from gozero.snapshots import canonical_json, read_json, verify
from qualify_katago import kata_cells


def require(value, message):
    if not value:
        raise ValueError(message)


def vertex(action):
    return 'pass' if action == 81 else 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[action % 9] + str(9 - action // 9)


def raw_policy_mask(response, size):
    """Read the pinned KataGo raw-NN strict legality mask, without using its targets."""
    words = response.split()
    require(all(words.count(k) == 1 for k in ('symmetry', 'policy', 'policyPass'))
            and words[words.index('symmetry') + 1] == '0', 'Raw policy requires one symmetry-0 response')
    start = words.index('policy') + 1
    points = [float(x) for x in words[start:start + size * size]]
    values = points + [float(words[words.index('policyPass') + 1])]
    require(len(points) == size * size and math.isfinite(values[-1])
            and all(math.isnan(x) or math.isfinite(x) and 0 <= x <= 1 for x in values)
            and abs(sum(x for x in values if math.isfinite(x)) - 1.) < 1e-3, 'Invalid raw policy probability field')
    return [not math.isnan(x) for x in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'GTP qualification protocol changed')
    protocol = read_json(a.protocol)
    require(protocol['kind'] == 'state_expert_gtp_qualification' and protocol['snapshot'] == SOURCE.name
            and protocol['maximum_attempts'] == 1 and protocol['registered_unix'] < time.time(), 'Wrong GTP qualification identity')
    prerequisite = root / 'research/studies/state_expert_distillation/cpu_result.json'
    require(sha256(prerequisite) == protocol['cpu_qualification_sha256'] and read_json(prerequisite)['status'] == 'passed',
            'CPU model/recovery qualification differs')
    require(a.output.resolve() == root / protocol['output'] and not a.output.exists(), 'Wrong or existing GTP qualification output')
    for name, expected in protocol['files_sha256'].items():
        require(sha256(SOURCE / name) == expected, 'Registered GTP input changed: ' + name)
    native_receipt = artifact(root, protocol['native_receipt']); receipt = read_json(native_receipt)
    require(sha256(native_receipt) == protocol['native_receipt_sha256']
            and sha256(native_receipt.parent / receipt['filename']) == receipt['binary_sha256'], 'Native receipt or binary differs')
    native = load_library(native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
    kata = read_json(SOURCE / 'eval/katago_build.json'); weights = read_json(SOURCE / protocol['katago_weights'])
    require(sha256(artifact(root, kata['binary_path'])) == kata['binary_sha256']
            and weights['role'] == 'evaluation_only' and sha256(artifact(root, weights['path'])) == weights['sha256'], 'KataGo executable or weights differ')
    fixture = read_json(SOURCE / protocol['fixture']); infer = read_json(SOURCE / protocol['inference'])
    allowed = set(os.sched_getaffinity(0))
    candidate_cpus, kata_cpus = protocol['candidate_cpus'], protocol['katago_cpus']
    require(candidate_cpus and kata_cpus and not set(candidate_cpus) & set(kata_cpus)
            and set(candidate_cpus + kata_cpus) <= allowed, 'CPU groups overlap or escape the controller')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': protocol['kind'], 'status': 'failed', 'snapshot': SOURCE.name,
              'protocol_sha256': a.expected_protocol_sha256, 'started_unix': time.time(), 'arms': {},
              'new_competitive_games': 0, 'claims_strength': False, 'claims_throughput': False,
              'scope': 'Searched exact-state cases and forced replay of a scoring fixture; not a playing-strength benchmark.'}
    deadline = time.monotonic() + protocol['maximum_seconds']
    def command(client, text):
        remaining = deadline - time.monotonic(); require(remaining > 0, 'GTP qualification deadline expired')
        return client.command(text, timeout=min(60., remaining))
    try:
        for mode in ('history', 'state'):
            candidate = SOURCE / protocol['candidates'][mode]; descriptor = read_json(candidate); trained = validate(root, descriptor)
            require(trained['config']['model']['architecture'] == mode, 'Candidate architecture differs')
            folder = a.output / mode; folder.mkdir(); checked = 0; rejected = 0; cases = []
            rejection_cases = []
            for name, expected_error in (('architecture', 'Expert architecture'), ('behavior_enabled', 'Expert architecture'),
                                          ('observer_weights', 'Fixed observer identity')):
                synthetic = read_json(artifact(root, descriptor['training_result_path']))
                if name == 'architecture':
                    synthetic['expert_architecture'] = 'state' if mode == 'history' else 'history'
                elif name == 'behavior_enabled':
                    synthetic['behavior_training_enabled'] = True
                else:
                    synthetic['fixed_observer']['weights_sha256'] = '0' * 64
                synthetic_path = folder / ('invalid-' + name + '.json'); synthetic_path.write_bytes(canonical_json(synthetic))
                bad_descriptor = {**descriptor, 'training_result_path': str(synthetic_path.relative_to(root)),
                                  'training_result_sha256': sha256(synthetic_path)}
                try:
                    validate(root, bad_descriptor)
                except ValueError as error:
                    require(expected_error in str(error), 'Invalid artifact rejected for an unexpected reason')
                    rejection_cases.append({'case': name, 'error': str(error)})
                else:
                    raise ValueError('Invalid expert artifact was accepted')
            candidate_argv = ['taskset', '-c', ','.join(map(str, candidate_cpus)), sys.executable, '-B', str(SOURCE / 'eval/state_expert_gtp.py'),
                '--candidate', str(candidate), '--native-receipt', str(native_receipt), '--artifacts-root', str(root),
                '--simulations', str(infer['simulations']), '--cpuct', str(infer['cpuct']),
                '--inference-config', str(SOURCE / protocol['inference']), '--inference-native-snapshot', receipt['snapshot_id']]
            kata_argv = ['taskset', '-c', ','.join(map(str, kata_cpus)), str(artifact(root, kata['binary_path'])), 'gtp',
                '-model', str(artifact(root, weights['path'])), '-config', str(SOURCE / protocol['katago_config'])]
            env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            with GTPClient(candidate_argv, folder / 'candidate', environment=env) as engine, GTPClient(kata_argv, folder / 'katago') as referee:
                require(command(engine, 'version') == 'gozero-state-expert-native-leaf-v1', 'Wrong candidate adapter')
                actual_kata_version = command(referee, 'version')
                for client in (engine, referee):
                    command(client, 'boardsize 9'); command(client, 'komi 7.5')
                rules = json.loads(command(referee, 'kata-get-rules'))
                require(rules['ko'] == 'POSITIONAL' and rules['scoring'] == 'AREA' and rules['suicide'], 'KataGo rules differ')
                def check_board():
                    nonlocal checked
                    require(command(engine, 'showboard') == kata_cells(command(referee, 'showboard'), 9), 'Native and KataGo boards differ')
                    checked += 1
                def replay(moves):
                    for client in (engine, referee):
                        command(client, 'clear_board')
                    check_board()
                    for ply, move in enumerate(moves):
                        for client in (engine, referee):
                            command(client, 'play ' + ('B' if ply % 2 == 0 else 'W') + ' ' + move)
                        check_board()
                for case in protocol['search_cases']:
                    moves = [vertex(x) for x in case['actions']]; replay(moves)
                    color = 'B' if len(moves) % 2 == 0 else 'W'
                    # KataGo GTP play uses tolerant legality. Its NN evaluator
                    # masks using BoardHistory::isLegal, which is strict.
                    external_mask = raw_policy_mask(command(referee, 'kata-raw-nn 0'), 9)
                    import numpy as np
                    tape = np.asarray(case['actions'] + [81], np.int32)
                    _, native_masks, _ = native.replay_observations(json.dumps({'size': 9, 'komi': 7.5, 'scoring': 'pass_alive_area'}),
                                                                  tape, np.asarray([0, len(tape)], np.int64))
                    expected_mask = native_masks.reshape(-1, 82)[-1].tolist()
                    require(external_mask == expected_mask, 'KataGo strict policy mask differs from native replay')
                    if 'illegal_action' in case:
                        require(not external_mask[case['illegal_action']], 'KataGo strict mask permits declared superko')
                        try:
                            command(engine, 'play ' + color + ' ' + vertex(case['illegal_action']))
                        except GTPCommandError:
                            rejected += 1
                        else:
                            raise ValueError('Native superko action was accepted')
                        check_board()
                    action = command(engine, 'genmove ' + color)
                    require(action.lower() != 'resign', 'Unexpected resignation')
                    command(referee, 'play ' + color + ' ' + action); check_board()
                    stats = json.loads(command(engine, 'gozero-search-stats'))
                    require(stats['simulations'] == infer['simulations']
                            and stats['neural_evaluations'] + stats['terminal_evaluations'] == infer['simulations'] + 1,
                            'Actual search work differs')
                    cases.append({'id': case['id'], 'action': action, 'stats': stats, 'strict_legal_mask': external_mask})
                replay(fixture['action_vertices'])
                actual, reference = command(engine, 'final_score'), command(referee, 'final_score')
                require(actual == reference == score_string(fixture['katago_adjudicated_white_minus_black']), 'Terminal scoring fixture differs')
            require(engine.process.returncode == referee.process.returncode == 0, 'GTP process did not exit cleanly')
            ready = [json.loads(line) for line in (folder / 'candidate/stderr.log').read_text().splitlines() if line.startswith('{')]
            ready = [r for r in ready if r.get('kind') == 'engine_ready']; require(len(ready) == 1, 'Missing or repeated model identity')
            identity = ready[0]
            require(identity['candidate_sha256'] == sha256(candidate) and identity['training_snapshot'] == descriptor['training_snapshot']
                    and identity['weights_sha256'] == descriptor['model_export_sha256'] and identity['expert_architecture'] == mode
                    and identity['model_code_sha256'] == trained['model_code_sha256'] and identity['native_sha256'] == receipt['binary_sha256']
                    and identity['native_snapshot'] == receipt['snapshot_id'] and identity['inference_config_sha256'] == sha256(SOURCE / protocol['inference'])
                    and identity['fixed_observer'] == trained['training_result']['fixed_observer'] and identity['observer_used_as_search_policy'] is False
                    and identity['backend'] == 'cpu' and identity['state_feature_channels'] == 9
                    and identity['simulations_excluding_root'] == infer['simulations'] and identity['cpuct'] == infer['cpuct']
                    and identity['context_tokens'] == trained['config']['model']['max_tokens']
                    and identity['maximum_game_moves'] == infer['max_game_moves']
                    and set(identity['helper_code_sha256']) == {'history_model.py', 'spatial_model.py', 'state_features.py'},
                    'Actual loaded model, search, native or observer identity differs')
            for name, digest in identity['helper_code_sha256'].items():
                require(sha256(trained['snapshot'] / trained['manifest']['recipe'] / name) == digest, 'Loaded model helper differs')
            report['arms'][mode] = {'candidate_sha256': sha256(candidate), 'loaded_identity': identity, 'checked_boards': checked,
                'superko_rejections': rejected, 'searched_cases': cases, 'terminal_score': actual, 'katago_version': actual_kata_version,
                'strict_legal_mask_entries_checked': 82 * len(cases),
                'artifact_rejection_cases': rejection_cases,
                'raw_files_sha256': {str(f.relative_to(folder)): sha256(f) for f in sorted(folder.rglob('*')) if f.is_file()}}
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(report))
        verify(SOURCE); require(sha256(a.protocol) == a.expected_protocol_sha256, 'GTP qualification protocol changed during execution')
        print(json.dumps({k: v for k, v in report.items() if k != 'arms'}), flush=True)


if __name__ == '__main__':
    main()
