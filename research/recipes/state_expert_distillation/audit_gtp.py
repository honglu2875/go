#!/usr/bin/env python3
"""Audit the retained GTP qualification, including the failed parent attempt."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'eval'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
from qualify_katago import kata_cells


def require(value, message):
    if not value:
        raise ValueError(message)


def transcript(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    commands = {r['id']: r for r in rows if r['kind'] == 'command'}
    responses = {r['id']: r for r in rows if r['kind'] == 'response'}
    require(len(commands) == sum(r['kind'] == 'command' for r in rows)
            and len(responses) == sum(r['kind'] == 'response' for r in rows)
            and set(commands) == set(responses), 'Missing or duplicate protocol exchange')
    require(rows[0]['kind'] == 'start' and rows[-1]['kind'] == 'exit'
            and rows[-1]['returncode'] == 0 and not any(r['kind'] == 'transport_error' for r in rows),
            'Unclean protocol process')
    require(commands[max(commands)]['text'] == 'quit' and responses[max(commands)]['success'], 'Missing clean quit')
    return [(commands[k]['text'], responses[k]) for k in sorted(commands)]


def strict_mask(text):
    # Separate line-oriented reconstruction from the runner's token parser.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(lines.count('symmetry 0') == 1 and lines.count('policy') == 1, 'Unexpected raw-NN layout')
    first = lines.index('policy') + 1
    rows = [[float(x) for x in line.split()] for line in lines[first:first + 9]]
    require(len(rows) == 9 and all(len(row) == 9 for row in rows), 'Wrong policy board shape')
    passes = [line.split() for line in lines if line.startswith('policyPass ')]
    require(len(passes) == 1 and len(passes[0]) == 2, 'Wrong pass field')
    values = [x for row in rows for x in row] + [float(passes[0][1])]
    require(math.isfinite(values[-1]) and all(math.isnan(x) or math.isfinite(x) and 0 <= x <= 1 for x in values)
            and abs(sum(x for x in values if math.isfinite(x)) - 1) < 1e-3, 'Invalid raw probabilities')
    return [math.isfinite(x) for x in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'Protocol changed')
    spec = read_json(a.protocol); source = root / '.gozero/snapshots' / spec['snapshot']; verify(source)
    result_path = root / spec['output'] / 'result.json'; result = read_json(result_path)
    require(result['status'] == 'passed' and result['snapshot'] == source.name
            and result['protocol_sha256'] == sha256(a.protocol)
            and spec['registered_unix'] < result['started_unix'] < result['finished_unix']
            and result['finished_unix'] - result['started_unix'] < spec['maximum_seconds'], 'Wrong run identity or timing')
    for name, expected in spec['files_sha256'].items():
        require(sha256(source / name) == expected, 'Changed frozen input: ' + name)
    parent_path = a.protocol.parent / 'gtp_spec.json'; parent = read_json(parent_path)
    parent_result_path = root / parent['output'] / 'result.json'; parent_result = read_json(parent_result_path)
    require(sha256(parent_path) == spec['parent_protocol_sha256']
            and sha256(parent_result_path) == spec['parent_result_sha256']
            and parent_result['status'] == 'failed'
            and parent_result['finished_unix'] < spec['registered_unix'], 'Failed parent was not retained')
    parent_exchanges = [transcript(root / parent['output'] / 'history' / arm / 'gtp.jsonl') for arm in ('candidate', 'katago')]
    parent_responses = [[r for command, r in rows if command == 'play W C7'][-1] for rows in parent_exchanges]
    require(not parent_responses[0]['success'] and parent_responses[1]['success'], 'Parent failure interpretation differs')
    parser_path = root / 'runs/qualification/state-expert-raw-policy-7b89a249/result.json'; parser = read_json(parser_path)
    require(sha256(parser_path) == spec['parser_tests_sha256'] and parser['status'] == 'passed'
            and parser['returncode'] == 0 and parser['snapshot'] == source.name, 'Parser qualification differs')
    require(sha256(parser_path.parent / 'tests.log') == parser['log_sha256'], 'Parser log changed')
    cpu_path = a.protocol.parent / 'cpu_result.json'
    require(sha256(cpu_path) == spec['cpu_qualification_sha256'] and read_json(cpu_path)['status'] == 'passed', 'CPU qualification differs')
    summaries = {}
    for arm in ('history', 'state'):
        raw = result_path.parent / arm; reported = result['arms'][arm]
        for name, digest in reported['raw_files_sha256'].items():
            require(sha256(raw / name) == digest, 'Changed raw file')
        descriptor = read_json(source / spec['candidates'][arm]); trained = validate(root, descriptor)
        ready = [json.loads(line) for line in (raw / 'candidate/stderr.log').read_text().splitlines() if line.startswith('{')]
        ready = [row for row in ready if row.get('kind') == 'engine_ready']
        require(ready == [reported['loaded_identity']] and ready[0]['expert_architecture'] == arm
                and ready[0]['weights_sha256'] == descriptor['model_export_sha256']
                and ready[0]['training_snapshot'] == descriptor['training_snapshot']
                and ready[0]['fixed_observer'] == trained['training_result']['fixed_observer']
                and ready[0]['observer_used_as_search_policy'] is False, 'Loaded identity differs')
        engine, kata = [transcript(raw / actor / 'gtp.jsonl') for actor in ('candidate', 'katago')]
        ours = [r['text'] for c, r in engine if c == 'showboard']
        theirs = [kata_cells(r['text'], 9) for c, r in kata if c == 'showboard']
        require(ours == theirs and len(ours) == reported['checked_boards'] == 148, 'Board replay differs')
        masks = [strict_mask(r['text']) for c, r in kata if c == 'kata-raw-nn 0']
        require(masks == [case['strict_legal_mask'] for case in reported['searched_cases']]
                and len(masks) == 4 and not masks[2][20], 'Strict-mask reconstruction differs')
        scores = [[r['text'] for c, r in rows if c == 'final_score'] for rows in (engine, kata)]
        require(scores == [['W+88.5'], ['W+88.5']], 'Score differs')
        rejected = [(c, r) for c, r in engine if not r['success']]
        require(len(rejected) == 1 and rejected[0][0] == 'play W C7'
                and all(r['success'] for _, r in kata), 'Unexpected rejection')
        stats = [json.loads(r['text']) for c, r in engine if c == 'gozero-search-stats']
        require(stats == [case['stats'] for case in reported['searched_cases']]
                and all(s['simulations'] == 16 and s['neural_evaluations'] + s['terminal_evaluations'] == 17 for s in stats), 'Search accounting differs')
        require(len(reported['artifact_rejection_cases']) == 3, 'Missing artifact rejection case')
        summaries[arm] = {'checked_boards': len(ours), 'strict_mask_entries': sum(map(len, masks)),
                          'completed_fixture_scores': 1, 'searched_positions': len(stats), 'clean_process_exits': 2,
                          'weights_sha256': descriptor['model_export_sha256']}
    report = {'schema_version': 1, 'kind': 'state_expert_gtp_independent_audit', 'status': 'passed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': sha256(a.protocol), 'result_sha256': sha256(result_path),
              'retained_parent_result_sha256': sha256(parent_result_path), 'parser_tests_sha256': sha256(parser_path),
              'arms': summaries, 'new_competitive_games': 0, 'claims_strength': False, 'claims_throughput': False}
    with a.output.open('xb') as f:
        f.write(canonical_json(report))
    a.output.chmod(0o444); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
