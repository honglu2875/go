#!/usr/bin/env python3
"""Check frozen registration, opening provenance and per-update matched samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--expected-training-audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    require(sha256(a.protocol) == a.expected_protocol_sha256 and sha256(a.training_audit) == a.expected_training_audit_sha256,
            'Registration or training audit changed')
    spec, training = read_json(a.protocol), read_json(a.training_audit)
    require(training['status'] == 'passed' and training['protocol_sha256'] == a.expected_protocol_sha256, 'Training audit differs')
    for identity in {SOURCE.name, training['analysis_snapshot'], *[arm['snapshot_id'] for arm in spec['arms'].values()]}:
        source = root / '.gozero/snapshots' / identity; verify(source); verify_inputs(source, spec['evaluation_input_closure'])
        for group in ('cpu_qualified_code_sha256', 'evaluation_code_sha256', 'analysis_code_sha256'):
            for name, digest in spec[group].items():
                require(sha256(source / name) == digest, 'Registered code differs: ' + name)
    tests_path = root / spec['external_criterion_tests_path']; tests = read_json(tests_path)
    require(sha256(tests_path) == spec['external_criterion_tests_sha256'] and tests['status'] == 'passed'
            and tests['returncode'] == 0 and tests['finished_unix'] < spec['registered_unix']
            and sha256(tests_path.parent / 'tests.log') == tests['log_sha256'], 'External criterion tests differ')
    book_path = SOURCE / 'eval/state_expert_distillation/openings.json'; book = read_json(book_path)
    require(sha256(book_path) == spec['openings_sha256'], 'Opening book changed')
    used = set()
    for name, digest in book['exclusion_inputs_sha256'].items():
        path = SOURCE / name; require(sha256(path) == digest, 'Historical opening input changed')
        for opening in read_json(path)['openings']:
            if len(opening) == 2 and all(isinstance(v, str) for v in opening):
                used.add(tuple(v.upper() for v in opening))
    require(len(used) == book['excluded_previous_ordered_pairs'], 'Opening exclusion count differs')
    fresh = book['early'] + book['strong']; tuples = [tuple(v.upper() for v in o) for o in fresh]
    require(len(tuples) == len(set(tuples)) == 34 and all(t not in used and t[::-1] not in used for t in tuples)
            and not set(tuples) & {t[::-1] for t in tuples}, 'Opening pairs or reversals were reused')
    work = {}
    for arm in ('history', 'state'):
        directory = root / 'runs' / training['arms'][arm]['attempt']; streams = []
        for rank in range(4):
            rows = [json.loads(line) for line in (directory / f'rank-{rank}/artifacts/metrics.jsonl').read_text().splitlines()]
            require([r['turn'] for r in rows] == list(range(1, spec['steps'] + 1)), 'Update coverage differs')
            counts = [[r['turn'], int(r['expert_tokens']), int(r['value_tokens'])] for r in rows]
            require(all(r['expert_tokens'] == x[1] and r['value_tokens'] == x[2] for r, x in zip(rows, counts)), 'Fractional work counts')
            require(sum(r[1] for r in counts) == training['arms'][arm]['global_exposures']['expert_token_exposures'], 'Exposure sum differs')
            streams.append(counts)
        require(all(s == streams[0] for s in streams), 'Replicated update work differs')
        work[arm] = streams[0]
    require(work['history'] == work['state'], 'Per-update matched work differs')
    report = {'schema_version': 1, 'kind': 'state_expert_registration_audit', 'status': 'passed',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': sha256(a.protocol), 'training_audit_sha256': sha256(a.training_audit),
              'registered_code_groups_verified': ['cpu_qualified_code_sha256', 'evaluation_code_sha256', 'analysis_code_sha256'],
              'unique_fresh_opening_pairs': len(tuples), 'excluded_previous_pairs': len(used),
              'per_update_global_work_exact': True, 'updates_per_arm': len(work['history']),
              'work_stream_sha256': hashlib.sha256(canonical_json(work['history'])).hexdigest(),
              'scope': 'Input identities and deterministic per-update work; no new predictions, training, games or scientific thresholds.'}
    with a.output.open('xb') as stream:
        stream.write(canonical_json(report))
    a.output.chmod(0o444); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
