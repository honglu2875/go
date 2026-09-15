#!/usr/bin/env python3
"""Run one registered final-model KataGo suite, retaining incomplete games."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value: raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--arm', choices=('parent', 'inherit', 'anneal'), required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--training-audit', type=Path)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec); spec_sha = sha256(a.spec)
    require(spec['kind'] == 'online_annealing_pilot' and spec['maximum_eval_attempts_each_arm'] == 1, 'Wrong evaluation registration')
    verify_inputs(SOURCE, spec['evaluation_input_closure'])
    for path, expected in spec['evaluation_code_sha256'].items():
        require(sha256(SOURCE / path) == expected, 'Registered evaluator changed: ' + path)
    candidate_path = SOURCE / f'eval/online_annealing/{a.arm}.json'; candidate = read_json(candidate_path)
    identity = validate_candidate(root, candidate)
    if a.arm == 'parent':
        require(candidate['training_snapshot'] == spec['parent_snapshot'] and identity['checkpoint_turn'] == spec['parent_turn'], 'Parent checkpoint differs')
    else:
        require(a.training_audit is not None, 'Trained arm requires an independent training audit')
        audit = read_json(a.training_audit)
        require(audit['status'] == 'passed' and audit['spec_sha256'] == spec_sha
                and audit['arms'][a.arm]['model_export_sha256'] == candidate['model_export_sha256']
                and candidate['training_snapshot'] == spec['arms'][a.arm]['snapshot']
                and identity['checkpoint_turn'] == spec['final_turn'], 'Trained candidate differs from audited final checkpoint')
    for path in (root / 'runs/eval').glob('*/result.json'):
        previous = read_json(path)
        require(not (previous.get('kind') == 'online_annealing_katago_suite' and previous.get('arm') == a.arm
                     and previous.get('spec_sha256') == spec_sha), 'This arm already has an evaluation attempt')
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'online_annealing_katago_suite', 'status': 'running', 'snapshot': SOURCE.name,
              'spec_sha256': spec_sha, 'arm': a.arm, 'candidate_sha256': sha256(candidate_path),
              'model_export_sha256': candidate['model_export_sha256'], 'started_unix': time.time(), 'panels': []}
    (output / 'result.json').write_bytes(canonical_json(report))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    try:
        for group, path in zip(('primary', 'strong'), spec['evaluation_panels'][a.arm], strict=True):
            command = [sys.executable, '-B', str(SOURCE / 'eval/panel.py'), '--spec', str(SOURCE / path),
                       '--artifacts-root', str(root), '--output', str(output / group)]
            print('Starting ' + a.arm + ' ' + group, flush=True)
            with (output / (group + '.log')).open('xb') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            panel = read_json(output / group / 'result.json')
            require(process.returncode in (0, 1) and 'error' not in panel
                    and all(not child['timed_out'] and child['returncode'] in (0, 1)
                            and all(g['status'] in ('completed', 'truncated') for g in child['games']) for child in panel['matches']),
                    'Panel has a process or command failure')
            report['panels'].append({'group': group, 'path': path, 'spec_sha256': sha256(SOURCE / path),
                                    'returncode': process.returncode, 'result_sha256': sha256(output / group / 'result.json'),
                                    'summaries': panel['summaries']})
            (output / 'result.json').write_bytes(canonical_json(report))
            print(canonical_json(report['panels'][-1]).decode(), flush=True)
        require(time.time() - report['started_unix'] <= spec['maximum_eval_seconds_each_arm'], 'Suite exceeded registered time budget')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (output / 'result.json').write_bytes(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
