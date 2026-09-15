#!/usr/bin/env python3
"""Run one registered fixed-checkpoint value-loss evaluation arm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True); p.add_argument('--arm', choices=('mse', 'bce2'), required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'Registration changed')
    protocol = read_json(a.protocol); arm = protocol['arms'][a.arm]
    require(protocol['kind'] == 'value_logit_distillation_pilot' and protocol['maximum_eval_attempts_per_arm'] == 1, 'Study identity differs')
    verify_inputs(SOURCE, protocol['evaluation_input_closure'])
    for name, expected in protocol['evaluation_code_sha256'].items():
        require(sha256(SOURCE / name) == expected, 'Registered evaluator code changed')
    descriptor = read_json(SOURCE / arm['descriptor']); trained = validate(root, descriptor)
    require(descriptor['training_snapshot'] == arm['snapshot_id'] and trained['config']['model']['value_objective'] == a.arm
            and descriptor['network_version'] == protocol['steps']
            and sha256(trained['snapshot'] / 'resolved_config.json') == arm['config_sha256'], 'Completed fixed checkpoint differs')
    a.output = a.output.resolve(); require(a.output.is_relative_to(root / 'runs/eval'), 'Evaluation output escapes run store')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'value_logit_katago_suite', 'status': 'running', 'snapshot_id': SOURCE.name,
              'arm': a.arm, 'protocol_sha256': a.expected_protocol_sha256, 'candidate_sha256': sha256(SOURCE / arm['descriptor']),
              'model_export_sha256': descriptor['model_export_sha256'], 'started_unix': time.time(), 'panels': []}
    try:
        for item in arm['evaluation_panels']:
            require(time.time() - report['started_unix'] + item['maximum_seconds'] <= protocol['maximum_eval_seconds_per_arm'],
                    'Remaining evaluation budget cannot cover panel')
            path = SOURCE / item['path']; require(sha256(path) == item['sha256'], 'Panel changed')
            command = [sys.executable, '-B', str(SOURCE / 'eval/panel.py'), '--spec', str(path),
                       '--artifacts-root', str(root), '--output', str(a.output / item['id'])]
            print(json.dumps({'kind': 'value_logit_panel_start', 'arm': a.arm, 'panel': item['id']}), flush=True)
            with (a.output / (item['id'] + '.log')).open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=item['maximum_seconds'])
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise
            result_path = a.output / item['id'] / 'result.json'; result = read_json(result_path)
            require(code in (0, 1) and 'error' not in result
                    and sum(s['scheduled_games'] for s in result['summaries'].values()) == item['games']
                    and not any(s['failed_games'] for s in result['summaries'].values())
                    and not any(m['timed_out'] or not m['result_sha256'] for m in result['matches']), 'Panel process or integrity failure; no retry')
            report['panels'].append({'id': item['id'], 'spec_sha256': sha256(path), 'returncode': code,
                                     'result_sha256': sha256(result_path), 'summaries': result['summaries']})
            (a.output / 'progress.json').write_bytes(canonical_json(report))
            print(json.dumps(report['panels'][-1]), flush=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished_unix'] = time.time(); (a.output / 'result.json').write_bytes(canonical_json(report)); verify(SOURCE)
        require(sha256(a.protocol) == a.expected_protocol_sha256, 'Protocol changed during evaluation')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
