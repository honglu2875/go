#!/usr/bin/env python3
"""Run the registered two-architecture four-host continuation qualification once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--expected-protocol-sha256', required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    require(sha256(a.protocol) == a.expected_protocol_sha256, 'Protocol changed')
    spec = read_json(a.protocol)
    require(spec['kind'] == 'state_expert_tpu_recovery_qualification' and spec['runner_snapshot'] == SOURCE.name
            and spec['maximum_attempts'] == 4 and spec['steps'] == 16 and spec['resume_turn'] == 8
            and spec['expected_hosts'] == 4 and spec['expected_devices'] == 16, 'Invalid qualification protocol')
    for name, expected in spec['prerequisites_sha256'].items():
        require(sha256(root / name) == expected, 'Prerequisite changed: ' + name)
        data = read_json(root / name)
        require(data['status'] == 'passed', 'Prerequisite did not pass: ' + name)
    preflight = read_json(root / spec['preflight'])
    require(0 <= time.time() - preflight['finished_unix'] <= spec['preflight_maximum_age_seconds'], 'Preflight is stale')
    require({r['host'] for r in preflight['hosts']} == {f'go-user@worker-{i}.example.invalid' for i in range(4)}
            and all(r['status'] == 'passed' for r in preflight['hosts']), 'Preflight host coverage differs')
    configs = {}
    for arm, entry in spec['arms'].items():
        source = root / '.gozero/snapshots' / entry['snapshot']; manifest = verify(source)
        require(sha256(source / 'resolved_config.json') == entry['config_sha256'], 'Configuration differs')
        for name, digest in spec['cpu_qualified_code_sha256'].items():
            require(sha256(source / name) == digest, 'Scientific code differs from CPU qualification')
        config = read_json(source / 'resolved_config.json'); configs[arm] = config
        require(config['model']['architecture'] == arm and config['steps'] == spec['steps']
                and config['expected_processes'] == 4 and config['expected_devices'] == 16
                and config['platform'] == 'tpu' and config['checkpoint_every'] == spec['resume_turn']
                and config['model']['value_objective'] == 'mse' and config['model']['board_mode'] == 'exact', 'Training shape differs')
        require(manifest['recipe'] == 'research/recipes/state_expert_distillation', 'Wrong recipe')
    h, s = (json.loads(json.dumps(configs[k])) for k in ('history', 'state'))
    h['model'].pop('architecture'); s['model'].pop('architecture')
    require(h == s, 'Control and candidate differ beyond architecture')
    output = root / spec['output']; output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': spec['kind'], 'status': 'failed', 'runner_snapshot': SOURCE.name,
              'protocol_sha256': sha256(a.protocol), 'started_unix': time.time(), 'attempts': [], 'arms': {},
              'claims_strength': False, 'claims_host_loss_recovery': False, 'claims_mfu': False}
    try:
        for arm in ('history', 'state'):
            source = root / '.gozero/snapshots' / spec['arms'][arm]['snapshot']; baseline = None
            for mode in ('baseline', 'resumed'):
                before = {p.name for p in (root / 'runs').glob('pod-*')}
                command = [sys.executable, '-B', str(source / 'ops/pod_run.py'), '--snapshot', str(source),
                           '--workspace-root', str(root), '--timeout', str(spec['maximum_seconds_per_attempt']),
                           '--prepare-timeout', str(spec['prepare_timeout_seconds']), '--controller-cpus', '32']
                if mode == 'resumed':
                    command += ['--resume-attempt', baseline, '--resume-turn', str(spec['resume_turn'])]
                started = time.time()
                with (output / f'{arm}-{mode}.log').open('x') as stream:
                    done = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                created = {p.name for p in (root / 'runs').glob('pod-*')} - before
                require(len(created) == 1, 'Expected one uniquely recorded pod attempt')
                attempt = created.pop(); folder = root / 'runs' / attempt
                entry = {'arm': arm, 'mode': mode, 'attempt': attempt, 'command': command, 'returncode': done.returncode,
                         'controller_started_unix': started, 'controller_finished_unix': time.time(),
                         'launch_sha256': sha256(folder / 'launch.json'), 'result_sha256': sha256(folder / 'result.json'),
                         'log_sha256': sha256(output / f'{arm}-{mode}.log')}
                pod = read_json(folder / 'result.json'); entry['recorded_attempt_chip_hours'] = pod['reserved_chip_hours']
                report['attempts'].append(entry)
                print(json.dumps({'kind': 'attempt_closed', **entry}), flush=True)
                require(done.returncode == 0 and pod['status'] == 'passed' and pod['snapshot_id'] == source.name
                        and pod['start_unix_time'] >= spec['registered_unix'], 'Pod attempt failed or preceded registration')
                if mode == 'baseline':
                    baseline = attempt
                else:
                    recovery = output / f'{arm}-recovery.json'
                    command = [sys.executable, '-B', str(SOURCE / 'research/recipes/state_expert_distillation/qualify_recovery.py'),
                               '--workspace-root', str(root), '--baseline', str(root / 'runs' / baseline),
                               '--resumed', str(folder), '--hosts', '4', '--output', str(recovery)]
                    with (output / f'{arm}-recovery.log').open('x') as stream:
                        done = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
                    require(done.returncode == 0, 'Exact continuation audit failed')
                    audit = read_json(recovery)
                    count = sum(r['arrays_exact'] for r in audit['ranks'])
                    require(audit['status'] == 'passed' and count == spec['arms'][arm]['expected_recovery_arrays']
                            and all(r['turn'] == 16 and r['resume_turn'] == 8 for r in audit['ranks']), 'Unexpected recovery coverage')
                    report['arms'][arm] = {'baseline': baseline, 'resumed': attempt, 'arrays_exact': count,
                                           'recovery_sha256': sha256(recovery), 'recovery_command': command}
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        report['recorded_attempt_chip_hours'] = sum(row['recorded_attempt_chip_hours'] for row in report['attempts'])
        with (output / 'result.json').open('xb') as stream:
            stream.write(canonical_json(report))
        verify(SOURCE)
        print(json.dumps({k: v for k, v in report.items() if k != 'attempts'}), flush=True)


if __name__ == '__main__':
    main()
