"""Run the three fixed pilot arms sequentially, retaining two verified RAM copies.

No architecture/LR/horizon selection is performed here. Stop after the paired
pilot, or immediately after a failed closure/audit/retention check. Each arm
gets a new immutable launch plan referencing all prerequisite outcomes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from execute_run import ROOT, STUDY, publish, read, require, sha
from pilot_observation import observe, METRICS


def inspect(path, digest):
    require(sha(path) == digest, 'Pilot registration changed')
    r = read(path)
    require(r['kind'] == 'joint19_three_arm_pilot' and r['order'] == ['source_cnn', 'cnn_adamw', 'transformer'],
            'Unexpected pilot scope or order')
    for name, expected in r['operators'].items():
        require(sha(ROOT / name) == expected, 'Pilot operator changed: ' + name)
    for name, expected in r['prerequisites'].items():
        require(sha(ROOT / name) == expected and read(ROOT / name)['status'] in ('passed', 'prepared'),
                'Pilot prerequisite differs: ' + name)
    from gozero.snapshots import verify
    configs = {}
    for arm in r['order']:
        item = r['arms'][arm]; snapshot = ROOT / '.gozero/snapshots' / item['snapshot']
        verify(snapshot); c = read(snapshot / 'resolved_config.json'); configs[arm] = c
        require(sha(snapshot / 'resolved_config.json') == item['config_sha256']
                and c['training']['purpose'] == 'learning' and c['steps'] == c['checkpoint_every'] == 108
                and c['eval_every'] == 9 and not c['evaluation']['run_test']
                and c['evaluation']['games_per_bucket'] >= 146 and c['evaluation']['training_probe_games'] == 128
                and c['learner']['games_per_host'] == 32 and c['learner']['augmentation'] == 'd4'
                and c['expected_processes'] == 4 and c['expected_devices'] == 16, 'Pilot config differs')
    for key in ('seed', 'dataset', 'evaluation', 'value_model'):
        require(all(c[key] == configs['source_cnn'][key] for c in configs.values()), 'Paired input differs: ' + key)
    require(configs['source_cnn']['model'] == configs['cnn_adamw']['model'], 'CNN optimizer control changed model')
    require(configs['cnn_adamw']['learner'] == configs['transformer']['learner'], 'AdamW settings differ')
    require(configs['transformer']['model']['encoder_layer_scale'] == .01, 'Confirmed encoder initialization differs')
    return r


def compare_draws(attempt, reference):
    for host in range(4):
        directory = ROOT / 'runs' / attempt / f'rank-{host}/artifacts'
        rank = read(directory / 'result.json')['jax_rank']
        rows = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines()]
        require(len(rows) == len(reference['draws']) == 108, 'Incomplete registered draw replay')
        for actual, expected in zip(rows, reference['draws']):
            local = expected['ranks'][rank]
            require(local['jax_rank'] == rank and actual['turn'] == expected['turn']
                    and actual['positions'] == expected['positions'] and actual['bucket'] == expected['bucket']
                    and actual['local_entries_sha256'] == local['local_entries_sha256']
                    and actual['local_symmetries'] == local['local_symmetries'], 'Predeclared draw replay differs')


def run(path, digest):
    r = inspect(path, digest); folder = STUDY / 'pilot-sequence-001'; folder.mkdir(exist_ok=False)
    reference = read(ROOT / r['draw_reference']); population = read(ROOT / r['population_reference'])
    outcomes = {}; audits = {}; started = time.time()
    def event(kind, **values):
        value = dict(kind=kind, time=time.time(), **values)
        with (folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n'); stream.flush()
        print(json.dumps(value), flush=True)
    def small(argv, label):
        with (folder / (label + '.log')).open('xb') as log:
            subprocess.run(list(map(str, argv)), cwd=ROOT, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    try:
        for arm in r['order']:
            inspect(path, digest)
            item = r['arms'][arm]; name = item['output_directory']
            plan = dict(kind='registered_joint19_run', created=time.time(), snapshot=item['snapshot'],
                config_sha256=item['config_sha256'], purpose='learning', steps=108, parameters=item['parameters'],
                expected_positions=reference['total_positions'], validation_positions=population['validation_positions'],
                timeout_seconds=r['timeout_seconds'], replica_peer=r['replica_peer'],
                checkpoint_reserve_bytes=r['checkpoint_reserve_bytes'], shm_floor_bytes=64*(1<<30),
                memory_floor_bytes=96*(1<<30), disk_floor_bytes=1<<30,
                producer_growth_reserve_bytes=r['producer_growth_reserve_bytes'],
                additional_reserve_by_host={str(i):0 for i in range(4)}, output_directory=name,
                audit_python=r['audit_python'], operators={k:v for k,v in r['operators'].items()
                    if k in ('research/studies/strong19_scaling/execute_run.py',
                             'research/studies/strong19_scaling/audit_run.py',
                             'research/studies/strong19_source_muon/replicate_full_size.py')},
                prerequisites={**r['prerequisites'], str(path.relative_to(ROOT)):digest},
                scope='One registered fixed-data joint learnability pilot arm. No test targets; no playing-strength claim.')
            for previous in outcomes:
                output = folder / (previous + '-retention.json')
                plan['prerequisites'][str(output.relative_to(ROOT))] = sha(output)
            plan_path = STUDY / (name.replace('-001', '-plan-001')+'.json'); publish(plan_path, plan)
            command = [sys.executable, '-B', STUDY / 'execute_run.py', '--plan', plan_path,
                       '--plan-sha256', sha(plan_path), '--run']
            event('arm_started', arm=arm, snapshot=item['snapshot'], plan_sha256=sha(plan_path))
            monitor_errors = []; attempt = None
            with (folder / (arm + '-controller.log')).open('xb') as log:
                child = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    controller_log = STUDY / name / 'controller.log'
                    if controller_log.exists():
                        for line in controller_log.read_text().splitlines():
                            if line.startswith('{'):
                                value = json.loads(line)
                                if value.get('kind') == 'pod_attempt':
                                    attempt = value['attempt_id']; break
                    if attempt:
                        try:
                            value = observe(ROOT, attempt, item['snapshot'], validation_positions=population['validation_positions'])
                            temp = folder / (arm + '-current.tmp')
                            temp.write_text(json.dumps(value, sort_keys=True)+'\n')
                            temp.replace(folder / (arm + '-current.json'))
                            event('arm_progress', arm=arm, attempt=attempt, updates=value['completed_updates'],
                                  positions=value['position_exposures'], rank0_learning_seconds=value['rank0_learning_seconds'],
                                  overfit=value['provisional_overfit_flags'])
                        except Exception as error:
                            # Preserve a complete bounded child closure before deciding
                            # whether any later arm is allowed to start.
                            monitor_errors.append(repr(error)); event('monitor_error', arm=arm, error=repr(error))
                    time.sleep(60)
                require(child.returncode == 0, 'Registered arm failed: ' + arm)
            require(not monitor_errors, 'Live observation failed; review before the next arm')
            output = STUDY / name; result = read(output / 'result.json')
            require(result['status'] == 'passed' and result['plan_sha256'] == sha(plan_path), 'Wrong arm closure')
            attempt = result['attempt']; audit = read(output / 'audit.json')
            require(sha(output / 'audit.json') == result['audit_sha256'], 'Audited observations changed')
            compare_draws(attempt, reference)
            for key, population_key in [('validation_history','validation_episode_ids_sha256'),
                                        ('training_probe_history','probe_episode_ids_sha256')]:
                require(all(x['episode_ids_sha256'] == population[population_key] for x in audit[key]), 'Fixed population differs')
            if arm == 'cnn_adamw':
                require(audit['initial_parameters_sha256'] == audits['source_cnn']['initial_parameters_sha256'],
                        'CNN optimizer control did not start with the same parameters')
            peer = folder / (arm + '-peer3.json')
            small([sys.executable, '-B', ROOT / 'research/studies/strong19_source_muon/replicate_full_size.py',
                   '--workspace-root', ROOT, '--attempt', attempt, '--peer', '3', '--output', peer], arm+'-third-copy')
            retention = folder / (arm + '-retention.json')
            small([sys.executable, '-B', ROOT / 'ops/cold_checkpoint_storage/storage_registered_joint.py',
                   '--attempt', attempt, '--peer-receipts', output / 'replica.json', peer,
                   '--operation', 'evict', '--output', retention], arm+'-retention')
            require(read(retention)['status'] == 'passed', 'Checkpoint retention did not pass')
            audits[arm] = audit
            outcomes[arm] = dict(attempt=attempt, result_sha256=sha(output / 'result.json'),
                audit_sha256=result['audit_sha256'], retention_sha256=sha(retention),
                endpoint={k:audit['validation_history'][-1]['metrics'][k] for k in METRICS},
                overfit_observations=audit['overfit_observations'])
            publish(folder / (arm + '-review.json'), dict(status='passed', registered_draws_and_populations_match=True,
                    scope='Completed fixed horizon; no extension or further seed selected.', **outcomes[arm]))
            event('arm_passed', arm=arm, attempt=attempt, endpoint=outcomes[arm]['endpoint'])
        baseline = outcomes['cnn_adamw']['endpoint']
        comparisons = {arm:{metric:dict(value=value, cnn_adamw=baseline[metric],
            relative_gain=1-value/baseline[metric] if baseline[metric] > 0 else None)
            for metric,value in outcomes[arm]['endpoint'].items()} for arm in ('source_cnn','transformer')}
        result = dict(status='passed', arms=outcomes, comparisons=comparisons,
            scope='Single-seed, fixed-horizon joint learnability pilot. Review before another horizon or seed. '
                  'Real trained KataGo matches remain required; validation does not measure playing strength.')
    except BaseException as error:
        result = dict(status='failed', arms=outcomes, error=repr(error))
        publish(folder / 'result.json', dict(kind=r['kind'], registration_sha256=digest,
                                           started=started, finished=time.time(), **result))
        event('pilot_failed', error=repr(error)); raise
    publish(folder / 'result.json', dict(kind=r['kind'], registration_sha256=digest,
                                       started=started, finished=time.time(), **result))
    event('pilot_passed', comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        inspect(args.registration, args.registration_sha256)
        print(json.dumps(dict(status='prepared', learning_jobs_started=False))); return
    with (STUDY / '.pilot-sequence.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.registration, args.registration_sha256)


if __name__ == '__main__':
    main()
