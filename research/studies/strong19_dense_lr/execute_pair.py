"""Guard the fixed dense learning-rate sequence, with audits between arms."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from execute_run import ROOT, STUDY, publish, read, require, sha
from observation import observe
from monitor import resource_observation, request_cancellation
from compare import METRICS, compare_draws, report as comparison_report
from gozero.snapshots import verify


def inspect(path, digest):
    require(sha(path) == digest, 'Registration changed')
    r = read(path)
    require(r['kind'] == 'dense_learning_rate_grid' and r['order'] == ['lr06', 'lr15']
            and r['steps'] == 128 and r['schedule_steps'] == 512, 'Unexpected experiment')
    for field in ('operators', 'prerequisites'):
        for name, expected in r[field].items():
            require(sha(ROOT / name) == expected, 'Changed ' + field + ': ' + name)
    require(r['operators'].get(str(Path(__file__).relative_to(ROOT))) == sha(Path(__file__)), 'Runner not pinned')
    for name in r['prerequisites']:
        require(read(ROOT / name)['status'] in ('passed', 'prepared'), 'Prerequisite failed')
    for arm, item in r['arms'].items():
        snapshot = ROOT / '.gozero/snapshots' / item['snapshot']; verify(snapshot)
        c = read(snapshot / 'resolved_config.json')
        require(sha(snapshot / 'resolved_config.json') == item['config_sha256']
                and c['steps'] == 512 and c['checkpoint_every'] == 128 and c['eval_every'] == 16
                and c['dataset']['manifest_sha256'] == r['dataset_manifest_sha256']
                and c['training']['optimizer'] == 'adamw'
                and c['training']['value_objective'] == 'signed_target_cross_entropy'
                and c['checkpoint_disk']['peer'] == item['replica_peer']
                and not c['evaluation']['run_test'], 'Configuration changed: ' + arm)
    require(sha(ROOT / r['draw_reference']) == r['draw_sha256'], 'Draw replay changed')
    return r


def run(path, digest, backup_path, backup_digest):
    r = inspect(path, digest)
    require(sha(backup_path) == backup_digest, 'Registration backup changed')
    backup = read(backup_path)
    require(backup['status'] == 'passed' and backup['registration_sha256'] == digest
            and len(backup['copies']) == 2, 'Missing two-peer source backup')
    require(not any(not (p.parent / 'result.json').exists() for p in (ROOT / 'runs').glob('pod-*/launch.json')),
            'Another pod attempt remains open')
    admission = resource_observation(ROOT / '.gozero/snapshots' / r['arms']['lr06']['snapshot'])
    for observed in admission:
        require(observed['disk_free'] > r['aggregate_disk_minimum_by_host'][str(observed['rank'])]
                and observed['shm_free'] > (64 << 30) and observed['memory_available'] > (96 << 30),
                'Sequence admission failed on rank ' + str(observed['rank']))
    folder = STUDY / 'sequence-001'; folder.mkdir(exist_ok=False)
    started = time.time(); outcomes = {}; reference = read(ROOT / r['draw_reference'])
    publish(folder / 'admission.json', dict(status='passed', created=started, hosts=admission,
            registration_sha256=digest, backup_sha256=backup_digest))

    def event(kind, **values):
        row = dict(kind=kind, time=time.time(), **values)
        with (folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n'); stream.flush()
        print(json.dumps(row), flush=True)

    try:
        for arm in r['order']:
            inspect(path, digest); item = r['arms'][arm]; steps = r['steps']
            prerequisites = {**r['prerequisites'], str(path.relative_to(ROOT)): digest,
                             str(backup_path.relative_to(ROOT)): backup_digest}
            for previous in outcomes:
                p = folder / (previous + '-review.json'); prerequisites[str(p.relative_to(ROOT))] = sha(p)
            peer = item['replica_peer']; reserve = r['checkpoint_bytes_per_arm'][arm]
            plan = dict(kind='registered_joint19_run', created=time.time(), snapshot=item['snapshot'],
                config_sha256=item['config_sha256'], purpose='learning', steps=steps,
                schedule_steps=512, checkpoint_every=128, parameters=item['parameters'],
                timeout_seconds=item['timeout_seconds'], replica_peer=peer,
                checkpoint_reserve_bytes=0, shm_floor_bytes=64 << 30, memory_floor_bytes=96 << 30,
                disk_floor_bytes=2 << 30, producer_growth_reserve_bytes=0,
                disk_checkpoint_reserve_by_host={str(h): reserve if h == 0 else
                    reserve + (6 << 30) if h == peer else 0 for h in range(4)},
                additional_reserve_by_host={str(h): 0 for h in range(4)},
                output_directory=item['output_directory'], audit_python=r['audit_python'],
                operators={k: v for k, v in r['operators'].items()
                           if Path(k).name in ('execute_run.py', 'audit_run.py', 'replicate.py')},
                prerequisites=prerequisites, expected_positions=r['expected_positions'],
                validation_positions=r['validation_positions'],
                scope='Registered fixed-data 128-update prefix; complete disk checkpoint and peer at 128.')
            plan_path = STUDY / (arm + '-plan-001.json'); publish(plan_path, plan)
            event('stage_started', arm=arm, snapshot=item['snapshot'], plan_sha256=sha(plan_path))
            argv = [sys.executable, '-B', STUDY / 'execute_run.py', '--plan', plan_path,
                    '--plan-sha256', sha(plan_path), '--run']
            attempt = None; last_observed = -1; errors = 0; resource_errors = 0
            last_progress = time.monotonic(); last_resources = 0.; cancelled = False
            startup_checked = False; startup_failure = None
            with (folder / (arm + '-controller.log')).open('xb') as log:
                child = subprocess.Popen(list(map(str, argv)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    controller = STUDY / item['output_directory'] / 'controller.log'
                    if attempt is None and controller.exists():
                        for line in controller.read_text().splitlines():
                            if line.startswith('{'):
                                row = json.loads(line)
                                if row.get('kind') == 'pod_attempt': attempt = row['attempt_id']; break
                    if attempt:
                        try:
                            value = observe(ROOT, attempt, item['snapshot'], validation_positions=r['validation_positions'])
                            temp = folder / (arm + '-current.tmp'); temp.write_text(json.dumps(value) + '\n')
                            temp.replace(folder / (arm + '-current.json')); errors = 0
                            if value['completed_updates'] and not startup_checked:
                                first = json.loads((ROOT / 'runs' / attempt / 'rank-0/artifacts/metrics.jsonl')
                                                   .read_text().splitlines()[0])
                                expected = item['first_update_metrics']
                                differences = {k: dict(expected=v, actual=first.get(k)) for k, v in expected.items()
                                    if k not in first or not math.isclose(first[k], v, rel_tol=2e-6, abs_tol=2e-6)}
                                startup_checked = True
                                if differences: startup_failure = 'Fresh first update differs from systems qualification'
                                publish(folder / (arm + '-startup.json'), dict(status='failed' if differences else 'passed',
                                    attempt=attempt, metrics_checked=len(expected), differences=differences))
                                event('startup_checked', arm=arm, metrics_checked=len(expected), differences=differences)
                            if value['completed_updates'] != last_observed:
                                last_observed = value['completed_updates']; last_progress = time.monotonic()
                                event('progress', arm=arm, attempt=attempt, updates=last_observed,
                                    positions=value['position_exposures'], learning_seconds=value['rank0_learning_seconds'],
                                    overfit=value['provisional_overfit_flags'])
                        except Exception as error:
                            errors += 1; event('monitor_error', arm=arm, error=repr(error), consecutive=errors)
                        if not cancelled and not (ROOT / 'runs' / attempt / 'result.json').exists():
                            reason = None
                            if startup_failure: reason = startup_failure
                            if errors >= 5: reason = 'Five consecutive observation failures'
                            if time.monotonic() - last_progress > 1800: reason = 'No accepted update for 30 minutes'
                            if time.monotonic() - last_resources >= 600:
                                try:
                                    resource_rows = resource_observation(ROOT / '.gozero/snapshots' / item['snapshot'])
                                    event('resources', arm=arm, hosts=resource_rows)
                                    last_resources = time.monotonic(); resource_errors = 0
                                    if any(x['disk_free'] < ((2 if x['rank'] == 0 else 8) << 30)
                                           or x['shm_free'] < (64 << 30) or x['memory_available'] < (24 << 30)
                                           for x in resource_rows): reason = 'Storage or memory reserve exhausted'
                                except Exception as error:
                                    resource_errors += 1; last_resources = time.monotonic() - 570
                                    event('resource_monitor_error', arm=arm, error=repr(error), consecutive=resource_errors)
                                    if resource_errors >= 3: reason = 'Three consecutive resource observation failures'
                            if reason:
                                event('cancellation_requested', arm=arm, attempt=attempt, reason=reason)
                                request_cancellation(ROOT / '.gozero/snapshots' / item['snapshot'], attempt, folder, reason)
                                cancelled = True
                    time.sleep(30)
                require(child.returncode == 0 and not cancelled, 'Stage failed or was cancelled: ' + arm)
            stage = STUDY / item['output_directory']; result = read(stage / 'result.json'); audit = read(stage / 'audit.json')
            require(result['status'] == 'passed' and result['plan_sha256'] == sha(plan_path)
                    and sha(stage / 'audit.json') == result['audit_sha256'], 'Stage closure differs')
            attempt = result['attempt']
            final = observe(ROOT, attempt, item['snapshot'], validation_positions=r['validation_positions'])
            require(final['closure'] == 'passed' and final['completed_updates'] == steps, 'Final observation incomplete')
            compare_draws(attempt, reference, steps)
            for field, expected in (('validation_history', r['validation_population_sha256']),
                                    ('training_probe_history', r['probe_population_sha256'])):
                require(all(x['episode_ids_sha256'] == expected for x in audit[field]), 'Evaluation population differs')
            endpoint = read(ROOT / 'runs' / attempt / 'rank-0/artifacts/result.json')
            outcome = dict(status='passed', attempt=attempt, steps=steps, parameters=audit['parameters'],
                positions=audit['positions'], result_sha256=sha(stage / 'result.json'),
                audit_sha256=result['audit_sha256'], replica_sha256=result['replica_sha256'],
                endpoint={k: audit['validation_history'][-1]['metrics'][k] for k in METRICS},
                learning_seconds=endpoint['segment_timing']['learning_seconds'],
                sustained_overfit_flags=sum(x['sustained'] for x in audit['overfit_observations']))
            publish(folder / (arm + '-review.json'), outcome); outcomes[arm] = outcome
            event('stage_passed', arm=arm, **outcome)
        comparison_report(r, outcomes, folder)
        publish(folder / 'result.json', dict(status='passed', registration_sha256=digest,
            started=started, finished=time.time(), comparison_sha256=sha(folder / 'comparison.json'), arms=outcomes))
        event('sequence_passed', arms=outcomes)
    except BaseException as error:
        if not (folder / 'result.json').exists():
            publish(folder / 'result.json', dict(status='failed', registration_sha256=digest,
                started=started, finished=time.time(), arms=outcomes, error=repr(error)))
        event('sequence_failed', error=repr(error)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--backup', type=Path); p.add_argument('--backup-sha256'); p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not args.run:
        inspect(args.registration, args.sha256)
        print(json.dumps(dict(status='prepared', accelerator_jobs_started=False))); return
    require(args.backup is not None and args.backup_sha256, 'Backup receipt required')
    with (STUDY / '.pair.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.registration.resolve(), args.sha256, args.backup.resolve(), args.backup_sha256)


if __name__ == '__main__': main()
