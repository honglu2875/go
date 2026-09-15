"""Monitor one registered scalar intervention and conditionally confirm its seed.

This continuation owns closure of the already running first seed. It cannot
select another rate or architecture. A failure stops only this continuation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scalar_compare import read, sha
import launch_scalar

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def confirmed_screen(result, registration, stage, registration_sha):
    if (result['kind'] != 'strong9_scalar_paired_contrast' or result['status'] != 'passed'
            or result['plan_sha256'] != registration_sha or result['stage'] != stage
            or result['candidate']['snapshot'] != registration['stages'][stage]['snapshot']
            or result['parent']['snapshot'] != registration['stages'][stage]['parent_snapshot']
            or result['screen_min_relative_gain'] != registration['screen_min_relative_gain']):
        raise ValueError('Wrong registered contrast')
    gains = result['endpoint_and_tail_gains']
    if set(gains) != {'expert_kl', 'family_kl'}:
        raise ValueError('Missing co-primary measure')
    values = [g[k] for g in gains.values() for k in ('relative_endpoint_gain', 'relative_last_three_gain')]
    if not all(type(x) in (int, float) and math.isfinite(x) for x in values):
        raise ValueError('Invalid gain')
    passed = all(g['relative_endpoint_gain'] >= registration['screen_min_relative_gain']
                 and g['relative_last_three_gain'] >= 0 for g in gains.values()) and not result['sustained_overfit']
    if type(result['screen_passed']) is not bool or result['screen_passed'] != passed:
        raise ValueError('Screen flag disagrees with registered evidence')
    return passed


def inspect(path, expected):
    if sha(path) != expected:
        raise ValueError('Continuation plan changed')
    plan = read(path)
    if (plan['kind'] != 'strong9_scalar_conditional_continuation' or plan['poll_seconds'] != 60
            or plan['monitor_seconds'] != 300 or plan['max_seconds'] != 21600):
        raise ValueError('Continuation scope changed')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Continuation source changed: '+name)
    item = plan['registration']; registration_path = ROOT/item['path']
    if sha(registration_path) != item['sha256']:
        raise ValueError('Scientific registration changed')
    registration = read(registration_path); registration['_sha256'] = item['sha256']
    launch_scalar.inspect(registration, 'seed1')
    for control in registration['cnn_controls'].values():
        if sha(ROOT/control['path']) != control['sha256']:
            raise ValueError('Pinned CNN control changed')
    receipt = ROOT/plan['existing_seed1_receipt']['path']
    if sha(receipt) != plan['existing_seed1_receipt']['sha256']:
        raise ValueError('Existing seed 1 receipt changed')
    if read(receipt)['snapshot'] != registration['stages']['seed1']['snapshot']:
        raise ValueError('Wrong first seed')
    return plan, registration_path, registration


class Continuation:
    def __init__(self, path, digest):
        self.path, self.digest = path, digest
        self.plan, self.registration_path, self.registration = inspect(path, digest)
        self.directory = path.parent
        self.trial = STUDY/self.registration['trial']
        self.started = time.time(); self.calls = 0; self.last_monitor = {}

    def event(self, kind, **values):
        row = dict(kind=kind, time=time.time(), **values)
        with (self.directory/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n'); stream.flush()
        print(json.dumps(row), flush=True)

    def status(self, **values):
        row = dict(updated=time.time(), started=self.started, pid=os.getpid(), plan_sha256=self.digest,
                   process_identity=Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19], **values)
        temp = self.directory/'.status.partial'
        temp.write_text(json.dumps(row, indent=2)+'\n'); temp.replace(self.directory/'status.json')

    def check(self):
        inspect(self.path, self.digest)
        if (self.directory/'stop').exists():
            raise RuntimeError('Continuation stop requested; learner remains under its controller')
        if time.time()-self.started >= self.plan['max_seconds']:
            raise TimeoutError('Continuation time bound reached; learner remains under its controller')

    def command(self, argv, label, timeout=900, accelerator=False):
        self.check(); self.calls += 1
        log = self.directory/f'command-{self.calls:03d}-{label}.log'
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/gozero-mpl-scalar')
        if accelerator:
            env.pop('JAX_PLATFORMS', None)
        else:
            env['JAX_PLATFORMS'] = 'cpu'
        self.event('command_started', label=label, log=str(log.relative_to(ROOT)))
        with log.open('xb') as stream:
            subprocess.run(['taskset','-c','0,1',*map(str,argv)], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        self.event('command_passed', label=label, log_sha256=sha(log))

    def attempt(self, stage):
        receipt_path = self.trial/(stage+'-process-001.json')
        if not receipt_path.exists():
            return None
        receipt = read(receipt_path)
        if (receipt['stage'] != stage or receipt['plan_sha256'] != self.registration['_sha256']
                or receipt['snapshot'] != self.registration['stages'][stage]['snapshot']):
            raise ValueError('Wrong stage receipt')
        log = self.trial/(stage+'-controller-001.log')
        if receipt['log'] != str(log):
            raise ValueError('Wrong stage log')
        for line in log.read_text().splitlines(keepends=True)[:10]:
            if not line.endswith('\n'):
                continue
            value = json.loads(line)
            if value.get('kind') != 'pod_attempt':
                continue
            attempt = Path(value['attempt'])
            if (attempt.parent != ROOT/'runs' or attempt.name != value['attempt_id']
                    or value['snapshot_id'] != receipt['snapshot']):
                raise ValueError('Wrong stage attempt')
            launch = read(attempt/'launch.json')
            if launch['snapshot_id'] != receipt['snapshot'] or not -30 <= launch['start_unix_time']-receipt['started'] <= 180:
                raise ValueError('Attempt launch does not match receipt')
            return attempt
        if time.time()-receipt['started'] > 180:
            raise RuntimeError('Controller did not publish an attempt')
        return None

    def monitor(self, stage, attempt, final=False):
        if not (attempt/'rank-0/artifacts/metrics.jsonl').exists():
            return
        if not final and time.time()-self.last_monitor.get(stage, 0) < self.plan['monitor_seconds']:
            return
        self.command([sys.executable,'-B',STUDY.parent/'strong9_scaling/monitor.py',
                      '--attempt',attempt.name],stage+'-monitor',timeout=60)
        self.last_monitor[stage] = time.time()
        parent = read(ROOT/self.registration['stages'][stage]['parent_audit']['path'])
        cnn = read(ROOT/self.registration['cnn_controls'][stage]['path'])
        self.command([ROOT/'.gozero/analysis-environments/plotting/bin/python','-B',
            STUDY.parent/'strong9_scaling/plot_progress.py',
            '--arm','CNN',ROOT/'runs'/cnn['attempt'],
            '--arm',self.registration['parent_label'],ROOT/'runs'/parent['attempt'],
            '--arm',self.registration['candidate_label'],attempt,
            '--output',self.trial/(stage+('-final.png' if final else '-progress.png'))],stage+'-plot',timeout=60)

    def wait(self, stage):
        while True:
            self.check(); attempt = self.attempt(stage)
            if attempt is not None:
                result_path = attempt/'result.json'
                if result_path.exists():
                    result = read(result_path)
                    if result['status'] != 'passed' or result['snapshot_id'] != self.registration['stages'][stage]['snapshot']:
                        raise RuntimeError('Registered Scalar attempt failed')
                    self.monitor(stage, attempt, final=True)
                    return attempt
                launch = read(attempt/'launch.json')
                if time.time() > launch['start_unix_time']+launch['timeout_seconds']+launch['prepare_timeout_seconds']+300:
                    raise TimeoutError('Scalar attempt did not close within declared bounds')
                self.monitor(stage, attempt)
            self.status(status='waiting', stage=stage, attempt=attempt.name if attempt else None)
            time.sleep(self.plan['poll_seconds'])

    def run(self):
        self.event('continuation_started', registration_sha256=self.registration['_sha256'])
        outcomes = {}
        try:
            for stage in ('seed1', 'seed2'):
                if stage == 'seed2':
                    if not outcomes['seed1']['screen_passed']:
                        break
                    self.command([sys.executable,'-B',STUDY/'launch_scalar.py','--plan',self.registration_path,
                        '--plan-sha256',self.registration['_sha256'],'--stage',stage],stage+'-launch',timeout=180,accelerator=True)
                attempt = self.wait(stage)
                self.command([sys.executable,'-B',STUDY/'finalize_scalar.py','--plan',self.registration_path,
                    '--plan-sha256',self.registration['_sha256'],'--stage',stage],stage+'-finalize',timeout=2700)
                contrast = self.trial/(stage+'-contrast-001.json'); result = read(contrast)
                passed = confirmed_screen(result, self.registration, stage, self.registration['_sha256'])
                outcomes[stage] = dict(screen_passed=passed,contrast_sha256=sha(contrast),attempt=attempt.name,
                                        gains=result['endpoint_and_tail_gains'])
                self.event('stage_reviewed',stage=stage,**outcomes[stage])
            accepted = len(outcomes) == 2 and all(x['screen_passed'] for x in outcomes.values())
            outcome = dict(kind='strong9_scalar_continuation_outcome',status='passed',created=time.time(),
                registration_sha256=self.registration['_sha256'],continuation_plan_sha256=self.digest,
                intervention_accepted=accepted,stages=outcomes,
                next='Review full curves before selecting another LR or encoder; no further experiment is queued.')
            path = self.directory/'outcome-001.json'
            with path.open('x') as stream:
                json.dump(outcome,stream,indent=2);stream.write('\n')
            path.chmod(0o444)
            self.status(status='completed',intervention_accepted=accepted,outcome_sha256=sha(path))
            self.event('continuation_completed',intervention_accepted=accepted,outcome_sha256=sha(path))
        except BaseException as error:
            self.status(status='stopped',error=repr(error),active_learner_terminated=False)
            self.event('continuation_stopped',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    inspect(args.plan,args.plan_sha256)
    if args.inspect:
        print(json.dumps(dict(status='passed',stages=['seed1','seed2_if_screen_passes'],mutations=False)));return
    with (args.plan.parent/'.queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.plan.parent/'status.json').exists():
            raise ValueError('Continuation already started; inspect before recovery')
        Continuation(args.plan,args.plan_sha256).run()


if __name__=='__main__':main()
