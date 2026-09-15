"""Complete only the preregistered two-seed comparison through audited stages.

Any failed run, changed source, missing evidence, or resource gate stops this
queue. Stopping the queue never terminates an already launched learner.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/go');STUDY=ROOT/'research/studies/strong9_scaling'
STAGES=('cnn_seed1','transformer_qualification','transformer_seed1','cnn_seed2','transformer_seed2')
REGISTRATION_SHA='600b818e0ffc6649351e3fcaa50756c7c80c8f6bfed9b91c720c3beda908e96e'
CACHE_SNAPSHOT='86f6d0c7ef737b35985a94edb653b8e0be191848d6778c5447b2e55ca7e76df7'


def read(path):return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inspect(plan):
    if plan['kind']!='registered_strong9_continuation' or tuple(plan['stages'])!=STAGES:
        raise ValueError('Unexpected continuation scope')
    registration=ROOT/plan['registration']['path']
    if plan['registration']['sha256']!=REGISTRATION_SHA or sha(registration)!=REGISTRATION_SHA:
        raise ValueError('Scientific registration changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Continuation source changed: '+name)
    for name,digest in plan['existing_receipt_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Existing launch receipt changed')
    if not 3600<=plan['max_seconds']<=14*3600 or plan['poll_seconds']!=60:
        raise ValueError('Invalid continuation time bounds')
    if plan['replica_peer']!=2 or plan['qualification_extra_peer']!=3:
        raise ValueError('Unexpected checkpoint destinations')
    reg=read(registration)
    for group in ('protocol','objective_details','preceding_decision'):
        if sha(ROOT/reg[group]['path'])!=reg[group]['sha256']:raise ValueError('Registered evidence changed')
    return registration,reg


def attempt_for(stage,reg):
    arm,phase=stage.split('_',1);name=stage.replace('_','-')
    path=STUDY/(name+'-process-001.json')
    if not path.exists():return None
    process=read(path)
    if process['stage']!=stage or process['snapshot']!=reg['snapshots'][arm][phase] or process['registration_sha256']!=REGISTRATION_SHA:
        raise ValueError('Wrong stage process receipt')
    expected_log=STUDY/(name+'-controller-001.log')
    if process['log']!=str(expected_log):raise ValueError('Unexpected controller log')
    if not expected_log.exists():return None
    with expected_log.open() as f:
        for _ in range(10):
            line=f.readline()
            if not line:break
            if not line.endswith('\n'):continue
            value=json.loads(line)
            if value.get('kind')!='pod_attempt':continue
            attempt=Path(value['attempt'])
            if attempt.parent!=ROOT/'runs' or attempt.name!=value['attempt_id'] or value['snapshot_id']!=process['snapshot']:
                raise ValueError('Controller launched wrong attempt')
            launch=read(attempt/'launch.json')
            if launch['snapshot_id']!=process['snapshot'] or not -30<=launch['start_unix_time']-process['started']<=180:
                raise ValueError('Attempt does not belong to this launch')
            return attempt
    if time.time()-process['started']>180:raise ValueError('Controller did not publish an attempt')
    return None


class Queue:
    def __init__(self,plan,path):
        self.plan,self.path=plan,path;self.registration,self.reg=inspect(plan)
        self.directory=path.parent;self.started=time.time();self.deadline=self.started+plan['max_seconds']
        self.calls=0;self.last_plot={};self.last_monitor={}

    def event(self,kind,**values):
        row=dict(kind=kind,time=time.time(),**values)
        with (self.directory/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)

    def status(self,**values):
        row=dict(updated=time.time(),pid=os.getpid(),process_identity=Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19],
                 plan_sha256=sha(self.path),started=self.started,**values)
        temp=self.directory/'.status.partial';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(self.directory/'status.json')

    def check(self):
        inspect(self.plan)
        if (self.directory/'stop').exists():raise RuntimeError('Continuation stop requested; active learner left running')
        if time.time()>=self.deadline:raise TimeoutError('Continuation time bound reached; active learner left running')

    def invoke(self,argv,label,*,timeout=900,cpu=True):
        self.check();self.calls+=1;log=self.directory/f'command-{self.calls:03d}-{label}.log'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        if cpu:env['JAX_PLATFORMS']='cpu'
        else:env.pop('JAX_PLATFORMS',None)
        self.event('command_started',label=label,argv=list(map(str,argv)),log=str(log))
        with log.open('xb') as stream:
            subprocess.run(['taskset','-c','0,1',*map(str,argv)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        self.event('command_passed',label=label,log_sha256=sha(log))

    def monitor(self,stage,attempt):
        if not (attempt/'rank-0/artifacts/metrics.jsonl').exists():return
        if time.time()-self.last_monitor.get(stage,0)<300:return
        self.invoke([sys.executable,'-B',STUDY/'monitor.py','--attempt',attempt.name],stage+'-monitor',timeout=60)
        self.last_monitor[stage]=time.time()
        if stage.endswith('qualification'):return
        seed=stage[-1];arms=[]
        for arm in ('cnn','transformer'):
            other=attempt_for(arm+'_seed'+seed,self.reg)
            if other is not None and (other/'rank-0/artifacts/evaluations.jsonl').exists():
                arms+=['--arm',arm,str(other)]
        if arms:
            env=os.environ.get('MPLCONFIGDIR');os.environ['MPLCONFIGDIR']='/tmp/gozero-mpl-cache'
            try:self.invoke([ROOT/'.gozero/analysis-environments/plotting/bin/python','-B',STUDY/'plot_progress.py',*arms,
                             '--output',STUDY/('seed'+seed+'-progress.png')],stage+'-plot',timeout=60)
            finally:
                if env is None:os.environ.pop('MPLCONFIGDIR',None)
                else:os.environ['MPLCONFIGDIR']=env

    def wait(self,stage):
        while True:
            self.check();attempt=attempt_for(stage,self.reg)
            if attempt is not None:
                result=attempt/'result.json'
                if result.exists():
                    closed=read(result)
                    if closed['status']!='passed':raise RuntimeError('Registered attempt failed: '+attempt.name)
                    return attempt
                launch=read(attempt/'launch.json')
                if time.time()>launch['start_unix_time']+launch['timeout_seconds']+launch['prepare_timeout_seconds']+300:
                    raise TimeoutError('Attempt did not close within its declared bounds')
                self.monitor(stage,attempt)
            self.status(status='waiting',stage=stage,attempt=attempt.name if attempt else None)
            time.sleep(self.plan['poll_seconds'])

    def validate_receipt(self,path,attempt):
        value=read(path)
        if value['status']!='passed' or value['attempt']!=attempt.name:
            raise ValueError('Wrong completed-stage evidence: '+str(path))
        snapshot=value.get('snapshot',value.get('training_snapshot'))
        if snapshot is not None and snapshot!=read(attempt/'launch.json')['snapshot_id']:
            raise ValueError('Evidence names another source snapshot')
        return value

    def finalize(self,stage,attempt):
        arm,phase=stage.split('_',1);name=stage.replace('_','-')
        snapshot=ROOT/'.gozero/snapshots'/self.reg['snapshots'][arm][phase]
        recipe=snapshot/read(snapshot/'manifest.json')['recipe']
        replica=STUDY/(name+'-ram-replicas-001.json')
        if not replica.exists():
            self.invoke([sys.executable,'-B',snapshot/'ops/replicate_ram_checkpoint.py','--workspace-root',ROOT,
                '--attempt',attempt.name,'--peer','2','--output',replica],name+'-replica')
        self.validate_receipt(replica,attempt)
        audit=STUDY/(name+'-audit-001.json')
        if not audit.exists():
            command=[sys.executable,'-B',recipe/('audit_qualification.py' if phase=='qualification' else 'audit_learning.py'),
                     '--workspace-root',ROOT,'--attempt',attempt,'--output',audit]
            if phase=='qualification':
                command+=['--cpu-budget',ROOT/self.reg['cpu_budgets'][arm]['path'],
                          '--checkpoint-replicas',replica,'--audit-output',STUDY/(name+'-learning-audit-001.json')]
            self.invoke(command,name+'-audit')
        self.validate_receipt(audit,attempt)
        if phase=='qualification':
            third=STUDY/(name+'-third-ram-copy-001.json')
            if not third.exists():
                self.invoke([sys.executable,'-B',snapshot/'ops/replicate_ram_checkpoint.py','--workspace-root',ROOT,
                    '--attempt',attempt.name,'--peer','3','--output',third],name+'-third-copy')
            self.validate_receipt(third,attempt)
            eviction=STUDY/(name+'-cache-eviction-001.json')
            if not eviction.exists():
                self.invoke([sys.executable,'-B',ROOT/'.gozero/snapshots'/CACHE_SNAPSHOT/'ops/qualification_ram_cache.py',
                    '--attempt',attempt.name,'--peer-receipts',replica,third,'--operation','evict','--output',eviction],name+'-eviction')
            self.validate_receipt(eviction,attempt)
        self.event('stage_audited',stage=stage,attempt=attempt.name,audit_sha256=sha(audit),replica_sha256=sha(replica))

    def contrast(self,seed):
        output=STUDY/(f'seed{seed}-contrast-001.json')
        cnn=STUDY/(f'cnn-seed{seed}-audit-001.json');transformer=STUDY/(f'transformer-seed{seed}-audit-001.json')
        if not output.exists():
            self.invoke([sys.executable,'-B',STUDY/'compare.py','--workspace-root',ROOT,'--registration',self.registration,
                '--registration-sha256',REGISTRATION_SHA,'--cnn-audit',cnn,'--cnn-audit-sha256',sha(cnn),
                '--transformer-audit',transformer,'--transformer-audit-sha256',sha(transformer),'--output',output],f'seed{seed}-contrast')
        result=read(output)
        if result['status']!='passed' or result['registration_sha256']!=REGISTRATION_SHA:
            raise ValueError('Paired contrast failed')
        self.event('paired_seed_contrasted',seed=seed,contrast_sha256=sha(output),gains=result['relative_endpoint_kl_improvement'])

    def run(self):
        self.event('continuation_started',stages=STAGES,plan_sha256=sha(self.path))
        try:
            for stage in STAGES:
                self.check();name=stage.replace('_','-')
                if not (STUDY/(name+'-process-001.json')).exists():
                    self.invoke([sys.executable,'-B',STUDY/'launch.py','--registration',self.registration,
                        '--registration-sha256',REGISTRATION_SHA,'--stage',stage],name+'-launch',timeout=180,cpu=False)
                attempt=self.wait(stage);self.finalize(stage,attempt)
                if stage in ('transformer_seed1','transformer_seed2'):self.contrast(int(stage[-1]))
            self.status(status='registered_comparison_runs_completed',next='Review both paired contrasts; 19x19 training remains unregistered')
            self.event('continuation_completed')
        except BaseException as error:
            self.status(status='stopped',error=repr(error),active_learner_terminated=False)
            self.event('continuation_stopped',error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
    plan=read(a.plan);inspect(plan)
    if a.inspect:
        print(json.dumps(dict(status='passed',stages=STAGES,mutations=False)));return
    with (a.plan.parent/'.queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.plan.parent/'status.json').exists():raise ValueError('Continuation already started; inspect existing state before recovery')
        Queue(plan,a.plan).run()


if __name__=='__main__':main()
