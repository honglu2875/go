"""Execute only the registered context-readout qualification, screen and optional replication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
SOURCE=Path(__file__).resolve().parents[3];RECIPE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256

Q='e538c2da1cbc5b5f49a3d30c35bb3eea90cae5ce1daa39be8975b6c91c8beccd'
FIRST='aee857e7db076a499fc40588d5103445445ef7233c4c24c446bd50433767bb30'
SECOND='d91ebf5226faefd3459250966754a640465e38ebf2a482b4587bee828a550dcd'
STORAGE='e903cbc5b3bb5d9b9cf2a06698393c9d6ad372e7cad3dd48e4202ff686754318'
REGISTRATION='e6e97d5ecb52a25c6ca0be1d47cc8f61c8a8895ef03e1d87787391346af50656'
REFERENCES=[('runs/shared-spatial-fff3608e/spatial-audit.json','b92d0ff04e3411cc61ac73aa58c55e0e813b33d8be7708824df81a0e3cedf040'),
 ('runs/shared-spatial-6ce8224f/analysis-pipeline/spatial-seed2-audit.json','ce8c9e85a104a9acb39d74757aeb6a18f426e8f337cbf192d79bb098d5a6a433')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.workspace_root.resolve();verify(SOURCE)
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();commands=[];attempts=[]
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    cpu={**env,'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/gozero-spatial-followups-mpl'}
    python=root/'.venv/bin/python';plot=root/'.gozero/analysis-environments/plotting/bin/python'
    registration=root/'research/studies/spatial_followups/registration_001.json'
    budget=root/'research/studies/spatial_followups/context_cpu_budget_001.json'
    if sha256(registration)!=REGISTRATION:raise ValueError('Prospective sequence changed')
    for name,h in REFERENCES:
        if sha256(root/name)!=h:raise ValueError('Paired reference changed')
    for name in (Q,FIRST,SECOND,STORAGE):verify(root/'.gozero/snapshots'/name)
    initial={'operator_snapshot':SOURCE.name,'created_unix':started,'registration_sha256':REGISTRATION,
        'qualification_snapshot':Q,'first_snapshot':FIRST,'second_snapshot':SECOND,'references':REFERENCES,
        'cpu_budget_sha256':sha256(budget),'automatic_retry':False,'max_new_tpu_attempts':3}
    (a.output/'launch.json').write_bytes(canonical_json(initial));(a.output/'launch.json').chmod(0o444)
    def execute(name,argv,*,timeout=600,is_cpu=True):
        command={'phase':name,'argv':list(map(str,argv)),'timeout_seconds':timeout};commands.append(command)
        (a.output/(name+'-command.json')).write_bytes(canonical_json(command))
        print(json.dumps({'phase':name,'status':'starting'}),flush=True)
        with (a.output/(name+'.log')).open('xb') as f:
            subprocess.run(list(map(str,argv)),cwd=root,env=cpu if is_cpu else env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        print(json.dumps({'phase':name,'status':'passed'}),flush=True)
    def analysis(name,script,args,plotting=False):
        execute(name,['taskset','-c','112-119',plot if plotting else python,'-B',RECIPE/script,*args])
    def relocate(label,attempt,host):
        execute(label,[python,'-B',root/'.gozero/snapshots'/STORAGE/'ops/retain_checkpoint_replicas.py',
            '--workspace-root',root,'--output',a.output/(label+'-receipt.json'),'--apply','--attempt',attempt,
            '--copy-missing-replicas','--retained-hosts',str(host),'--relocate-single-owner'],timeout=1500,is_cpu=False)
    def pod(label,snapshot):
        if any(not (p.parent/'result.json').exists() for p in (root/'runs').glob('pod-*/launch.json')):raise ValueError('Another TPU attempt is open')
        s=os.statvfs(root)
        if s.f_bavail*s.f_frsize<3*2**30:raise ValueError('Less than 3 GiB persistent checkpoint space')
        source=root/'.gozero/snapshots'/snapshot
        execute(label,[python,'-B',source/'ops/pod_run.py','--snapshot',source,'--workspace-root',root,
            '--timeout','7200','--prepare-timeout','180','--controller-cpus','32'],timeout=7500,is_cpu=False)
        entries=[json.loads(x) for x in (a.output/(label+'.log')).read_text().splitlines() if x.startswith('{')]
        launch=next(x for x in entries if x.get('kind')=='pod_attempt');attempt=Path(launch['attempt'])
        closed=read_json(attempt/'result.json')
        if closed['status']!='passed' or closed['snapshot_id']!=snapshot:raise ValueError('Attempt did not pass')
        attempts.append(attempt.name);return attempt
    def finish_pair(label,attempt,index):
        audit=a.output/(label+'-audit.json')
        analysis(label+'-audit','audit_learning.py',['--workspace-root',root,'--attempt',attempt,'--output',audit])
        folder=a.output/(label+'-comparison');ref,h=REFERENCES[index]
        analysis(label+'-compare','report_comparison.py',['--workspace-root',root,'--audit','Spatial control',root/ref,h,
            '--audit','Context readout',audit,sha256(audit),'--output',folder],plotting=True)
        contrast=a.output/(label+'-contrast.json')
        analysis(label+'-contrast','compare_context.py',['--workspace-root',root,'--comparison',folder/'comparison.json',
            '--comparison-sha256',sha256(folder/'comparison.json'),'--control','Spatial control','--candidate','Context readout','--output',contrast])
        return contrast
    try:
        gate=root/'runs/spatial-followups-cnn-main/analysis-recovery/result.json';deadline=time.monotonic()+7300
        print(json.dumps({'phase':'waiting_for_cnn_audit'}),flush=True)
        while not gate.exists():
            if time.monotonic()>deadline:raise TimeoutError('CNN analysis has not completed')
            time.sleep(10)
        if read_json(gate)['status']!='passed':raise ValueError('CNN analysis failed; inspect before proceeding')
        relocate('relocate-cnn-qualification','pod-20260913T174317Z-18817a9f',1)
        relocate('relocate-old-control','pod-20260913T143017Z-44bc36ef',2)
        qualification=pod('qualification',Q)
        qfile=a.output/'qualification.json';qaudit=a.output/'qualification-learning-audit.json'
        analysis('qualification-audit','audit_qualification.py',['--workspace-root',root,'--attempt',qualification,
            '--cpu-budget',budget,'--output',qfile,'--audit-output',qaudit])
        q=read_json(qfile)
        for snapshot in (FIRST,SECOND):
            source=root/'.gozero/snapshots'/snapshot;c=read_json(source/'resolved_config.json')
            if c['model']!=q['model'] or c['dataset']!=q['dataset']:raise ValueError('Qualified model/data differ')
            for name,h in q['numerical_sources'].items():
                if sha256(source/'research/recipes/context_spatial'/name)!=h:raise ValueError('Qualified numerical source differs')
        first=pod('first-seed',FIRST);contrast=finish_pair('first-seed',first,0);screen=read_json(contrast)
        if screen['replication_screen_passed']:
            replication_registration=a.output/'replication-registration.json'
            r={'kind':'prospective_context_replication','created_unix':time.time(),'screen_contrast_sha256':sha256(contrast),
                'seed':91312428,'control_snapshot':'6ce8224facc784e702f7df8f8e4378aef431c5b77f9035937c8de1450435f53a',
                'candidate_snapshot':SECOND,'selection_rule':'Retain both seeds; require both to pass >=1% KL improvement and <=15% initial decode slowdown before combining.'}
            replication_registration.write_bytes(canonical_json(r));replication_registration.chmod(0o444)
            relocate('relocate-context-qualification',qualification.name,3)
            second=pod('second-seed',SECOND);second_contrast=finish_pair('second-seed',second,1)
            analysis('replication','compare_replication.py',['--workspace-root',root,'--pair',contrast,sha256(contrast),
                '--pair',second_contrast,sha256(second_contrast),'--registration',replication_registration,
                '--registration-sha256',sha256(replication_registration),'--output',a.output/'replication'],plotting=True)
            replicated=read_json(a.output/'replication/replication.json')['each_seed_passed_one_percent_latency_screen']
        else:replicated=False
        result={'status':'passed','first_contrast':str(contrast),'first_contrast_sha256':sha256(contrast),
            'first_seed_screen_passed':screen['replication_screen_passed'],'replicated_gain':replicated,
            'next_action':'Review the CNN and context outcomes, select/register the attention parent, and qualify it before learning. This controller launches no attention or auxiliary run.'}
    except BaseException as error:
        result={'status':'failed','error':repr(error)};raise
    finally:
        result.update(operator_snapshot=SOURCE.name,started_unix=started,ended_unix=time.time(),attempts=attempts,commands=commands,automatic_retry=False)
        with (a.output/'result.json').open('xb') as f:f.write(canonical_json(result))
        for path in a.output.iterdir():
            if path.is_file():path.chmod(0o444)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
