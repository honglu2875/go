"""Run two registered MoE arms sequentially; preserve all existing checkpoints."""
import argparse, fcntl, json, math, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shlex
from execute_run import ROOT,STUDY,read,sha,publish,require,resources
from gozero.snapshots import verify
from gozero.pod import load_hosts, SSH_OPTIONS
from gozero.disk_mirror import inventory,publish as mirror
from observation import observe
from monitor import resource_observation,request_cancellation
from promote import promote
from compare import report
PYTHON=ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
RESERVE=math.ceil((317001492*12+4)*1.01)+(32<<20)


def backup(paths,label):
    files=inventory(ROOT,sorted({str(p.relative_to(ROOT)) for p in paths}));copies=[]
    hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')}
    for rank in (1,3):
        copies.append(dict(peer_rank=rank,**mirror(ROOT,ROOT/'.gozero/retained-source-archives'/('20261006-moe-batch64-'+label),
            files=files,peer=hosts[rank].ssh,python=str(PYTHON),floor_bytes=2<<30)))
    answer=dict(status='passed',files=files,copies=copies);publish(STUDY/(label+'-backup.json'),answer);return answer


def idle():
    code="""import os,json
from pathlib import Path
holders=[];unreadable=0
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try: ds=list((p/'fd').iterdir())
 except (PermissionError,FileNotFoundError,ProcessLookupError):unreadable+=1;continue
 for d in ds:
  try: target=os.readlink(d)
  except OSError:continue
  if target.startswith(('/dev/accel','/dev/vfio')):holders.append(int(p.name));break
print(json.dumps(dict(holders=holders,unreadable=unreadable)))
"""
    def one(h):
        output=subprocess.check_output(['ssh',*SSH_OPTIONS,h.ssh,shlex.join(['taskset','-c','0,1','python3','-c',code])],text=True,timeout=45)
        value=json.loads(output);require(not value['holders'],'Accelerator is occupied');return dict(rank=h.rank,**value)
    require(not any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json')),'An attempt is open')
    with ThreadPoolExecutor(4) as pool:return list(pool.map(one,load_hosts(ROOT/'ops/hosts.json')))


class Campaign:
    def __init__(self,path,digest):
        require(sha(path)==digest,'Registration changed');self.path=path;self.digest=digest;self.r=read(path)
        require(self.r['kind']=='tuned_default_moe_campaign' and self.r['status']=='prepared','Wrong campaign')
        self.started=time.time();self.deadline=self.started+14*3600;self.records={'dense':self.r['dense_reference']};self.check()

    def check(self):
        require(time.time()<self.deadline,'Campaign deadline reached');require(sha(self.path)==self.digest,'Registration changed')
        for name,digest in self.r['inputs'].items():require(sha(ROOT/name)==digest,'Input changed: '+name)

    def event(self,kind,**values):
        row=dict(kind=kind,time=time.time(),**values)
        with (STUDY/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)

    def system(self):
        self.check();self.event('occupancy',hosts=idle())
        snapshot=ROOT/'.gozero/snapshots'/self.r['system_snapshot'];verify(snapshot)
        cfg=read(snapshot/'resolved_config.json');self.event('system_started',snapshot=snapshot.name)
        argv=[str(PYTHON),'-B',str(snapshot/'ops/pod_run.py'),'--snapshot',str(snapshot),'--workspace-root',str(ROOT),
            '--timeout','6000','--prepare-timeout','600','--controller-cpus','6','--controller-cpu-list','2,3,4,5,6,7']
        log=STUDY/'system-controller.log'
        with log.open('xb') as f:subprocess.run(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=6900)
        heads=[json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
        heads=[h for h in heads if h.get('kind')=='pod_attempt'];require(len(heads)==1,'Ambiguous system attempt')
        attempt=ROOT/'runs'/heads[0]['attempt_id'];closed=read(attempt/'result.json')
        require(closed['status']=='passed' and closed['snapshot_id']==snapshot.name,'Systems attempt failed')
        reports=[]
        for rank in range(4):
            process=read(attempt/f'rank-{rank}/result.json');r=read(attempt/f'rank-{rank}/artifacts/result.json')
            require(process['status']=='passed' and process['source_integrity'] and process['returncode']==0
                and r['status']=='passed' and r['snapshot_id']==snapshot.name,'Rank systems evidence failed')
            require([{k:v[k] for k in cfg['cases'][0]} for v in r['cases']]==cfg['cases'],'Systems case differs')
            require(all(x['global_peak_bytes']<=31<<30 and len(x['updates'])==2 and all(u['metrics']['accepted']==1
                and u['metrics'].get('moe_dropped_tokens',0)==0 for u in x['updates']) for x in r['cases']),'Systems budget or acceptance failed')
            reports.append(r)
        require(sorted(r['jax_rank'] for r in reports)==list(range(4)),'Missing logical rank')
        for r in reports[1:]:require([[u['metrics'] for u in x['updates']] for x in r['cases']]==[[u['metrics'] for u in x['updates']] for x in reports[0]['cases']],'Systems metrics differ by rank')
        control=json.loads((ROOT/'runs'/self.r['dense_reference']['attempt']/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()[0])
        expected=next(x for x in reports[0]['cases'] if x['arm']=='dense')['updates'][0]['metrics']
        differences={k:dict(expected=v,actual=control.get(k)) for k,v in expected.items() if k not in control or not math.isclose(v,control[k],rel_tol=2e-6,abs_tol=2e-6)}
        require(not differences,'Dense first update differs: '+str(differences))
        receipt=STUDY/'system-audit.json';publish(receipt,dict(status='passed',snapshot=snapshot.name,attempt=attempt.name,reports=reports,dense_first_update_verified=True))
        backup([receipt,*[p for p in attempt.rglob('*') if p.is_file()]],'system-closure')
        self.event('system_passed',attempt=attempt.name)

    def learning(self,label):
        self.check();self.event('occupancy',hosts=idle())
        plan=dict(self.r['arms'][label]);plan['prerequisites'][str((STUDY/'system-audit.json').relative_to(ROOT))]=sha(STUDY/'system-audit.json')
        plan['prerequisites'][str(self.path.relative_to(ROOT))]=self.digest
        require(time.time()+plan['timeout_seconds']+1800<self.deadline,'Insufficient campaign time')
        path=STUDY/(label+'-plan.json');publish(path,plan);resources(plan)
        backup([path],label+'-plan');self.event('learning_started',label=label,snapshot=plan['snapshot'])
        self.guarded_learning(path,plan,label)
        folder=STUDY/plan['output_directory'];result=read(folder/'result.json');audit=folder/'audit.json'
        require(result['status']=='passed' and sha(audit)==result['audit_sha256'] and sha(folder/'replica.json')==result['replica_sha256'],'Learning closure identity differs')
        record=dict(attempt=result['attempt'],audit=str(audit.relative_to(ROOT)),audit_sha256=sha(audit))
        self.records[label]=record
        attempt=ROOT/'runs'/result['attempt']
        paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='arrays.npz']
        paths += [p for p in attempt.rglob('*') if p.is_file() and p.name!='arrays.npz' and p.stat().st_size<16<<20]
        backup(paths,label+'-closure');self.event('learning_passed',label=label,record=record)

    def run(self):
        self.check();hosts=resource_observation(ROOT/'.gozero/snapshots'/self.r['system_snapshot'])
        for h in hosts:
            ram=2*RESERVE if h['rank'] in (0,1,3) else 0;disk=RESERVE if h['rank'] in (1,3) else 0
            require(h['shm_free']>(64<<30)+ram and h['disk_free']>(2<<30)+disk+(512<<20) and h['memory_available']>96<<30,'Campaign storage reserve insufficient')
        publish(STUDY/'capacity-001.json',dict(status='passed',hosts=hosts,ram_endpoints=2,selected_disk_endpoints=1,existing_files_preserved=True))
        self.system()
        for arm in ('temporal','balance_low'):self.learning(arm)
        comparison=report(self.r,self.records);selected=comparison['selected_moe']
        promote(self.records[selected],'selected-moe');self.event('selected_disk_copies_verified',selected=selected)
        final=backup([p for p in STUDY.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md')],'campaign-closure')
        publish(STUDY/'result.json',dict(status='passed',started=self.started,finished=time.time(),registration_sha256=self.digest,
            records=self.records,comparison_sha256=sha(STUDY/'comparison.json'),closure_backup_sha256=sha(STUDY/'campaign-closure-backup.json'),selected_moe=selected,provisional_winner=comparison['provisional_winner']))
        self.event('campaign_passed',selected_moe=selected,provisional_winner=comparison['provisional_winner'])

    def guarded_learning(self,path,plan,label):
        argv=[str(PYTHON),'-B',str(STUDY/'execute_run.py'),'--plan',str(path),'--plan-sha256',sha(path),'--run']
        attempt=None;errors=0;resource_errors=0;last=-1;last_progress=time.monotonic();last_resources=0.;cancelled=False
        snapshot=ROOT/'.gozero/snapshots'/plan['snapshot']
        config=read(snapshot/'resolved_config.json');startup_checked=False;startup_failure=None
        with (STUDY/(label+'-controller.log')).open('xb') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                controller=STUDY/plan['output_directory']/'controller.log'
                if attempt is None and controller.exists():
                    for line in controller.read_text().splitlines():
                        if line.startswith('{'):
                            row=json.loads(line)
                            if row.get('kind')=='pod_attempt':attempt=row['attempt_id'];break
                if attempt:
                    reason=None
                    try:
                        value=observe(ROOT,attempt,plan['snapshot'],validation_positions=64371);errors=0
                        temp=STUDY/(label+'-current.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(STUDY/(label+'-current.json'))
                        if value['completed_updates']!=last:
                            last=value['completed_updates'];last_progress=time.monotonic()
                            self.event('progress',label=label,attempt=attempt,updates=last,positions=value['position_exposures'])
                        if last and not startup_checked:
                            startup_checked=True
                            first=json.loads((ROOT/'runs'/attempt/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()[0])
                            system=read(STUDY/'system-audit.json')['reports'][0]['cases']
                            expected=next(r for r in system if r['arm']==label and r['bucket']==512)['updates'][0]['metrics']
                            differences={k:dict(expected=v,actual=first.get(k)) for k,v in expected.items()
                                if k not in first or not math.isclose(first[k],v,rel_tol=2e-6,abs_tol=2e-6)}
                            publish(STUDY/(label+'-startup.json'),dict(status='failed' if differences else 'passed',differences=differences))
                            if differences:startup_failure='Fresh first update differs from systems qualification'
                    except Exception as error:
                        errors+=1;self.event('monitor_error',label=label,error=repr(error),consecutive=errors)
                    if errors>=5:reason='Five consecutive observation failures'
                    if startup_failure:reason=startup_failure
                    if time.monotonic()-last_progress>1800:reason='No accepted update for 30 minutes'
                    if time.time()>self.deadline:reason='Campaign deadline'
                    if time.monotonic()-last_resources>=600:
                        try:
                            observed=resource_observation(snapshot);resource_errors=0;last_resources=time.monotonic()
                            self.event('resources',label=label,hosts=observed)
                            if any(r['disk_free']<(2<<30) or r['shm_free']<64<<30
                                   or r['memory_available']<24<<30 for r in observed):reason='Resource reserve exhausted'
                        except Exception as error:
                            resource_errors+=1;last_resources=time.monotonic()-570
                            self.event('resource_error',label=label,error=repr(error),consecutive=resource_errors)
                            if resource_errors>=3:reason='Repeated resource inspection failure'
                    if reason and not cancelled and not (ROOT/'runs'/attempt/'result.json').exists():
                        request_cancellation(snapshot,attempt,STUDY,reason);cancelled=True
                        self.event('cancel_requested',label=label,reason=reason)
                time.sleep(30)
            require(child.returncode==0 and not cancelled,'Learning run failed: '+label)

def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--backup-sha256');p.add_argument('--run',action='store_true');a=p.parse_args();c=Campaign(a.registration,a.sha256)
    if not a.run:print(json.dumps(dict(status='prepared',accelerator_jobs_started=False)));return
    proof=STUDY/'registration-backup.json';require(a.backup_sha256 and sha(proof)==a.backup_sha256,'Registration backup differs')
    b=read(proof);require(b['status']=='passed' and len(b['copies'])==2 and b['files'][str(a.registration.resolve().relative_to(ROOT))]['sha256']==a.sha256,'Source backup incomplete')
    with (STUDY/'.campaign.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:c.run()
        except BaseException as error:
            if not (STUDY/'result.json').exists():publish(STUDY/'result.json',dict(status='failed',error=repr(error),records=c.records,started=c.started,finished=time.time()))
            c.event('campaign_failed',error=repr(error));raise

if __name__=='__main__':main()
