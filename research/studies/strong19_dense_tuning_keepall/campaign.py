"""Run the prospectively bounded LR refinement, confirmation and batch study."""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from execute_run import ROOT,STUDY,read,sha,publish,require,resources
from gozero.snapshots import freeze,verify
from gozero.pod import load_hosts
from gozero.disk_mirror import inventory,publish as mirror
from observation import observe
from monitor import resource_observation,request_cancellation
from decision import METRICS,summarize,select,next_lr,gates
from promote import promote

INITIAL=ROOT/'research/studies/strong19_dense_lr'
REPLAY=ROOT/'research/studies/strong19_recovery/draw-replay-001.json'
PYTHON=ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
RESERVE=math.ceil((232011540*12+4)*1.01)+(32<<20)


def backup(paths,label):
    files=inventory(ROOT,sorted({str(p.relative_to(ROOT)) for p in paths}))
    copies=[];hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')}
    for rank in (1,3):
        copies.append(dict(peer_rank=rank,**mirror(ROOT,ROOT/'.gozero/retained-source-archives'/('20261005-dense-tuning-keepall-'+label),
            files=files,peer=hosts[rank].ssh,python=str(PYTHON),floor_bytes=2<<30)))
    receipt=dict(status='passed',files=files,copies=copies)
    path=STUDY/(label+'-backup.json');publish(path,receipt)
    return path,receipt


def closure_backup(record,label):
    folder=(ROOT/record['stage_result']).parent;attempt=ROOT/'runs'/record['attempt']
    paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='arrays.npz']
    paths += [p for p in attempt.rglob('*') if p.is_file() and p.name!='arrays.npz' and p.stat().st_size<16<<20]
    _,receipt=backup(paths,label+'-closure')
    return receipt


class Campaign:
    def __init__(self,path,digest):
        require(sha(path)==digest,'Campaign registration changed')
        self.path=path;self.digest=digest;self.r=read(path);self.started=time.time();self.deadline=self.started+30*3600
        require(self.r['status']=='prepared' and self.r['kind']=='bounded_dense_lr_batch_campaign','Unexpected campaign registration')
        self.records={};self.backups={}
        self.check()

    def check(self):
        require(time.time()<self.deadline,'Campaign time budget exhausted')
        require(sha(self.path)==self.digest,'Registration changed')
        for name,digest in self.r['inputs'].items():require(sha(ROOT/name)==digest,'Pinned input changed: '+name)

    def event(self,kind,**values):
        row=dict(kind=kind,time=time.time(),**values)
        with (STUDY/'campaign-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)

    def record(self,stage,lr,games=128):
        result=read(stage/'result.json');audit=stage/'audit.json';a=read(audit)
        require(result['status']=='passed' and result['audit_sha256']==sha(audit) and a['status']=='passed','Unclosed trial')
        require(a['dataset_manifest_sha256']==self.r['dataset_manifest_sha256'],'Wrong corpus')
        require(a['initial_parameters_sha256']==self.r['initial_parameters_sha256'],'Initial weights differ')
        baseline=read(ROOT/self.r['control']['audit'])
        for key in ('validation_history','training_probe_history'):
            require(all(math.isclose(a[key][0]['metrics'][m],baseline[key][0]['metrics'][m],rel_tol=2e-6,abs_tol=2e-6)
                        for m in METRICS),'Initial evaluation differs')
        return dict(attempt=result['attempt'],audit=str(audit.relative_to(ROOT)),audit_sha256=sha(audit),peak_lr=lr,batch_games=games,
            stage_result=str((stage/'result.json').relative_to(ROOT)),stage_result_sha256=sha(stage/'result.json'))

    def launch_learning(self,label,config,recipe,equivalent):
        self.check();steps=equivalent*128//(4*config['learner']['games_per_host'])
        config['checkpoint_every']=config['steps']
        config.pop('checkpoint_disk',None)
        config.update(checkpoint_temporary=True,checkpoint_minimum_free_bytes=64<<30)
        config_path=STUDY/(label+'-config.json');publish(config_path,config)
        snapshot=freeze(ROOT,Path('research/recipes')/recipe,config_path,ROOT/'.gozero/snapshots');verify(snapshot)
        plan=dict(kind='registered_joint19_run',created=time.time(),snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),
            purpose='learning',steps=steps,schedule_steps=config['steps'],checkpoint_every=config['steps'],parameters=232011540,
            timeout_seconds=equivalent*100+1800,replica_peer=1,checkpoint_reserve_bytes=RESERVE,
            shm_floor_bytes=64<<30,memory_floor_bytes=96<<30,disk_floor_bytes=2<<30,producer_growth_reserve_bytes=0,
            disk_checkpoint_reserve_by_host={str(h):0 for h in range(4)},
            additional_reserve_by_host={str(h):RESERVE if h in (0,1,3) else 0 for h in range(4)},output_directory=label+'-stage',audit_python=str(PYTHON),
            expected_positions=read(REPLAY)['draws'][equivalent-1]['cumulative_positions'],validation_positions=64371,
            operators={str((STUDY/n).relative_to(ROOT)):sha(STUDY/n) for n in ('execute_run.py','audit_run.py','replicate.py','batch_audit.py','ram_copy.py')},
            prerequisites={str(self.path.relative_to(ROOT)):self.digest,
                str((STUDY/'batch-cpu-qualification-002.json').relative_to(ROOT)):sha(STUDY/'batch-cpu-qualification-002.json')},
            scope='Prospectively selected fresh run; fixed model and canonical game exposure. No test access.')
        for prerequisite in [INITIAL/'sequence-001/result.json',*sorted(STUDY.glob('lr-decision-*.json')),
                             STUDY/'lr-confirmation-review.json',STUDY/'lr-selection.json']:
            if prerequisite.exists():plan['prerequisites'][str(prerequisite.relative_to(ROOT))]=sha(prerequisite)
        if recipe=='strong19_dense_batch_keepall':
            proof=STUDY/'batch-system-audit.json';require(read(proof)['status']=='passed','Batch systems qualification missing')
            plan['prerequisites'][str(proof.relative_to(ROOT))]=sha(proof)
        # Require enough wall-time for the complete bounded run before allocating the pod.
        require(time.time()+plan['timeout_seconds']<self.deadline,'Insufficient remaining campaign time for this run')
        path=STUDY/(label+'-plan.json');publish(path,plan)
        resources(plan)
        paths=[path,config_path,self.path,*[ROOT/n for n in self.r['inputs']]]
        paths += [p for p in snapshot.rglob('*') if p.is_file()]
        backup(paths,label+'-registration')
        self.event('learning_registered',label=label,snapshot=snapshot.name,steps=steps,peak_lr=config['learner']['learning_rate'],games=4*config['learner']['games_per_host'])
        self.guarded_learning(path,plan,label)
        record=self.record(STUDY/(label+'-stage'),config['learner']['learning_rate'],4*config['learner']['games_per_host'])
        self.records[label]=record;self.backups[label]=closure_backup(record,label)
        self.event('learning_audited',label=label,record=record,summary=summarize(record,equivalent))
        return record

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
                            if 'batch_replay' not in config:
                                first=json.loads((ROOT/'runs'/attempt/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()[0])
                                reference=json.loads((ROOT/'runs'/self.r['control']['attempt']/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()[0])
                                factor=config['learner']['learning_rate']/.001
                                ignored={'turn','bucket','local_entries_sha256','local_symmetries','cumulative_learning_seconds','cumulative_sampling_seconds'}
                                expected={k:v*(factor if k=='learning_rate' or k.startswith('update_norm_') else 1) for k,v in reference.items() if k not in ignored}
                                differences={k:dict(expected=v,actual=first.get(k)) for k,v in expected.items()
                                             if k not in first or not math.isclose(first[k],v,rel_tol=2e-6,abs_tol=2e-6)}
                                publish(STUDY/(label+'-startup.json'),dict(status='failed' if differences else 'passed',differences=differences))
                                if differences:startup_failure='Fresh first update differs from the scaled reference'
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

    def system_qualification(self,peak):
        self.check();c=read(STUDY/'batch-system-template-001.json')
        for cfg in c['configurations'].values():cfg['learner'].update(learning_rate=peak,end_learning_rate=.3*peak)
        path=STUDY/'batch-system-config.json';publish(path,c)
        snapshot=freeze(ROOT,Path('research/recipes/strong19_dense_batch_keepall'),path,ROOT/'.gozero/snapshots');verify(snapshot)
        plan=dict(status='prepared',snapshot=snapshot.name,configuration_sha256=sha(snapshot/'resolved_config.json'),
                  timeout_seconds=6000,scope='Four shape cases, two repeated updates each. No learning endpoint.')
        planpath=STUDY/'batch-system-plan.json';publish(planpath,plan)
        backup([path,planpath,self.path,*[ROOT/n for n in self.r['inputs']],*[p for p in snapshot.rglob('*') if p.is_file()]],'batch-system-registration')
        require(not any(not (p.parent/'result.json').exists() for p in (ROOT/'runs').glob('pod-*/launch.json')),'An attempt is open')
        resources(dict(snapshot=snapshot.name,shm_floor_bytes=64<<30,memory_floor_bytes=96<<30,
                       producer_growth_reserve_bytes=0,additional_reserve_by_host={str(h):0 for h in range(4)},
                       disk_floor_bytes=2<<30,disk_checkpoint_reserve_by_host={str(h):0 for h in range(4)}))
        argv=[str(PYTHON),'-B',str(snapshot/'ops/pod_run.py'),'--snapshot',str(snapshot),'--workspace-root',str(ROOT),
              '--timeout','6000','--prepare-timeout','600','--controller-cpus','6','--controller-cpu-list','2,3,4,5,6,7']
        log=STUDY/'batch-system-controller.log';self.event('batch_system_started',snapshot=snapshot.name)
        with log.open('xb') as f:subprocess.run(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=6900)
        headers=[json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
        headers=[r for r in headers if r.get('kind')=='pod_attempt'];require(len(headers)==1,'Ambiguous qualification attempt')
        attempt=ROOT/'runs'/headers[0]['attempt_id'];closed=read(attempt/'result.json')
        require(closed['status']=='passed' and closed['snapshot_id']==snapshot.name,'Qualification did not close')
        reports=[]
        for rank in range(4):
            process=read(attempt/f'rank-{rank}/result.json');report=read(attempt/f'rank-{rank}/artifacts/result.json')
            require(process['status']=='passed' and process['source_integrity'] and process['returncode']==0
                    and report['status']=='passed' and report['snapshot_id']==snapshot.name,'Qualification rank failed')
            require([{k:r[k] for k in c['cases'][0]} for r in report['cases']]==c['cases'],'Qualification cases changed')
            require(all(r['global_peak_bytes']<=31<<30 and len(r['updates'])==2 and all(u['metrics']['accepted']==1 for u in r['updates']) for r in report['cases']),'Shape admission failed')
            reports.append(report)
        require(sorted(r['jax_rank'] for r in reports)==list(range(4)),'Logical ranks incomplete')
        for r in reports[1:]:
            require([[u['metrics'] for u in x['updates']] for x in r['cases']]==[[u['metrics'] for u in x['updates']] for x in reports[0]['cases']],'Replicated qualification metrics differ')
        answer=dict(status='passed',snapshot=snapshot.name,attempt=attempt.name,reports=reports,plan_sha256=sha(planpath))
        receipt=STUDY/'batch-system-audit.json';publish(receipt,answer)
        backup([receipt,planpath,*[p for p in attempt.rglob('*') if p.is_file()]],'batch-system-closure')
        self.event('batch_system_passed',attempt=attempt.name)

    def run(self):
        self.event('waiting_for_initial_grid',initial_registration_sha256=self.r['initial_registration_sha256'])
        result_path=INITIAL/'sequence-001/result.json'
        while not result_path.exists():self.check();time.sleep(30)
        result=read(result_path);require(result['status']=='passed' and result['registration_sha256']==self.r['initial_registration_sha256'],'Initial grid failed')
        require(sha(INITIAL/'sequence-001/comparison.json')==result['comparison_sha256'],'Initial comparison changed')
        observed=resource_observation(ROOT/'.gozero/snapshots'/read(INITIAL/'pair-registration-001.json')['arms']['lr06']['snapshot'])
        for row in observed:
            ram_reserve=5*RESERVE if row['rank'] in (0,1,3) else 0
            disk_reserve=2*RESERVE if row['rank'] in (1,3) else 0
            require(row['shm_free']>(64<<30)+ram_reserve and row['disk_free']>(2<<30)+disk_reserve+(512<<20)
                    and row['memory_available']>96<<30,'Worst-case non-destructive campaign storage does not fit')
        publish(STUDY/'campaign-capacity.json',dict(status='passed',hosts=observed,max_new_ram_checkpoints=5,
            selected_disk_slots_per_peer=2,scope='All existing checkpoint payloads preserved; capacity reserved for new files only.'))
        self.records['control']=self.r['control']
        for label,peak in (('lr06',.0006),('lr15',.0015)):
            self.records[label]=self.record(INITIAL/(label+'-stage-001'),peak)
            self.backups[label]=closure_backup(self.records[label],label)
        table={k:summarize(v) for k,v in self.records.items()};previous=None
        for index in range(2):
            self.check();decision=next_lr(table,index,previous);decision.update(status='prepared',round=index+1,table=table)
            publish(STUDY/f'lr-decision-{index+1:03d}.json',decision);self.event('lr_decision',**decision)
            if decision['action']=='stop':break
            config=read(INITIAL/'lr06-config-001.json');config['learner'].update(learning_rate=decision['peak_lr'],end_learning_rate=.3*decision['peak_lr'])
            label=f'lr-extra-{index+1:03d}'
            self.launch_learning(label,config,'strong19_dense_lr_keepall',128);previous=label
            table[label]=summarize(self.records[label])
        chosen=select(table)
        if chosen!='control':
            config=read(INITIAL/'lr06-config-001.json');peak=self.records[chosen]['peak_lr']
            config['learner'].update(learning_rate=peak,end_learning_rate=.3*peak)
            confirmation=self.launch_learning('lr-confirmation',config,'strong19_dense_lr_keepall',256)
            original=read(ROOT/self.records[chosen]['audit']);fresh=read(ROOT/confirmation['audit'])
            for key in ('validation_history','training_probe_history'):
                for before,after in zip(original[key],fresh[key][:len(original[key])],strict=True):
                    require(before['turn']==after['turn'] and before['episode_ids_sha256']==after['episode_ids_sha256'],'Confirmation prefix differs')
                    require(all(math.isclose(before['metrics'][m],after['metrics'][m],rel_tol=1e-5,abs_tol=2e-6) for m in METRICS),'Confirmation prefix metrics differ')
            later={k:summarize(self.records[k],256) for k in ('control','lr-confirmation')}
            settled=select(later)
            publish(STUDY/'lr-confirmation-review.json',dict(status='passed',table=later,gates=gates(later['lr-confirmation'],later['control']),selected=settled))
        else:settled='control'
        promotion=promote(self.records[settled],'selected-lr')
        self.event('selected_lr_disk_copies_verified',proof=promotion)
        peak=self.records[settled]['peak_lr'];self.event('lr_settled_for_batch_study',selected=settled,peak_lr=peak)
        publish(STUDY/'lr-selection.json',dict(status='passed',selected=settled,peak_lr=peak,screen_table=table,
            record=self.records[settled],production_defaults_changed=False))
        self.system_qualification(peak)
        batch_records={'control':self.records[settled]}
        for games in (64,256):
            cfg=read(STUDY/f'batch{games}-template-001.json');cfg['learner'].update(learning_rate=peak,end_learning_rate=.3*peak)
            label=f'batch{games}';batch_records[label]=self.launch_learning(label,cfg,'strong19_dense_batch_keepall',128)
        batch_table={k:summarize(v) for k,v in batch_records.items()};quality=select(batch_table)
        base=batch_table['control'];fast=[]
        for label,row in batch_table.items():
            if label=='control':continue
            if (all(row[part][k]<=1.01*base[part][k] for part in ('endpoint','tail') for k in ('expert_kl','family_kl'))
                and row['tail']['value_mse']<=1.05*base['tail']['value_mse'] and not row['sustained_overfit']
                and row['learning_seconds']<=.95*base['learning_seconds']):fast.append(label)
        speed=min(fast,key=lambda k:batch_table[k]['learning_seconds']) if fast else 'control'
        keep=quality if quality!='control' else speed
        if keep!='control':
            promotion=promote(self.records[keep],'selected-batch')
            self.event('selected_batch_disk_copies_verified',proof=promotion)
        answer=dict(status='passed',started=self.started,finished=time.time(),learning_rate=peak,lr_record=self.records[settled],
            batch_table=batch_table,batch_records=batch_records,batch_quality_candidate=quality,batch_throughput_candidate=speed,
            retained_batch_candidate=keep,existing_checkpoints_deleted=False,production_defaults_changed=False,
            scope='Single-seed bounded fixed-data tuning. Later LR check at 256 updates; batch comparisons at equal 7,001,181 position exposures. No strength, convergence or MFU claim.')
        publish(STUDY/'comparison.json',answer);write_report(answer)
        write_curves(self.records)
        closure_path,_=backup([STUDY/'comparison.json',STUDY/'RESULTS.md',STUDY/'curves.csv',*[p for p in STUDY.glob('*.json')],
                ],'campaign-closure')
        answer['closure_backup_sha256']=sha(closure_path);publish(STUDY/'result.json',answer)
        self.event('campaign_passed',peak_lr=peak,quality_candidate=quality,throughput_candidate=speed)


def write_report(result):
    lines=['# Dense LR and batch tuning','',f"Settled study peak LR: {result['learning_rate']:.3g}. Production defaults unchanged.",'',
           '| Global games | Updates | Policy KL | Family KL | Value MSE | Learning hours |',
           '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for label in ('batch64','control','batch256'):
        row=result['batch_table'][label];m=row['endpoint']
        lines.append(f"| {row['batch_games']} | {row['steps']} | {m['expert_kl']:.6f} | {m['family_kl']:.6f} | {m['value_mse']:.6f} | {row['learning_seconds']/3600:.3f} |")
    lines+=['',f"Quality candidate: {result['batch_quality_candidate']}. Throughput candidate: {result['batch_throughput_candidate']}.",'',result['scope'],'',
            'See each stage audit for every validation/probe checkpoint and overfit observation. '
            'Existing checkpoints are preserved. New trial states have RAM replicas; selected new endpoints have two verified disk copies.']
    with (STUDY/'RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')


def write_curves(records):
    with (STUDY/'curves.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['label','peak_lr','global_games','split','actual_update','canonical_update',
                                             'position_exposures','learning_seconds',*METRICS])
        for label,record in records.items():
            audit=read(ROOT/record['audit']);rows=[json.loads(x) for x in (ROOT/'runs'/record['attempt']/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()]
            positions=0;clocks={0:(0,0)}
            for row in rows:
                positions+=row['positions'];clocks[row['turn']]=(positions,row['cumulative_learning_seconds'])
            for key,split in (('validation_history','validation'),('training_probe_history','training_probe')):
                for row in audit[key]:
                    turn=row['turn'];writer.writerow([label,record['peak_lr'],record['batch_games'],split,turn,
                        turn*record['batch_games']/128,*clocks[turn],*[row['metrics'][m] for m in METRICS]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--backup-sha256')
    p.add_argument('--run',action='store_true');a=p.parse_args()
    c=Campaign(a.registration,a.sha256)
    if not a.run:print(json.dumps(dict(status='prepared',accelerator_jobs_started=False)));return
    b=STUDY/'campaign-registration-backup.json'
    require(a.backup_sha256 and sha(b)==a.backup_sha256,'Campaign source backup changed')
    proof=read(b)
    require(proof['status']=='passed' and len(proof['copies'])==2
            and proof['files'][str(a.registration.resolve().relative_to(ROOT))]['sha256']==a.sha256,'Missing verified campaign source copies')
    with (STUDY/'.campaign.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:c.run()
        except BaseException as error:
            if not (STUDY/'result.json').exists():publish(STUDY/'result.json',dict(status='failed',error=repr(error),records=c.records,started=c.started,finished=time.time()))
            c.event('campaign_failed',error=repr(error));raise


if __name__=='__main__':main()
