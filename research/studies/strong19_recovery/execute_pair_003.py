"""Retry the frozen pair after repairing idle lockfile permissions.

Only output names differ from the qualified original pair controller.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from execute_run import ROOT, STUDY, publish, read, require, sha
from observation import observe, METRICS


def inspect(path,digest):
    require(sha(path)==digest,'Registration changed')
    r=read(path)
    require(r['kind']=='joint19_attention_pool_pair' and r['order']==['attention','flat']
            and r['steps']==256 and r['schedule_steps']==512,'Unexpected experiment')
    for field in ('operators','prerequisites'):
        for name,expected in r[field].items():
            require(sha(ROOT/name)==expected,'Changed '+field+': '+name)
    for name in r['prerequisites']:
        require(read(ROOT/name)['status'] in ('passed','prepared'),'Prerequisite did not pass')
    from gozero.snapshots import verify
    configs={}
    for arm,item in r['arms'].items():
        snapshot=ROOT/'.gozero/snapshots'/item['snapshot'];verify(snapshot)
        c=read(snapshot/'resolved_config.json');configs[arm]=c
        require(sha(snapshot/'resolved_config.json')==item['config_sha256']
            and c['steps']==512 and c['checkpoint_every']==64 and c['eval_every']==16
            and c['training']['optimizer']=='adamw' and c['training']['skip_padding']
            and c['training']['value_objective']=='signed_target_cross_entropy'
            and c['model']['board_pool_kind']==arm and not c['evaluation']['run_test'],'Configuration differs')
    for key in configs['flat']:
        if key!='model':require(configs['flat'][key]==configs['attention'][key],'Paired setting differs: '+key)
    trim=lambda m:{k:v for k,v in m.items() if not k.startswith('board_pool_')}
    require(trim(configs['flat']['model'])==trim(configs['attention']['model']),'Backbone differs')
    require(sha(ROOT/r['draw_reference'])==r['draw_sha256'],'Draw replay changed')
    return r


def compare_draws(attempt,reference,steps):
    for host in range(4):
        folder=ROOT/'runs'/attempt/f'rank-{host}/artifacts'
        rank=read(folder/'result.json')['jax_rank']
        rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        require(len(rows)==steps,'Wrong update count')
        for actual,expected in zip(rows,reference['draws'][:steps]):
            local=expected['ranks'][rank]
            for key in ('turn','bucket','positions'):
                require(actual[key]==expected[key],'Draw clock differs: '+key)
            for key in ('local_entries_sha256','local_symmetries'):
                require(actual[key]==local[key],'Sampler differs: '+key)


def update_ledger(folder,label):
    ledger=ROOT/'research/studies/runtime_qualification/reservation_ledger.json'
    script=ROOT/'.gozero/snapshots/ed15249055a11bac5b86806505d0d14ae75a4aee2ea327e86b6a422e49a7259c/ops/update_reservation_ledger.py'
    with (folder/(label+'-ledger.log')).open('xb') as log:
        subprocess.run([sys.executable,'-B',str(script),'--workspace-root',str(ROOT),
            '--expected-previous-sha256',sha(ledger)],stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)


def resource_observation(snapshot):
    from gozero.pod import load_hosts,SSH_OPTIONS
    hosts=load_hosts(snapshot/'ops/hosts.json')
    code="import os,json;from pathlib import Path;s=os.statvfs('/dev/shm');d=os.statvfs('.');m={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(dict(shm_free=s.f_bavail*s.f_frsize,disk_free=d.f_bavail*d.f_frsize,memory_available=m['MemAvailable'])))"
    def one(host):
        command='cd '+shlex.quote(str(ROOT))+' && taskset -c 0,1 python3 -c '+shlex.quote(code)
        value=json.loads(subprocess.check_output(['ssh',*SSH_OPTIONS,host.ssh,command],text=True,timeout=30))
        return dict(rank=host.rank,**value)
    with ThreadPoolExecutor(4) as pool:return list(pool.map(one,hosts))


def request_cancellation(snapshot,attempt,folder,reason):
    """Ask the existing token-aware remote supervisors to close this attempt."""
    from gozero.pod import load_hosts,SSH_OPTIONS
    hosts=load_hosts(snapshot/'ops/hosts.json')
    def one(host):
        argv=['python3',str(snapshot/'ops/cancel_host.py'),'--snapshot',str(snapshot),
              '--attempt',str(ROOT/'runs'/attempt)]
        result=subprocess.run(['ssh',*SSH_OPTIONS,host.ssh,shlex.join(argv)],capture_output=True,text=True,timeout=45)
        return dict(rank=host.rank,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,hosts))
    publish(folder/(attempt+'-cancellation.json'),dict(reason=reason,created=time.time(),ranks=rows))


def review(r,outcomes,audits,folder):
    base,candidate=audits['flat'],audits['attention']
    contrasts={}
    for metric in METRICS:
        b,c=base['validation_history'][-1]['metrics'][metric],candidate['validation_history'][-1]['metrics'][metric]
        bm=sum(x['metrics'][metric] for x in base['validation_history'][-3:])/3
        cm=sum(x['metrics'][metric] for x in candidate['validation_history'][-3:])/3
        contrasts[metric]=dict(flat=b,attention=c,relative_attention_gain=1-c/b,
            last_three_flat=bm,last_three_attention=cm,last_three_relative_gain=1-cm/bm)
    # Equal learning-time nearest completed validation points are explicitly
    # discrete; no interpolation or target-selected checkpoint is introduced.
    clocks={};equal_time={}
    for arm,outcome in outcomes.items():
        path=ROOT/'runs'/outcome['attempt']/'rank-0/artifacts/metrics.jsonl'
        logs=[json.loads(x) for x in path.read_text().splitlines()]
        clocks[arm]={0:0.,**{x['turn']:x['cumulative_learning_seconds'] for x in logs}}
    common_time=min(q['learning_seconds'] for q in outcomes.values())
    for arm,a in audits.items():
        rows=[x for x in a['validation_history'] if clocks[arm][x['turn']]<=common_time]
        point=rows[-1]
        equal_time[arm]=dict(turn=point['turn'],learning_seconds=clocks[arm][point['turn']],metrics=point['metrics'])
    for host in range(4):
        def logs(arm):
            p=ROOT/'runs'/outcomes[arm]['attempt']/f'rank-{host}/artifacts/metrics.jsonl'
            return [json.loads(x) for x in p.read_text().splitlines()]
        require(all(a['learning_rate']==b['learning_rate'] for a,b in zip(logs('flat'),logs('attention'))),'LR clocks differ')
    result=dict(status='passed',arms=outcomes,comparisons=contrasts,equal_learning_time=equal_time,
        common_learning_time=common_time,steps=r['steps'],schedule_steps=r['schedule_steps'],
        continuation_ready=True,complete_512_step_training=False,
        scope='One paired seed, fixed data and targets; fixed 256-update endpoint of a 512-update schedule. No playing-strength or final-512 claim.')
    publish(folder/'comparison.json',result)
    rows=['# Single-token attention pooling: matched 19x19 stage','',
        '| Arm | Parameters | Policy KL | Value MSE | Learning hours |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('flat','attention'):
        q=outcomes[arm];m=q['endpoint']
        rows.append(f"| {arm} | {q['parameters']:,} | {m['expert_kl']:.6f} | {m['value_mse']:.6f} | {q['learning_seconds']/3600:.3f} |")
    rows+=['',result['scope'],'',
        'All-rank state, optimizer, sampler, D4 draws and fixed evaluation populations were audited. Each endpoint has a hash-verified disk peer copy including every rank state.',
        '',f"Endpoint policy KL relative gain: {contrasts['expert_kl']['relative_attention_gain']:.2%}; last-three mean gain: {contrasts['expert_kl']['last_three_relative_gain']:.2%}.",
        '','Use a second seed and a longer continuation before treating a small difference as established. Review the train/validation curves and all overfit flags before extending.']
    (STUDY/'RESULTS-003.md').write_text('\n'.join(rows)+'\n')
    # A portable numeric curve artifact needs no plotting dependency.
    with (STUDY/'curves-003.csv').open('x') as f:
        f.write('arm,split,turn,learning_seconds,expert_kl,family_kl,value_mse,value_family_mse\n')
        for arm,a in audits.items():
            for field,split in (('validation_history','validation'),('training_probe_history','train_probe')):
                for row in a[field]:
                    f.write(','.join(map(str,[arm,split,row['turn'],clocks[arm][row['turn']],*[row['metrics'][k] for k in METRICS]]))+'\n')
    return result


def run(path,digest):
    r=inspect(path,digest);folder=STUDY/'sequence-003';folder.mkdir(exist_ok=False)
    started=time.time();outcomes={};audits={};reference=read(ROOT/r['draw_reference'])
    def event(kind,**values):
        row=dict(kind=kind,time=time.time(),**values)
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    try:
        for stage in r['order']:
            inspect(path,digest);item=r['arms'][stage];steps=r['steps']
            prereqs={**r['prerequisites'],str(path.relative_to(ROOT)):digest}
            for previous in outcomes:
                p=folder/(previous+'-review.json');prereqs[str(p.relative_to(ROOT))]=sha(p)
            name=stage+'-stage-003'
            plan=dict(kind='registered_joint19_run',created=time.time(),snapshot=item['snapshot'],
                config_sha256=item['config_sha256'],purpose='learning',steps=steps,schedule_steps=r['schedule_steps'],
                checkpoint_every=64,parameters=item['parameters'],timeout_seconds=24000,
                replica_peer=2,checkpoint_reserve_bytes=0,shm_floor_bytes=65*(1<<30),
                memory_floor_bytes=96*(1<<30),disk_floor_bytes=3*(1<<30),producer_growth_reserve_bytes=0,
                disk_checkpoint_reserve_by_host={str(i):(3 if i==0 else 4 if i==2 else 0)*r['checkpoint_bytes_per_arm'][stage]+(5*(1<<30) if i==2 else 0) for i in range(4)},
                additional_reserve_by_host={str(i):r['packed_array_bytes'] if i else 0 for i in range(4)},
                output_directory=name,audit_python=r['audit_python'],
                operators={k:v for k,v in r['operators'].items() if Path(k).name in ('execute_run.py','audit_run.py','replicate.py')},
                prerequisites=prereqs,expected_positions=sum(x['positions'] for x in reference['draws'][:steps]),
                validation_positions=r['validation_positions'],scope='Fixed connector-only pair on newly collected data; disk full-state checkpoint and complete disk peer copy every 64 updates, bounded local retention.')
            plan_path=STUDY/(stage+'-plan-003.json');publish(plan_path,plan)
            event('stage_started',stage=stage,snapshot=item['snapshot'],plan_sha256=sha(plan_path))
            command=[sys.executable,'-B',STUDY/'execute_run.py','--plan',plan_path,'--plan-sha256',sha(plan_path),'--run']
            attempt=None;last_observed=-1;consecutive_errors=0
            last_progress=time.monotonic();last_resources=0.;cancelled=False;resource_errors=0
            with (folder/(stage+'-controller.log')).open('xb') as log:
                child=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    controller=STUDY/name/'controller.log'
                    if attempt is None and controller.exists():
                        for line in controller.read_text().splitlines():
                            if line.startswith('{'):
                                value=json.loads(line)
                                if value.get('kind')=='pod_attempt':attempt=value['attempt_id'];break
                    if attempt:
                        try:
                            value=observe(ROOT,attempt,item['snapshot'],validation_positions=r['validation_positions'])
                            tmp=folder/(stage+'-current.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(folder/(stage+'-current.json'))
                            consecutive_errors=0
                            if value['completed_updates']!=last_observed:
                                last_observed=value['completed_updates']
                                last_progress=time.monotonic()
                                event('progress',arm=stage,attempt=attempt,updates=last_observed,
                                    positions=value['position_exposures'],learning_seconds=value['rank0_learning_seconds'],
                                    overfit=value['provisional_overfit_flags'])
                        except Exception as error:
                            consecutive_errors+=1
                            event('monitor_error',stage=stage,error=repr(error),consecutive=consecutive_errors)
                            # A bounded pod controller owns cleanup. Do not kill
                            # that controller and strand remote processes. A
                            # final authoritative audit must succeed before the
                            # next stage even after a transient observation error.
                        if not cancelled and not (ROOT/'runs'/attempt/'result.json').exists():
                            reason=None
                            if consecutive_errors>=5:reason='Five consecutive live observation failures'
                            if time.monotonic()-last_progress>1800:reason='No accepted-update progress for 30 minutes'
                            if time.monotonic()-last_resources>=600:
                                try:
                                    resources=resource_observation(ROOT/'.gozero/snapshots'/item['snapshot'])
                                    event('resources',stage=stage,hosts=resources);last_resources=time.monotonic();resource_errors=0
                                    if any(x['disk_free']<3*(1<<30) or x['shm_free']<65*(1<<30)
                                           or x['memory_available']<24*(1<<30) for x in resources):
                                        reason='Live storage or memory reserve exhausted'
                                except Exception as error:
                                    resource_errors+=1;last_resources=time.monotonic()-570
                                    event('resource_monitor_error',stage=stage,error=repr(error),consecutive=resource_errors)
                                    if resource_errors>=3:reason='Three consecutive resource observation failures'
                            if reason:
                                event('cancellation_requested',stage=stage,attempt=attempt,reason=reason)
                                request_cancellation(ROOT/'.gozero/snapshots'/item['snapshot'],attempt,folder,reason)
                                cancelled=True
                    time.sleep(30)
                require(child.returncode==0 and not cancelled,'Stage failed or monitoring requested cancellation: '+stage)
            output=STUDY/name;result=read(output/'result.json');audit=read(output/'audit.json')
            require(result['status']=='passed' and result['plan_sha256']==sha(plan_path)
                and sha(output/'audit.json')==result['audit_sha256'],'Stage closure/audit differs')
            attempt=result['attempt'];report=read(ROOT/'runs'/attempt/'rank-0/artifacts/result.json')
            final_observation=observe(ROOT,attempt,item['snapshot'],validation_positions=r['validation_positions'])
            require(final_observation['closure']=='passed' and final_observation['completed_updates']==steps,'Final observation incomplete')
            compare_draws(attempt,reference,steps)
            for key,expected in [('validation_history',r['validation_population_sha256']),
                                 ('training_probe_history',r['probe_population_sha256'])]:
                require(all(x['episode_ids_sha256']==expected for x in audit[key]),'Evaluation population differs')
            audits[stage]=audit
            item_review=dict(status='passed',attempt=attempt,steps=steps,parameters=audit['parameters'],positions=audit['positions'],
                result_sha256=sha(output/'result.json'),audit_sha256=result['audit_sha256'],replica_sha256=result['replica_sha256'],
                endpoint={k:audit['validation_history'][-1]['metrics'][k] for k in METRICS},
                probe_endpoint={k:audit['training_probe_history'][-1]['metrics'][k] for k in METRICS},
                learning_seconds=report['segment_timing']['learning_seconds'],
                sustained_overfit_flags=sum(x['sustained'] for x in audit['overfit_observations']))
            publish(folder/(stage+'-review.json'),item_review);outcomes[stage]=item_review
            update_ledger(folder,stage)
            event('stage_passed',stage=stage,**item_review)
        comparison=review(r,outcomes,audits,folder)
        publish(folder/'result.json',dict(status='passed',registration_sha256=digest,started=started,finished=time.time(),
            comparison_sha256=sha(folder/'comparison.json'),arms=outcomes))
        event('pair_passed',comparisons=comparison['comparisons'])
    except BaseException as error:
        if not (folder/'result.json').exists():
            publish(folder/'result.json',dict(status='failed',registration_sha256=digest,started=started,
                finished=time.time(),arms=outcomes,error=repr(error)))
        event('pair_failed',error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:
        inspect(a.registration,a.registration_sha256);print(json.dumps(dict(status='prepared',accelerator_jobs_started=False)));return
    with (STUDY/'.pair.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);run(a.registration,a.registration_sha256)


if __name__=='__main__':main()
