"""Own qualification, both long arms, live diagnostics and the final comparison."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from execute_run import ROOT, STUDY, publish, read, require, sha
from observation import observe, METRICS


def inspect(path,digest):
    require(sha(path)==digest,'Registration changed')
    r=read(path)
    require(r['kind']=='paired_long_joint19_adamw_ce' and r['order']==['cnn','transformer'],'Unexpected scope')
    for field in ('operators','prerequisites'):
        for name,expected in r[field].items():
            require(sha(ROOT/name)==expected,'Changed '+field+': '+name)
    for name in r['prerequisites']:
        require(read(ROOT/name)['status'] in ('passed','prepared'),'Prerequisite did not pass')
    configs={}
    from gozero.snapshots import verify
    for arm,item in r['arms'].items():
        snapshot=ROOT/'.gozero/snapshots'/item['snapshot'];verify(snapshot)
        c=read(snapshot/'resolved_config.json');configs[arm]=c
        require(sha(snapshot/'resolved_config.json')==item['config_sha256'] and c['steps']==512
            and c['checkpoint_every']==256 and c['eval_every']==16
            and c['training']['optimizer']=='adamw'
            and c['training']['value_objective']=='signed_target_cross_entropy'
            and not c['evaluation']['run_test'],'Scientific configuration differs')
    for key in ('seed','dataset','learner','evaluation','value_model'):
        require(configs['cnn'][key]==configs['transformer'][key],'Paired input differs: '+key)
    require(sha(ROOT/r['draw_reference'])==r['draw_sha256'],'Predeclared draw replay differs')
    return r


def compare_draws(attempt,reference):
    for host in range(4):
        folder=ROOT/'runs'/attempt/f'rank-{host}/artifacts'
        rank=read(folder/'result.json')['jax_rank']
        rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        require(len(rows)==len(reference['draws'])==512,'Wrong horizon')
        for actual,expected in zip(rows,reference['draws']):
            local=expected['ranks'][rank]
            for key in ('turn','bucket','positions'):
                require(actual[key]==expected[key],'Draw clock differs: '+key)
            for key in ('local_entries_sha256','local_symmetries'):
                require(actual[key]==local[key],'Rank sampler differs: '+key)


def run(path,digest):
    r=inspect(path,digest);folder=STUDY/'sequence-002';folder.mkdir(exist_ok=False)
    started=time.time();outcomes={};audits={};reference=read(ROOT/r['draw_reference'])
    def event(kind,**values):
        row=dict(kind=kind,time=time.time(),**values)
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    try:
        for stage in ('qualification',*r['order']):
            inspect(path,digest)
            qualification=stage=='qualification';item=r['qualification'] if qualification else r['arms'][stage]
            steps=4 if qualification else 512;checkpoint_every=4 if qualification else 256
            prereqs={**r['prerequisites'],str(path.relative_to(ROOT)):digest}
            for previous in outcomes:
                p=folder/(previous+'-review.json');prereqs[str(p.relative_to(ROOT))]=sha(p)
            name='cnn-qualification-002' if qualification else stage+'-long-002'
            plan=dict(kind='registered_joint19_run',created=time.time(),snapshot=item['snapshot'],
                config_sha256=item['config_sha256'],purpose='qualification' if qualification else 'learning',
                steps=steps,checkpoint_every=checkpoint_every,parameters=item['parameters'],
                timeout_seconds=2400 if qualification else 60000,
                replica_peer=2 if qualification else 3,checkpoint_reserve_bytes=3*(1<<30),
                shm_floor_bytes=64*(1<<30),memory_floor_bytes=96*(1<<30),disk_floor_bytes=1<<30,
                producer_growth_reserve_bytes=2*(1<<30),
                additional_reserve_by_host={str(i):3*(1<<30) if i==0 and not qualification else 0 for i in range(4)},
                output_directory=name,audit_python=r['audit_python'],
                operators={k:v for k,v in r['operators'].items() if k in (
                    'research/studies/strong19_long_pair/execute_run.py','research/studies/strong19_long_pair/audit_run.py',
                    'research/studies/strong19_source_muon/replicate_full_size.py')},prerequisites=prereqs,
                scope='Common-objective AdamW architecture comparison. Two RAM checkpoints explicitly reserved; no test targets.')
            if not qualification:plan.update(expected_positions=reference['total_positions'],validation_positions=60284)
            plan_path=STUDY/(name.replace('-002','-plan-002')+'.json');publish(plan_path,plan)
            event('stage_started',stage=stage,snapshot=item['snapshot'],plan_sha256=sha(plan_path))
            command=[sys.executable,'-B',STUDY/'execute_run.py','--plan',plan_path,'--plan-sha256',sha(plan_path),'--run']
            attempt=None;last_observed=-1;monitor_errors=[]
            with (folder/(stage+'-controller.log')).open('xb') as log:
                child=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    controller=STUDY/name/'controller.log'
                    if attempt is None and controller.exists():
                        for line in controller.read_text().splitlines():
                            if line.startswith('{'):
                                value=json.loads(line)
                                if value.get('kind')=='pod_attempt':attempt=value['attempt_id'];break
                    if attempt and not qualification:
                        try:
                            value=observe(ROOT,attempt,item['snapshot'],validation_positions=60284)
                            tmp=folder/(stage+'-current.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(folder/(stage+'-current.json'))
                            if value['completed_updates']!=last_observed:
                                last_observed=value['completed_updates']
                                event('progress',arm=stage,attempt=attempt,updates=last_observed,
                                    positions=value['position_exposures'],learning_seconds=value['rank0_learning_seconds'],
                                    overfit=value['provisional_overfit_flags'])
                        except Exception as error:
                            monitor_errors.append(repr(error));event('monitor_error',stage=stage,error=repr(error))
                    time.sleep(30)
                require(child.returncode==0,'Stage failed: '+stage)
            require(not monitor_errors,'Live observation failed; review before starting another arm')
            output=STUDY/name;result=read(output/'result.json');audit=read(output/'audit.json')
            require(result['status']=='passed' and result['plan_sha256']==sha(plan_path)
                and sha(output/'audit.json')==result['audit_sha256'],'Stage closure/audit differs')
            attempt=result['attempt'];report=read(ROOT/'runs'/attempt/'rank-0/artifacts/result.json')
            if not qualification:
                compare_draws(attempt,reference)
                for key,expected in [('validation_history',r['validation_population_sha256']),
                                     ('training_probe_history',r['probe_population_sha256'])]:
                    require(all(x['episode_ids_sha256']==expected for x in audit[key]),'Population differs')
                parent=read(ROOT/r['initialization_controls'][stage])
                require(audit['initial_parameters_sha256']==parent['initial_parameters_sha256'],'Initialization changed')
                audits[stage]=audit
            review=dict(status='passed',attempt=attempt,steps=steps,parameters=audit['parameters'],positions=audit['positions'],
                result_sha256=sha(output/'result.json'),audit_sha256=result['audit_sha256'],replica_sha256=result['replica_sha256'],
                endpoint={k:audit['validation_history'][-1]['metrics'][k] for k in METRICS},
                probe_endpoint={k:audit['training_probe_history'][-1]['metrics'][k] for k in METRICS},
                learning_seconds=report['segment_timing']['learning_seconds'],
                sustained_overfit_flags=sum(x['sustained'] for x in audit['overfit_observations']))
            publish(folder/(stage+'-review.json'),review);outcomes[stage]=review
            event('stage_passed',stage=stage,**review)
        baseline=outcomes['cnn']['endpoint'];candidate=outcomes['transformer']['endpoint']
        comparisons={k:dict(cnn=baseline[k],transformer=candidate[k],relative_transformer_gain=1-candidate[k]/baseline[k]) for k in METRICS}
        for host in range(4):
            def logs(arm):
                p=ROOT/'runs'/outcomes[arm]['attempt']/f'rank-{host}/artifacts/metrics.jsonl'
                return [json.loads(x) for x in p.read_text().splitlines()]
            for a,b in zip(logs('cnn'),logs('transformer')):
                require(a['learning_rate']==b['learning_rate'],'Paired LR clock differs')
        result=dict(kind=r['kind'],status='passed',registration_sha256=digest,started=started,finished=time.time(),
            arms=outcomes,comparisons=comparisons,
            scope='Single paired seed on fixed data with a common AdamW/CE recipe. Endpoint and complete curves retained; no playing-strength claim.')
        publish(folder/'result.json',result)
        lines=['# Completed longer 19×19 comparison','',
            '| Arm | Policy KL | Family KL | Value MSE | Learning minutes |',
            '| --- | ---: | ---: | ---: | ---: |']
        for arm in r['order']:
            q=outcomes[arm];m=q['endpoint'];lines.append(f"| {arm} | {m['expert_kl']:.6f} | {m['family_kl']:.6f} | {m['value_mse']:.6f} | {q['learning_seconds']/60:.2f} |")
        lines+=['',result['scope'],'','Both complete-state audits, registered draws, fixed evaluation populations, initialization controls and final peer copies passed.']
        (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        event('pair_passed',comparisons=comparisons)
    except BaseException as error:
        result=dict(kind=r['kind'],status='failed',registration_sha256=digest,started=started,finished=time.time(),arms=outcomes,error=repr(error))
        if not (folder/'result.json').exists():publish(folder/'result.json',result)
        event('pair_failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--registration-sha256',required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:
        inspect(args.registration,args.registration_sha256);print(json.dumps(dict(status='prepared',accelerator_jobs_started=False)));return
    with (STUDY/'.pair.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.registration,args.registration_sha256)


if __name__=='__main__':main()
