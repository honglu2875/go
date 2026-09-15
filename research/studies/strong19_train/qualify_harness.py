"""Run full versus fresh-process split joint training, then audit all state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,read_json,verify


def sha(path):return checkpoints.sha256(path)


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def run_stage(python,snapshot,folder,*,stop=None,resume=None):
    manifest=verify(snapshot)
    argv=[str(python),'-B',str(snapshot/manifest['recipe']/'train.py'),'--config',str(snapshot/'resolved_config.json'),
          '--output',str(folder/'artifacts')]
    if stop is not None:argv+=['--stop-after-turn',str(stop)]
    if resume is not None:argv+=['--resume',str(resume)]
    folder.mkdir(parents=True,exist_ok=False)
    print(json.dumps(dict(stage=str(folder.relative_to(ROOT)),status='starting')),flush=True)
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=4',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',GOZERO_HOST_RANK='0')
    start=time.monotonic()
    with (folder/'process.log').open('xb') as log:
        child=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    receipt=dict(argv=argv,returncode=child.returncode,seconds=time.monotonic()-start,log_sha256=sha(folder/'process.log'))
    with (folder/'process.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    if child.returncode:raise ValueError('Joint harness subprocess failed: '+str(folder.relative_to(ROOT)))
    result=read_json(folder/'artifacts/result.json')
    if result['status']!='passed':raise ValueError('Harness result did not pass')
    print(json.dumps(dict(stage=str(folder.relative_to(ROOT)),status='passed',seconds=receipt['seconds'])),flush=True)
    return result


def audit_model(folder,snapshot,full,prefix,resumed):
    import numpy as np
    c=read_json(snapshot/'resolved_config.json')
    if (full['turn']!=4 or not full['training_complete'] or prefix['turn']!=2 or prefix['training_complete']
            or resumed['turn']!=4 or not resumed['training_complete']):raise ValueError('Wrong continuation boundary')
    def saved(result):
        cp=result['latest_checkpoint'];path=Path(cp['path'])
        if sha(path.with_suffix('.group.json'))!=cp['group_sha256']:raise ValueError('Group identity differs')
        group=read_json(path.with_suffix('.group.json'))
        if group['snapshot_id']!=snapshot.name or group['turn']!=result['turn']:raise ValueError('Group source/turn differs')
        state,arrays,_=checkpoints.read(path,expected_manifest_sha256=cp['manifest_sha256'])
        if state['config_sha256']!=hashlib.sha256(canonical_json(c)).hexdigest():raise ValueError('Wrong saved configuration')
        return state,arrays
    a,aa=saved(full);b,bb=saved(resumed);boundary,_=saved(prefix)
    if a!=b:raise ValueError('Complete rank state differs after continuation')
    if set(aa)!=set(bb):raise ValueError('Checkpoint array coverage differs')
    maximum=0.;elements=0
    for key in aa:
        np.testing.assert_array_equal(aa[key],bb[key],err_msg=key);elements+=aa[key].size
        if not np.isfinite(aa[key]).all():raise ValueError('Nonfinite continued array')
    if a['optimizer_metadata']['step']!=4 or int(aa['step'])!=4:raise ValueError('Optimizer step differs')
    f=rows(folder/'full/artifacts/metrics.jsonl')
    s=rows(folder/'prefix/artifacts/metrics.jsonl')+rows(folder/'resumed/artifacts/metrics.jsonl')
    ignore={'cumulative_learning_seconds','cumulative_sampling_seconds'}
    trim=lambda r:{k:v for k,v in r.items() if k not in ignore}
    if list(map(trim,f))!=list(map(trim,s)):raise ValueError('Draws, symmetries or subsequent metrics differ')
    for key in ('validation_history','training_probe_history','overfit_observations'):
        if full[key]!=resumed[key]:raise ValueError('Diagnostic history differs: '+key)
    if [x['turn'] for x in resumed['validation_history']]!=[0,2,4]:raise ValueError('Validation history was lost')
    if [x['turn'] for x in resumed['training_probe_history']]!=[0,2,4]:raise ValueError('Training-probe history was lost')
    if len(resumed['overfit_observations'])!=8:raise ValueError('Policy/value overfit checks missing')
    for row in resumed['validation_history']+resumed['training_probe_history']:
        if row['raw_totals']['expert_count']!=row['raw_totals']['value_count']:raise ValueError('Policy/value populations differ')
        if not all(np.isfinite(x) for x in row['metrics'].values()):raise ValueError('Nonfinite validation metric')
    # Replay draws from the frozen full corpus metadata independently of saved RNGs.
    from gozero.corpus_sequence_batches import Dataset
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);entries=data.bucket_entries(c['dataset']['buckets'])
    random=np.random.Generator(np.random.PCG64(c['seed']+1))
    bucket_random=np.random.Generator(np.random.PCG64(c['seed']+9143))
    augmentation=np.random.Generator(np.random.PCG64(c['seed']+400003))
    positions=0
    for i,row in enumerate(f):
        warm=c['dataset']['warmup_buckets']
        bucket=warm[i] if i<len(warm) else int(bucket_random.choice(c['dataset']['buckets'],p=c['dataset']['bucket_probabilities']))
        pool=entries['expert',bucket]
        chosen=[pool[int(j)] for j in random.integers(len(pool),size=c['learner']['games_per_host'])]
        symmetries=augmentation.integers(0,8,len(chosen)).tolist()
        count=sum(int(data.game_info(e)['length']) for e in chosen);positions+=count
        if row['bucket']!=bucket or row['local_entries_sha256']!=hashlib.sha256(canonical_json(chosen)).hexdigest() or row['local_symmetries']!=symmetries:
            raise ValueError('Independent game/D4 draws differ')
        if row['positions']!=count or not row['accepted']:raise ValueError('Wrong global training count/update')
    if a['numpy_rng']!=random.bit_generator.state or a['bucket_rng']!=bucket_random.bit_generator.state or a['augmentation_rng']!=augmentation.bit_generator.state:
        raise ValueError('Final sampler state differs from independent replay')
    if a['counters']['expert_positions']!=positions:raise ValueError('Position exposure counter differs')
    return dict(status='passed',checkpoint_arrays=len(aa),checkpoint_elements=elements,complete_rank_state_exact=True,
        all_update_metrics_exact_except_timings=True,diagnostic_histories_exact=True,independent_draws_verified=4,
        position_exposures=positions,final_validation=resumed['validation_history'][-1]['metrics'],
        checkpoint_manifest_sha256=full['latest_checkpoint']['manifest_sha256'],
        resumed_checkpoint_manifest_sha256=resumed['latest_checkpoint']['manifest_sha256'],
        snapshot=snapshot.name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Harness plan changed')
    plan=read_json(a.plan)
    if sha(Path(__file__))!=plan['operator_sha256']:raise ValueError('Harness operator changed')
    started=time.monotonic();report=dict(kind='joint_training_cpu_continuation_qualification',created=time.time(),
        plan_sha256=a.plan_sha256,models={},status='running')
    try:
        for name,identity in plan['snapshots'].items():
            snapshot=ROOT/'.gozero/snapshots'/identity;verify(snapshot)
            folder=ROOT/plan['run_directory']/name
            full=run_stage(a.python,snapshot,folder/'full')
            prefix=run_stage(a.python,snapshot,folder/'prefix',stop=2)
            resumed=run_stage(a.python,snapshot,folder/'resumed',resume=Path(prefix['latest_checkpoint']['path']))
            report['models'][name]=audit_model(folder,snapshot,full,prefix,resumed)
            print(json.dumps(dict(model=name,continuation='passed')),flush=True)
        left,right=report['models'].values()
        if left['position_exposures']!=right['position_exposures']:raise ValueError('Architectures consumed different positions')
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report.update(seconds=time.monotonic()-started,
            scope='Small complete joint models, four simulated devices in one CPU process, actual fixed-corpus whole-game sampling, joint validation and fresh-process 2+2 continuation. Not a full-size TPU or multi-process recovery result, optimizer selection, architecture learning experiment or Go-strength test.')
        with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        a.output.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
