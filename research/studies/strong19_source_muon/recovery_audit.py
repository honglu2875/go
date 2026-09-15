"""Clone of the qualified joint recovery audit for registered source-runtime horizons."""
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
        child=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
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
    horizon=c['steps']
    if (full['turn']!=horizon or not full['training_complete'] or prefix['turn']!=6 or prefix['training_complete']
            or resumed['turn']!=horizon or not resumed['training_complete']):raise ValueError('Wrong continuation boundary')
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
    if a['optimizer_metadata']['step']!=horizon or int(aa['step'])!=horizon:raise ValueError('Optimizer step differs')
    f=rows(folder/'full/artifacts/metrics.jsonl')
    s=rows(folder/'prefix/artifacts/metrics.jsonl')+rows(folder/'resumed/artifacts/metrics.jsonl')
    ignore={'cumulative_learning_seconds','cumulative_sampling_seconds'}
    trim=lambda r:{k:v for k,v in r.items() if k not in ignore}
    if list(map(trim,f))!=list(map(trim,s)):raise ValueError('Draws, symmetries or subsequent metrics differ')
    for key in ('validation_history','training_probe_history','overfit_observations'):
        if full[key]!=resumed[key]:raise ValueError('Diagnostic history differs: '+key)
    epoch_ends=[];count=0
    for epoch in c['learner']['source_runtime']['epochs']:
        count+=sum(epoch);epoch_ends.append(count)
    evaluation_turns=sorted(set([0,horizon,*range(c['eval_every'],horizon+1,c['eval_every']),*epoch_ends]))
    if [x['turn'] for x in resumed['validation_history']]!=evaluation_turns:raise ValueError('Validation history was lost')
    if [x['turn'] for x in resumed['training_probe_history']]!=evaluation_turns:raise ValueError('Training-probe history was lost')
    if len(resumed['overfit_observations'])!=4*(len(evaluation_turns)-1):raise ValueError('Policy/value overfit checks missing')
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
        all_update_metrics_exact_except_timings=True,diagnostic_histories_exact=True,independent_draws_verified=horizon,
        position_exposures=positions,final_validation=resumed['validation_history'][-1]['metrics'],
        checkpoint_manifest_sha256=full['latest_checkpoint']['manifest_sha256'],
        resumed_checkpoint_manifest_sha256=resumed['latest_checkpoint']['manifest_sha256'],
        snapshot=snapshot.name)

