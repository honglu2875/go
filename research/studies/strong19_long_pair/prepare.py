"""Freeze the paired long horizon, common objective and independent draw replay."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json,freeze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path,value):
    if path.exists():
        if path.read_bytes()!=canonical_json(value):raise ValueError('Existing preparation differs: '+str(path))
        return
    with path.open('xb') as f:f.write(canonical_json(value))
    path.chmod(0o444)


def main():
    import numpy as np
    old=ROOT/'research/studies/strong19_scaling'
    cohort=json.loads((old/'cohort-plan-002.json').read_text())
    assert sha(old/'cohort-plan-002.json')=='9270f03a2080e19e3036d30d3d184dd6d56fdfeb4309824f87c64046350bc5f6'
    parents=dict(cnn='779336b7884695f35cfff4fa578b3779ade63515755cb873f0527d798c25b0f8',
        transformer='5eba4db502e1d3b39e1c9385db0f4ec6b2e7a85ea1b290a6eba2177b71972cc6')
    configs={}
    for arm,parent in parents.items():
        c=json.loads((ROOT/'.gozero/snapshots'/parent/'resolved_config.json').read_text())
        c.update(steps=512,checkpoint_every=256,eval_every=16,log_every=1)
        c['training']['value_objective']='signed_target_cross_entropy'
        assert c['training']['optimizer']=='adamw' and c['learner']['warmup_steps']==40
        configs[arm]=c
    for key in ('seed','dataset','evaluation','learner','value_model','steps','checkpoint_every','eval_every'):
        assert configs['cnn'][key]==configs['transformer'][key],key
    entries={b:[] for b in cohort['buckets']};info={}
    for index,row in enumerate(cohort['records']):
        key=('expert',index//32,index%32);info[key]=row
        if row['split']=='train':entries[512 if row['rows']<=512 else 768].append(key)
    c=configs['cnn'];seed=c['seed'];samplers=[np.random.Generator(np.random.PCG64(seed+1+104729*r)) for r in range(4)]
    augment=[np.random.Generator(np.random.PCG64(seed+400003+104729*r)) for r in range(4)]
    buckets=np.random.Generator(np.random.PCG64(seed+9143));draws=[];positions=slots=0
    for turn in range(1,513):
        bucket=cohort['buckets'][turn-1] if turn<=2 else int(buckets.choice(cohort['buckets'],p=c['dataset']['bucket_probabilities']))
        ranks=[];count=0
        for rank,(rng,aug) in enumerate(zip(samplers,augment)):
            chosen=[entries[bucket][int(i)] for i in rng.integers(len(entries[bucket]),size=32)]
            syms=aug.integers(0,8,32).tolist();count+=sum(info[e]['rows'] for e in chosen)
            ranks.append(dict(jax_rank=rank,local_entries_sha256=hashlib.sha256(canonical_json(chosen)).hexdigest(),local_symmetries=syms))
        positions+=count;slots+=128*bucket
        draws.append(dict(turn=turn,bucket=bucket,positions=count,cumulative_positions=positions,ranks=ranks))
    old_draws=json.loads((old/'pilot-draw-replay-001.json').read_text())
    assert draws[:108]==old_draws['draws']
    replay=dict(kind='paired_joint19_long_draw_replay',status='prepared',seed=seed,draws=draws,total_positions=positions,
        padded_position_slots=slots,training_population_equivalent_passes=positions/cohort['populations']['train']['positions'],
        cohort_plan_sha256=sha(old/'cohort-plan-002.json'),first_108_match_pilot=True,model_or_targets_read=False)
    publish(STUDY/'draw-replay-001.json',replay)
    recipe=Path('research/recipes/strong19_long_ce');sys.path.insert(0,str(ROOT/recipe));import train_config
    snapshots={}
    for arm,c in configs.items():
        train_config.validate(c);path=STUDY/(arm+'-config-001.json');publish(path,c)
        snapshot=freeze(ROOT,recipe,path,ROOT/'.gozero/snapshots')
        snapshots[arm]=dict(snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),parent=parents[arm],
            parameters=233220870 if arm=='cnn' else 232011540)
    qc=json.loads(json.dumps(configs['cnn']));qc.update(steps=4,checkpoint_every=4,eval_every=2)
    qc['learner']['warmup_steps']=2;qc['training']['purpose']='qualification'
    qc['evaluation'].update(games_per_bucket=2,training_probe_games=4)
    train_config.validate(qc);path=STUDY/'cnn-qualification-config-001.json';publish(path,qc)
    qs=freeze(ROOT,recipe,path,ROOT/'.gozero/snapshots')
    result=dict(kind='common_adamw_joint19_long_preparation',status='prepared',created=time.time(),arms=snapshots,
        qualification=dict(snapshot=qs.name,config_sha256=sha(qs/'resolved_config.json'),parameters=233220870),
        recipe=str(recipe),steps=512,checkpoint_every=256,eval_every=16,draw_replay_sha256=sha(STUDY/'draw-replay-001.json'),
        position_exposures=positions,training_population_equivalent_passes=replay['training_population_equivalent_passes'],
        changes_from_pilot=['Common CE value objective','512-update cosine horizon from scratch','Evaluation every 16 updates; checkpoints at 256 and 512'],
        unchanged=['Model shapes, parameters and deployment FLOPs','AdamW peak/end LR 0.001/0.0003 and 40-update warmup',
                   'Seed, frozen corpus, full histories, global batch 128 games and D4 draws','Validation population and fixed 128-game training probe'],
        test_targets_read=False,operator_sha256=sha(Path(__file__)))
    publish(STUDY/'preparation-001.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
