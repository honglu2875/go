"""Register the unchanged matched architecture pair on the replacement corpus."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).parent
RECIPE=ROOT/'research/recipes/strong19_recovery'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
from gozero.corpus_sequence_batches import Dataset
from gozero.durable_files import atomic_json,sha256
from gozero.snapshots import canonical_json,freeze,verify


def main():
    import numpy as np
    from train_config import validate
    import joint
    import training_probe
    old=ROOT/'research/studies/strong19_attention_pool'
    cpu=json.loads((STUDY/'harness-qualification-001.json').read_text())
    pod=json.loads((STUDY/'pod-recovery-qualification-002.json').read_text())
    assert cpu['status']==pod['status']=='passed'
    assert pod['disk_peer_restore'] and pod['all_rank_states_exact']
    qualified=ROOT/'.gozero/snapshots'/pod['snapshot']
    verify(qualified)
    for p in RECIPE.glob('*.py'):
        assert sha256(p)==sha256(qualified/p.relative_to(ROOT)),p.name
    for name in ('checkpoint_disk.py','disk_mirror.py','durable_files.py'):
        p=ROOT/'packages/gozero/src/gozero'/name
        assert sha256(p)==sha256(qualified/p.relative_to(ROOT)),name
    # Carry forward the already qualified model/cached-decoding mathematics.
    for name in ('cpu-qualification-002.json','decode-qualification-001.json'):
        q=json.loads((old/name).read_text());assert q['status']=='passed'
        for relative,digest in q['source_sha256'].items():
            # These two transport/validation files have their own new exact
            # CPU and real-pod recovery qualifications above. Mathematical
            # model, objective, optimizer and decoding files must be unchanged.
            if relative not in ('train_config.py','train_joint.py'):
                assert sha256(RECIPE/relative)==digest,relative
    budget=json.loads((old/'budget-001.json').read_text());assert budget['status']=='passed'
    packed=json.loads((STUDY/'cohort-result-001.json').read_text())
    copies=json.loads((STUDY/'cohort-disk-backups-001.json').read_text())
    assert packed['status']==copies['status']=='passed' and copies['manifest_sha256']==packed['manifest_sha256']
    data=Dataset(packed['target'],packed['manifest_sha256']);buckets=[512,768]
    assert data.size==19 and not data.manifest['qualification_only']
    train=data.bucket_entries(buckets);val=data.bucket_entries(buckets,split=1)
    assert all(train['expert',b] and val['expert',b] for b in buckets)
    count=sum(len(v) for v in train.values());prob=[len(train['expert',b])/count for b in buckets]
    configs={}
    for arm in ('flat','attention'):
        c=json.loads((old/f'{arm}-config-001.json').read_text())
        for key in ('checkpoint_temporary','checkpoint_temporary_uncompressed','checkpoint_minimum_free_bytes'):c.pop(key,None)
        c['checkpoint_every']=64
        c['checkpoint_disk']=dict(peer=2,root=str(ROOT/'.gozero/checkpoint-disk-replicas'),keep=2,floor_bytes=2<<30,peer_floor_bytes=8<<30)
        c['dataset']=dict(path=packed['target'],manifest_sha256=packed['manifest_sha256'],
            buckets=buckets,bucket_probabilities=prob,warmup_buckets=buckets)
        validate(c);configs[arm]=c
        assert sum(p['elements'] for p in joint.parameter_schema(c['model'],c['value_model']))==budget['arms'][arm]['parameters']
    c=configs['flat'];seed=c['seed'];samplers=[np.random.Generator(np.random.PCG64(seed+1+104729*r)) for r in range(4)]
    augmentation=[np.random.Generator(np.random.PCG64(seed+400003+104729*r)) for r in range(4)]
    chooser=np.random.Generator(np.random.PCG64(seed+9143));draws=[];positions=slots=0
    for turn in range(1,513):
        bucket=buckets[turn-1] if turn<=2 else int(chooser.choice(buckets,p=prob))
        ranks=[];n=0
        for rank,(rng,aug) in enumerate(zip(samplers,augmentation)):
            pool=train['expert',bucket];chosen=[pool[int(i)] for i in rng.integers(len(pool),size=32)]
            syms=aug.integers(0,8,32).tolist();n+=sum(int(data.game_info(e)['length']) for e in chosen)
            ranks.append(dict(jax_rank=rank,local_entries_sha256=hashlib.sha256(canonical_json(chosen)).hexdigest(),local_symmetries=syms))
        positions+=n;slots+=128*bucket
        draws.append(dict(turn=turn,bucket=bucket,positions=n,cumulative_positions=positions,ranks=ranks))
    replay=dict(status='prepared',kind='replacement_joint19_draw_replay',seed=seed,draws=draws,total_positions=positions,
        dataset_manifest_sha256=packed['manifest_sha256'],padded_position_slots=slots,
        model_or_targets_read=False,old_lost_dataset_reused=False)
    atomic_json(STUDY/'draw-replay-001.json',replay,replace=False)
    selected_val=[e for b in buckets for e in val['expert',b][:c['evaluation']['games_per_bucket']]]
    probe=training_probe.select({b:train['expert',b] for b in buckets},c['evaluation']['training_probe_games'])
    selected_probe=[e for b in buckets for e in probe[b]]
    snapshots={}
    for arm,c in configs.items():
        cfg=STUDY/f'{arm}-config-001.json';atomic_json(cfg,c,replace=False)
        snapshot=freeze(ROOT,RECIPE,cfg,ROOT/'.gozero/snapshots');verify(snapshot)
        snapshots[arm]=dict(snapshot=snapshot.name,config_sha256=sha256(snapshot/'resolved_config.json'),parameters=budget['arms'][arm]['parameters'])
    manifests=[json.loads((ROOT/'.gozero/snapshots'/s['snapshot']/'manifest.json').read_text())['files'] for s in snapshots.values()]
    assert {k:v for k,v in manifests[0].items() if k!='resolved_config.json'}=={k:v for k,v in manifests[1].items() if k!='resolved_config.json'}
    operators={str((STUDY/n).relative_to(ROOT)):sha256(STUDY/n) for n in
        ('prepare_pair.py','execute_pair.py','execute_run.py','audit_run.py','replicate.py','observation.py','PROTOCOL.md')}
    paths=[STUDY/n for n in ('harness-qualification-001.json','pod-recovery-qualification-002.json',
        'pod-stage-audit-001.json','pod-peer-verification-001.json','cohort-result-001.json','cohort-disk-backups-001.json')]
    paths += [old/n for n in ('cpu-qualification-002.json','decode-qualification-001.json','budget-001.json')]
    r=dict(status='prepared',kind='joint19_attention_pool_pair',created=time.time(),arms=snapshots,
        steps=256,schedule_steps=512,order=['attention','flat'],operators=operators,
        prerequisites={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        draw_reference=str((STUDY/'draw-replay-001.json').relative_to(ROOT)),draw_sha256=sha256(STUDY/'draw-replay-001.json'),
        validation_population_sha256=hashlib.sha256(canonical_json(selected_val)).hexdigest(),
        probe_population_sha256=hashlib.sha256(canonical_json(selected_probe)).hexdigest(),
        validation_positions=sum(int(data.game_info(e)['length']) for e in selected_val),
        audit_python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),
        paired_positions=sum(d['positions'] for d in draws[:256]),packed_array_bytes=packed['packed_array_bytes'],
        checkpoint_bytes_per_arm={k:12*v['parameters']+4 for k,v in snapshots.items()},
        primary='Fixed endpoint policy KL with family KL and last-three support; complete validation/probe curves retained.',
        horizon_policy='256 updates each of the unchanged 512-step AdamW schedule, new paired data draws; no adaptive extra queue.',
        recovery_change='New dataset identity, disk checkpoints every 64 updates, verified all-rank disk peer copies. Historical lost-corpus results remain separate.')
    atomic_json(STUDY/'registration-001.json',r,replace=False)
    print(json.dumps(dict(status='prepared',arms=snapshots,paired_positions=r['paired_positions'],validation_positions=r['validation_positions'])),flush=True)


if __name__=='__main__':main()
