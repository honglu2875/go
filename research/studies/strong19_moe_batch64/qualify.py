"""Qualify the composition of tested MoE numerics and tested game rebatching."""
import ast, copy, json, sys, time
from execute_run import ROOT, STUDY, read, sha, require, publish
RECIPE=ROOT/'research/recipes/strong19_moe_batch64'
sys.path.insert(0,str(RECIPE))


def main():
    import jax
    import joint, train_config, batch_replay
    from batch_audit import expected
    from gozero.corpus_sequence_batches import Dataset
    from gozero.snapshots import canonical_json
    require(all(d.platform=='cpu' for d in jax.devices()),'CPU qualification only')
    original=ROOT/'research/studies/strong19_dense_tuning_keepall'
    batch_proof=read(original/'batch-cpu-qualification-002.json')
    model_proof=read(ROOT/'research/studies/strong19_moe_temporal/cpu-qualification-002.json')
    require(batch_proof['status']==model_proof['status']=='passed','Inherited qualification failed')
    for name,digest in model_proof['source_sha256'].items():require(sha(ROOT/name)==digest,'MoE evidence source changed')
    lineage={}
    for p in RECIPE.glob('*.py'):
        ast.parse(p.read_text())
        if p.name=='qualify_batches.py':continue
        parent=ROOT/'research/recipes/strong19_dense_batch_keepall'/p.name
        require(sha(p)==sha(parent)==batch_proof['source_files'][str(parent.relative_to(ROOT))],'Batch recipe source changed: '+p.name)
        temporal=ROOT/'research/recipes/strong19_moe_temporal'/p.name
        if p.name not in ('batch_replay.py','train.py','train_config.py','train_joint.py','qualify_system.py'):
            require(sha(p)==sha(temporal),'Numerical source differs from tested temporal MoE: '+p.name)
        lineage[str(p.relative_to(ROOT))]=sha(p)
    configs={'dense':read(RECIPE/'dense.json'),**{a:read(STUDY/(a+'-config.json')) for a in ('temporal','balance_low')}}
    require(configs['dense']==read(original/'batch64-config.json'),'Dense reference config differs')
    low=copy.deepcopy(configs['balance_low']);low['model']['moe']['balance_weight']*=3
    require(low==configs['temporal'],'Low balance changes something else')
    budgets={}
    for arm,c in configs.items():
        train_config.validate(c)
        no_moe=copy.deepcopy(c);no_moe['model'].pop('moe',None)
        require(no_moe==configs['dense'],'Non-MoE scientific configuration differs')
        schema=joint.parameter_schema(c['model'],c['value_model'])
        count=sum(x['elements'] for x in schema)
        require(count==(232011540 if arm=='dense' else 317001492),'Parameter count differs')
        prior=read(ROOT/'research/studies/strong19_moe_temporal/budget-001.json')['budgets']['dense' if arm=='dense' else 'temporal_experts']
        require(schema==prior['schema'],'Tensor schema differs from qualified budget')
        budgets[arm]={k:prior[k] for k in ('parameters','encoder_parameters','expert_parameters','decode')}
    c=configs['dense'];data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);pools=data.bucket_entries(c['dataset']['buckets'])
    positions=0;digests={}
    for rank in range(4):
        rows,digest=batch_replay.build(c,pools,rank);independent,games=expected(c,data,rank)
        require(canonical_json(rows)==canonical_json(independent),'Replay differs')
        require(digest==batch_proof['counts']['64']['plan_sha256_by_rank'][str(rank)],'Replay identity differs')
        positions+=sum(g['positions'] for g in games[:256*16]);digests[str(rank)]=digest
    require(positions==7001181,'Exposure count differs')
    cases=[dict(arm=arm,games=64,bucket=bucket,draw_index=0 if bucket==512 else 2,plan_sha256_by_rank=digests)
           for arm in configs for bucket in ((512,) if arm=='dense' else (512,768))]
    publish(STUDY/'system-config.json',dict(kind='dense_batch_qualification',updates_per_case=2,
        compiled_memory_limit_bytes=31<<30,cases=cases,configurations=configs))
    from compare import gates, select
    good=dict(endpoint=dict(expert_kl=1.,family_kl=1.,value_mse=1.),tail=dict(expert_kl=1.,family_kl=1.,value_mse=1.),
              sustained_overfit=False,all_tokens_retained=True,learning_seconds=100.)
    better=copy.deepcopy(good);better['endpoint'].update(expert_kl=.9,family_kl=.9)
    better['tail'].update(expert_kl=.9,family_kl=.9)
    require(select({'dense':good,'temporal':better,'balance_low':good})=='temporal','Selection fails improvement')
    for field in ('sustained_overfit','all_tokens_retained'):
        bad=copy.deepcopy(better);bad[field]=field=='sustained_overfit'
        require(not all(gates(bad,good).values()),'Safety gate ignored')
    bad=copy.deepcopy(better);bad['endpoint']['value_mse']=1.06
    require(not all(gates(bad,good).values()),'Value regression accepted')
    for p in STUDY.glob('*.py'):ast.parse(p.read_text())
    publish(STUDY/'cpu-qualification-001.json',dict(status='passed',created=time.time(),positions=positions,budgets=budgets,
        checks=['exact inherited numerical sources','full abstract tensor schema and encoder-inclusive budgets',
          'only balance coefficient changes between MoE arms','complete canonical game/D4 replay',
          'matching LR/evaluation clocks','candidate selection rejects value/overfit/token failures'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [*RECIPE.glob('*.py'),*STUDY.glob('*.py')]},
        inherited={str(p.relative_to(ROOT)):sha(p) for p in (original/'batch-cpu-qualification-002.json',ROOT/'research/studies/strong19_moe_temporal/cpu-qualification-002.json',ROOT/'research/studies/strong19_moe_temporal/budget-001.json')},
        scope='Unchanged computation inherits complete-model tests; new replay/configuration composition checked on CPU. Full-size TPU qualification remains mandatory.'))
    print(json.dumps(dict(status='passed',positions=positions,parameters={k:v['parameters'] for k,v in budgets.items()})))

if __name__=='__main__':main()
