"""One-shot owned trainer clone from the frozen larger9 transport/harness."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[3]
PARENT=ROOT/'.gozero/snapshots/23c9dfe60c0a4c1c376e74f634b568fd1c85f3b13a0c7fef9172859ad38a40fb/research/recipes/strong9_policy'
TARGET=ROOT/'research/recipes/strong19_train'


def main():
    original=(PARENT/'train_policy.py').read_bytes()
    if hashlib.sha256(original).hexdigest()!='e9dacf3da9c59e1b4235627ade469aff271bf95c971cc16af1cc3b6487f82bd5':
        raise ValueError('Parent trainer changed')
    text=original.decode()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('Ambiguous parent edit: '+old[:80])
        text=text.replace(old,new)
    def section(first,last,new):
        nonlocal text
        if text.count(first)!=1 or text.count(last)!=1:raise ValueError('Ambiguous parent section')
        a=text.index(first);b=text.index(last,a);text=text[:a]+new+text[b:]
    replace('"""Cloneable pure-JAX policy learning with shared replicated checkpoints.\n\nDerived from the previously recovery-qualified visual learner. Sampling and\nobjectives are narrowed to the expert population; source snapshots retain both.\n"""',
            '"""Owned joint policy/value harness over fixed complete expert histories.\n\nThe shared checkpoint transport follows the recovery-qualified policy learner.\nThe model/update functions are cloned from the qualified joint prototype.\n"""')
    replace('from policy_config import validate','from train_config import validate')
    replace('    import policy_optimizer as learner\n    import policy_model as model',
            '    import adamw as optimizer\n    import learner as joint_learner\n    import joint as model\n    import optimizer_io\n    import evaluation')
    section('    def global_batch(batch):','    net, opt = ',
        "    def global_batch(batch):\n        return {k:jax.make_array_from_process_local_data(batched,v) for k,v in batch.items()}\n\n")
    replace('    schema = model.parameter_schema(net)','    schema = model.parameter_schema(net,c[\'value_model\'])')
    replace("model.initialize(c['seed'], net)","model.initialize(c['seed'], net,c['value_model'])")
    replace('jax.jit(learner.initialize, out_shardings=replicated)','jax.jit(optimizer.initialize, out_shardings=replicated)')
    replace('    leaves, definition = jax.tree.flatten(params)\n','')
    text,count=re.subn(r'^[ \t]+# spatial study provenance begin\n.*?^[ \t]+# spatial study provenance end\n','',text,flags=re.S|re.M)
    if count!=3:raise ValueError('Parent provenance sections differ')
    section("    if net.get('first_pass_aux_weight',0):\n        import qualify_draft_runtime",'    random = ','')
    replace('    if args.resume is not None:\n',
        '    history=[];training_history=[];overfit_history=[]\n    if args.resume is not None:\n')
    section("        required = {f'{kind}_{i:04d}'", "        turn = saved['turn']; counters = saved['counters']",
        """        if host!=0 and local_arrays:raise ValueError('Nonowner checkpoint contains arrays')
        if owner_state['optimizer_metadata']!=saved['optimizer_metadata']:
            raise ValueError('Optimizer metadata differs across ranks')
        expected_schema=[{key:item[key] for key in ('path','shape','dtype')} for item in schema]
        restored,restored_state=optimizer_io.restore(saved['optimizer_metadata'],arrays,schema=expected_schema,
            configuration_sha256=config_sha,source_sha256=SOURCE.name)
        if int(restored_state['step'])!=saved['turn']:raise ValueError('Optimizer step differs from rank progress')
        params=jax.tree.map(lambda x:jax.device_put(x,replicated),restored)
        state=jax.tree.map(lambda x:jax.device_put(x,replicated),restored_state)
        history=saved['validation_history'];training_history=saved['training_probe_history']
        overfit_history=saved['overfit_history']
        for rows in (history,training_history):
            turns=[row['turn'] for row in rows]
            if turns!=sorted(set(turns)) or any(t<0 or t>saved['turn'] for t in turns):
                raise ValueError('Checkpoint diagnostic history differs')
""")
    section('    def schedule(iteration):','    compiled_steps, compiled_evaluations = {}, {}',
        """    optimizer_config={k:v for k,v in opt.items() if k not in ('games_per_host','augmentation')}
    optimizer_config['horizon_steps']=c['steps']
    def specs(batch):return {k:P('data') for k in batch}
""")
    section("            objective = jax.shard_map(local_loss", "            print(json.dumps({'kind': 'visual_compile_update'",
        """            update=joint_learner.step(net,optimizer_config,value_weight=c['training']['value_weight'],
                path=c['training']['path'],chunk_frames=c['training']['chunk_frames'],mesh=mesh)
""")
    section('    def evaluate(split,*,draft=False,entries_by_bucket=None):','    def save():',
        """    def evaluate(split,*,entries_by_bucket=None):
        start=time.perf_counter();sums={};all_ids=[]
        selected_buckets=data.bucket_entries(buckets,split=split)
        chosen_by_bucket={bucket:(entries_by_bucket[bucket] if entries_by_bucket is not None else
            selected_buckets['expert',bucket][:c['evaluation']['games_per_bucket']]) for bucket in buckets}
        population=[entry for bucket in buckets for entry in chosen_by_bucket[bucket]]
        if not population:raise ValueError('Evaluation population is empty')
        family_weights=data.evaluation_family_weights(population);batch_size=opt['games_per_host']
        for bucket in buckets:
            entries=chosen_by_bucket[bucket];all_ids.extend(entries);local=entries[rank::world]
            for begin in range(0,math.ceil(len(entries)/world),batch_size):
                chosen=local[begin:begin+batch_size]
                batch=global_batch(data.batch(chosen+[None]*(batch_size-len(chosen)),positions=bucket,family_weights=family_weights))
                if bucket not in compiled_evaluations:
                    start_compile=time.perf_counter()
                    fn=jax.shard_map(lambda p,b:evaluation.totals(p,b,net,chunk_frames=c['training']['chunk_frames'],axis_name='data'),
                        mesh=mesh,in_specs=(P(),specs(batch)),out_specs=P(),check_vma=False)
                    print(json.dumps(dict(kind='joint_compile_evaluation',bucket=bucket,host=host)),flush=True)
                    compiled_evaluations[bucket]=jax.jit(fn).lower(params,batch).compile()
                    timing['compilation_seconds']+=time.perf_counter()-start_compile
                totals={k:float(v) for k,v in replica(compiled_evaluations[bucket](params,batch)).items()}
                if not all(math.isfinite(v) for v in totals.values()):raise FloatingPointError('Nonfinite joint evaluation')
                for k,v in totals.items():sums[k]=sums.get(k,0.)+v
        if sums['expert_count']!=sums['value_count']:raise ValueError('Policy/value population differs')
        result=dict(turn=turn,split=split,episode_ids_sha256=hashlib.sha256(canonical_json(all_ids)).hexdigest(),
            raw_totals=sums,metrics={k:float(v) for k,v in evaluation.averages(sums).items()})
        timing['evaluation_seconds']+=time.perf_counter()-start
        kind='joint_training_probe' if entries_by_bucket is not None else 'joint_heldout'
        with (args.output/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(kind=kind,**result),sort_keys=True)+'\\n');stream.flush()
        print(json.dumps(dict(kind=kind,host=host,**result)),flush=True)
        return result
""")
    section("        arrays = {f'{kind}_{i:04d}'",'        digest = digest_arrays(arrays)',
        """        optimizer_metadata,arrays=optimizer_io.flatten(replica(params),replica(state),
            configuration_sha256=config_sha,source_sha256=SOURCE.name)
        if optimizer_metadata['step']!=turn:raise ValueError('Optimizer and sampler progress differ')
""")
    replace("                 'counters': counters, 'owns_replicated_arrays': host == 0}",
        """                 'counters': counters, 'owns_replicated_arrays': host == 0,
                 'optimizer_metadata':optimizer_metadata,'validation_history':history,
                 'training_probe_history':training_history,'overfit_history':overfit_history}""")
    replace('    history = []; training_history = []; overfit_history = []\n','')
    section("    if 'draft_reference' in c and not args.resume:", '    if args.resume is None:', '')
    replace("        report['initial_validation'] = evaluate(1)",
        "        report['initial_validation'] = evaluate(1)\n        history.append(report['initial_validation'])")
    replace("                counters[role + '_positions'] += int(observed[role + '_positions'])",
        "                counters[role + '_positions'] += int(observed['positions'])")
    replace("                    paired_validation=([report['initial_validation']] if 'initial_validation' in report else [])+history",
        "                    paired_validation=history")
    replace("                    for metric in ('expert_kl','family_kl'):",
        "                    for metric in ('expert_kl','family_kl','value_mse','value_family_mse'):")
    section("    if c.get('profile_decode', False) and net['architecture']=='causal_visual_policy':",'    # The owner checkpoint already contains exact parameters;', '')
    replace("              'input_kind': 'Pinned exact complete expert histories, V7 features; one offline policy target.'}",
        "              'input_kind': 'Pinned complete expert histories and exact V7 features; raw policy and signed player-to-move value targets.',\n              'training_purpose':c['training']['purpose'],'optimizer_family':'adamw'}")
    outputs={'train_joint.py':text,'policy_config.py':(PARENT/'policy_config.py').read_text()}
    receipt={}
    for name,body in outputs.items():
        ast.parse(body,filename=name);path=TARGET/name
        with path.open('x') as f:f.write(body)
        receipt[name]=hashlib.sha256(body.encode()).hexdigest()
    result=dict(kind='owned_joint_trainer_clone',created=time.time(),parent_snapshot=PARENT.parents[2].name,
        parent_train_sha256=hashlib.sha256(original).hexdigest(),generated_files=receipt,
        scope='New joint integration over qualified checkpoint/sampling transport; requires complete CPU and TPU continuation qualification')
    with (Path(__file__).parent/'trainer-clone-001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
