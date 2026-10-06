"""CPU evidence for source lineage, complete-game rebatching and sample clocks."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.snapshots import canonical_json


def configuration(games, peak=.001):
    c=read(ROOT/'research/studies/strong19_recovery/flat-config-001.json')
    c['learner'].update(games_per_host=games//4,learning_rate=peak,end_learning_rate=.3*peak,warmup_steps=40*128//games)
    c.update(steps=512*128//games,eval_every=16*128//games,checkpoint_every=128*128//games,
             batch_replay=dict(source_games_per_host=32,source_steps=512))
    c['checkpoint_disk']['peer']=2
    return c


def main():
    recipe=ROOT/'research/recipes/strong19_dense_batch';parent=ROOT/'research/recipes/strong19_dense_lr'
    sys.path.insert(0,str(recipe))
    import batch_replay,train_config,learner
    import jax,jax.numpy as jnp,numpy as np
    from gozero.corpus_sequence_batches import Dataset,augment
    from batch_audit import expected,audit_replay
    require(all(d.platform=='cpu' for d in jax.devices()),'CPU only')
    source={};unchanged=[]
    for p in recipe.glob('*.py'):
        ast.parse(p.read_text());source[str(p.relative_to(ROOT))]=sha(p)
        if p.name not in ('train_config.py','train_joint.py','batch_replay.py','train.py','qualify_batches.py'):
            require(sha(p)==sha(parent/p.name),'Numerical source changed: '+p.name);unchanged.append(p.name)
    configurations={n:configuration(n) for n in (64,128,256)}
    for c in configurations.values():train_config.validate(c)
    data=Dataset(configurations[128]['dataset']['path'],configurations[128]['dataset']['manifest_sha256'])
    pools=data.bucket_entries([512,768]);plans={};counts={};checks=[]
    for games,c in configurations.items():
        reports={};metrics={};states={};stop=128*128//games
        for rank in range(4):
            rows,digest=batch_replay.build(c,pools,rank);independent,stream=expected(c,data,rank)
            require(canonical_json(rows)==canonical_json(independent),'Independent full stream differs')
            plans[games,rank]=rows
            reports[rank]=dict(jax_rank=rank,batch_replay=dict(plan_sha256=digest))
            states[rank]=dict(batch_replay_state=batch_replay.state(c,digest,stop))
            for key,offset in (('numpy_rng',1+104729*rank),('bucket_rng',9143),('augmentation_rng',400003+104729*rank)):
                states[rank][key]=np.random.Generator(np.random.PCG64(c['seed']+offset)).bit_generator.state
            size=games//4
            metrics[rank]=[dict(bucket=r['bucket'],local_entries_sha256=hashlib.sha256(canonical_json(r['entries'])).hexdigest(),
                    local_symmetries=r['symmetries'],canonical_update_equivalent=(i+1)*size/32)
                    for i,r in enumerate(rows[:stop])]
        position_counts=[sum(int(data.game_info(e)['length']) for rank in range(4) for e in plans[games,rank][i]['entries']) for i in range(stop)]
        slots=sum((games//4)*r['bucket'] for rank in range(4) for r in plans[games,rank][:stop])
        for rank in range(4):
            states[rank]['counters']=dict(updates=stop,expert_positions=sum(position_counts),padded_position_slots=slots)
            for i,row in enumerate(metrics[rank]): row['positions']=position_counts[i]
        positions,audited_slots=audit_replay(c,data,reports,metrics,states,stop)
        require(positions==7001181 and audited_slots==slots,'Exposure budget changed')
        broken=copy.deepcopy(states);broken[0]['batch_replay_state']['next_update']-=1
        try:audit_replay(c,data,reports,metrics,broken,stop)
        except ValueError:pass
        else:raise AssertionError('Wrong saved cursor accepted')
        counts[games]=dict(updates=stop,positions=positions,padded_slots=slots,plan_sha256_by_rank={r:reports[r]['batch_replay']['plan_sha256'] for r in reports})
    checks.extend(['all_512_reference_updates_all_logical_ranks','identical_games_and_D4_order','equal_7001181_position_budget',
                   'independent_saved_cursor_and_draw_audit','reject_bad_replay_cursor'])
    base=configurations[128]
    def opt(c):return dict(horizon_steps=c['steps'],**{k:v for k,v in c['learner'].items() if k not in ('games_per_host','augmentation')})
    for games,c in configurations.items():
        steps=jnp.arange(1,c['steps']+1,dtype=jnp.int32)
        got=np.asarray(learner.schedule(steps,opt(c)))
        expected_rates=np.asarray(learner.schedule(steps*(games/128),opt(base)))
        np.testing.assert_allclose(got,expected_rates,rtol=3e-7,atol=1e-10)
        require(c['eval_every']*games==16*128,'Evaluation clock changed')
    checks.append('LR_and_evaluation_follow_canonical_game_clock')
    # Merging short and long draws pads the short game; active data/D4 must be identical.
    entry=plans[128,0][0]['entries'][0];sym=plans[128,0][0]['symmetries'][0]
    a=augment(data.batch([entry],positions=512),np.asarray([sym]))
    b=augment(data.batch([entry],positions=768),np.asarray([sym]))
    for k in a:
        rhs=b[k][:,:512] if a[k].ndim>=2 and a[k].shape[1]==512 else b[k]
        np.testing.assert_array_equal(a[k],rhs)
    checks.append('padding_preserves_features_actions_targets_and_D4')
    for games,c in configurations.items():
        path=STUDY/f'batch{games}-template-001.json'
        if path.exists():require(read(path)==c,'Existing template changed')
        else:publish(path,c)
    cases=[]
    for games in (64,256):
        for bucket in (512,768):
            index=next(i for i,r in enumerate(plans[games,0]) if r['bucket']==bucket)
            cases.append(dict(games=games,bucket=bucket,draw_index=index,
                              plan_sha256_by_rank=counts[games]['plan_sha256_by_rank']))
    publish(STUDY/'batch-system-template-001.json',dict(kind='dense_batch_qualification',updates_per_case=2,
        compiled_memory_limit_bytes=31<<30,cases=cases,configurations={str(n):configurations[n] for n in (64,256)}))
    result=dict(status='passed',created=time.time(),checks=checks,source_files=source,unchanged_numerical_files=unchanged,
        configurations={str(n):sha(STUDY/f'batch{n}-template-001.json') for n in configurations},counts=counts,
        operator_sha256=sha(Path(__file__)),auditor_sha256=sha(STUDY/'batch_audit.py'),
        scope='CPU loader/state/schedule evidence only. Full-size batch TPU memory/runtime qualification is required before learning.')
    publish(STUDY/'batch-cpu-qualification-002.json',result)
    print(json.dumps(dict(status='passed',checks=checks,unchanged_numerical_files=len(unchanged),counts=counts)))


if __name__=='__main__':main()
