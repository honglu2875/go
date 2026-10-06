"""Bounded full-model A/B benchmark on two replayed, complete training draws.

Each case starts from identical model/AdamW state. One untimed warmup and two
timed updates repeat the same real batch. This is a systems qualification, not
a learning experiment. Full final parameter/optimizer trees are compared on
the host, then discarded; no persistent model payload is written.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify


def publish(path,value):
    with path.open('xb') as f:f.write(canonical_json(value));f.flush();os.fsync(f.fileno())


def validate(c):
    if c['kind']!='joint19_throughput_benchmark' or c['platform']!='tpu':raise ValueError('Wrong benchmark')
    base=c['reference_config']
    if hashlib.sha256(canonical_json(base)).hexdigest()!=c['reference_config_sha256']:
        raise ValueError('Scientific configuration changed')
    from train_config import validate as validate_training
    validate_training(base)
    if (base['steps']!=512 or base['learner']['games_per_host']!=32
            or base['model']['architecture']!='causal_visual_policy'
            or c['warmup_updates']!=1 or c['timed_updates']!=2
            or [d['bucket'] for d in c['draws']]!=[512,768]
            or c['variants']!=[dict(name='baseline',chunk_frames=8,skip_padding=False),
                               dict(name='skip_padding',chunk_frames=8,skip_padding=True)]):
        raise ValueError('Benchmark scope differs')
    return base


def run(args,c,report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset,augment
    import adamw
    import joint
    import learner
    base=validate(c);rank=jax.process_index();world=jax.process_count()
    if world!=4 or len(jax.devices())!=16 or any(d.platform!='tpu' for d in jax.devices()):
        raise ValueError('Unexpected topology')
    mesh=Mesh(np.asarray(jax.devices()),('data',));replicated=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    def host_tree(tree):return jax.tree.map(lambda x:np.asarray(x.addressable_shards[0].data),tree)
    def emit(kind,**values):
        row=dict(kind=kind,time=time.time(),jax_rank=rank,**values)
        with (args.output/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        print(json.dumps(row,allow_nan=False),flush=True)
    data=Dataset(base['dataset']['path'],base['dataset']['manifest_sha256'])
    entries_by_bucket=data.bucket_entries(base['dataset']['buckets'])
    random=np.random.Generator(np.random.PCG64(base['seed']+1+104729*rank))
    augmentation=np.random.Generator(np.random.PCG64(base['seed']+400003+104729*rank))
    batches={};draw_records=[]
    for draw in c['draws']:
        bucket=draw['bucket'];pool=entries_by_bucket['expert',bucket]
        entries=[pool[int(i)] for i in random.integers(len(pool),size=base['learner']['games_per_host'])]
        symmetries=augmentation.integers(0,8,len(entries))
        expected=draw['ranks'][rank]
        entries_sha=hashlib.sha256(canonical_json(entries)).hexdigest()
        if entries_sha!=expected['local_entries_sha256'] or symmetries.tolist()!=expected['local_symmetries']:
            raise ValueError('Registered draw changed')
        batch=augment(data.batch(entries,positions=bucket),symmetries);batches[bucket]=batch
        all_counts=np.asarray(mh.process_allgather(batch['counts'])).reshape(-1)
        if int(all_counts.sum())!=draw['positions']:raise ValueError('Exposure count differs')
        device_counts=all_counts.reshape(16,8)
        live_chunks=np.sum((device_counts+7)//8,axis=1)
        draw_records.append(dict(bucket=bucket,local_entries_sha256=entries_sha,local_counts=batch['counts'].tolist(),
            local_symmetries=symmetries.tolist(),global_positions=int(all_counts.sum()),
            max_device_live_chunks=int(live_chunks.max()),dense_chunks_per_device=bucket,
            live_fraction=float(all_counts.sum()/(128*bucket))))
    report.update(jax_rank=rank,host_rank=int(os.environ['GOZERO_HOST_RANK']),draws=draw_records,
        parameter_count=sum(row['elements'] for row in joint.parameter_schema(base['model'],base['value_model'])),
        cases=[],reference_config_sha256=c['reference_config_sha256'],jax_version=jax.__version__)
    if report['parameter_count']!=232011540:raise ValueError('Model changed')
    opt={k:v for k,v in base['learner'].items() if k not in ('games_per_host','augmentation')};opt['horizon_steps']=512
    references={}
    def initialize():
        p=joint.initialize(base['seed'],base['model'],base['value_model']);return p,adamw.initialize(p)
    def compare(reference,actual):
        a,at=jax.tree.flatten(reference);b,bt=jax.tree.flatten(actual)
        if at!=bt:raise ValueError('Optimizer or parameter tree changed')
        squared=0.;norm=0.;maximum=0.;mismatched=0;exact=True
        for x,y in zip(a,b):
            if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(y).all():raise ValueError('State schema/nonfinite value')
            exact=exact and np.array_equal(x,y)
            delta=x.astype(np.float64)-y.astype(np.float64)
            squared+=float(np.sum(delta*delta));norm+=float(np.sum(x.astype(np.float64)**2))
            maximum=max(maximum,float(np.max(np.abs(delta))))
            mismatched+=int(np.count_nonzero(np.abs(delta)>c['state_atol']+c['state_rtol']*np.abs(x)))
        relative=math.sqrt(squared/max(norm,1e-30))
        return dict(status='passed' if mismatched==0 and relative<=c['state_relative_l2'] else 'failed',
                    exactly_equal=exact,max_abs=maximum,relative_l2=relative,outside_tolerance=mismatched)
    for variant in c['variants']:
        for draw in c['draws']:
            bucket=draw['bucket'];name=variant['name'];mh.sync_global_devices(f'{name}-{bucket}-start')
            emit('case_start',variant=name,bucket=bucket)
            init=jax.jit(initialize,out_shardings=replicated)
            params,state=init();jax.block_until_ready((params,state))
            batch={k:jax.make_array_from_process_local_data(batched,v) for k,v in batches[bucket].items()}
            jax.block_until_ready(batch)
            update=learner.step(base['model'],opt,value_weight=base['training']['value_weight'],path='bounded',
                chunk_frames=variant['chunk_frames'],mesh=mesh,skip_padding=variant['skip_padding'])
            start=time.perf_counter();lowered=jax.jit(update,donate_argnums=(0,1)).lower(params,state,batch)
            executable=lowered.compile();compile_seconds=time.perf_counter()-start
            memory=executable.memory_analysis();fields=['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']
            sizes={k:int(getattr(memory,k)) for k in fields}
            peak=sizes['argument_size_in_bytes']+sizes['output_size_in_bytes']+sizes['temp_size_in_bytes']-sizes['alias_size_in_bytes']
            global_peak=int(np.max(np.asarray(mh.process_allgather(np.asarray(peak,np.int64)))))
            hlo_hash=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest()
            case=dict(variant=name,bucket=bucket,compile_seconds=compile_seconds,memory_bytes=sizes,
                estimated_peak_bytes=peak,global_peak_bytes=global_peak,hlo_sha256=hlo_hash,updates=[])
            emit('case_compiled',variant=name,bucket=bucket,compile_seconds=compile_seconds,estimated_peak_bytes=peak)
            if global_peak>c['compiled_memory_limit_bytes']:
                case['status']='skipped_memory_limit';report['cases'].append(case)
                emit('case_skipped',variant=name,bucket=bucket,reason='compiled_memory_limit')
                del params,state,batch,executable,lowered,update,init
                jax.clear_caches();gc.collect();continue
            mh.sync_global_devices(f'{name}-{bucket}-compiled')
            for step in range(c['warmup_updates']+c['timed_updates']):
                started=time.perf_counter();params,state,metrics=executable(params,state,batch)
                jax.block_until_ready((params,state,metrics));seconds=time.perf_counter()-started
                scalars={k:float(v) for k,v in host_tree(metrics).items()}
                if not all(math.isfinite(v) for v in scalars.values()) or scalars['accepted']!=1:
                    raise ValueError('Nonfinite/rejected benchmark update')
                row=dict(update=step+1,seconds=seconds,timed=step>=c['warmup_updates'],metrics=scalars)
                case['updates'].append(row);emit('update',variant=name,bucket=bucket,update=step+1,seconds=seconds,timed=row['timed'])
            saved=host_tree((params,state))
            if name=='baseline':
                references[bucket]=dict(state=saved,metrics=[r['metrics'] for r in case['updates']]);case['equivalence']={'status':'reference'}
            else:
                case['equivalence']=compare(references[bucket]['state'],saved)
                for actual,expected in zip(case['updates'],references[bucket]['metrics']):
                    for key,value in actual['metrics'].items():
                        if not math.isclose(value,expected[key],rel_tol=c['metric_rtol'],abs_tol=c['metric_atol']):
                            case['equivalence'].update(status='failed',metric_mismatch=key)
                emit('equivalence',variant=name,bucket=bucket,**case['equivalence'])
                del references[bucket]
            case['status']='passed' if case['equivalence']['status'] in ('passed','reference') else 'failed'
            report['cases'].append(case)
            publish(args.output/f'{name}-{bucket}.json',case)
            del saved,params,state,batch,executable,lowered,update,init,metrics
            jax.clear_caches();gc.collect()
    report['status']='passed' if all(case['status']=='passed' for case in report['cases']) else 'failed'
    verify(SOURCE);mh.sync_global_devices('throughput-benchmark-complete')
    if report['status']!='passed':raise ValueError('A performance candidate did not qualify')


def entry(args):
    c=read_json(args.config);validate(c);verify(SOURCE)
    if canonical_json(c)!=canonical_json(read_json(SOURCE/'resolved_config.json')):raise ValueError('Unfrozen config')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'resolved_config.json',c)
    report=dict(kind=c['kind'],snapshot_id=SOURCE.name,status='running',started=time.time(),
        claims_learning_improvement=False,claims_strength=False,claims_mfu=False,
        scope='Two real fixed training draws, identical initialization and AdamW. One warmup plus two timed updates per bucket/variant. Full host state comparison. No new scientific training or checkpoint payload.')
    import jax
    initialized=False
    try:
        jax.distributed.initialize(initialization_timeout=90);initialized=True
        run(args,c,report)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished']=time.time();publish(args.output/'result.json',report)
        if initialized:jax.distributed.shutdown()
