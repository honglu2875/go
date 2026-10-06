"""Full-size fresh-state dense/sparse updates on pinned complete-game draws."""
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
from qualify_kernels import publish


def run(args,c,report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset,augment
    import joint,learner,adamw,train_config
    base=c['reference_config'];train_config.validate(base)
    if hashlib.sha256(canonical_json(base)).hexdigest()!=c['reference_config_sha256']:raise ValueError('Reference differs')
    if jax.process_count()!=4 or len(jax.devices())!=16 or any(d.platform!='tpu' for d in jax.devices()):raise ValueError('Topology differs')
    if c['updates_per_case']!=2 or c['compiled_memory_limit_bytes']!=31*(1<<30):raise ValueError('Qualification scope differs')
    mesh=Mesh(np.asarray(jax.devices()),('data',));replicated=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    rank=jax.process_index();host=int(os.environ['GOZERO_HOST_RANK'])
    def emit(kind,**values):
        row=dict(kind=kind,time=time.time(),jax_rank=rank,**values)
        with (args.output/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(row,allow_nan=False),flush=True)
    def host_metrics(tree):return jax.tree.map(lambda x:float(np.asarray(x.addressable_shards[0].data)),tree)
    data=Dataset(base['dataset']['path'],base['dataset']['manifest_sha256']);pools=data.bucket_entries(base['dataset']['buckets'])
    rng=np.random.Generator(np.random.PCG64(base['seed']+1+104729*rank));aug=np.random.Generator(np.random.PCG64(base['seed']+400003+104729*rank))
    batches={}
    for draw in c['draws']:
        bucket=draw['bucket'];pool=pools['expert',bucket]
        entries=[pool[int(i)] for i in rng.integers(len(pool),size=base['learner']['games_per_host'])];symmetries=aug.integers(0,8,len(entries))
        expected=draw['ranks'][rank]
        if hashlib.sha256(canonical_json(entries)).hexdigest()!=expected['local_entries_sha256'] or symmetries.tolist()!=expected['local_symmetries']:raise ValueError('Draw replay differs')
        local=augment(data.batch(entries,positions=bucket),symmetries)
        if int(np.asarray(mh.process_allgather(local['counts'])).sum())!=draw['positions']:raise ValueError('Positions differ')
        batches[bucket]=local
    report.update(jax_rank=rank,host_rank=host,jax_version=jax.__version__,cases=[])
    opt={k:v for k,v in base['learner'].items() if k not in ('games_per_host','augmentation')};opt['horizon_steps']=base['steps']
    for case in c['cases']:
        name,bucket=case['name'],case['bucket'];cfg=dict(base['model'])
        if case['moe']:cfg['moe']=c['moe']
        mh.sync_global_devices(f'{name}-{bucket}-start');emit('case_start',**case)
        schema=joint.parameter_schema(cfg,base['value_model']);count=sum(x['elements'] for x in schema)
        def initialize():
            p=joint.initialize(base['seed'],cfg,base['value_model']);return p,adamw.initialize(p)
        init=jax.jit(initialize,out_shardings=replicated);params,state=init();jax.block_until_ready((params,state))
        batch={k:jax.make_array_from_process_local_data(batched,v) for k,v in batches[bucket].items()};jax.block_until_ready(batch)
        update=learner.step(cfg,opt,value_weight=base['training']['value_weight'],path='bounded',chunk_frames=case['chunk_frames'],mesh=mesh,skip_padding=True)
        start=time.perf_counter();lowered=jax.jit(update,donate_argnums=(0,1)).lower(params,state,batch);executable=lowered.compile();seconds=time.perf_counter()-start
        mem=executable.memory_analysis();sizes={k:int(getattr(mem,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        peak=sizes['argument_size_in_bytes']+sizes['output_size_in_bytes']+sizes['temp_size_in_bytes']-sizes['alias_size_in_bytes']
        global_peak=1024*max(int(v) for v in np.asarray(mh.process_allgather(np.asarray((peak+1023)//1024,np.int32))).reshape(-1))
        row=dict(**case,parameters=count,memory_bytes=sizes,global_peak_bytes=global_peak,compile_seconds=seconds,
                 hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),updates=[])
        emit('case_compiled',name=name,bucket=bucket,parameters=count,global_peak_bytes=global_peak,compile_seconds=seconds)
        if global_peak>c['compiled_memory_limit_bytes']:raise MemoryError('Compiled allocation exceeds research memory limit')
        mh.sync_global_devices(f'{name}-{bucket}-compiled')
        for step in range(c['updates_per_case']):
            start=time.perf_counter();params,state,metrics=executable(params,state,batch);jax.block_until_ready((params,state,metrics));seconds=time.perf_counter()-start
            observed=host_metrics(metrics)
            if observed['accepted']!=1 or not all(math.isfinite(v) for v in observed.values()):raise ValueError('Nonfinite/rejected full model update')
            if case['moe'] and observed['moe_dropped_tokens']!=0:raise ValueError('Token dropping')
            row['updates'].append(dict(step=step+1,seconds=seconds,metrics=observed));emit('update',name=name,bucket=bucket,step=step+1,seconds=seconds,metrics=observed)
        row['status']='passed';report['cases'].append(row);publish(args.output/f'{name}-{bucket}.json',row)
        del params,state,batch,executable,lowered,update,init,metrics
        jax.clear_caches();gc.collect()
    report['status']='passed';verify(SOURCE);mh.sync_global_devices('moe-systems-qualified')


def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json') or c['kind']!='moe_system_qualification':raise ValueError('Unfrozen configuration')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False);publish(args.output/'resolved_config.json',c)
    report=dict(kind=c['kind'],snapshot_id=SOURCE.name,status='running',started=time.time(),claims_learning=False,claims_strength=False,
        scope='Two updates per fresh-state case on repeated real batches. No scientific learning or checkpoint payload; donated compiled memory guard and measured wall time.')
    import jax
    initialized=False
    try:
        jax.distributed.initialize(initialization_timeout=90);initialized=True;run(args,c,report)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished']=time.time();publish(args.output/'result.json',report)
        if initialized:jax.distributed.shutdown()
