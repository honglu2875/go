"""Full-size batch-shape admission using pinned complete-game replay draws."""
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


def decode_benchmark(params,c,mesh,batched,rep):
    """Synthetic cached move timing, fixed global B=128; no quality claim."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import joint
    n=32;d=c['width']//c['heads'];rows=[]
    make=lambda x:jax.make_array_from_process_local_data(batched,x)
    spatial=make(np.ones((n,19,19,22),np.float32));glob=make(np.zeros((n,19),np.float32))
    action=make(np.zeros(n,np.int32))
    for history in (128,512):
        shape=(n,c['layers'],2*(history+1),c['kv_heads'],d)
        cache=dict(keys=make(np.zeros(shape,jnp.bfloat16)),values=make(np.zeros(shape,jnp.bfloat16)),
            lengths=make(np.full(n,2*history-1,np.int32)),valid=make(np.ones(n,bool)),
            network_version=jax.device_put(np.asarray(0,np.uint32),rep))
        f=jax.jit(lambda p,k,a,s,g:joint.append_move(p,k,a,s,g,c,attention_positions=history+1))
        start=time.perf_counter();lowered=f.lower(params,cache,action,spatial,glob);exe=lowered.compile()
        compile_seconds=time.perf_counter()-start
        result=exe(params,cache,action,spatial,glob);jax.block_until_ready(result)
        times=[]
        for _ in range(5):
            start=time.perf_counter();result=exe(params,cache,action,spatial,glob);jax.block_until_ready(result)
            times.append(time.perf_counter()-start)
        rows.append(dict(history_positions=history,global_batch=128,compile_seconds=compile_seconds,
            seconds=times,median_seconds=float(np.median(times)),cost_analysis=exe.cost_analysis(),
            hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
            scope='Synthetic finite inputs, repeated same cache; latency only. No sparse-kernel or MFU claim.'))
        del cache,result,exe,lowered,f
    return rows


def run(args,c,report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset,augment
    import batch_replay,joint,learner,adamw,train_config
    if jax.process_count()!=4 or len(jax.devices())!=16 or any(d.platform!='tpu' for d in jax.devices()):
        raise ValueError('Topology differs')
    rank=jax.process_index();host=int(os.environ['GOZERO_HOST_RANK'])
    mesh=Mesh(np.asarray(jax.devices()),('data',));rep=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    reference=c['configurations']['dense'];data=Dataset(reference['dataset']['path'],reference['dataset']['manifest_sha256'])
    pools=data.bucket_entries(reference['dataset']['buckets'])
    report.update(jax_rank=rank,host_rank=host,cases=[])
    if c['updates_per_case']!=2 or c['compiled_memory_limit_bytes']!=31<<30:raise ValueError('Scope differs')
    for case in c['cases']:
        cfg=c['configurations'][case['arm']];train_config.validate(cfg)
        rows,digest=batch_replay.build(cfg,pools,rank);draw=rows[case['draw_index']]
        if draw['bucket']!=case['bucket'] or digest!=case['plan_sha256_by_rank'][str(rank)]:
            raise ValueError('Qualification draw changed')
        raw=augment(data.batch(draw['entries'],positions=draw['bucket']),np.asarray(draw['symmetries']))
        batch={k:jax.make_array_from_process_local_data(batched,v) for k,v in raw.items()}
        del raw,rows
        net=cfg['model'];opt={k:v for k,v in cfg['learner'].items() if k not in ('games_per_host','augmentation')}
        opt['horizon_steps']=cfg['steps']
        def init():
            p=joint.initialize(cfg['seed'],net,cfg['value_model']);return p,adamw.initialize(p)
        initializer=jax.jit(init,out_shardings=rep);params,state=initializer();jax.block_until_ready((params,state,batch))
        update=learner.step(net,opt,value_weight=cfg['training']['value_weight'],path='bounded',chunk_frames=8,
                            mesh=mesh,skip_padding=True)
        start=time.perf_counter();lowered=jax.jit(update,donate_argnums=(0,1)).lower(params,state,batch)
        executable=lowered.compile();memory=executable.memory_analysis()
        sizes={k:int(getattr(memory,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        peak=sizes['argument_size_in_bytes']+sizes['output_size_in_bytes']+sizes['temp_size_in_bytes']-sizes['alias_size_in_bytes']
        peak=1024*max(int(v) for v in np.asarray(mh.process_allgather(np.asarray((peak+1023)//1024,np.int32))).reshape(-1))
        row=dict(**case,compile_seconds=time.perf_counter()-start,global_peak_bytes=peak,memory_bytes=sizes,
                 hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),updates=[])
        print(json.dumps(dict(kind='batch_case_compiled',arm=case['arm'],games=case['games'],bucket=case['bucket'],peak_bytes=peak)),flush=True)
        if peak>c['compiled_memory_limit_bytes']:raise MemoryError('Batch exceeds donated memory admission')
        for step in range(2):
            start=time.perf_counter();params,state,metrics=executable(params,state,batch);jax.block_until_ready((params,state,metrics))
            values={k:float(np.asarray(v.addressable_shards[0].data)) for k,v in metrics.items()}
            if values['accepted']!=1 or not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('Rejected batch qualification update')
            row['updates'].append(dict(step=step+1,seconds=time.perf_counter()-start,metrics=values))
        if case['bucket']==512:row['cached_decode']=decode_benchmark(params,net,mesh,batched,rep)
        row['status']='passed';report['cases'].append(row)
        publish(args.output/f"{case['arm']}-{case['bucket']}.json",row)
        del params,state,batch,metrics,executable,lowered,update,initializer
        jax.clear_caches();gc.collect()
    report['status']='passed';verify(SOURCE);mh.sync_global_devices('dense-batches-qualified')


def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json') or c['kind']!='dense_batch_qualification':raise ValueError('Unfrozen qualification')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'resolved_config.json',c)
    report=dict(kind=c['kind'],snapshot_id=SOURCE.name,status='running',started=time.time(),
                claims_learning=False,scope='Two fresh-state updates on each repeated real replay batch. Memory/numerical/runtime admission only.')
    import jax
    initialized=False
    try:
        jax.distributed.initialize(initialization_timeout=90);initialized=True;run(args,c,report)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished']=time.time();publish(args.output/'result.json',report)
        if initialized:jax.distributed.shutdown()
