"""Bounded attention-kernel timing and gradient checks; no learning or data selection."""
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
from gozero.snapshots import read_json,canonical_json,verify
from gozero.checkpoints import sha256


def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json') or c['kind']!='attention_tile_qualification':raise ValueError('Unfrozen kernel probe')
    os.environ['JAX_PLATFORMS']='tpu'
    args.output.mkdir(parents=True,exist_ok=False)
    from train_policy import publish
    publish(args.output/'resolved_config.json',c)
    report={'kind':c['kind'],'snapshot_id':SOURCE.name,'status':'running','started_unix':time.time(),
        'scope':'Kernel forward/backward timing on synthetic dynamic Q/K/V/cotangent arrays. No architecture learning, labels or validation selection.'}
    import jax
    try:
        jax.distributed.initialize(initialization_timeout=90)
        run(c,report)
        report['status']='passed';verify(SOURCE)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();publish(args.output/'result.json',report)
        jax.distributed.shutdown()


def run(c,report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from jax.experimental.pallas.ops.tpu import splash_attention as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as kernel_source
    import causal
    from observation_attention import ObservationMask
    observation_stride=c.get('observation_stride',0)
    if len(jax.devices())!=16 or jax.process_count()!=4:raise ValueError('Wrong topology')
    mesh=Mesh(np.asarray(jax.devices()),('data',));batched=NamedSharding(mesh,P('data'))
    report.update(jax_rank=jax.process_index(),host_rank=int(os.environ['GOZERO_HOST_RANK']),
        kernel_source_sha256=sha256(Path(kernel_source.__file__)),jax_version=jax.__version__,
        observation_stride=observation_stride,cases=[])

    def arrays(length):
        def create(seed):
            keys=jax.random.split(seed,4)
            return tuple(jax.random.normal(key,(128,length,heads,64),jnp.bfloat16)*.3 for key,heads in zip(keys,[12,4,4,12]))
        return jax.jit(create,out_shardings=(batched,)*4)(jax.random.key(193+length))

    def attention(tile,length):
        if tile==0:
            return lambda q,k,v:causal.dense_attention(q,k,v,jnp.broadcast_to(jnp.arange(length),q.shape[:2]),jnp.full((q.shape[0],),length),observation_stride)
        extent=math.ceil(length/tile)*tile
        one=(ObservationMask((extent,extent),observation_stride,length) if observation_stride else splash.CausalMask((extent,extent)))
        mask=splash.MultiHeadMask([one for _ in range(12)])
        blocks=splash.BlockSizes(block_q=tile,block_kv=tile,block_kv_compute=tile,block_q_dkv=tile,block_kv_dkv=tile,
            block_kv_dkv_compute=tile,block_q_dq=tile,block_kv_dq=tile)
        with jax.ensure_compile_time_eval():fn=splash.make_splash_mha(mask,block_sizes=blocks,head_shards=1,q_seq_shards=1)
        def execute(q,k,v):
            def pad(x):return jnp.pad(x.transpose(0,2,1,3),((0,0),(0,0),(0,extent-length),(0,0)))
            return jax.vmap(fn)(pad(q)/8,pad(k),pad(v)).transpose(0,2,1,3)[:,:length].astype(jnp.float32)
        return execute

    def functions(tile,length):
        attend=attention(tile,length)
        out=jax.shard_map(attend,mesh=mesh,in_specs=(P('data'),)*3,out_specs=P('data'),check_vma=False)
        def loss(q,k,v,do):
            return jax.lax.psum(jnp.sum(attend(q,k,v)*do),'data')/(128*length*12*64)
        objective=jax.shard_map(loss,mesh=mesh,in_specs=(P('data'),)*4,out_specs=P(),check_vma=False)
        return jax.jit(out),jax.jit(jax.value_and_grad(objective,argnums=(0,1,2)))

    inputs=arrays(257);refout,refgrad=functions(0,257)
    expected=refout(*inputs[:3]);_,grad_expected=refgrad(*inputs);jax.block_until_ready((expected,grad_expected))
    @jax.jit
    def errors(a,b):return jnp.max(jnp.abs(a.astype(jnp.float32)-b.astype(jnp.float32))),jnp.sqrt(jnp.sum((a.astype(jnp.float32)-b)**2)/jnp.maximum(jnp.sum(b.astype(jnp.float32)**2),1e-30))
    def host(x):return np.asarray(x.addressable_shards[0].data)
    for tile in c['tiles']:
        out,grad=functions(tile,257);actual=out(*inputs[:3]);_,got=grad(*inputs)
        discrepancies=[list(map(lambda z:float(host(z)),errors(x,y))) for x,y in zip([actual,*got],[expected,*grad_expected])]
        if any(absolute>.05 or relative>.03 for absolute,relative in discrepancies):raise ValueError('Attention output/gradient tolerance exceeded')
        report.setdefault('numerics',[]).append({'tile':tile,'length':257,'output_q_k_v_errors':discrepancies,'absolute_tolerance':.05,'relative_l2_tolerance':.03})
        print(json.dumps({'kind':'attention_numerics','tile':tile,'errors':discrepancies}),flush=True)
    del inputs,expected,grad_expected,actual,got,refout,refgrad,out,grad
    gc.collect()
    for length in c['lengths']:
        inputs=arrays(length);jax.block_until_ready(inputs)
        for tile in c['tiles']:
            _,gradient=functions(tile,length)
            start=time.perf_counter();lowered=gradient.lower(*inputs);compiled=lowered.compile();compilation=time.perf_counter()-start
            jax.block_until_ready(compiled(*inputs));times=[]
            for _ in range(c['repeats']):
                start=time.perf_counter();values=compiled(*inputs);jax.block_until_ready(values);times.append(time.perf_counter()-start)
            memory=compiled.memory_analysis()
            row={'tile':tile,'tokens':length,'padded_tokens':math.ceil(length/tile)*tile,'global_batch':128,
                'compile_seconds':compilation,'seconds':times,
                'memory_bytes':{key:int(getattr(memory,key)) for key in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']},
                'hlo_sha256':hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest()}
            report['cases'].append(row);print(json.dumps({'kind':'attention_timing',**row}),flush=True)
            del gradient,compiled,lowered,values
        del inputs
        gc.collect()
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices('attention-tiles-qualified')
