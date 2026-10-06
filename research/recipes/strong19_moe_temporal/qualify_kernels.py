"""Bounded TPU grouped-expert numerical qualification and latency measurements."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,canonical_json,verify


def publish(path,value):
    with path.open('xb') as stream:stream.write(canonical_json(value));stream.flush();os.fsync(stream.fileno())


def run(args,c,report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from gozero import moe
    if jax.process_count()!=4 or len(jax.devices())!=16 or any(d.platform!='tpu' for d in jax.devices()):raise ValueError('Topology differs')
    if c['kind']!='moe_kernel_qualification' or c['repetitions']!=10:raise ValueError('Scope differs')
    report.update(host_rank=int(os.environ['GOZERO_HOST_RANK']),jax_rank=jax.process_index(),jax_version=jax.__version__,cases=[])
    def reference(p,x,k):
        logits=jnp.matmul(x,p['router'],precision=jax.lax.Precision.HIGHEST)
        chosen=jax.lax.top_k(logits,k)[1]
        mixture=jax.nn.softmax(jnp.take_along_axis(logits,chosen,-1),-1)
        proj=lambda weights:jnp.einsum('nd,edh->neh',x.astype(jnp.bfloat16),weights.astype(jnp.bfloat16),preferred_element_type=jnp.float32)
        y=proj(p['up'])
        if 'up_bias' in p:y=y+p['up_bias']
        y=jax.nn.silu(proj(p['gate']))*y if 'gate' in p else jax.nn.gelu(y)
        y=jnp.einsum('neh,ehd->ned',y.astype(jnp.bfloat16),p['down'].astype(jnp.bfloat16),preferred_element_type=jnp.float32)
        if 'down_bias' in p:y=y+p['down_bias']
        return jnp.sum(jnp.take_along_axis(y,chosen[...,None],axis=1)*mixture[...,None],axis=1)
    for case_index,case in enumerate(c['cases']):
        if c.get('partition_cases_by_rank',False) and case_index % 4 != jax.process_index():continue
        n,d,h=case['tokens'],case['width'],case['hidden'];e,k=4,2
        with jax.default_device(jax.local_devices()[0]):
            p=moe.initialize(jax.random.key(731),width=d,hidden=h,experts=e,gated=case['gated'],bias=not case['gated'])
            x=jax.random.normal(jax.random.key(732),(n,d));weights=jax.random.normal(jax.random.key(733),(n,d))/n
            if case['collapsed']:p={**p,'router':jnp.zeros_like(p['router'])}
            def sparse(p,x):return moe.feed_forward(p,x,top_k=k,dtype=jnp.bfloat16,
                backend=case.get('backend','pallas'),tiling=tuple(case.get('tiling',(128,128,128))))[0]
            loss=lambda fn,p,x:(jnp.sum(fn(p,x)*weights),fn(p,x))
            actual=jax.jit(jax.value_and_grad(lambda p,x:loss(sparse,p,x),(0,1),has_aux=True))
            ref=jax.jit(jax.value_and_grad(lambda p,x:loss(lambda p,x:reference(p,x,k),p,x),(0,1),has_aux=True))
            started=time.perf_counter();out=actual(p,x);jax.block_until_ready(out);compile_and_first=time.perf_counter()-started
            expected=ref(p,x);jax.block_until_ready(expected)
            errors=[]
            for a,b in zip(jax.tree.leaves(out),jax.tree.leaves(expected)):
                a,b=np.asarray(a),np.asarray(b)
                np.testing.assert_allclose(a,b,rtol=.025,atol=.008)
                errors.append(dict(max_abs=float(np.max(np.abs(a-b))),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20))))
            compiled=actual.lower(p,x).compile();memory=compiled.memory_analysis()
            timings=[]
            for _ in range(c['repetitions']):
                started=time.perf_counter();jax.block_until_ready(compiled(p,x));timings.append(time.perf_counter()-started)
            # Independent dense FFN of the original active hidden width, same
            # precision/activation and full parameter+input reverse-mode work.
            dense={name:value[0] for name,value in moe.initialize(jax.random.key(734),width=d,hidden=h*k,experts=1,gated=case['gated'],bias=not case['gated']).items() if name!='router'}
            def dense_fn(p,x):
                dot=lambda a,b:jnp.matmul(a.astype(jnp.bfloat16),b.astype(jnp.bfloat16),preferred_element_type=jnp.float32)
                y=dot(x,p['up'])
                if 'up_bias' in p:y=y+p['up_bias']
                y=jax.nn.silu(dot(x,p['gate']))*y if 'gate' in p else jax.nn.gelu(y)
                result=dot(y,p['down'])
                return result+p['down_bias'] if 'down_bias' in p else result
            dense_exec=jax.jit(jax.value_and_grad(lambda p,x:loss(dense_fn,p,x),(0,1),has_aux=True)).lower(dense,x).compile()
            jax.block_until_ready(dense_exec(dense,x));dense_times=[]
            for _ in range(c['repetitions']):
                started=time.perf_counter();jax.block_until_ready(dense_exec(dense,x));dense_times.append(time.perf_counter()-started)
            row=dict(**case,status='passed',errors=errors,compile_and_first_seconds=compile_and_first,
                sparse_forward_backward_seconds=timings,dense_forward_backward_seconds=dense_times,
                sparse_median_seconds=float(np.median(timings)),dense_median_seconds=float(np.median(dense_times)),
                memory_bytes={name:int(getattr(memory,name)) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
                hlo_sha256=hashlib.sha256(actual.lower(p,x).compiler_ir('hlo').as_hlo_text().encode()).hexdigest())
            report['cases'].append(row);publish(args.output/('case-'+str(len(report['cases']))+'.json'),row)
            print(json.dumps(dict(kind='moe_kernel_case',case=case,status='passed',sparse_seconds=row['sparse_median_seconds'],dense_seconds=row['dense_median_seconds'])),flush=True)
            del out,expected,actual,ref,compiled,dense_exec,p,x,dense
            jax.clear_caches()
    mh.sync_global_devices('moe-kernels-qualified');report['status']='passed';verify(SOURCE)


def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json'):raise ValueError('Unfrozen configuration')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    publish(args.output/'resolved_config.json',c)
    report=dict(kind=c['kind'],status='running',snapshot_id=SOURCE.name,started=time.time(),claims_learning=False,claims_strength=False)
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
