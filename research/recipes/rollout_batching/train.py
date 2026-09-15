#!/usr/bin/env python3
"""Fixed-game, fixed-network native rollout throughput; inference microbatch is explicit."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json,read_json,verify


def validate(c):
    fields='schema_version platform expected_processes expected_devices seed actors model model_artifact inference_batch warmup_turns measured_turns controller_cpus verify_trace'
    if not set(fields.split())<=set(c) or set(c)-set(fields.split())-{'dispatch_mode','verify_outputs'} or c['schema_version']!=1 or c['platform'] not in ('cpu','tpu'):
        raise ValueError('Invalid benchmark configuration')
    if c.get('dispatch_mode','serial') not in ('serial','async','scan','wide') or type(c.get('verify_outputs',False)) is not bool:
        raise ValueError('Invalid inference dispatch or output verification mode')
    for key in ('expected_processes','expected_devices','inference_batch','measured_turns'):
        if type(c[key]) is not int or c[key]<1: raise ValueError('Invalid '+key)
    if not 0<=c['warmup_turns']<=1024 or c['measured_turns']>4096 or type(c['verify_trace']) is not bool:
        raise ValueError('Invalid measurement window or trace switch')
    a=c['actors'];local=c['expected_devices']//c['expected_processes']
    if not local or c['expected_devices']%c['expected_processes'] or c['inference_batch']%local or a['games']%c['inference_batch']:
        raise ValueError('Inference batches must divide games and local devices')
    if not 1<=a['size']<=25 or not 1<=a['games']<=4096 or not 1<=a['history']<=64 or a['actor_offset']!=0:
        raise ValueError('Actor dimensions exceed benchmark scope')
    controllers=c['controller_cpus'];workers=a['worker_cpus']
    if not controllers or len(set(controllers))!=len(controllers) or set(controllers)&set(workers):
        raise ValueError('Controller and native workers need separate CPU sets')
    return c


def run(args,c,report,allowed):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.native import Actors
    import model
    rank=jax.process_index();world=jax.process_count()
    if world!=c['expected_processes'] or jax.device_count()!=c['expected_devices'] or jax.default_backend()!=c['platform']:
        raise ValueError('Discovered topology differs')
    host=int(os.environ.get('GOZERO_HOST_RANK','0'))
    mesh=Mesh(np.array(jax.local_devices()),('data',));rep=NamedSharding(mesh,P());batch_shard=NamedSharding(mesh,P('data'))
    a=dict(c['actors']);a['actor_offset']=rank*a['games'];size=a['size'];channels=2*a['history']+4;actions=size*size+1
    receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=ROOT.name: raise ValueError('Native source differs')
    initial=model.initialize(c['seed'],channels,c['model']);leaves,tree=jax.tree.flatten(initial)
    artifact=c['model_artifact'];network=0
    if artifact is not None:
        workspace=ROOT.parents[2]
        source=workspace/'.gozero/snapshots'/artifact['training_snapshot'];manifest=verify(source)
        if sha256(source/manifest['recipe']/'model.py')!=sha256(ROOT/'research/recipes/rollout_batching/model.py'):
            raise ValueError('Benchmark model implementation differs from checkpoint')
        trained=read_json(source/'resolved_config.json')
        if trained['model']!=c['model'] or any(trained['actors'][k]!=a[k] for k in ('size','komi','history','scoring')):
            raise ValueError('Model observation/scoring contract differs')
        path=workspace/artifact['export_path_template'].format(host_rank=host)
        if sha256(path)!=artifact['export_sha256']: raise ValueError('Model export hash differs')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={f'p_{i:04d}' for i in range(len(leaves))}: raise ValueError('Model parameter tree differs')
            saved=[]
            for i,expected in enumerate(leaves):
                value=data[f'p_{i:04d}']
                if value.shape!=expected.shape or str(value.dtype)!=str(expected.dtype) or not np.isfinite(value).all():
                    raise ValueError('Invalid model parameter')
                saved.append(value)
        initial=tree.unflatten(saved);network=artifact['network_version']
    params=jax.device_put(jax.tree.map(np.asarray,initial),rep)
    mb=c['inference_batch'];sample=np.zeros((mb,size,size,channels),np.float32)
    mode=c.get('dispatch_mode','serial');grouped_shard=NamedSharding(mesh,P(None,'data'))
    before=time.perf_counter()
    forward=jax.jit(lambda p,x:model.apply(p,x,c['model']),in_shardings=(rep,batch_shard),out_shardings=(batch_shard,batch_shard)).lower(
        params,jax.device_put(sample,batch_shard)).compile()
    jax.block_until_ready(forward(params,jax.device_put(sample,batch_shard)))
    grouped=None;wide=None
    if mode=='scan':
        grouped=jax.jit(lambda p,x:jax.lax.map(lambda block:model.apply(p,block,c['model']),x),
            in_shardings=(rep,grouped_shard),out_shardings=(grouped_shard,grouped_shard)).lower(
            params,jax.device_put(np.zeros((a['games']//mb,*sample.shape),np.float32),grouped_shard)).compile()
        jax.block_until_ready(grouped(params,jax.device_put(np.zeros((a['games']//mb,*sample.shape),np.float32),grouped_shard)))
    if mode=='wide':
        wide_sample=np.zeros((a['games'],size,size,channels),np.float32)
        wide=jax.jit(lambda p,x:model.apply(p,x,c['model']),in_shardings=(rep,batch_shard),out_shardings=(batch_shard,batch_shard)).lower(
            params,jax.device_put(wide_sample,batch_shard)).compile()
        jax.block_until_ready(wide(params,jax.device_put(wide_sample,batch_shard)))
    report.update(jax_rank=rank,host_rank=host,world_size=world,devices=[str(d) for d in jax.local_devices()],
                  compile_and_warmup_seconds=time.perf_counter()-before,compiler_cost_estimate=forward.cost_analysis(),native=receipt,
                  model_artifact=artifact,parameter_count=sum(x.size for x in jax.tree.leaves(params)),dispatch_mode=mode)
    if grouped is not None:report['grouped_compiler_cost_estimate']=grouped.cost_analysis()
    if wide is not None:report['wide_compiler_cost_estimate']=wide.cost_analysis()
    def evaluate(features,method):
        if method=='wide':return jax.device_get(wide(params,jax.device_put(features,batch_shard)))
        if method=='scan':
            shaped=features.reshape((a['games']//mb,mb,size,size,channels))
            pi,v=jax.device_get(grouped(params,jax.device_put(shaped,grouped_shard)))
            return pi.reshape((a['games'],actions)),v.reshape(a['games'])
        if method=='async':
            # All calls use the original executable and data placement. Submit
            # independent requests before materializing any result on the host.
            futures=[forward(params,jax.device_put(features[first:first+mb],batch_shard)) for first in range(0,a['games'],mb)]
            ready=jax.device_get(futures)
            return np.concatenate([x[0] for x in ready]),np.concatenate([x[1] for x in ready])
        logits=np.empty((a['games'],actions),np.float32);values=np.empty(a['games'],np.float32)
        for first in range(0,a['games'],mb):
            logits[first:first+mb],values[first:first+mb]=jax.device_get(forward(params,jax.device_put(features[first:first+mb],batch_shard)))
        return logits,values
    if c.get('verify_outputs',False):
        report['output_comparison']={'reference':'serial dispatch of the same microbatch executable','batches':0,'exact_batches':0,'max_logit_error':0.,'max_value_error':0.}
    request_hash=hashlib.sha256();target_hash=hashlib.sha256();game_hash=hashlib.sha256()
    def counters():
        return {key:0 for key in ('real_moves','neural_dispatches','neural_slots','host_materialization_calls','active_neural_evaluations','completed_games','truncated_games','eligible_rows',
                                 'verification_neural_slots','verification_neural_dispatches',
                                 'native_seconds','inference_seconds','verification_seconds')}
    counts=counters();report['counters']=counts
    os.sched_setaffinity(0,allowed)
    try:
        engine=Actors(a,args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
    finally:
        os.sched_setaffinity(0,c['controller_cpus'])
    with engine:
        masks=Counter()
        for path in Path('/proc/self/task').glob('*/status'):
            try:
                line=next(x for x in path.read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
                masks[line.split(':',1)[1].strip()]+=1
            except (FileNotFoundError,StopIteration):pass
        report['observed_thread_affinity_counts']=dict(masks)
        for turn in range(c['warmup_turns']+c['measured_turns']):
            if turn==c['warmup_turns']:
                mh.sync_global_devices('measurement-start')
                counts=counters();report['counters']=counts;start=time.perf_counter();cpu_start=time.process_time()
            before=time.perf_counter();pending=engine.start(network);counts['native_seconds']+=time.perf_counter()-before
            while pending.active_count:
                if c['verify_trace']:
                    before=time.perf_counter();request_hash.update(pending.features.tobytes());request_hash.update(pending.active.tobytes())
                    counts['verification_seconds']+=time.perf_counter()-before
                before=time.perf_counter()
                logits,values=evaluate(pending.features,mode)
                counts['neural_dispatches']+=1 if mode in ('scan','wide') else a['games']//mb
                counts['host_materialization_calls']+=a['games']//mb if mode=='serial' else 1
                counts['neural_slots']+=a['games']
                counts['inference_seconds']+=time.perf_counter()-before
                if c.get('verify_outputs',False):
                    before=time.perf_counter();ref_pi,ref_v=evaluate(pending.features,'serial');comparison=report['output_comparison']
                    counts['verification_neural_slots']+=a['games'];counts['verification_neural_dispatches']+=a['games']//mb
                    comparison['batches']+=1;exact=np.array_equal(logits,ref_pi) and np.array_equal(values,ref_v)
                    comparison['exact_batches']+=int(exact)
                    comparison['max_logit_error']=max(comparison['max_logit_error'],float(np.max(np.abs(logits-ref_pi))))
                    comparison['max_value_error']=max(comparison['max_value_error'],float(np.max(np.abs(values-ref_v))))
                    if not exact and 'first_difference' not in comparison:
                        path=args.output/'first_output_difference.npz'
                        np.savez(path,features=pending.features,reference_logits=ref_pi,reference_values=ref_v,logits=logits,values=values)
                        comparison['first_difference']={'turn':turn,'file':path.name,'sha256':sha256(path)}
                    counts['verification_seconds']+=time.perf_counter()-before
                counts['active_neural_evaluations']+=pending.active_count
                before=time.perf_counter();pending=engine.evaluate(pending,logits,values);counts['native_seconds']+=time.perf_counter()-before
            before=time.perf_counter();rows=engine.commit();counts['native_seconds']+=time.perf_counter()-before
            counts['real_moves']+=a['games'];counts['eligible_rows']+=len(rows.outcomes)
            for game in rows.games:counts['truncated_games' if game['truncated'] else 'completed_games']+=1
            if c['verify_trace']:
                before=time.perf_counter()
                for value in (rows.features,rows.policies,rows.outcomes,rows.metadata,rows.ownership):target_hash.update(value.tobytes())
                game_hash.update(canonical_json(rows.games));counts['verification_seconds']+=time.perf_counter()-before
            if (turn+1)%32==0:print(json.dumps({'kind':'rollout_progress','turn':turn+1,'rank':rank,**counts}),flush=True)
        report['elapsed_segment_seconds']=time.perf_counter()-start
        report['process_cpu_segment_seconds']=time.process_time()-cpu_start
        if c['verify_trace']:
            path=args.output/'actors.json.gz'
            with gzip.open(path,'wt',compresslevel=1) as f:f.write(engine.checkpoint())
            report['validation']={'request_features_sha256':request_hash.hexdigest(),'training_targets_sha256':target_hash.hexdigest(),
                                  'game_records_sha256':game_hash.hexdigest(),'actors_gzip_sha256':sha256(path),'includes_warmup':True}
    report['status']='passed';verify(ROOT);mh.sync_global_devices('rollouts-finished')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-receipt',type=Path,default=os.environ.get('GOZERO_NATIVE_RECEIPT'))
    args=p.parse_args();verify(ROOT);c=validate(read_json(args.config))
    if canonical_json(c)!=canonical_json(read_json(ROOT/'resolved_config.json')): raise ValueError('Configuration is not frozen')
    if args.native_receipt is None:p.error('Native receipt required')
    allowed=os.sched_getaffinity(0)
    if not set(c['controller_cpus']+c['actors']['worker_cpus'])<=allowed:raise ValueError('CPU contract exceeds process allowance')
    os.sched_setaffinity(0,c['controller_cpus'])
    os.environ['JAX_PLATFORMS']=c['platform'];args.native_receipt=Path(args.native_receipt).resolve()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'resolved_config.json').write_bytes(canonical_json(c))
    report={'schema_version':1,'kind':'fixed_network_rollout_batching','status':'running','snapshot_id':ROOT.name,
            'config_sha256':hashlib.sha256(canonical_json(c)).hexdigest(),'claims_go_strength':False,'claims_mfu':False,'platform':c['platform']}
    distributed=False
    try:
        import jax
        if c['platform']=='tpu':jax.distributed.initialize(initialization_timeout=90);distributed=True
        run(args,c,report,allowed)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        (args.output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k not in ('native','compiler_cost_estimate')}),flush=True)
        if distributed:jax.distributed.shutdown()


if __name__=='__main__':main()
