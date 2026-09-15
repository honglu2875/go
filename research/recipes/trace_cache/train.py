#!/usr/bin/env python3
"""Equal-prefix native/JAX rollout timing; the fixed model is never trained."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json,read_json,verify

MODES={'sequential':(1,1,'independent',False),
       'behavior':(4,4,'independent',False),
       'joint':(4,1,'shared',True)}
MODES.update({'cached_'+name:value for name,value in list(MODES.items())})


def validate(c):
    fields='schema_version platform seed games horizon samples controller_cpus actors model candidate expected_processes expected_devices qualification_moves_per_game timed_moves_per_game orders warmup_calls profile profile_moves_per_game matmul_precision'
    if set(c)!=set(fields.split()) or c['schema_version']!=1 or c['platform'] not in ('cpu','tpu'):
        raise ValueError('Unknown or missing benchmark configuration')
    def integer(x,lo,hi):return type(x) is int and lo<=x<=hi
    for name,lo,hi in [('seed',0,2**32-1),('games',1,256),('expected_processes',1,4),('expected_devices',1,16),
                       ('qualification_moves_per_game',1,2048),('timed_moves_per_game',1,2048),
                       ('warmup_calls',1,8),('profile_moves_per_game',1,256)]:
        if not integer(c[name],lo,hi):raise ValueError('Invalid '+name)
    if c['horizon']!=4 or c['samples']!=4:raise ValueError('This registered comparison uses horizon4 and forecast k4')
    if type(c['profile']) is not bool or c['profile_moves_per_game']>c['timed_moves_per_game']:
        raise ValueError('Invalid profiling scope')
    if not 1<=len(c['orders'])<=6 or any(sorted(order)!=sorted(MODES) for order in c['orders']):
        raise ValueError('Every repetition must contain every mode exactly once')
    if c['games']*c['timed_moves_per_game']*len(MODES)*len(c['orders'])>2**21:
        raise ValueError('Timing event archive exceeds scope')
    if c['expected_devices']%c['expected_processes'] or c['games']%(c['expected_devices']//c['expected_processes']):
        raise ValueError('Games must divide local device count')
    a=c['actors'];m=c['model']
    if c['matmul_precision']!='highest' or m['dtype']!='float32':
        raise ValueError('This numerical control requires float32 and highest matmul precision')
    if set(a)!=set('size komi history scoring max_game_moves root_legal_mask'.split()) or not a['root_legal_mask'] or a['scoring']!='pass_alive_area':
        raise ValueError('Unexpected native inference contract')
    if not integer(a['size'],1,25) or not integer(a['history'],1,64) or not integer(a['max_game_moves'],2,1024):
        raise ValueError('Invalid game dimensions')
    if m['size']!=a['size'] or m['komi']!=a['komi'] or m['max_tokens']!=a['max_game_moves']+c['horizon']+1 or m['board_mode']!='exact':
        raise ValueError('Exact-state model and actor contracts differ')
    if any(not isinstance(m[x],(int,float)) or not math.isfinite(m[x]) or not 0<=m[x]<=10 for x in ('expert_temperature','behavior_temperature')):
        raise ValueError('Invalid sampling temperatures')
    bytes_per_element=2 if m['dtype']=='bfloat16' else 4
    if c['games']*2*c['samples']*m['blocks']*2*m['max_tokens']*m['width']*bytes_per_element>2**30:
        raise ValueError('Expanded local KV exceeds 1 GiB')
    cpus=c['controller_cpus']
    if not isinstance(cpus,list) or not cpus or len(set(cpus))!=len(cpus) or any(not integer(x,0,65535) for x in cpus):
        raise ValueError('Invalid CPU affinity')
    return c


def write_payload(path,payload):
    path.write_bytes(gzip.compress(canonical_json(payload),compresslevel=1,mtime=0))
    path.chmod(0o444)
    return sha256(path)


def compare_events(left,right):
    differences=[]
    for game,(a,b) in enumerate(zip(left,right)):
        if a!=b:
            first=next((i for i,(x,y)in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            differences.append({'game':game,'first_event':first,'reference':a[first]if first<len(a)else None,'candidate':b[first]if first<len(b)else None})
    return differences


def play(native,execute,params,key,model_id,c,mode,target,output,label,synchronize=None,allocate_cache=None):
    import jax
    import numpy as np
    from cache_protocol import CacheCoordinator
    horizon,samples,_,_=MODES[mode];cached=mode.startswith('cached_')
    setup=time.perf_counter();engine=native.BoundedTraceActors(json.dumps(c['actors']),c['games'])
    coordinator=CacheCoordinator(c['games'],horizon,samples,c['actors']['size']**2+1,c['model']['max_tokens'],[model_id,model_id])if cached else None
    carry=allocate_cache(mode)if cached else None
    carry_bytes=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(carry))if cached else 0
    if cached:jax.block_until_ready(carry)
    setup=time.perf_counter()-setup
    if synchronize is not None:synchronize(label)
    remaining=np.full(c['games'],target,np.int64)
    records=[];counts=Counter(cache_prepare_seconds=0.,cache_commit_seconds=0.,cache_drain_seconds=0.,
        cache_reset_games=0,cache_root_append_slots=0,retained_cache_output_bytes=0)
    with jax.profiler.TraceAnnotation('gozero_packet_loop',mode=mode,target_moves_per_game=target):
        before=time.perf_counter();cpu_before=time.process_time()
        for packet in range(target):
            if not remaining.any():break
            active=remaining>0;allowances=np.minimum(remaining,horizon)
            mark=time.perf_counter()
            encoded,tokens,lengths,legal,stones=engine.start([model_id,model_id],[0,0],c['horizon'],active.tolist())
            tickets=json.loads(encoded)
            episodes=np.asarray([t['episode'] if t is not None else 0 for t in tickets],np.uint32)
            counts['native_start_and_frame_decode_seconds']+=time.perf_counter()-mark
            control=None
            if cached:
                mark=time.perf_counter();control=coordinator.prepare(tickets,tokens,lengths)
                counts['cache_prepare_seconds']+=time.perf_counter()-mark
                counts['cache_reset_games']+=int(((control[:,0]==1)&(control[:,1]==1)).sum())
            mark=time.perf_counter();arrays,carry=execute(mode,params,tokens,lengths,episodes,key,stones,legal,carry,control)
            counts['inference_and_transfer_seconds']+=time.perf_counter()-mark
            counts['output_array_bytes']+=sum(a.nbytes for a in arrays)
            counts['frame_array_bytes']+=sum(a.nbytes for a in (tokens,lengths,episodes,stones,legal))
            counts['input_array_bytes']+=sum(a.nbytes for a in (control if cached else tokens,lengths,episodes,stones,legal))
            counts['request_json_bytes']+=len(encoded.encode())
            counts['prefill_token_slots']+=0 if cached else tokens.size
            counts['cache_root_append_slots']+=c['games']if cached else 0
            counts['retained_cache_output_bytes']+=carry_bytes
            counts['active_prefix_tokens']+=int(((lengths+1)*active).sum())
            counts['inactive_game_slots']+=int((~active).sum())
            mark=time.perf_counter()
            response=engine.resolve(encoded,*[np.ascontiguousarray(a)for a in arrays],allowances.tolist())
            resolved=json.loads(response)
            counts['native_resolve_and_response_decode_seconds']+=time.perf_counter()-mark
            certificates=None
            if cached:
                mark=time.perf_counter();certificates=coordinator.commit(resolved,arrays[0])
                counts['cache_commit_seconds']+=time.perf_counter()-mark
            counts['response_json_bytes']+=len(response.encode())
            advances=np.asarray([len(r['actions'])for r in resolved],np.int64)
            if ((advances>allowances)|(active&(advances==0))|((~active)&(advances!=0))).any():
                raise ValueError('Native work allowance was not respected')
            remaining-=advances
            records.append({'packet':packet,'tickets':tickets,'allowances':allowances.tolist(),'resolutions':resolved,
                'cache_controls':control,'cache_commits':certificates})
        if cached:
            mark=time.perf_counter();jax.block_until_ready(carry);counts['cache_drain_seconds']+=time.perf_counter()-mark
        elapsed=time.perf_counter()-before;cpu=time.process_time()-cpu_before
    if remaining.any():raise ValueError('Equal-work execution failed to finish within its packet bound')
    final=json.loads(engine.inspect());events=[[]for _ in range(c['games'])]
    for record in records:
        if record['cache_controls']is not None:record['cache_controls']=record['cache_controls'].tolist()
        for game,(ticket,resolution)in enumerate(zip(record['tickets'],record['resolutions'])):
            counts['stop_'+resolution['stop']]+=1
            counts['legality_corrections']+=resolution['legality_corrections']
            counts['resolved_depth_'+str(len(resolution['actions']))]+=1
            if ticket is None:
                if resolution['actions'] or resolution['stop']!='work_limit' or resolution['terminal_white_score'] is not None:
                    raise ValueError('Inactive prediction changed a game or assigned an outcome')
                continue
            for depth,action in enumerate(resolution['actions']):
                last=depth==len(resolution['actions'])-1
                events[game].append({'episode':ticket['episode'],'ply':ticket['move_number']+depth,'action':action,
                    'ending':resolution['stop']if last and resolution['stop']in ('terminal','move_limit')else None,
                    'terminal_white_score':resolution['terminal_white_score']if last else None})
    if any(len(e)!=target for e in events):raise ValueError('Per-game accepted event count differs')
    counts.update(dispatches=len(records),real_moves=c['games']*target,
        board_encode_rows=len(records)*c['games']*2*samples*horizon,
        decoder_append_token_slots=len(records)*c['games']*2*samples*(horizon-1))
    artifact_start=time.perf_counter();path=output/(label+'.json.gz')
    digest=write_payload(path,{'events':events,'final_states':final,'packets':records})
    report={'mode':mode,'label':label,'target_moves_per_game':target,'elapsed_seconds':elapsed,
        'process_cpu_seconds':cpu,'average_active_cpu_cores':cpu/elapsed,'setup_seconds':setup,'retained_cache_bytes':carry_bytes,
        'artifact_seconds':time.perf_counter()-artifact_start,'artifact':path.name,'artifact_sha256':digest,
        'moves_per_second':c['games']*target/elapsed,'moves_per_game_per_dispatch':target/len(records),'counters':dict(counts)}
    print(json.dumps({'kind':'trace_work_finished','label':label,'mode':mode,'moves':report['counters']['real_moves'],
        'seconds':elapsed,'dispatches':len(records)}),flush=True)
    return report,events,final


def run(args,c,report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.causal_artifacts import validate as validate_candidate
    from gozero.native import load_library
    import model
    from decode import decode
    from cached_decode import decode_cached,empty_cache
    jax.config.update('jax_default_matmul_precision',c['matmul_precision'])
    host=int(os.environ.get('GOZERO_HOST_RANK','0'));rank=jax.process_index();world=jax.process_count()
    if world!=c['expected_processes'] or jax.device_count()!=c['expected_devices'] or jax.default_backend()!=c['platform']:
        raise ValueError('Unexpected execution topology')
    receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=ROOT.name:raise ValueError('Native source differs')
    native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
    if getattr(native,'BOUNDED_STATE_TRACE_ABI_VERSION',None)!=1:raise ValueError('Bounded trace ABI differs')
    descriptor_path=ROOT/c['candidate'];descriptor=read_json(descriptor_path)
    trained=validate_candidate(args.artifacts_root,descriptor)
    if descriptor['kind']!='board_causal_history_policy' or trained['model_code_sha256']!=sha256(Path(__file__).with_name('model.py')):
        raise ValueError('Execution model differs from trained exact-state model')
    architecture=lambda d:{k:v for k,v in d.items()if not k.endswith('_temperature') and k!='dtype'}
    if architecture(c['model'])!=architecture(trained['config']['model']):
        raise ValueError('Execution architecture differs from training')
    template=jax.eval_shape(lambda:model.initialize(trained['config']['seed'],c['model']))
    leaves,tree=jax.tree.flatten(template)
    schema=[{'path':jax.tree_util.keystr(p),'shape':list(v.shape),'dtype':str(v.dtype)}for p,v in jax.tree_util.tree_flatten_with_path(template)[0]]
    if schema!=trained['model_schema']:raise ValueError('Parameter schema differs')
    mesh=Mesh(np.array(jax.local_devices()),('games',));rep=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('games'))
    parameters=tree.unflatten([trained['arrays'][f'p_{i:04d}']for i in range(len(leaves))])
    params=jax.device_put(parameters,rep)
    for expected,actual in zip(jax.tree.leaves(parameters),jax.tree.leaves(jax.device_get(params))):
        if expected.dtype!=actual.dtype or expected.shape!=actual.shape or expected.tobytes()!=actual.tobytes():
            raise ValueError('Device transfer changed trained parameter elements')
    parameter_hash=hashlib.sha256(b''.join(x.tobytes()for x in jax.tree.leaves(parameters))).hexdigest()
    model_id=int(parameter_hash[:16],16)
    report.update(host_rank=host,jax_rank=rank,world_size=world,jax_version=jax.__version__,native=receipt,
        local_devices=[str(d)for d in jax.local_devices()],global_devices=[str(d)for d in jax.devices()],
        candidate=descriptor,candidate_sha256=sha256(descriptor_path),parameter_elements_sha256=parameter_hash,
        parameter_count=sum(v.size for v in trained['arrays'].values()),rng_host_rank=host,
        training_dtype=trained['config']['model']['dtype'],execution_dtype=c['model']['dtype'],
        execution_matmul_precision=jax.config.jax_default_matmul_precision,
        inference_mesh_scope='Local devices only; cross-host barriers occur between segments.')
    actions=c['actors']['size']**2+1;games=c['games'];width=c['model']['max_tokens']
    tokens=np.full((games,width),actions+1,np.int32);tokens[:,0]=actions
    host_dummy=(tokens,np.zeros(games,np.int32),np.zeros(games,np.uint32),np.zeros((games,actions-1),np.uint8),np.ones((games,actions),np.bool_))
    key=jax.device_put(jax.random.fold_in(jax.random.key(c['seed']),host),rep)
    dummy=(params,jax.device_put(host_dummy[0],batched),jax.device_put(host_dummy[1],batched),
           jax.device_put(host_dummy[2],batched),key,jax.device_put(host_dummy[3],batched),jax.device_put(host_dummy[4],batched))
    executables={};initial_carries={};report['compilation']={}
    for name,(horizon,samples,coupling,oracle)in MODES.items():
        before=time.perf_counter()
        cached=name.startswith('cached_')
        if cached:
            if samples not in initial_carries:
                initial_carries[samples]=jax.jit(lambda:empty_cache(c['model'],games,samples),out_shardings=batched)()
                jax.block_until_ready(initial_carries[samples])
            controls=np.full((games,5+horizon),-1,np.int32);controls[:,:5]=[1,1,0,actions,0]
            current_dummy=(params,initial_carries[samples],jax.device_put(controls,batched),*dummy[2:])
            compiled=jax.jit(lambda p,state,ct,l,e,k,s,legal:decode_cached(p,state,ct,l,e,k,s,c['model'],
                horizon=horizon,samples=samples,coupling=coupling,oracle_behavior=oracle,root_legal=legal),
                in_shardings=(rep,batched,batched,batched,batched,rep,batched,batched),
                out_shardings=(batched,)*5).lower(*current_dummy).compile()
        else:
            current_dummy=dummy
            compiled=jax.jit(lambda p,t,l,e,k,s,legal:decode(p,t,l,e,k,s,c['model'],horizon=horizon,samples=samples,
                coupling=coupling,oracle_behavior=oracle,root_legal=legal),
                in_shardings=(rep,batched,batched,batched,rep,batched,batched),out_shardings=(batched,)*4).lower(*dummy).compile()
        hlo=args.output/(name+'.hlo.txt.gz');hlo.write_bytes(gzip.compress(compiled.as_text().encode(),compresslevel=1,mtime=0))
        cost=compiled.cost_analysis();cost_path=args.output/(name+'.cost.json');cost_path.write_bytes(canonical_json(cost))
        for _ in range(c['warmup_calls']):
            warm=compiled(*current_dummy);jax.device_get(warm[:4])
            if cached:jax.block_until_ready(warm[4])
        executables[name]=compiled
        report['compilation'][name]={'seconds_including_warmup':time.perf_counter()-before,'hlo_sha256':sha256(hlo),
            'cost_sha256':sha256(cost_path),'compiler_estimates':{k:cost[k]for k in ('flops','bytes accessed','transcendentals')if k in cost}}
    def execute(mode,p,t,l,e,k,s,legal,carry,control):
        tail=(jax.device_put(l,batched),jax.device_put(e,batched),k,jax.device_put(s,batched),jax.device_put(legal,batched))
        if mode.startswith('cached_'):
            result=executables[mode](p,carry,jax.device_put(control,batched),*tail)
            return jax.device_get(result[:4]),result[4]
        return jax.device_get(executables[mode](p,jax.device_put(t,batched),*tail)),None
    def allocate_cache(mode):return initial_carries[MODES[mode][1]]
    sync=(lambda label:mh.sync_global_devices('trace-throughput-'+label))if world>1 else None
    def run_mode(mode,target,label,stream):
        draw=jax.device_put(jax.random.fold_in(key,stream),rep)
        return play(native,execute,params,draw,model_id,c,mode,target,args.output,label,synchronize=sync,allocate_cache=allocate_cache)
    report['qualification']=[];reference=None;reference_final=None
    for mode in MODES:
        row,events,final=run_mode(mode,c['qualification_moves_per_game'],'qualification-'+mode,0)
        if reference is None:reference=events;reference_final=final
        differences=compare_events(reference,events);row['first_differences']=differences[:8]
        row['sequential_events_exact']=not differences;row['final_native_states_exact']=final==reference_final
        report['qualification'].append(row)
    if any(not r['sequential_events_exact']or not r['final_native_states_exact']for r in report['qualification']):
        raise ValueError('TPU/CPU numerical execution changed the equal-work reference')
    report['timings']=[];first_reference=None
    for repetition,order in enumerate(c['orders']):
        results={}
        for mode in order:
            row,events,final=run_mode(mode,c['timed_moves_per_game'],f'timing-{repetition}-{mode}',repetition+1)
            row['repetition']=repetition;report['timings'].append(row);results[mode]=(row,events,final)
        reference=results['sequential'][1];reference_final=results['sequential'][2]
        if repetition==0:first_reference=reference
        for mode,(row,events,final)in results.items():
            differences=compare_events(reference,events);row['first_differences']=differences[:8]
            row['sequential_events_exact']=not differences;row['final_native_states_exact']=final==reference_final
        if any(not row['sequential_events_exact']or not row['final_native_states_exact']for row,_,_ in results.values()):
            raise ValueError('Timed execution changed the common sequential events or inactive final state')
    report['profile']={'status':'disabled'}
    if c['profile']:
        if sync is not None:sync('profile-start')
        if host==0:
            profile_dir=args.output/'profile';profile_dir.mkdir()
            try:
                jax.profiler.start_trace(str(profile_dir),create_perfetto_trace=False)
            except Exception as error:
                report['profile']={'status':'failed_to_start','error':repr(error),'host_rank':host}
            else:
                stop_error=None
                try:
                    profiled,events,_=play(native,execute,params,jax.device_put(jax.random.fold_in(key,1),rep),
                        model_id,c,'cached_behavior',c['profile_moves_per_game'],args.output,'profile-cached_behavior',allocate_cache=allocate_cache)
                finally:
                    try:jax.profiler.stop_trace()
                    except Exception as error:stop_error=repr(error)
                expected=[e[:c['profile_moves_per_game']]for e in first_reference]
                if compare_events(expected,events):raise ValueError('Profiled work differs from the registered reference prefix')
                report['profile']={'status':'captured'if stop_error is None else 'failed_to_stop','error':stop_error,'host_rank':host,'run':profiled,
                    'files':{str(p.relative_to(args.output)):sha256(p)for p in profile_dir.rglob('*')if p.is_file()}}
        else:report['profile']={'status':'host0_only'}
        if sync is not None:sync('profile-end')
    report['status']='passed';verify(ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--native-receipt',type=Path)
    p.add_argument('--artifacts-root',type=Path,default=ROOT.parents[2]);args=p.parse_args();verify(ROOT)
    c=validate(read_json(args.config))
    if c!=read_json(ROOT/'resolved_config.json'):raise ValueError('Configuration is not frozen')
    if not set(c['controller_cpus'])<=os.sched_getaffinity(0):raise ValueError('Unavailable controller CPUs')
    os.sched_setaffinity(0,c['controller_cpus']);os.environ['JAX_PLATFORMS']=c['platform']
    if args.native_receipt is None:
        name=os.environ.get('GOZERO_NATIVE_RECEIPT')
        if not name:raise ValueError('Native receipt missing')
        args.native_receipt=Path(name)
    args.native_receipt=args.native_receipt.resolve();args.artifacts_root=args.artifacts_root.resolve()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'kind':'retained_kv_trace_throughput','status':'running','snapshot_id':ROOT.name,
        'config_sha256':sha256(ROOT/'resolved_config.json'),'started_unix':time.time(),'no_training_performed':True,
        'claims_go_strength':False,'claims_mfu':False,'claims_mcts_equivalence':False,
        'measurement_scope':'Complete repeated equal-prefix packet loops, including necessary Python/native scheduling and logical transfers. Compilation, final-state inspection, event reconstruction and artifact writes are outside measured segments and inside total attempt time.'}
    distributed=False
    try:
        import jax
        if c['platform']=='tpu':jax.distributed.initialize(initialization_timeout=90);distributed=True
        run(args,c,report)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();(args.output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items()if k in ('kind','status','snapshot_id','host_rank','jax_rank','error','started_unix','finished_unix')}),flush=True)
        if distributed:jax.distributed.shutdown()


if __name__=='__main__':main()
