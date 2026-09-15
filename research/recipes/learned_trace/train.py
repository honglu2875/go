#!/usr/bin/env python3
"""Execute a frozen trained causal policy with forecast and known-policy modes."""
import argparse
from collections import Counter
import hashlib
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


def validate(c):
    def fields(x,names):
        if not isinstance(x,dict) or set(x)!=set(names.split()):raise ValueError('Unknown or missing configuration fields')
    def integer(x,lo,hi):return type(x) is int and lo<=x<=hi
    def real(x,lo,hi):return type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi
    fields(c,'schema_version platform seed games packets horizon samples controller_cpus actors model candidate')
    if c['schema_version']!=1 or c['platform']!='cpu':raise ValueError('This qualification is CPU-only')
    for key,lo,hi in [('seed',0,2**32-3),('games',1,32),('packets',1,256),('horizon',2,8),('samples',1,16)]:
        if not integer(c[key],lo,hi):raise ValueError('Invalid '+key)
    if c['horizon']%2:raise ValueError('Paired trace horizon must be 2N')
    a=c['actors'];m=c['model']
    fields(a,'size komi history scoring max_game_moves root_legal_mask')
    if type(a['root_legal_mask']) is not bool:raise ValueError('Root masking must be explicit')
    fields(m,'size komi width heads blocks max_tokens dtype expert_temperature behavior_temperature')
    if not integer(a['size'],1,25) or not integer(a['history'],1,64) or not integer(a['max_game_moves'],2,1024):
        raise ValueError('Invalid actor dimensions')
    if not real(a['komi'],-100,100) or a['scoring'] not in ('raw_area','pass_alive_area'):raise ValueError('Invalid rules')
    if m['size']!=a['size'] or m['komi']!=a['komi'] or m['max_tokens']!=a['max_game_moves']+c['horizon']+1:
        raise ValueError('Model and actor input contracts differ')
    for key,lo,hi in [('width',8,256),('heads',1,32),('blocks',1,16)]:
        if not integer(m[key],lo,hi):raise ValueError('Invalid model '+key)
    if m['width']%m['heads'] or m['dtype'] not in ('float32','bfloat16'):raise ValueError('Invalid attention configuration')
    if any(not real(m[k],0,10) for k in ('expert_temperature','behavior_temperature')):raise ValueError('Invalid sampler temperature')
    # Root prefill is broadcast to every view/sample, including its padded KV.
    kv_bytes=c['games']*2*c['samples']*m['blocks']*2*m['max_tokens']*m['width']*4
    if kv_bytes>2**30:raise ValueError('Expanded KV exceeds 1 GiB probe limit')
    cpus=c['controller_cpus']
    if not isinstance(cpus,list) or not cpus or len(set(cpus))!=len(cpus) or any(not integer(x,0,65535) for x in cpus):
        raise ValueError('Invalid CPU affinity')
    return c


def run(args,c,report):
    import json
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gozero.native import load_library
    import model
    receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=ROOT.name:raise ValueError('Native source differs')
    library=args.native_receipt.parent/receipt['filename']
    native=load_library(library,receipt['binary_sha256'])
    if getattr(native,'DUAL_TRACE_ABI_VERSION',None)!=2:raise ValueError('Trace frame ABI differs')
    report['native']=receipt;report['jax_version']=jax.__version__
    if jax.default_backend()!='cpu' or jax.process_count()!=1:raise ValueError('Unexpected execution topology')
    from gozero.causal_artifacts import validate as validate_candidate
    descriptor_path=(ROOT/c['candidate']).resolve()
    if not descriptor_path.is_relative_to(ROOT):raise ValueError('Candidate descriptor must be frozen')
    descriptor=read_json(descriptor_path)
    trained=validate_candidate(args.artifacts_root,descriptor)
    if trained['model_code_sha256']!=sha256(Path(__file__).with_name('model.py')):
        raise ValueError('Decoder model implementation differs from training')
    original_model=trained['config']['model']
    if {k:v for k,v in c['model'].items() if not k.endswith('_temperature')}!={k:v for k,v in original_model.items() if not k.endswith('_temperature')}:
        raise ValueError('Decoder architecture differs from trained model')
    initial=model.initialize(trained['config']['seed'],c['model'])
    leaves,definition=jax.tree.flatten(initial)
    schema=[{'path':jax.tree_util.keystr(path),'shape':list(a.shape),'dtype':str(a.dtype)} for path,a in jax.tree_util.tree_flatten_with_path(initial)[0]]
    if schema!=trained['model_schema']:raise ValueError('Decoder parameter tree differs')
    params=definition.unflatten([jnp.asarray(trained['arrays'][f'p_{i:04d}']) for i in range(len(leaves))])
    untrained_behavior={k:(initial[k] if k.startswith('behavior') else v) for k,v in params.items()}
    key=jax.random.key(c['seed']+1)
    leaves=jax.tree.leaves(params)
    export=args.output/'loaded_parameters.npz'
    np.savez(export,**{f'p_{i:04d}':np.asarray(a) for i,a in enumerate(leaves)})
    report['parameter_count']=sum(x.size for x in leaves);report['parameter_sha256']=sha256(export)
    # Device transfers can change array strides and therefore NPY layout. The
    # trained file is independently hash-verified; require every loaded value,
    # dtype and canonical element byte to match its committed array exactly.
    for i,a in enumerate(leaves):
        actual=np.asarray(a);expected=trained['arrays'][f'p_{i:04d}']
        if actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes(order='C')!=expected.tobytes(order='C'):
            raise ValueError('Loaded parameter elements differ from trained export')
    report['trained_parameter_elements_exact']=True
    report['candidate']=descriptor;report['candidate_sha256']=sha256(descriptor_path)
    control_export=args.output/'untrained_behavior_parameters.npz'
    np.savez(control_export,**{f'p_{i:04d}':np.asarray(a) for i,a in enumerate(jax.tree.leaves(untrained_behavior))})
    report['untrained_behavior_parameter_sha256']=sha256(control_export)
    fake_tokens=np.full((c['games'],c['model']['max_tokens']),c['actors']['size']**2+2,np.int32)
    fake_tokens[:,0]=c['actors']['size']**2+1
    dummy=(params,fake_tokens,np.zeros(c['games'],np.int32),np.zeros(c['games'],np.uint32),key,
           np.ones((c['games'],c['actors']['size']**2+1),np.bool_))
    executables={};compiles={}
    modes=[('sequential',1,1,'independent',False),
           ('behavior_independent',c['horizon'],c['samples'],'independent',False),
           ('behavior_untrained',c['horizon'],c['samples'],'independent',False),
           ('known_policy_independent',c['horizon'],c['samples'],'independent',True),
           ('known_policy_joint',c['horizon'],1,'shared',True)]
    for name,horizon,samples,coupling,oracle in modes:
        before=time.perf_counter()
        if name=='behavior_untrained':
            executables[name]=executables['behavior_independent']
            compiles[name]={'same_executable_as':'behavior_independent'}
            continue
        function=jax.jit(lambda p,t,l,e,k,legal:model.decode(p,t,l,e,k,c['model'],horizon=horizon,
            samples=samples,coupling=coupling,oracle_behavior=oracle,
            root_legal=legal if c['actors']['root_legal_mask'] else None))
        lowered=function.lower(*dummy)
        hlo=args.output/(name+'.hlo.txt');hlo.write_text(lowered.compiler_ir('hlo').as_hlo_text())
        executable=lowered.compile();jax.block_until_ready(executable(*dummy))
        executables[name]=executable
        compiles[name]={'seconds':time.perf_counter()-before,'hlo_sha256':sha256(hlo),
                        'cost_estimate':executable.cost_analysis()}
    report['compilation']=compiles
    def play(name,targets=None):
        engine=native.DualTraceActors(json.dumps(c['actors']),c['games'])
        events=[[] for _ in range(c['games'])];counts=Counter();records=[]
        before=time.perf_counter();cpu_before=time.process_time()
        _,horizon,samples,_,_=next(m for m in modes if m[0]==name)
        parameters=untrained_behavior if name=='behavior_untrained' else params
        parameter_sha=report['untrained_behavior_parameter_sha256' if name=='behavior_untrained' else 'parameter_sha256']
        model_id=int(parameter_sha[:16],16)
        limit=c['packets'] if targets is None else max(map(len,targets))
        for packet in range(limit):
            if targets is not None and all(len(e)>=len(t) for e,t in zip(events,targets)):break
            start=time.perf_counter()
            tickets_text,tokens,lengths,legal=engine.start([model_id,model_id],[0,0],c['horizon'])
            tickets=json.loads(tickets_text)
            episodes=np.asarray([t['episode'] for t in tickets],np.uint32)
            counts['native_start_seconds']+=time.perf_counter()-start
            start=time.perf_counter()
            outputs=jax.device_get(executables[name](parameters,tokens,lengths,episodes,key,legal))
            counts['inference_and_transfer_seconds']+=time.perf_counter()-start
            counts['output_bytes']+=sum(a.nbytes for a in outputs)
            counts['prefill_token_slots']+=int(tokens.size)
            counts['valid_prefix_tokens']+=int((lengths+1).sum())
            counts['decoder_append_token_slots']+=c['games']*2*samples*(horizon-1)
            for game in range(c['games']):
                for view in range(2):
                    depth=0 if view!=int(lengths[game])%2 else 1
                    if depth<horizon:
                        counts['first_forecast_samples']+=samples
                        counts['first_forecast_distinct_actions']+=len(set(map(int,outputs[0][game,view,:,depth])))
            start=time.perf_counter()
            resolved=json.loads(engine.resolve(tickets_text,*[np.ascontiguousarray(a) for a in outputs]))
            counts['native_resolve_seconds']+=time.perf_counter()-start
            counts['dispatches']+=1
            for game,(ticket,result) in enumerate(zip(tickets,resolved)):
                if not result['actions']:raise ValueError('Every live root must advance')
                counts['real_moves']+=len(result['actions']);counts['legality_corrections']+=result['legality_corrections']
                counts['stop_'+result['stop']]+=1
                counts['resolved_depth_'+str(len(result['actions']))]+=1
                for depth,action in enumerate(result['actions']):
                    last=depth+1==len(result['actions'])
                    ending=result['stop'] if last and result['stop'] in ('terminal','move_limit') else None
                    events[game].append({'episode':ticket['episode'],'ply':ticket['move_number']+depth,
                        'action':action,'ending':ending,'terminal_white_score':result['terminal_white_score'] if last else None})
            records.append({'packet':packet,'tickets':tickets,'resolutions':resolved})
        counts['elapsed_seconds']=time.perf_counter()-before;counts['process_cpu_seconds']=time.process_time()-cpu_before
        path=args.output/(name+'-events.json');path.write_bytes(canonical_json({'events':events,'packets':records}))
        counts['moves_per_game_per_dispatch']=counts['real_moves']/(c['games']*counts['dispatches'])
        counts['event_sha256']=sha256(path)
        counts['parameter_sha256']=parameter_sha
        return events,dict(counts)
    report['modes']={};all_targets=[]
    for name,*_ in modes[1:]:
        events,counts=play(name);all_targets.append((name,events))
        report['modes'][name]=counts
        print(json.dumps({'kind':'dual_trace_probe','mode':name,**counts}),flush=True)
    longest=[max((events[g] for _,events in all_targets),key=len) for g in range(c['games'])]
    baseline,counts=play('sequential',longest);report['modes']['sequential']=counts
    for name,events in all_targets:
        differences=[]
        for game,(actual,reference) in enumerate(zip(events,baseline)):
            if actual!=reference[:len(actual)]:
                first=next((i for i,(a,b) in enumerate(zip(actual,reference)) if a!=b),min(len(actual),len(reference)))
                differences.append({'game':game,'first_event':first})
        report['modes'][name]['sequential_prefix_exact']=not differences
        report['modes'][name]['first_differences']=differences
    verify(ROOT)
    report['status']='passed' if all(report['modes'][n]['sequential_prefix_exact'] for n,_ in all_targets) else 'failed'
    if report['status']!='passed':raise ValueError('Paired traces changed the sequential action stream')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--artifacts-root',type=Path,required=True)
    args=parser.parse_args();verify(ROOT);c=validate(read_json(args.config))
    if canonical_json(c)!=canonical_json(read_json(ROOT/'resolved_config.json')):raise ValueError('Configuration is not frozen')
    if not set(c['controller_cpus'])<=os.sched_getaffinity(0):raise ValueError('Unavailable CPU affinity')
    os.sched_setaffinity(0,c['controller_cpus']);os.environ['JAX_PLATFORMS']=c['platform']
    args.native_receipt=args.native_receipt.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'resolved_config.json').write_bytes(canonical_json(c))
    report={'schema_version':1,'kind':'paired_causal_policy_qualification','status':'running','snapshot_id':ROOT.name,
            'config_sha256':hashlib.sha256(canonical_json(c)).hexdigest(),'started_unix':time.time(),
            'claims_go_strength':False,'claims_training':False,'claims_mfu':False,
            'limitations':['Fixed MCTS-distilled student and same own-policy parameters for both players.',
                'Policy-only legality-corrected execution; no MCTS decisions are performed in this probe.',
                'CPU correctness probe; timings include instrumentation and concurrent work.',
                'Known-policy joint continuation uses both self-play policies and their actual counter-based draws; external opponent forecasts do not have this information.',
                'Known-policy independent uses true policy probabilities but independently forecasts the other player random draw.',
                'Untrained-behavior control replaces only the behavior parameters; the own policy stays fixed.',
                'KV stays on device within a packet; each new packet prefills the actual full tape.',
                'Native batch resolver is serial in this prototype; no CPU scaling claim.']}
    try:run(args,c,report)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();(args.output/'result.json').write_bytes(canonical_json(report))
        print(__import__('json').dumps({k:v for k,v in report.items() if k not in ('compilation','native')}),flush=True)


if __name__=='__main__':main()
