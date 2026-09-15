"""Evaluate pinned completed checkpoints on the original validation population."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero import checkpoints
from gozero.model_artifacts import artifact

def implementation(model):
    import katago,causal,linear_reference
    return katago if model['architecture']=='katago_nested_policy' else linear_reference if model['encoder']=='overlap_linear' else causal

def array_digest(values):
    digest=hashlib.sha256()
    for key,value in sorted(values.items()):
        digest.update(canonical_json([key,list(value.shape),str(value.dtype)]));digest.update(value.tobytes(order='C'))
    return digest.hexdigest()

def logits(params,batch,model):
    module=implementation(model)
    if model['architecture']=='causal_visual_policy':
        return module.forward(params,batch['spatial'],batch['global_features'],batch['actions'],batch['counts'],model)
    n,t,h,w,ch=batch['spatial'].shape
    return module.forward(params,batch['spatial'].reshape(n*t,h,w,ch),batch['global_features'].reshape(n*t,-1),model).reshape(n,t,-1)

def owner_parameters(root,case):
    """Read p_ arrays only; the normal reader still authenticates the whole file."""
    source=artifact(root,'.gozero/snapshots/'+case['training_snapshot']);manifest=verify(source)
    config=read_json(source/'resolved_config.json')
    if config['kind']!='fixed_policy_learning' or config['model']!=case['model']:raise ValueError('Training model differs')
    result_path=artifact(root,case['training_result_path'])
    if checkpoints.sha256(result_path)!=case['training_result_sha256']:raise ValueError('Training result changed')
    result=read_json(result_path)
    if (result['status']!='passed' or not result['training_complete'] or result['turn']!=1024 or result['host_rank']!=0
            or result['snapshot_id']!=source.name or result['config_sha256']!=checkpoints.sha256(source/'resolved_config.json')):
        raise ValueError('Expected completed owner checkpoint')
    saved=result['latest_checkpoint'];path=artifact(root,saved['owner_checkpoint_path']);group_path=path.with_suffix('.group.json')
    if checkpoints.sha256(group_path)!=saved['group_sha256']:raise ValueError('Checkpoint group changed')
    group=read_json(group_path)
    if (group['snapshot_id']!=source.name or group['turn']!=1024 or group['host_manifests']['0']!=saved['manifest_sha256']
            or group['config_sha256']!=result['config_sha256'] or artifact(root,group['owner_checkpoint_path'])!=path):
        raise ValueError('Checkpoint identity differs')
    state,arrays,_=checkpoints.read(path,expected_manifest_sha256=saved['manifest_sha256'],array_prefix='p_')
    schema=case['model_schema']
    if (state['model_schema']!=schema or result['model_schema']!=schema or state['snapshot_id']!=source.name
            or state['turn']!=1024 or state['host_rank']!=0 or not state['owns_replicated_arrays']
            or state['config_sha256']!=group['config_sha256']):
        raise ValueError('Saved model state differs')
    if state['dataset_manifest_sha256']!=config['dataset']['manifest_sha256']:raise ValueError('Dataset differs')
    if set(arrays)!={f'p_{i:04d}' for i in range(len(schema))}:raise ValueError('Parameter coverage differs')
    params={}
    for i,item in enumerate(schema):
        value=arrays[f'p_{i:04d}']
        if list(value.shape)!=item['shape'] or str(value.dtype)!=item['dtype'] or not np.isfinite(value).all():raise ValueError('Invalid parameter array')
        params[item['path']]=value
    filename='katago.py' if case['model']['architecture']=='katago_nested_policy' else 'causal.py'
    if checkpoints.sha256(source/manifest['recipe']/filename)!=case['model_code_sha256']:raise ValueError('Original model code differs')
    return params

def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json') or c['kind']!='paired_policy_evaluation':raise ValueError('Unfrozen evaluation')
    if c['evaluation']!={'split':1,'games_per_host':32,'games_per_bucket':10000}:raise ValueError('Validation population changed')
    os.environ['JAX_PLATFORMS']=c['platform']
    from train_policy import publish
    args.output.mkdir(parents=True,exist_ok=False);publish(args.output/'resolved_config.json',c)
    report={'kind':c['kind'],'snapshot_id':SOURCE.name,'status':'running','started_unix':time.time()}
    import jax
    try:
        jax.distributed.initialize(initialization_timeout=90)
        run(c,args.output,report);report['status']='passed';verify(SOURCE)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();publish(args.output/'result.json',report);jax.distributed.shutdown()

def run(c,output,report):
    import jax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from jax.experimental import multihost_utils as mh
    from gozero.katago_sequence_batches import Dataset
    from train_policy import publish
    import game_metrics
    rank,world=jax.process_index(),jax.process_count();host=int(os.environ['GOZERO_HOST_RANK'])
    if world!=4 or len(jax.devices())!=16:raise ValueError('Unexpected topology')
    root=SOURCE.parents[2];mesh=Mesh(np.asarray(jax.devices()),('data',));replicated=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);selected=data.bucket_entries(c['dataset']['buckets'],split=1)
    report.update(host_rank=host,jax_rank=rank,dataset_manifest_sha256=c['dataset']['manifest_sha256'],cases=[])
    for case in c['cases']:
        name=case['label'];model=case['model'];module=implementation(model)
        if checkpoints.sha256(Path(module.__file__))!=case['model_code_sha256'] or module.parameter_schema(model)!=case['model_schema']:
            raise ValueError('Evaluator model implementation differs from trained source')
        start=time.perf_counter()
        values=owner_parameters(root,case) if host==0 else {s['path']:np.zeros(s['shape'],s['dtype']) for s in case['model_schema']}
        original_digest=array_digest(values) if host==0 else None
        values=mh.broadcast_one_to_all(values,is_source=host==0);digest=array_digest(values)
        if host==0 and digest!=original_digest:raise ValueError('Broadcast changed parameter bytes')
        params=jax.device_put(values,replicated);jax.block_until_ready(params);del values
        rows=[];all_ids=[];compiled={};profiles={}
        for bucket in c['dataset']['buckets']:
            entries=selected['expert',bucket][:10000];all_ids.extend(entries);local=entries[rank::world]
            for begin in range(0,math.ceil(len(entries)/world),32):
                chosen=local[begin:begin+32];raw=data.batch(chosen+[None]*(32-len(chosen)),positions=bucket)
                batch={k:jax.make_array_from_process_local_data(batched,v) for k,v in raw.items()}
                if bucket not in compiled:
                    fn=jax.shard_map(lambda p,b:game_metrics.totals(logits(p,b,model),b),mesh=mesh,
                        in_specs=(P(),{k:P('data') for k in batch}),out_specs=P('data'),check_vma=False)
                    lowered=jax.jit(fn).lower(params,batch);compiled[bucket]=lowered.compile()
                    hlo=lowered.compiler_ir('hlo').as_hlo_text();path=output/f'{name}-{bucket}.hlo'
                    with path.open('x') as f:f.write(hlo)
                    path.chmod(0o444);profiles[str(bucket)]={'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest()}
                totals=compiled[bucket](params,batch)
                local_totals=jax.tree.map(lambda value:np.concatenate([np.asarray(s.data) for s in sorted(value.addressable_shards,key=lambda s:s.index[0].start)],axis=0),totals)
                for i,entry in enumerate(chosen):
                    row={'entry':entry,'bucket':bucket,**{key:float(value[i]) for key,value in local_totals.items()}}
                    if any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):raise ValueError('Nonfinite game metric')
                    rows.append(row)
        ids=hashlib.sha256(canonical_json(all_ids)).hexdigest()
        if ids!=case['validation_episode_ids_sha256']:raise ValueError('Validation episode identity changed')
        result={'label':name,'parameter_elements_sha256':digest,'episode_ids_sha256':ids,'rows':rows,'profiles':profiles,
            'elapsed_seconds':time.perf_counter()-start}
        publish(output/f'{name}.json',result);report['cases'].append({k:v for k,v in result.items() if k!='rows'})
        print(json.dumps({'kind':'paired_evaluation_case','label':name,'host':host,'games':len(rows),'seconds':result['elapsed_seconds']}),flush=True)
        del params,compiled,totals,batch,raw;gc.collect();jax.clear_caches()
    mh.sync_global_devices('paired-validation-complete')
