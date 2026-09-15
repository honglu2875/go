"""ABBA comparison of complete cached move graphs, preserving model arithmetic."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.katago_sequence_batches import Dataset


def entry(args):
    c=read_json(args.config);verify(SOURCE)
    if c!=read_json(SOURCE/'resolved_config.json') or c['kind']!='cached_decode_qualification':raise ValueError('Unfrozen cache comparison')
    if c['cases']!=['reference','carried','carried','reference']:raise ValueError('Unregistered case order')
    os.environ['JAX_PLATFORMS']='tpu'
    import causal
    causal.validate(c['model'])
    from train_policy import publish
    args.output.mkdir(parents=True,exist_ok=False);publish(args.output/'resolved_config.json',c)
    report={'kind':c['kind'],'snapshot_id':SOURCE.name,'status':'running','started_unix':time.time(),
        'scope':'Complete compute-only cached move, encoder and observable cache outputs included. Same initial model/real prefixes, ABBA order, no training or Go strength claim.'}
    import jax
    try:
        jax.distributed.initialize(initialization_timeout=90)
        run(c,report,args.output);report['status']='passed';verify(SOURCE)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();publish(args.output/'result.json',report);jax.distributed.shutdown()


def run(c,report,output):
    import numpy as np
    import jax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from jax.experimental import multihost_utils
    import causal
    import profile_causal
    from train_policy import publish
    if len(jax.devices())!=16 or jax.process_count()!=4:raise ValueError('Wrong topology')
    mesh=Mesh(np.asarray(jax.devices()),('data',));replicated=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    def replica(x):return jax.tree.map(lambda a:np.asarray(a.addressable_shards[0].data),x)
    def global_batch(b):return {k:jax.make_array_from_process_local_data(batched,v) for k,v in b.items()}
    params=jax.jit(lambda:causal.initialize(c['seed'],c['model']),out_shardings=replicated)();jax.block_until_ready(params)
    arrays=replica(params);digest=hashlib.sha256()
    for i,x in enumerate(jax.tree.leaves(arrays)):
        digest.update(canonical_json([f'p_{i:04d}',list(x.shape),str(x.dtype)]));digest.update(x.tobytes(order='C'))
    del arrays
    data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256'])
    report.update(jax_rank=jax.process_index(),host_rank=int(os.environ['GOZERO_HOST_RANK']),
        initial_parameter_elements_sha256=digest.hexdigest(),dataset_manifest_sha256=c['dataset']['manifest_sha256'],
        parameter_count=sum(x['elements'] for x in causal.parameter_schema(c['model'])),cases=[])
    for index,name in enumerate(c['cases']):
        destination=output/f'case-{index}-{name}';destination.mkdir()
        implementation=causal.append_move_reference if name=='reference' else causal.append_move
        result=profile_causal.run(params,data,c['model'],global_batch,replica,4,mesh,destination,implementation)
        publish(destination/'result.json',result);report['cases'].append({'case':name,'index':index,**result})
        print(json.dumps({'kind':'cached_graph_comparison','case':name,'index':index,
            'seconds':result['host_dispatch_latency_seconds'],'memory_bytes':result['memory_bytes'],
            'max_policy_tv':result['max_policy_tv']}),flush=True)
    multihost_utils.sync_global_devices('cached-graph-comparison-complete')
