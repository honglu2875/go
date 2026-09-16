"""Frozen, read-only inspection of an audited joint model checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,verify


def digest_arrays(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=arrays[key]
        h.update(canonical_json([key,list(a.shape),str(a.dtype)]));h.update(a.tobytes(order='C'))
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    c=json.loads(args.config.read_text());verify(SOURCE)
    if c!=json.loads((SOURCE/'resolved_config.json').read_text()) or c['kind']!='joint_value_diagnostic':
        raise ValueError('Expected frozen read-only diagnostic configuration')
    os.environ['JAX_PLATFORMS']=c['platform']
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset
    import compact
    import evaluation
    import heads
    import joint
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(kind=c['kind'],snapshot_id=SOURCE.name,status='running',started=time.time(),
                training_updates=0,test_targets_read=False,parent=c['checkpoint'],selection=c['selection'])
    initialized=False
    try:
        if c['platform']=='tpu':
            jax.distributed.initialize(initialization_timeout=90);initialized=True
        host=int(os.environ.get('GOZERO_HOST_RANK','0'));rank=jax.process_index();world=jax.process_count()
        if world!=c['expected_processes'] or len(jax.devices())!=c['expected_devices']:
            raise ValueError('Unexpected topology')
        schema=joint.parameter_schema(c['model'],c['value_model'])
        if schema!=c['checkpoint']['model_schema']:raise ValueError('Model schema changed')
        if host==0:
            saved,arrays,_=checkpoints.read(Path(c['checkpoint']['owner_path']),
                expected_manifest_sha256=c['checkpoint']['manifest_sha256'],array_prefix='p_')
            if saved['snapshot_id']!=c['checkpoint']['snapshot'] or saved['turn']!=108:
                raise ValueError('Wrong checkpoint lineage')
            params={r['path']:arrays[f'p_{i:04d}'] for i,r in enumerate(schema)}
            if digest_arrays(arrays)!=c['checkpoint']['parameters_sha256']:
                raise ValueError('Wrong parameter tensors')
            del saved,arrays
        else:params={r['path']:np.zeros(r['shape'],np.float32) for r in schema}
        params=mh.broadcast_one_to_all(params,is_source=host==0)
        observed=digest_arrays({f'p_{i:04d}':params[r['path']] for i,r in enumerate(schema)})
        if observed!=c['checkpoint']['parameters_sha256']:raise ValueError('Broadcast changed parameters')
        mesh=Mesh(np.asarray(jax.devices()),('data',));rep=NamedSharding(mesh,P());bat=NamedSharding(mesh,P('data'))
        params=jax.device_put(params,rep)
        data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256'])
        entries=[tuple(r['entry']) for r in c['selection']]
        if len(entries)%world:raise ValueError('Uneven selection')
        local=entries[rank::world]
        local_batch=data.batch(local,positions=c['positions'])
        batch={k:jax.make_array_from_process_local_data(bat,v) for k,v in local_batch.items()}
        results=[]
        for mode in ('evaluation','training'):
            def forward(p,b):
                if mode=='evaluation':
                    out=evaluation.forward(p,b,c['model'],chunk_frames=c['training']['chunk_frames'])
                    return {'logits':out['value_logits'],'latent':out['latent']}
                out=compact.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c['model'],
                    chunk_frames=c['training']['chunk_frames'],inner_rematerialize=False,with_hidden=True)
                hp=joint.value_params(p)
                return dict(logits=heads.temporal(hp,out['latent']),latent=out['latent'],
                    aux_logits=heads.temporal(hp,out['aux_latent']),aux_latent=out['aux_latent'])
            f=jax.shard_map(forward,mesh=mesh,in_specs=(P(),P('data')),out_specs=P('data'),check_vma=False)
            start=time.perf_counter();exe=jax.jit(f).lower(params,batch).compile()
            compile_seconds=time.perf_counter()-start
            start=time.perf_counter();out=exe(params,batch);jax.block_until_ready(out)
            seconds=time.perf_counter()-start
            local_out={k:np.concatenate([np.asarray(s.data) for s in v.addressable_shards],axis=0) for k,v in out.items()}
            local_out.update(values=local_batch['values'],counts=local_batch['counts'],
                opponent=local_batch['opponent'],split=np.asarray([c['selection'][i]['split'] for i in range(rank,len(entries),world)]))
            path=args.output/(mode+'.npz');np.savez_compressed(path,**local_out)
            row=dict(mode=mode,entries=local,positions=int(local_batch['counts'].sum()),
                compile_seconds=compile_seconds,forward_seconds=seconds,artifact=path.name,
                sha256=checkpoints.sha256(path),bytes=path.stat().st_size)
            results.append(row)
            print(json.dumps(dict(kind='value_diagnostic_forward',host=host,**row)),flush=True)
        report.update(status='passed',host_rank=host,jax_rank=rank,parameters_sha256=observed,results=results)
        verify(SOURCE);mh.sync_global_devices('value-diagnostic-complete')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished']=time.time()
        (args.output/'result.json').write_bytes(canonical_json(report))
        if initialized:jax.distributed.shutdown()


if __name__=='__main__':main()
