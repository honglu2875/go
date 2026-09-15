"""Exercise repeated profiler calls, observable cache donation and file outputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,verify
from gozero.checkpoints import sha256
import causal
import profile_causal


class SyntheticPrefixes:
    size=3;time=130
    indices={('expert',0):[(0,i) for i in range(128)]}
    shards=[{'expert_offsets':np.arange(129)*130}]
    def batch(self,choices,positions):
        random=np.random.default_rng(582);n=len(choices)
        return {'spatial':random.integers(0,2,(n,positions,3,3,22)).astype(np.float32),
                'global_features':random.normal(0,.1,(n,positions,19)).astype(np.float32),
                'actions':random.integers(0,10,(n,positions),dtype=np.int32)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();verify(SOURCE)
    if len(jax.devices())!=4 or jax.default_backend()!='cpu':raise ValueError('Four virtual CPU devices required')
    args.output.mkdir(parents=True,exist_ok=False)
    c=dict(architecture='causal_visual_policy',width=32,layers=2,mlp_hidden=48,heads=4,kv_heads=2,max_board_size=9,
        max_positions=512,dtype='float32',rematerialize=True,norm_epsilon=1e-6,rope_theta=10000.,attention_backend='xla',encoder='overlap_linear')
    mesh=Mesh(np.asarray(jax.devices()),('data',));replicated=NamedSharding(mesh,P());batched=NamedSharding(mesh,P('data'))
    params=jax.jit(lambda:causal.initialize(73,c),out_shardings=replicated)()
    def global_batch(b):return {k:jax.make_array_from_process_local_data(batched,v) for k,v in b.items()}
    def replica(x):return jax.tree.map(lambda a:np.asarray(a.addressable_shards[0].data),x)
    rows=[]
    for i,name in enumerate(['reference','carried','carried','reference']):
        directory=args.output/f'{i}-{name}';directory.mkdir()
        fn=causal.append_move_reference if name=='reference' else causal.append_move
        r=profile_causal.run(params,SyntheticPrefixes(),c,global_batch,replica,1,mesh,directory,fn)
        r.update(case=name);rows.append(r)
    if len({r['analytical']['multiply_add_flops_per_batch'] for r in rows})!=1:raise ValueError('Arithmetic changed')
    result={'kind':'cpu_cache_profiler_pipeline_qualification','status':'passed','operator_snapshot':SOURCE.name,'cases':rows,
        'scope':'Synthetic 3x3 observations, full 128-move prefixes, four virtual CPUs. Verifies the profiling/donation pipeline, not TPU latency or Go strength.'}
    (args.output/'result.json').write_bytes(canonical_json(result));(args.output/'result.json').chmod(0o444)
    print(json.dumps({'status':'passed','sha256':sha256(args.output/'result.json')}),flush=True)


if __name__=='__main__':main()
