"""CPU mechanism probe: which observed boards receive a final-policy gradient?"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--config-name',default='c128_d2_mean.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.snapshot.resolve();recipe=source/'research/recipes/visual_token_encoder'
    sys.path.insert(0,str(source/'packages/gozero/src'));sys.path.insert(0,str(recipe))
    from gozero.snapshots import verify,read_json
    from gozero.checkpoints import sha256
    from gozero.checkpoint_archive import publish
    from gozero.katago_sequence_batches import Dataset
    verify(source);c=read_json(recipe/a.config_name);data=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256'])
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    import numpy as np
    causal=importlib.import_module('causal')
    for shard,game in data.indices['expert',0]:
        off=data.shards[shard]['expert_offsets'];length=int(off[game+1]-off[game])
        if 32<=length<=128:break
    else:raise ValueError('No eligible training episode')
    raw=data.batch([('expert',shard,game)],positions=128);moves=16
    spatial=jnp.asarray(raw['spatial'][:,:moves]);glob=jnp.asarray(raw['global_features'][:,:moves])
    actions=jnp.asarray(raw['actions'][:,:moves]);counts=jnp.asarray([moves],jnp.int32)
    target=jnp.asarray(raw['policies'][0,moves-1]);legal=jnp.asarray(raw['legal'][0,moves-1])
    net={**c['model'],'dtype':'float32','attention_backend':'xla'}
    start=time.time();params=jax.jit(lambda:causal.initialize(c['seed'],net))();jax.block_until_ready(params)
    cases=[]
    for pooling in ['learned','board_mean']:
        model={**net,'readout_pooling':pooling}
        def objective(params,spatial):
            logits=causal.forward(params,spatial,glob,actions,counts,model)[0,moves-1]
            logp=jax.nn.log_softmax(jnp.where(legal,logits,-1e9))
            return -jnp.sum(target*logp)
        began=time.time();loss,gradient=jax.jit(jax.value_and_grad(objective,argnums=1))(params,spatial)
        jax.block_until_ready((loss,gradient));norm=np.sqrt(np.sum(np.asarray(gradient,dtype=np.float64)**2,axis=(0,2,3,4)))
        row={'pooling':pooling,'cross_entropy':float(loss),'gradient_l2_by_frame':norm.tolist(),
             'current_frame_fraction_of_summed_frame_norms':float(norm[-1]/norm.sum()),'seconds':time.time()-began}
        cases.append(row);print(json.dumps(row),flush=True)
    publish(a.output,{'kind':'initial_policy_gradient_frame_allocation','status':'passed','model_snapshot':source.name,
        'config_sha256':sha256(recipe/a.config_name),'driver_sha256':sha256(Path(__file__)),
        'dataset_manifest_sha256':c['dataset']['manifest_sha256'],'training_episode':[shard,game],
        'original_game_length':length,'prefix_moves':moves,'seed':c['seed'],'execution_model':net,'cases':cases,'seconds':time.time()-start,
        'scope':'Full-sized untrained model, same parameters and one fixed training episode, float32 CPU execution. Differentiate the final policy CE with respect to each observed spatial frame. This probes a gradient path, not learnability, playing strength or a training result.'})
    print(a.output,sha256(a.output),flush=True)


if __name__=='__main__':main()
