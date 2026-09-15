"""Read-only current-board refinement diagnostic using an unchanged full-history cache.

This deliberately small CPU diagnostic is not a learning or TPU latency result.
One-step policy overlap on recorded prefixes is not speculative rollout acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero import checkpoints
from gozero.katago_sequence_batches import Dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--training-result',type=Path,required=True)
    parser.add_argument('--training-result-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=8)
    parser.add_argument('--past-moves',type=int,default=64)
    a=parser.parse_args();verify(SOURCE);root=a.workspace_root.resolve()
    if a.output.exists() or not 1<=a.games<=32 or not 1<=a.past_moves<=256:
        raise ValueError('Expected a new output and bounded diagnostic dimensions')
    if checkpoints.sha256(a.training_result)!=a.training_result_sha256:raise ValueError('Training result identity changed')
    result=read_json(a.training_result)
    if result['status']!='passed' or not result['training_complete'] or result['host_rank']!=0:raise ValueError('Completed owner training result required')
    source=root/'.gozero/snapshots'/result['snapshot_id'];manifest=verify(source)
    recipe=source/manifest['recipe'];sys.path.insert(0,str(recipe))
    # Import the model belonging to the checkpoint, never this operator's model.
    import causal
    import compute_budget
    import numpy as np
    import jax
    import jax.numpy as jnp
    if Path(causal.__file__).resolve()!=recipe/'causal.py':raise ValueError('Wrong model implementation imported')
    if len(jax.devices())!=1 or jax.devices()[0].platform!='cpu':raise ValueError('Diagnostic must use one CPU device')
    config=read_json(source/'resolved_config.json');c={**config['model'],'attention_backend':'xla'}
    if c['encoder_passes']!=2 or c['architecture']!='causal_visual_policy':raise ValueError('Expected the two-pass control family')
    data=Dataset(config['dataset']['path'],config['dataset']['manifest_sha256'])
    eligible=[]
    for shard,game in data.indices['expert',1]:
        off=data.shards[shard]['expert_offsets']
        if int(off[game+1]-off[game])>a.past_moves:eligible.append(('expert',shard,game))
    if len(eligible)<a.games:raise ValueError('Too few eligible validation games')
    rng=np.random.Generator(np.random.PCG64(91313611))
    entries=[eligible[int(i)] for i in rng.choice(len(eligible),size=a.games,replace=False)]
    raw=data.batch(entries,positions=max(data.time,a.past_moves+1))
    owner=Path(result['latest_checkpoint']['owner_checkpoint_path'])
    owner_manifest=checkpoints.sha256(owner/'manifest.json')
    if owner_manifest!=result['latest_checkpoint']['manifest_sha256']:raise ValueError('Checkpoint manifest differs from the training result')
    state,arrays,_=checkpoints.read(owner,expected_manifest_sha256=owner_manifest)
    if state['snapshot_id']!=source.name or state['turn']!=result['turn']:raise ValueError('Checkpoint lineage changed')
    params={item['path']:jnp.asarray(arrays[f'p_{i:04d}']) for i,item in enumerate(state['model_schema'])}
    del arrays
    past=a.past_moves;s=jnp.asarray(raw['spatial'][:,:past+1]);g=jnp.asarray(raw['global_features'][:,:past+1]);actions=jnp.asarray(raw['actions'][:,:past+1])
    counts=jnp.full((a.games,),past,jnp.int32)
    started=time.perf_counter()
    prefill=jax.jit(lambda p,s,g,act,n:causal.forward(p,s,g,act,n,c,with_cache=True,cache_positions=past+1))
    _,cache=prefill(params,s[:,:past],g[:,:past],actions[:,:past],counts)
    jax.block_until_ready(cache)
    def cache_digest(ca):
        h=hashlib.sha256()
        for k in sorted(ca):h.update(k.encode());h.update(np.asarray(ca[k]).tobytes())
        return h.hexdigest()
    original_cache_sha=cache_digest(cache);outputs={};budgets={}
    for passes in (1,2):
        cc={**c,'encoder_passes':passes}
        fn=jax.jit(lambda p,ca,act,s,g:causal.append_move(p,ca,act,s,g,cc,attention_positions=past+1))
        logits,updated=fn(params,cache,actions[:,past-1],s[:,past],g[:,past]);jax.block_until_ready((logits,updated))
        if not np.all(np.asarray(updated['valid'])):raise ValueError('Cache guard failed')
        if cache_digest(cache)!=original_cache_sha:raise ValueError('A diagnostic modified the common prefix cache')
        outputs[str(passes)]=np.asarray(logits)
        budgets[str(passes)]=compute_budget.transformer(cc,data.size,a.games,past)
    legal=np.asarray(raw['legal'][:,past]);targets=np.asarray(raw['policies'][:,past],np.float64)
    if not np.all(legal.any(-1)) or not np.allclose(targets.sum(-1),1,atol=1e-5):raise ValueError('Invalid policy targets')
    distributions={};metrics={}
    target_entropy=-np.sum(targets*np.log(np.maximum(targets,1e-30)),-1)
    for key,logits in outputs.items():
        if not np.isfinite(logits).all():raise ValueError('Nonfinite diagnostic logits')
        scores=np.where(legal,logits.astype(np.float64),-1e30);scores-=scores.max(-1,keepdims=True)
        logp=scores-np.log(np.exp(scores).sum(-1,keepdims=True));p=np.exp(logp);distributions[key]=p
        kl=-np.sum(targets*logp,-1)-target_entropy
        metrics[key]={'mean_teacher_kl':float(kl.mean()),'teacher_kl_per_position':kl.tolist(),
            'teacher_top1':float(np.mean(p.argmax(-1)==targets.argmax(-1))),
            'current_move_dense_gflops':budgets[key]['multiply_add_flops_per_move']/1e9}
    variation=.5*np.abs(distributions['1']-distributions['2']).sum(-1)
    out={'status':'passed','kind':'current_board_encoder_pass_diagnostic','operator_snapshot':SOURCE.name,
        'training_snapshot':source.name,'training_result':str(a.training_result),'training_result_sha256':a.training_result_sha256,
        'checkpoint_manifest_sha256':owner_manifest,'dataset_manifest_sha256':config['dataset']['manifest_sha256'],
        'selection':{'split':'validation','eligible_games':len(eligible),'games':a.games,'past_moves':past,
                     'selection_seed':91313611,'entries':entries},
        'scope':'Exact recorded two-pass history cache reused for both current-board policies; only current-board encoder pass count changes. No parameter update. CPU xla attention replaces TPU Splash.',
        'limits':'Eight default sampled midgame positions, not full validation. No CPU timing used as TPU speed; one-step probability overlap is not measured multi-step or MCTS acceptance.',
        'common_prefix_cache_sha256':original_cache_sha,'metrics':metrics,
        'mean_policy_tv':float(variation.mean()),'policy_tv_per_position':variation.tolist(),
        'mean_one_step_policy_overlap':float((1-variation).mean()),
        'greedy_agreement':float(np.mean(distributions['1'].argmax(-1)==distributions['2'].argmax(-1))),
        'cpu_diagnostic_wall_seconds':time.perf_counter()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:f.write(canonical_json(out))
    a.output.chmod(0o444);verify(SOURCE);verify(source)
    print(json.dumps({'status':'passed','output':str(a.output),'metrics':metrics,'mean_policy_overlap':out['mean_one_step_policy_overlap'],'greedy_agreement':out['greedy_agreement']}),flush=True)


if __name__=='__main__':main()
