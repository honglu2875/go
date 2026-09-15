"""Complete cached-decoding qualification, including encoder and observable KV writes."""
import gc
import hashlib
import json
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import causal
import draft_model
import compute_budget


def run(params,data,c,global_batch,replica,world,mesh,output):
    from gozero.jaxpr_cost import analyze
    past=128;local_n=128//world;stride=causal.layout(data.size,c)['stride']
    cheap={**c,'encoder_passes':1,'first_pass_aux_weight':0}
    # Use real complete game prefixes. Repeat them only to fix the batch shape;
    # model inputs remain runtime arguments, not constants folded into the graph.
    choices=[]
    for shard,game in data.indices['expert',0]:
        offsets=data.shards[shard]['expert_offsets']
        if int(offsets[game+1]-offsets[game])>=past+1:choices.append(('expert',shard,game))
        if len(choices)==local_n:break
    if len(choices)<local_n:raise ValueError('Need real long-game prefixes for decode qualification')
    raw=data.batch(choices,positions=max(data.time,past+1))
    spatial=global_batch({'s':raw['spatial'][:,:past+1]})['s']
    glob=global_batch({'g':raw['global_features'][:,:past+1]})['g']
    actions=global_batch({'a':raw['actions'][:,:past+1]})['a']
    counts=global_batch({'n':np.full(local_n,past,np.int32)})['n']
    cache_specs={k:P('data') for k in ('keys','values','lengths','valid')}
    cache_specs['network_version']=P()
    inputs=(P(),P('data'),P('data'),P('data'),P('data'))
    reference=jax.jit(jax.shard_map(lambda p,s,g,a,n:draft_model.forward(p,s,g,a,n,c),
        mesh=mesh,in_specs=inputs,out_specs=(P('data'),P('data')),check_vma=False))
    print(json.dumps({'kind':'decode_qualification_phase','phase':'reference','host':os.environ.get('GOZERO_HOST_RANK')}),flush=True)
    joint_main,joint_draft=reference(params,spatial,glob,actions,counts+1)
    expected=joint_draft[:,-1]
    ordinary=jax.jit(jax.shard_map(lambda p,s,g,a,n:causal.forward(p,s,g,a,n,c),
        mesh=mesh,in_specs=inputs,out_specs=P('data'),check_vma=False))
    ordinary_main=ordinary(params,spatial,glob,actions,counts+1)
    @jax.jit
    def main_errors(a,b):return jnp.max(jnp.abs(a-b)),jnp.max(.5*jnp.sum(jnp.abs(jax.nn.softmax(a)-jax.nn.softmax(b)),-1))
    main_error,main_tv=map(float,replica(main_errors(joint_main,ordinary_main)))
    if main_error>.04 or main_tv>.005:raise ValueError('Auxiliary training changed the full-policy semantics')
    del joint_main,joint_draft,ordinary_main,ordinary
    prefill=jax.jit(jax.shard_map(lambda p,s,g,a,n:causal.forward(p,s,g,a,n,c,with_cache=True,cache_positions=c['max_positions']),
        mesh=mesh,in_specs=inputs,out_specs=(P('data'),cache_specs),check_vma=False))
    start=time.perf_counter()
    _,cache=prefill(params,spatial[:,:past],glob[:,:past],actions[:,:past],counts)
    jax.block_until_ready(cache);prefill_seconds=time.perf_counter()-start
    print(json.dumps({'kind':'decode_qualification_phase','phase':'cache_ready','host':os.environ.get('GOZERO_HOST_RANK')}),flush=True)
    a,s,g=actions[:,past-1],spatial[:,past],glob[:,past]
    fn=lambda p,ca,a,s,g:causal.append_move(p,ca,a,s,g,cheap,attention_positions=past+1)
    graph=jax.make_jaxpr(fn)(params,cache,a,s,g);arithmetic=analyze(graph)
    analytical=compute_budget.transformer(cheap,data.size,128,past)
    if arithmetic['counts']['multiply_add_flops']!=analytical['multiply_add_flops_per_batch'] or arithmetic['unaccounted_primitives']:
        raise ValueError('Complete cached decode arithmetic is not qualified')
    mapped=jax.shard_map(fn,mesh=mesh,in_specs=(P(),cache_specs,P('data'),P('data'),P('data')),
        out_specs=(P('data'),cache_specs),check_vma=False)
    start=time.perf_counter();lowered=jax.jit(mapped,donate_argnums=(1,)).lower(params,cache,a,s,g);executable=lowered.compile()
    compile_seconds=time.perf_counter()-start;hlo=lowered.compiler_ir('hlo').as_hlo_text()
    with (output/'decode.hlo').open('x') as f:f.write(hlo);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    logits,cache=executable(params,cache,a,s,g);jax.block_until_ready((logits,cache))
    @jax.jit
    def compare(actual,expected,valid):
        return jnp.max(jnp.abs(actual-expected)),jnp.max(.5*jnp.sum(jnp.abs(jax.nn.softmax(actual)-jax.nn.softmax(expected)),-1)),jnp.all(valid)
    error,variation,valid=map(float,replica(compare(logits,expected,cache['valid'])))
    if not valid or error>.04 or variation>.005:raise ValueError(f'Cached/Splash policy mismatch: {error}, {variation}, {valid}')
    durations=[]
    for _ in range(10):
        pointers=global_batch({'lengths':np.full(local_n,past*stride-1,np.int32),'valid':np.ones(local_n,bool)})
        cache={**cache,**pointers};jax.block_until_ready(pointers)
        start=time.perf_counter();logits,cache=executable(params,cache,a,s,g);jax.block_until_ready((logits,cache));durations.append(time.perf_counter()-start)
    error2,variation2,valid2=map(float,replica(compare(logits,expected,cache['valid'])))
    if not valid2 or error2>.04 or variation2>.005:raise ValueError('Repeated decode corrupted prefix or policy')
    memory=executable.memory_analysis()
    result={'board_size':data.size,'batch_size':128,'past_moves':past,'allocated_cache_positions':c['max_positions'],
        'attention_key_extent':analytical['attention_key_extent'],'analytical':analytical,'jaxpr':arithmetic,
        'compiler_cost_estimate':executable.cost_analysis(),'compile_seconds':compile_seconds,
        'prefill_compile_and_execute_seconds':prefill_seconds,'host_dispatch_latency_seconds':durations,
        'max_logit_error':max(error,error2),'max_policy_tv':max(variation,variation2),
        'max_logit_tolerance':.04,'max_policy_tv_tolerance':.005,
        'memory_bytes':{k:int(getattr(memory,k)) for k in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']},
        'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest(),
        'cache_outputs_observed':True,'cache_donated':True,
        'full_pass_history_cache':True,'current_board_encoder_passes':1,
        'joint_main_max_logit_error':main_error,'joint_main_max_policy_tv':main_tv,
        'scope':'Warm one-pass current-board draft on exact full-pass history, checked against the masked teacher-forced auxiliary path. Encoder and pending action included; CPU rules/features, transfers and pointer reset excluded.'}
    # Do not retain the inference KV allocation during the subsequent learner.
    del cache,logits,expected,prefill,executable,lowered,spatial,glob,actions,counts,raw
    gc.collect()
    return result
