#!/usr/bin/env python3
"""Compare frozen native Gumbel search with the pinned authors' Mctx routines."""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify
from gozero.checkpoints import sha256

REVISION='88f92056a420c2673bed282f5a0c00211f126e78'
DEPENDENCIES={'chex':'0.1.91','absl-py':'2.3.1','toolz':'1.0.0','typing_extensions':'4.15.0'}


def fixtures(actions,count,rng,budget,rescale):
    import numpy as np
    nodes=1+actions+actions**2
    output=[]
    for seed in range(count):
        children=np.full((nodes,actions),-1,np.int32)
        for node in range(1+actions):
            if node and rng.uniform()<0.2:continue
            legal=rng.uniform(size=actions)>0.25
            legal[rng.integers(actions)]=True
            children[node,legal]=1+node*actions+np.arange(actions)[legal]
        logits=rng.normal(size=(nodes,actions)).astype(np.float32)
        values=rng.uniform(-0.95,0.95,size=nodes).astype(np.float32)
        gumbel=rng.gumbel(size=actions).astype(np.float32) if seed%4 else np.zeros(actions,np.float32)
        output.append({'children':[[None if a<0 else int(a) for a in row] for row in children],
                       'logits':logits.tolist(),'values':values.tolist(),'terminal':(children<0).all(-1).tolist(),
                       'gumbel':gumbel.tolist(),'budget':budget,'config':{
                           'max_considered_actions':16,'value_scale':0.1,'maxvisit_init':50.,'rescale_values':rescale}})
    return output


def oracle(cases):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mctx
    from mctx._src import action_selection,policies,qtransforms,search,seq_halving
    count=len(cases);budget=cases[0]['budget'];config=cases[0]['config']
    children=jnp.asarray([[[-1 if a is None else a for a in row] for row in case['children']] for case in cases],jnp.int32)
    logits=jnp.asarray([c['logits'] for c in cases],jnp.float32)
    values=jnp.asarray([c['values'] for c in cases],jnp.float32)
    terminal=jnp.asarray([c['terminal'] for c in cases],bool)
    draws=jnp.asarray([c['gumbel'] for c in cases],jnp.float32)
    invalid=children[:,0]<0
    root=mctx.RootFnOutput(prior_logits=policies._mask_invalid_actions(logits[:,0],invalid),
                          value=values[:,0],embedding=jnp.stack([jnp.arange(count),jnp.zeros(count,jnp.int32)],-1))
    def recurrent(params,key,action,embedding):
        del key
        child_table,logit_table,value_table,terminal_table=params
        batch,node=embedding[:,0],embedding[:,1]
        stopped=terminal_table[batch,node]
        child=jnp.where(stopped,node,child_table[batch,node,action])
        done=terminal_table[batch,child]
        reward=jnp.where(stopped,0.,jnp.where(done,-value_table[batch,child],0.))
        discount=jnp.where(stopped|done,0.,-1.)
        # Terminal nodes are absorbing in Mctx. Their entry edge carries the
        # exact outcome, whereas native search stops before requesting a NN.
        next_logits=jnp.where(done[:,None],jnp.zeros_like(logit_table[batch,child]),
                              policies._mask_invalid_actions(logit_table[batch,child],child_table[batch,child]<0))
        return mctx.RecurrentFnOutput(reward=reward,discount=discount,prior_logits=next_logits,
                                     value=jnp.where(done,0.,value_table[batch,child])),jnp.stack([batch,child],-1)
    transform=functools.partial(qtransforms.qtransform_completed_by_mix_value,
        value_scale=config['value_scale'],maxvisit_init=config['maxvisit_init'],rescale_values=config['rescale_values'])
    def compute():
        tree=search.search(params=(children,logits,values,terminal),rng_key=jax.random.key(27),root=root,
            recurrent_fn=recurrent,
            root_action_selection_fn=functools.partial(action_selection.gumbel_muzero_root_action_selection,
                num_simulations=budget,max_num_considered_actions=config['max_considered_actions'],qtransform=transform),
            interior_action_selection_fn=functools.partial(action_selection.gumbel_muzero_interior_action_selection,qtransform=transform),
            num_simulations=budget,invalid_actions=invalid,
            extra_data=action_selection.GumbelMuZeroExtraData(root_gumbel=draws))
        counts=tree.children_visits[:,0]
        qs=jax.vmap(transform,in_axes=[0,None])(tree,0)
        scores=seq_halving.score_considered(jnp.max(counts,-1,keepdims=True),draws,root.prior_logits,qs,counts)
        action=action_selection.masked_argmax(scores,invalid)
        policy=jax.nn.softmax(policies._mask_invalid_actions(root.prior_logits+qs,invalid))
        return counts,action,policy,jax.vmap(lambda t:t.qvalues(0))(tree)
    return tuple(np.asarray(a) for a in jax.jit(compute)())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE);root=args.workspace_root.resolve()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'kind':'native_gumbel_mctx_comparison','status':'failed',
            'snapshot_id':SOURCE.name,'reference_revision':REVISION,'started_unix':time.time(),
            'claims_go_strength':False,'claims_sample_efficiency':False}
    try:
        reference=root/'.gozero/external/mctx'/REVISION
        receipt=read_json(reference/'receipt.json')
        if receipt['commit']!=REVISION:raise ValueError('Reference revision differs')
        for name,digest in receipt['files'].items():
            if sha256(reference/name)!=digest:raise ValueError('Reference source differs')
        sys.path[:0]=[str(root/'.gozero/validation/mctx-deps'),str(reference)]
        deps={name:importlib.metadata.version(name) for name in DEPENDENCIES}
        if deps!=DEPENDENCIES:raise ValueError('Validation dependencies differ')
        import numpy as np
        import jax
        if jax.default_backend()!='cpu':raise ValueError('Reference audit requires CPU')
        report.update(reference_receipt_sha256=sha256(reference/'receipt.json'),dependencies=deps,
                      jax_version=jax.__version__,numpy_version=np.__version__)
        command=[str(root/'.gozero/cargo/bin/cargo'),'build','--release','--locked','--offline','-p','go-search',
                 '--example','gumbel_probe','--manifest-path',str(SOURCE/'Cargo.toml'),
                 '--target-dir',str(root/'.gozero/build/gumbel')]
        env={**os.environ,'RUSTUP_HOME':str(root/'.gozero/rustup'),'CARGO_HOME':str(root/'.gozero/cargo')}
        result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        (out/'build.log').write_bytes(result.stdout);result.check_returncode()
        binary=out/'gumbel_probe';shutil.copy2(root/'.gozero/build/gumbel/release/examples/gumbel_probe',binary)
        report.update(build_command=command,probe_sha256=sha256(binary),groups=[])
        rng=np.random.default_rng(912706)
        groups=[(9,budget,rescale) for budget in (1,3,7,16,31,64) for rescale in (False,True)]
        groups += [(actions,budget,True) for actions in (1,2,5,17) for budget in (7,31)]
        maximum_policy_error=0.;maximum_value_error=0.;bad=[];cases_total=0
        for group,(actions,budget,rescale) in enumerate(groups):
            cases=fixtures(actions,16,rng,budget,rescale)
            data=''.join(json.dumps(c,separators=(',',':'),allow_nan=False)+'\n' for c in cases).encode()
            (out/f'fixtures-{group:02d}.jsonl').write_bytes(data)
            result=subprocess.run([str(binary)],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
            (out/f'native-{group:02d}.jsonl').write_bytes(result.stdout)
            (out/f'native-{group:02d}.stderr').write_bytes(result.stderr);result.check_returncode()
            native=[json.loads(line) for line in result.stdout.splitlines()]
            expected_counts,expected_action,expected_policy,expected_values=oracle(cases)
            np.savez(out/f'mctx-{group:02d}.npz',visits=expected_counts,action=expected_action,policy=expected_policy,values=expected_values)
            if len(native)!=len(cases):raise ValueError('Native case count differs')
            group_bad=[]
            for i,(case,actual) in enumerate(zip(cases,native)):
                cases_total+=1;mask=expected_counts[i]>0
                actual_q=np.array([0. if v is None else v for v in actual['values']])
                policy_error=float(np.max(np.abs(np.array(actual['policy'])-expected_policy[i])))
                value_error=float(np.max(np.abs(actual_q[mask]-expected_values[i,mask])))
                maximum_policy_error=max(maximum_policy_error,policy_error);maximum_value_error=max(maximum_value_error,value_error)
                if (actual['completed']!=budget or sum(actual['visits'])!=budget
                        or actual['action']!=int(expected_action[i]) or not np.array_equal(actual['visits'],expected_counts[i])
                        or policy_error>2e-5 or value_error>2e-5):group_bad.append(i)
            entry={'actions':actions,'budget':budget,'rescale_values':rescale,'cases':len(cases),'mismatched_cases':group_bad,
                   'fixture_sha256':hashlib.sha256(data).hexdigest(),'native_sha256':sha256(out/f'native-{group:02d}.jsonl'),
                   'mctx_sha256':sha256(out/f'mctx-{group:02d}.npz')}
            report['groups'].append(entry);bad.extend((group,i) for i in group_bad)
            print(json.dumps({'group':group,'cases':len(cases),'mismatches':group_bad}),flush=True)
        report.update(cases=cases_total,mismatches=bad,maximum_policy_absolute_error=maximum_policy_error,
                      maximum_action_value_absolute_error=maximum_value_error,status='passed' if not bad else 'failed',
                      scope='Finite masked depth-two alternating-player trees, root perturbations and deterministic evaluation, odd budgets/action counts, and both Q-rescaling modes. Float64 native vs float32 Mctx; tolerance 2e-5, exact actions and visit counts. Not Go strength or complete algorithm equivalence.')
        verify(SOURCE)
    except Exception as error:
        report['error']=repr(error);raise
    finally:
        report['finished_unix']=time.time();(out/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
