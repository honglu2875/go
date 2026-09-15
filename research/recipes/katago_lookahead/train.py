"""Qualify source behavior and save complete fast/slow continuation state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,read_json,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after-events',type=int);p.add_argument('--resume',type=Path);a=p.parse_args()
    verify(SOURCE);c=read_json(a.config)
    if canonical_json(c)!=canonical_json(read_json(SOURCE/'resolved_config.json')):raise ValueError('Configuration is not frozen')
    if c['kind']!='katago_lookahead_cpu_qualification' or c['platform']!='cpu':raise ValueError('Expected CPU qualification')
    if c['rtol']!=2e-6 or c['atol']!=2e-7 or c['checkpoint_event']!=12:raise ValueError('Qualification bounds changed')
    reference=a.workspace_root/c['reference']['path']
    if checkpoints.sha256(reference)!=c['reference']['sha256']:raise ValueError('Original reference changed')
    ref=read_json(reference);data=reference.parent/ref['arrays']['path']
    if checkpoints.sha256(data)!=ref['arrays']['sha256']:raise ValueError('Reference arrays changed')
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import lookahead
    import state_io
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind=c['kind'],status='running',snapshot=SOURCE.name,reference_sha256=c['reference']['sha256'],cases=[])
    try:
        with np.load(data,allow_pickle=False) as arrays:
            for number,case in enumerate(ref['cases']):
                config=case['config'];params={k:jnp.array(arrays[v],copy=True) for k,v in ref['initial'].items()}
                state=lookahead.initialize(params,config);cursor=0
                schema=[dict(path=k,shape=list(v.shape),dtype='float32') for k,v in sorted(params.items())]
                if a.resume is not None:
                    saved,values,_=checkpoints.read(a.resume/f'case-{number:03d}')
                    if saved['case']!=number or saved['reference_sha256']!=c['reference']['sha256']:
                        raise ValueError('Wrong case/reference in resumed state')
                    params,state=state_io.restore(saved['lookahead'],values,config=config,schema=schema,source_sha256=SOURCE.name)
                    params,state=jax.tree.map(jnp.asarray,(params,state));cursor=saved['cursor']
                initial_cursor=cursor;stop=len(case['events']) if a.stop_after_events is None else a.stop_after_events
                if not cursor<stop<=len(case['events']):raise ValueError('Invalid continuation event boundary')
                def update(p,s,d):
                    candidate=jax.tree.map(lambda x,y:x+y,p,d)
                    return lookahead.after_step(candidate,s,config,accepted=True)
                advance=jax.jit(update,donate_argnums=(0,1));maximum=0.;comparisons=0
                for event in case['events'][cursor:stop]:
                    synchronized=False
                    if event['op']=='step':
                        delta={k:jnp.asarray(arrays[v]) for k,v in event['delta'].items()}
                        params,state,metrics=advance(params,state,delta);synchronized=bool(metrics['synchronized'])
                    elif event['op']=='begin_subepoch':state=lookahead.begin_subepoch(state)
                    elif event['op']=='end_epoch':params,state=lookahead.end_epoch(params,state,config)
                    else:raise ValueError('Unknown reference event')
                    jax.block_until_ready((params,state))
                    if int(state['counter'])!=event['counter'] or synchronized!=event['synchronized']:
                        raise ValueError('Clock/synchronization differs from source')
                    for role,actual in (('fast',params),('slow',state['slow'])):
                        if set(actual)!=set(event['expected'][role]):raise ValueError('Reference coverage differs')
                        for name,value in actual.items():
                            expected=arrays[event['expected'][role][name]];value=np.asarray(value)
                            np.testing.assert_allclose(value,expected,rtol=c['rtol'],atol=c['atol'],err_msg=f'{number}:{cursor}:{role}:{name}')
                            maximum=max(maximum,float(np.max(np.abs(value-expected))));comparisons+=1
                    cursor+=1
                params,state=jax.tree.map(np.asarray,(params,state))
                metadata,values=state_io.flatten(params,state,config,source_sha256=SOURCE.name)
                checkpoint=a.output/f'case-{number:03d}'
                identity=checkpoints.write(checkpoint,state=dict(case=number,cursor=cursor,reference_sha256=c['reference']['sha256'],lookahead=metadata),arrays=values,actors='{}',compress=True)
                report['cases'].append(dict(case=number,config=config,initial_cursor=initial_cursor,cursor=cursor,
                    comparisons=comparisons,maximum_absolute_error=maximum,checkpoint_arrays=len(values),
                    checkpoint_manifest_sha256=identity,checkpoint=str(checkpoint)))
        rejected=[]
        for name,bad in [('zero_period',dict(k=0,alpha=.5)),('unit_alpha_not_disabled',dict(k=6,alpha=1.)),
                         ('nonfinite_alpha',dict(k=6,alpha=float('nan'))),('unpaired_disabled',dict(k=None,alpha=.5)),
                         ('fractional_period',dict(k=1.5,alpha=.5))]:
            try:lookahead.validate(bad)
            except ValueError:rejected.append(name)
            else:raise ValueError('Invalid Lookahead setting accepted')
        for name,bad in [('missing_slow',{k:v for k,v in values.items() if k!='s_0000'}),
                         ('negative_clock',{**values,'counter':np.int32(-1)}),
                         ('expired_clock',{**values,'counter':np.int32(config['k'])}),
                         ('nonfinite_slow',{**values,'s_0000':np.full_like(values['s_0000'],np.nan)})]:
            try:state_io.restore(metadata,bad,config=config,schema=schema,source_sha256=SOURCE.name)
            except (ValueError,KeyError):rejected.append(name)
            else:raise ValueError('Invalid Lookahead state accepted')
        report.update(status='passed',rejected_invalid_inputs=rejected,
            all_events_complete=all(row['cursor']==len(case['events']) for row,case in zip(report['cases'],ref['cases'])))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report.update(seconds=time.monotonic()-started,scope='Actual source Lookahead state/boundaries with synthetic fast-optimizer increments and donated CPU execution. No neural learner, Muon schedule integration or accelerator result.')
        out=a.output/'result.json'
        with out.open('xb') as f:f.write(canonical_json(report))
        out.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=checkpoints.sha256(out))),flush=True)


if __name__=='__main__':main()
