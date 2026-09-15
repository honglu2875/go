"""Check host cadence against expressions/statements from the pinned trainer."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
RECIPE=ROOT/'research/recipes/strong19_source_muon'
sys.path.insert(0,str(RECIPE))
import source_runtime as runtime


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=Path(__file__).with_name('runtime-cpu-001.json')
    if output.exists():raise FileExistsError(output)
    source=ROOT/'.gozero/external/katago-source/92ee95c0a4b25fec214da00951ab69e97e207729/KataGo-92ee95c0a4b25fec214da00951ab69e97e207729/python/train.py'
    if sha(source)!='991cbfb2d5cea8180cd3daf818ef2a516aca048e0ae0eab68865e9a125967e9e':raise ValueError('Pinned trainer changed')
    tree=ast.parse(source.read_text())
    def at(line,kind):return next(n for n in ast.walk(tree) if isinstance(n,kind) and n.lineno==line)
    print_node=at(1546,ast.Assign)
    increment=[at(1644,ast.AugAssign),at(1646,ast.AugAssign)]
    refresh=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and '200000000' in ast.unparse(n.test)
        and 'batch_count_this_epoch' in ast.unparse(n.test))
    def code(nodes):return compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec')
    before_code=code([print_node]);increment_code=code(increment);refresh_code=code([refresh])
    base=json.loads((Path(__file__).with_name('cpu-config-001.json')).read_text())['learner']['source_runtime']
    checks=[];total=0;latest_state=None
    for initial,epochs in ((249000,[[4,3],[4]]),(200000001,[[53,7],[5]])):
        c=copy.deepcopy(base);c.update(initial_samples=initial,epochs=epochs)
        horizon=sum(map(sum,epochs));state=runtime.initialize(c,horizon,dict(input=1.,normal=2.))
        env=dict(train_state={'global_step_samples':initial},world_size=1,print_train_loss_every_batches=c['print_every'])
        calls=[];latest=None;refreshes=0;turn=0;exposures=0
        def update(log_if):
            calls.append(dict(samples=env['train_state']['global_step_samples'],batch=env['batch_count_this_epoch']))
            return 0.,0.
        env['update_and_return_lr_and_wd']=update
        for e,counts in enumerate(epochs):
            env['batch_count_this_epoch']=0
            for sub,count in enumerate(counts):
                for index in range(count):
                    state,b=runtime.before_step(state,c,horizon)
                    if b['epoch_entry']:
                        refreshes+=1
                        assert state['settings']==runtime.settings(c,env['train_state']['global_step_samples'],None if latest is None else {k:latest[k]/state['baseline'][k] for k in latest})
                    assert b['epoch_entry']==(sub==0 and index==0)
                    assert b['subepoch_entry']==(index==0)
                    exec(before_code,env);assert b['is_print_batch']==env['is_print_batch']
                    positions=(230,512,2048)[turn%3];env['batch_size']=positions;exposures+=positions
                    norms=dict(input=1.+turn*.001,normal=2.+turn*.003) if env['is_print_batch'] else None
                    exec(increment_code,env)
                    if norms is not None:latest=norms.copy()
                    previous=len(calls);exec(refresh_code,env);changed=len(calls)>previous
                    if changed:refreshes+=1
                    state,a=runtime.after_step(state,c,horizon,positions=positions,pre_update_norms=norms);turn+=1
                    assert a['refreshed']==changed and a['norm_snapshot']==env['is_print_batch']
                    assert a['subepoch_end']==(index+1==count)
                    assert a['epoch_end']==(sub+1==len(counts) and index+1==count)
                    assert state['samples']==env['train_state']['global_step_samples'] and state['refresh_count']==refreshes
                    assert state['latest_norms']==latest
                    runtime.validate_state(state,c,horizon,positions=exposures)
                    if changed:
                        assert state['settings']==runtime.settings(c,state['samples'],None if latest is None else {k:latest[k]/state['baseline'][k] for k in latest})
                    total+=1
        checks.append(dict(initial_samples=initial,updates=horizon,refreshes=refreshes,source_update_calls=calls))
        latest_state=(state,c,horizon,exposures)
    state,c,horizon,exposures=latest_state;rejected=[]
    mutations=[('sample count',lambda s:s.update(samples=s['samples']+1)),
        ('epoch counter',lambda s:s.update(epoch=0)),('stale settings tamper',lambda s:s['settings']['rates'].update(normal=42.)),
        ('missing norm',lambda s:s.update(norm_snapshot_turn=None)),
        ('nonfinite baseline',lambda s:s['baseline'].update(input=float('nan')))]
    for label,mutate in mutations:
        value=copy.deepcopy(state);mutate(value)
        try:runtime.validate_state(value,c,horizon,positions=exposures)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid runtime accepted: '+label)
    result=dict(kind='joint_source_runtime_cadence_qualification',status='passed',created=time.time(),
        operator_sha256=sha(Path(__file__)),runtime_sha256=sha(RECIPE/'source_runtime.py'),source_sha256=sha(source),
        events=total,cases=checks,rejected=rejected,source_ast_lines=[1546,1644,1646,refresh.lineno],
        scope='Actual pinned source print/sample/refresh statements, including both sides of the200M cadence threshold, irregular subepochs and lagged norm snapshots. No neural or playing-strength result.')
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    output.chmod(0o444)
    print(json.dumps(dict(status='passed',events=total,rejected=len(rejected),sha256=sha(output))))


if __name__=='__main__':main()
