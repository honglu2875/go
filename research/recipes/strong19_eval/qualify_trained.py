"""Qualify portable trained joint weights through full/cached/native inference."""
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero import checkpoints,joint_artifacts
from gozero.native import load_library
from gozero.snapshots import canonical_json,read_json,verify
from gozero.v7_replay import Replay
from gozero.v7_state import digest_from_native,row_digest
import inference
import joint


def run(args,c):
    started=time.monotonic();root=args.workspace_root.resolve();verify(SOURCE)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='trained_joint_native_inference_cpu_qualification',status='running',
                snapshot=SOURCE.name,created=time.time(),test_targets_decoded=False)
    try:
        if (jax.default_backend()!='cpu' or jax.device_count()!=4 or jax.process_count()!=1
                or c['expected_devices']!=4 or c['expected_processes']!=1):
            raise ValueError('Require four simulated CPU devices in one process')
        training_source=root/'.gozero/snapshots'/c['training_snapshot'];training_manifest=verify(training_source)
        config=read_json(training_source/'resolved_config.json');net=config['model'];value=config['value_model']
        if config['training']['purpose']!='qualification' or net['max_board_size']!=19:
            raise ValueError('Expected the real19 execution qualification checkpoint')
        schema=joint.parameter_schema(net,value)
        result_path=root/c['training_result']['path']
        export=joint_artifacts.export(root,training_source,result_path,c['training_result']['sha256'],
                                      args.output/'candidate',schema=schema)
        state,params=joint_artifacts.load(args.output/'candidate',export['manifest_sha256'])
        joint_artifacts.compatible(state,Path(__file__).parent,schema=schema)
        parent=read_json(result_path);owner=Path(parent['latest_checkpoint']['owner_checkpoint_path'])
        _,all_arrays,_=checkpoints.read(owner,expected_manifest_sha256=parent['latest_checkpoint']['manifest_sha256'],array_prefix='p_')
        full_params=joint_artifacts.parameters(all_arrays,schema,inference_only=False)
        initial=jax.device_get(joint.initialize(config['seed'],net,value))
        changed=[name for name,p in params.items() if not np.array_equal(p,initial[name])]
        if (not any(name.startswith('value_head.') for name in changed)
                or not any(not name.startswith('value_head.') for name in changed)):
            raise ValueError('Fixture must contain updated policy/backbone and value parameters')
        del initial
        report.update(export=export,architecture=net['architecture'],changed_inference_arrays=len(changed),
                      changed_value_arrays=sum(name.startswith('value_head.') for name in changed),
                      training_snapshot=training_source.name,training_result_sha256=c['training_result']['sha256'],
                      model=net,value_model=value,network_version=state['network_version'])
        feature=root/c['feature_binary']['path'];feature_sha=c['feature_binary']['sha256']
        feature_evidence=root/c['feature_equivalence']['path']
        if checkpoints.sha256(feature_evidence)!=c['feature_equivalence']['sha256']:
            raise ValueError('Feature compatibility evidence changed')
        joint_artifacts.compatible_features(state,feature_sha,evidence=read_json(feature_evidence))
        report['feature_equivalence_sha256']=c['feature_equivalence']['sha256']
        nr=c['native'];receipt_path=root/nr['receipt']
        if checkpoints.sha256(receipt_path)!=nr['receipt_sha256']:
            raise ValueError('Rust inference receipt changed')
        receipt=read_json(receipt_path)
        if receipt['binary_sha256']!=nr['binary_sha256']:
            raise ValueError('Rust inference binary identity differs')
        native=load_library(receipt_path.parent/receipt['filename'],nr['binary_sha256'])

        data=Path(config['dataset']['path']);manifest=read_json(data/'manifest.json')
        if checkpoints.sha256(data/'manifest.json')!=state['dataset_manifest_sha256']:
            raise ValueError('Execution fixture changed')
        if len(manifest['shards'])!=1:
            raise ValueError('This execution fixture expects one declared shard')
        arrays={}
        for name in ('actions','expert_offsets','games'):
            item=manifest['shards'][0]['files'][name];path=data/item['path']
            if checkpoints.sha256(path)!=item['sha256']:
                raise ValueError('Fixture history changed')
            arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        offsets=arrays['expert_offsets']
        indices=[i for i,row in enumerate(arrays['games']) if row['split']==0 and offsets[i+1]-offsets[i]>=20][:2]
        if len(indices)!=2:raise ValueError('Need two complete training histories of at least20 plies')
        tapes=[arrays['actions'][int(offsets[i]):int(offsets[i])+20].tolist() for i in indices]
        report['fixture_training_rows']=indices
        name='cnn' if net['architecture']=='katago_nested_policy' else 'transformer'
        maximum={};compared=0;loaded_reference_compared=0;events=[]
        def compare(actual,expected,scope):
            nonlocal compared
            if set(actual)!=set(expected) or set(actual)!={'policy','value_logits','value'}:
                raise ValueError('Unexpected joint prediction heads')
            for key in expected:
                error=float(np.max(np.abs(actual[key]-expected[key])))
                maximum[scope+'.'+key]=max(maximum.get(scope+'.'+key,0.),error)
                np.testing.assert_allclose(actual[key],expected[key],atol=5e-5,rtol=5e-4)
            logits=np.asarray(actual['value_logits'],np.float64);prob=np.exp(logits-logits.max());prob/=prob.sum()
            np.testing.assert_allclose(actual['value'],prob[0]-prob[1],atol=1e-7,rtol=1e-5)
            compared+=1
        forward=jax.jit(lambda p,b:joint.forward(p,b,net))
        def reference(rows,histories):
            nonlocal loaded_reference_compared
            length=1 if name=='cnn' else 1<<(max(len(h)+1 for h in histories)-1).bit_length()
            batch=dict(spatial=np.zeros((4,length,19,19,22),np.float32),
                       global_features=np.zeros((4,length,19),np.float32),
                       actions=np.zeros((4,length),np.int32),counts=np.zeros(4,np.int32))
            for slot,(row,history) in enumerate(zip(rows,histories)):
                n=1 if name=='cnn' else len(history)+1;batch['counts'][slot]=n
                batch['spatial'][slot,:n]=row['spatial'][-1:] if name=='cnn' else row['spatial']
                batch['global_features'][slot,:n]=row['global_features'][-1:] if name=='cnn' else row['global_features']
                if name!='cnn':batch['actions'][slot,:len(history)]=history
            batch=jax.tree.map(jnp.asarray,batch)
            original=jax.device_get(forward(full_params,batch));loaded=jax.device_get(forward(params,batch))
            result=[]
            for slot,history in enumerate(histories):
                index=0 if name=='cnn' else len(history)
                a={key:original[key][slot,index] for key in original}
                b={key:loaded[key][slot,index] for key in loaded}
                compare(b,a,'export_vs_checkpoint');loaded_reference_compared+=1;result.append(a)
            return result

        version=state['network_version'];komi=state['input_contract']['komi']
        with Replay(feature,feature_sha,size=19,komi=komi) as oracle:
            prefix=tapes[0][:6];row=oracle([prefix])[0]
            alternate=next(int(x) for x in np.flatnonzero(row['legal'][-1]) if x not in (tapes[0][6],361))
            batches=[{0:tapes[0][:5],1:tapes[1][:7],2:[],3:[361]},
                     {0:tapes[0][:12],1:tapes[1][:4],2:[0,361,1],3:[0]},
                     {0:prefix+[alternate],1:tapes[1][:9],2:[0,361,1,361,2],3:[]}]
            runner=inference.Runner(params,net,Replay(feature,feature_sha,size=19,komi=komi),
                                    slots=4,network_version=version,max_block=4)
            try:
                for histories in batches:
                    rows=oracle(list(histories.values()));expected=reference(rows,list(histories.values()))
                    request={slot:(h,row_digest(row,h,size=19,komi=komi)) for (slot,h),row in zip(histories.items(),rows)}
                    actual=runner.score(request)
                    for slot,want in zip(histories,expected):compare(actual[slot],want,'cached_vs_checkpoint')
                    previous=runner.stats.copy();again=runner.score(dict(reversed(list(request.items()))))
                    if any(runner.stats[key]!=previous[key] for key in ('dispatches','replay_calls')):
                        raise ValueError('Exact prefix hit unexpectedly dispatched work')
                    for slot,want in zip(histories,expected):compare(again[slot],want,'prefix_hit')

                games=[];pending={};root_values={}
                for slot,history in enumerate((tapes[0][:6],tapes[1][:7])):
                    game=native.Game(json.dumps(dict(size=19,komi=komi,scoring='pass_alive_area',history=1,
                                                     simulations=2,cpuct=1.5,max_search_edges=10000)))
                    for i,action in enumerate(history):game.play(1+i%2,action)
                    games.append(game);pending[slot]=game.start(version)
                while pending:
                    request={};histories={}
                    for slot,(handle,features) in pending.items():
                        history=games[slot].request_history(handle).tolist();histories[slot]=history
                        request[slot]=(history,digest_from_native(features,history=history,size=19,komi=komi))
                    rows=oracle(list(histories.values()));expected=reference(rows,list(histories.values()))
                    actual=runner.score(request)
                    for (slot,history),want,row in zip(histories.items(),expected,rows):
                        if request[slot][1]!=row_digest(row,history,size=19,komi=komi):
                            raise ValueError('Actual Rust leaf differs from native V7 replay')
                        compare(actual[slot],want,'rust_leaf')
                        root_values.setdefault(slot,float(actual[slot]['value']))
                        events.append(dict(slot=slot,depth=len(history),player=1+len(history)%2,
                                           value=float(actual[slot]['value']),state_digest=request[slot][1]))
                        handle,_=pending[slot]
                        following=games[slot].evaluate(handle,np.ascontiguousarray(actual[slot]['policy'],np.float32),float(actual[slot]['value']))
                        if following[0] is None:del pending[slot]
                        else:pending[slot]=following
                searches=[]
                for slot,game in enumerate(games):
                    audit=json.loads(game.inspect_search())
                    chosen,policy,value,simulations,neural,terminal=game.finish()
                    if chosen not in game.legal() or simulations!=2 or not np.isclose(policy.sum(),1.):
                        raise ValueError('Native search result is invalid')
                    np.testing.assert_allclose(audit['network_value'],root_values[slot],atol=1e-7,rtol=0)
                    searches.append(dict(action=chosen,simulations=simulations,neural=neural,terminal=terminal,
                                         value=value,network_value=audit['network_value']))
                report.update(comparisons=compared,loaded_reference_comparisons=loaded_reference_compared,
                              maximum_absolute_errors=maximum,searches=searches,search_events=events,
                              runner_stats=runner.stats,compilation=runner.compilation)
            finally:runner.close()
        report.update(status='passed',jax_version=jax.__version__,native_binary_sha256=nr['binary_sha256'],
                      feature_binary_sha256=feature_sha,
                      scope='Small trained joint19 execution checkpoints. Exact main parameter export/read-back, loaded/full/cached policy and value equivalence, prefix/branch reuse and two actual Rust MCTS searches. No new training, full-size TPU memory/throughput, RPC/GTP or playing-strength claim.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['seconds']=time.monotonic()-started
        path=args.output/'result.json'
        with path.open('xb') as stream:stream.write(canonical_json(report))
        path.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],result_sha256=checkpoints.sha256(path))),flush=True)
