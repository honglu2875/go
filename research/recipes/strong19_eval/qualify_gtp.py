"""Real concurrent GTP clients, independently reproduced native searches and JAX outputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
sys.path.insert(0,str(SOURCE/'eval'))
from gozero import joint_artifacts
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.joint_rpc import Client,Server
from gozero.native import load_library
from gozero.snapshots import canonical_json,read_json,verify
from gozero.v7_replay import Replay
from gozero.v7_state import digest_from_native,row_digest
from learned_gtp import action
from joint_gtp import contract,identity
import inference
import joint


def run(args,c):
    started=time.monotonic();root=args.workspace_root.resolve();verify(SOURCE)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='trained_joint_gtp_cpu_qualification',status='running',created=time.time(),snapshot=SOURCE.name)
    runner=oracle=server=pool=None
    try:
        if jax.default_backend()!='cpu' or jax.device_count()!=4 or jax.process_count()!=1:
            raise ValueError('Require four simulated CPU devices in one process')
        candidate_path=SOURCE/c['candidate'];search_path=SOURCE/c['search']
        if not candidate_path.resolve().is_relative_to(SOURCE) or not search_path.resolve().is_relative_to(SOURCE):
            raise ValueError('Candidate/search must be in the frozen recipe')
        descriptor=read_json(candidate_path);state,params=joint_artifacts.load_candidate(root,descriptor)
        if state['training_purpose']!='qualification':raise ValueError('CPU execution fixture expected')
        net=state['model'];schema=joint.parameter_schema(net,state['value_model'])
        joint_artifacts.compatible(state,Path(__file__).parent,schema=schema)
        search=contract(read_json(search_path));inputs=state['input_contract']
        if search['size']!=inputs['size'] or search['komi']!=inputs['komi'] or search['cache_positions']>net['max_positions']:
            raise ValueError('Search input/context differs from trained model')
        evidence=root/c['feature_equivalence']['path']
        if sha256(evidence)!=c['feature_equivalence']['sha256']:raise ValueError('Feature equivalence changed')
        feature=root/c['feature_binary']['path'];feature_sha=c['feature_binary']['sha256']
        joint_artifacts.compatible_features(state,feature_sha,evidence=read_json(evidence))
        nr=c['native'];receipt_path=root/nr['receipt']
        if sha256(receipt_path)!=nr['receipt_sha256'] or nr['receipt_sha256']!=search['native_receipt_sha256']:
            raise ValueError('Native receipt differs')
        receipt=read_json(receipt_path)
        if receipt['binary_sha256']!=nr['binary_sha256']:raise ValueError('Native binary differs')
        native=load_library(receipt_path.parent/receipt['filename'],nr['binary_sha256'])
        size,komi=inputs['size'],inputs['komi'];version=state['network_version']
        runner=inference.Runner(params,net,Replay(feature,feature_sha,size=size,komi=komi),slots=4,network_version=version,max_block=4)
        oracle=Replay(feature,feature_sha,size=size,komi=komi)
        forward=jax.jit(lambda p,b:joint.forward(p,b,net))
        comparisons=[];maximum={}
        class AuditedRunner:
            slots=4
            def __init__(self):self.size=size
            def score(self,requests):
                actual=runner.score(requests);histories=[h for h,_ in requests.values()];rows=oracle(histories)
                cnn=net['architecture']=='katago_nested_policy'
                length=1 if cnn else 1<<(max(len(h)+1 for h in histories)-1).bit_length()
                batch=dict(spatial=np.zeros((4,length,size,size,22),np.float32),global_features=np.zeros((4,length,19),np.float32),
                           actions=np.zeros((4,length),np.int32),counts=np.zeros(4,np.int32))
                for i,(row,history) in enumerate(zip(rows,histories)):
                    n=1 if cnn else len(history)+1;batch['counts'][i]=n
                    batch['spatial'][i,:n]=row['spatial'][-1:] if cnn else row['spatial']
                    batch['global_features'][i,:n]=row['global_features'][-1:] if cnn else row['global_features']
                    if not cnn:batch['actions'][i,:len(history)]=history
                full=jax.device_get(forward(params,jax.tree.map(jnp.asarray,batch)))
                for i,((slot,(history,digest)),row) in enumerate(zip(requests.items(),rows)):
                    if row_digest(row,history,size=size,komi=komi)!=digest:raise ValueError('RPC native leaf digest differs')
                    index=0 if cnn else len(history)
                    for key in ('policy','value_logits','value'):
                        expected=full[key][i,index];observed=actual[slot][key]
                        maximum[key]=max(maximum.get(key,0.),float(np.max(np.abs(observed-expected))))
                        np.testing.assert_allclose(observed,expected,atol=5e-5,rtol=5e-4)
                    comparisons.append(dict(slot=slot,plies=len(history),state_digest=digest))
                return actual
        socket=Path('/tmp')/('gozero-joint-'+hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:20]+'.sock')
        service_identity=identity(SOURCE,candidate_path,search_path)
        server=Server(socket,service_identity,AuditedRunner(),gather_seconds=.005,request_timeout=120.)
        native_config={k:search[k] for k in ('size','komi','scoring','simulations','cpuct','fpu_reduction','gumbel','max_search_edges')}
        native_config['history']=1
        def probe(index):
            argv=[sys.executable,'-B',str(SOURCE/'eval/joint_gtp.py'),'--candidate',str(candidate_path),
                  '--inference-config',str(search_path),'--native-receipt',str(receipt_path),'--socket',str(socket)]
            reference_client=Client(socket,service_identity,timeout=120.);records=[]
            def verify_board(client,game):
                wanted='\n'.join(''.join('.XO'[int(stone)] for stone in row) for row in game.state()[4].reshape(size,size))
                if client.command('showboard')!=wanted:raise ValueError('GTP board differs from independent native play')
            try:
                with GTPClient(argv,args.output/f'client-{index}',environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')) as client:
                    client.command('boardsize '+str(size));client.command('komi '+str(komi))
                    if client.command('version').strip()!='gozero-joint-v7-rpc-v1':raise ValueError('Wrong GTP adapter')
                    for round_index in range(2):
                        client.command('clear_board');game=native.Game(json.dumps(native_config,allow_nan=False))
                        if round_index:
                            for color,move,vertex in [(1,0,'A'+str(size)),(2,size*size,'pass')]:
                                client.command('play '+('B' if color==1 else 'W')+' '+vertex);game.play(color,move)
                                verify_board(client,game)
                        for turn in range(4):
                            if game.state()[2]:break
                            color=game.state()[1];handle,features=game.start(version)
                            while handle is not None:
                                history=game.request_history(handle).tolist()
                                digest=digest_from_native(features,history=history,size=size,komi=komi)
                                prediction=reference_client.predict(history,digest)
                                handle,features=game.evaluate(handle,np.ascontiguousarray(prediction['policy'],np.float32),float(prediction['value']))
                            inspect=json.loads(game.inspect_search())
                            chosen,_,root_value,simulations,neural,terminal=game.finish()
                            move=client.command('genmove '+('B' if color==1 else 'W'),timeout=120.).strip()
                            actual=action(move,size)
                            if actual!=chosen:raise ValueError('GTP and independent native searches chose different moves')
                            stats=json.loads(client.command('gozero-search-stats'))
                            if (stats['simulations']!=simulations or stats['neural_evaluations']!=neural
                                    or stats['terminal_evaluations']!=terminal or simulations!=search['simulations']):
                                raise ValueError('GTP search work differs')
                            np.testing.assert_allclose(stats['root_value'],root_value,atol=1e-6,rtol=1e-5)
                            game.play(color,chosen);verify_board(client,game)
                            records.append(dict(round=round_index,turn=turn,action=chosen,simulations=simulations,
                                                neural=neural,terminal=terminal,root_value=root_value,network_value=inspect['network_value']))
                        if game.state()[2]:
                            score=game.state()[3];expected='0' if score==0 else ('W' if score>0 else 'B')+'+'+str(abs(score))
                            if client.command('final_score').strip()!=expected:raise ValueError('GTP final score differs')
                return dict(client=index,reference_slot=reference_client.slot,searches=records)
            finally:reference_client.close()
        pool=ThreadPoolExecutor(2);futures=[pool.submit(probe,index) for index in range(2)]
        deadline=time.monotonic()+300
        while not all(f.done() for f in futures):
            if time.monotonic()>deadline:raise TimeoutError('GTP CPU qualification exceeded its bound')
            server.pump()
        clients=[f.result() for f in futures]
        if not comparisons or any(not row['searches'] for row in clients):raise ValueError('No complete GTP searches')
        report.update(status='passed',architecture=net['architecture'],clients=clients,comparisons=len(comparisons),
                      prediction_records=comparisons,maximum_absolute_errors=maximum,batches=server.records,
                      runner_stats=runner.stats,compilation=runner.compilation,
                      candidate_sha256=sha256(candidate_path),parameter_manifest_sha256=descriptor['parameters']['manifest_sha256'],
                      feature_equivalence_sha256=sha256(evidence),native_binary_sha256=nr['binary_sha256'],
                      scope='Small trained19 models and concurrent real GTP subprocesses. Every RPC prediction matches full JAX inference; independent Rust searches match moves, work, values and boards across resets/passes. No KataGo opponent, complete strength study or full-size TPU throughput result.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        if server is not None:server.close()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if runner is not None:runner.close()
        if oracle is not None:oracle.close()
        report['seconds']=time.monotonic()-started;path=args.output/'result.json'
        with path.open('xb') as stream:stream.write(canonical_json(report))
        path.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha256(path))),flush=True)
