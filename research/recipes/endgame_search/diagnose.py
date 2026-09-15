#!/usr/bin/env python3
"""Reproduce retained decisions and compare fixed-model value-rescaling searches."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(SOURCE/'packages/gozero/src'),str(SOURCE/'eval')]
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json,read_json,verify
from corpus import build,qualified_cases,require


def inputs(root,c):
    path=artifact(root,'.gozero/native/'+c['native_snapshot']+'/receipt.json');receipt=read_json(path)
    verify(artifact(root,'.gozero/snapshots/'+c['native_snapshot']))
    native=load_library(path.parent/receipt['filename'],receipt['binary_sha256'])
    require(native.SEARCH_INSPECTION_ABI_VERSION==1,'Inspection ABI differs')
    return path,receipt,native


def search(engine,contract,case,rescale,cache):
    import numpy as np
    config={**contract,'gumbel':{**contract['gumbel'],'rescale_values':rescale}}
    game=engine.native.Game(json.dumps(config));engine.game=game;engine.plies=case['ply']
    for ply,a in enumerate(case['prefix']):game.play(1+ply%2,a)
    state=game.state();require(state[1]==1+case['ply']%2 and not state[2]
        and state[4].tolist()==case['stones']and game.legal()==case['legal'],'Search root differs from qualified corpus')
    request,features=game.start(engine.network);leaves=[]
    while request is not None:
        history=tuple(int(a)for a in game.request_history(request));digest=hashlib.sha256(features.tobytes()).hexdigest()
        require(history[:case['ply']]==tuple(case['prefix']),'Search changed the root history')
        hit=history in cache
        if hit:
            logits,value,expected=cache[history];require(expected==digest,'Identical history changed neural inputs')
        else:
            logits,value=engine.evaluate_leaf(request,features);logits=np.ascontiguousarray(logits,np.float32);value=float(np.float32(value))
            require(logits.shape==(82,)and np.isfinite(logits).all()and np.isfinite(value)and abs(value)<=1.,'Invalid neural prediction')
            cache[history]=(logits,value,digest)
        leaves.append({'suffix':list(history[case['ply']:]),'features_sha256':digest,'logits':logits.tolist(),
            'value':value,'reused_prediction':hit,'is_root':bool(request[3])})
        request,features=game.evaluate(request,logits,value)
    inspection=json.loads(game.inspect_search());chosen,policy,value,simulations,neural,terminal=game.finish()
    require(chosen==inspection['chosen']and float(np.float32(inspection['root_value']))==value
        and np.array_equal(np.asarray(inspection['policy'],np.float32),policy),'Inspection changed finish')
    require(simulations==contract['simulations']==inspection['root_visits']==sum(inspection['visits'])
        and neural==len(leaves)and neural+terminal==simulations+1,'Search accounting differs')
    require(game.state()[:4]==state[:4]and np.array_equal(game.state()[4],state[4]),'Search changed its native root')
    require(leaves[0]['is_root']and leaves[0]['suffix']==[]and float(np.float32(inspection['network_value']))==leaves[0]['value'],'Root prediction differs')
    if inspection['terminal_child_values'][81]is not None:
        expected=case['pass_terminal_value'];require(expected is not None and -inspection['terminal_child_values'][81]==expected
            and inspection['action_values'][81]==expected and inspection['value_sums'][81]==expected*inspection['visits'][81],
            'Terminal pass backup has wrong value/perspective')
    return {'rescale_values':rescale,'inspection':inspection,'leaves':leaves}


def pass_value(native,contract,case):
    game=native.Game(json.dumps({**contract,'simulations':0,'gumbel':None}))
    for ply,a in enumerate(case['prefix']):game.play(1+ply%2,a)
    game.play(1+case['ply']%2,81);_,_,terminal,score,_=game.state()
    require(terminal==(case['category']=='after_pass'),'Counterfactual pass termination differs')
    if not terminal:return None,None
    white_value=float((score>0)-(score<0));return (white_value if case['ply']%2 else -white_value),score


def worker(a,c,corpus):
    os.environ['JAX_PLATFORMS']='cpu'
    import numpy as np
    from evaluators import student,teacher
    receipt_path,receipt,native=inputs(a.workspace_root,c)
    learner,inference=student(a.workspace_root,SOURCE,c,receipt_path)
    contract={k:inference[k]for k in ('size','komi','scoring','history','simulations','cpuct','fpu_reduction','gumbel','max_search_edges')}
    mentor,teacher_contract,teacher_identity=teacher(a.workspace_root,c,native)
    cases=qualified_cases(corpus,c['qualification_positions'])if a.phase=='qualify'else corpus['cases']
    cases=cases[a.worker::len(c['worker_cpus'])]
    report={'schema_version':1,'kind':'endgame_diagnostic_worker','status':'running','operator_snapshot':SOURCE.name,
        'phase':a.phase,'worker':a.worker,'config_sha256':sha256(SOURCE/'resolved_config.json'),'corpus_sha256':sha256(a.corpus),
        'native_receipt_sha256':sha256(receipt_path),'native':receipt,'teacher_identity':teacher_identity,
        'candidate_sha256':sha256(SOURCE/c['candidate']),'inference_sha256':sha256(SOURCE/c['inference']),
        'started_unix':time.time(),'case_ids':[],'recorded_decision_mismatches':[]}
    try:
        with (a.output/'cases.jsonl.gz').open('xb')as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0)as stream:
                for i,original in enumerate(cases):
                    case=dict(original);case['pass_terminal_value'],case['pass_terminal_white_score']=pass_value(native,contract,case)
                    results={}
                    for name,engine,settings in [('student',learner,contract),('teacher',mentor,teacher_contract)]:
                        cache={}
                        for rescale in (True,False):results[name+('_on'if rescale else'_off')]=search(engine,settings,case,rescale,cache)
                    original=case['recorded_search'];reproduced=results['student_on']['inspection']
                    same=(reproduced['chosen']==case['recorded_action']and np.float32(reproduced['root_value'])==np.float32(original['root_value'])
                        and reproduced['completed_simulations']==original['simulations']and reproduced['neural_evaluations']==original['neural_evaluations']
                        and reproduced['terminal_evaluations']==original['terminal_evaluations'])
                    if not same:report['recorded_decision_mismatches'].append(case['case_id'])
                    stream.write(canonical_json({'case':case,'recorded_decision_exact':bool(same),'searches':results}))
                    report['case_ids'].append(case['case_id'])
                    if (i+1)%25==0:print(json.dumps({'worker':a.worker,'cases':i+1,'mismatches':len(report['recorded_decision_mismatches'])}),flush=True)
        report.update(status='passed'if not report['recorded_decision_mismatches']else'failed',cases_sha256=sha256(a.output/'cases.jsonl.gz'))
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();(a.output/'result.json').write_bytes(canonical_json(report))
        for path in a.output.iterdir():
            if path.is_file():path.chmod(0o444)
        print(json.dumps({k:report[k]for k in ('worker','status','started_unix','finished_unix')}),flush=True)
    require(report['status']=='passed','Recorded decision reproduction failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','qualify','run'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path);p.add_argument('--expected-corpus-sha256');p.add_argument('--worker',type=int)
    p.add_argument('--protocol',type=Path);p.add_argument('--expected-protocol-sha256');a=p.parse_args();verify(SOURCE)
    a.workspace_root=a.workspace_root.resolve();a.output=a.output.resolve();c=read_json(SOURCE/'resolved_config.json')
    require(c['kind']=='pass_value_rescaling_diagnostic'and c['schema_version']==1,'Wrong diagnostic configuration')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.phase=='prepare':
        result={'schema_version':1,'kind':'endgame_corpus_preparation','operator_snapshot':SOURCE.name,'started_unix':time.time(),'status':'running'}
        try:
            receipt,_,native=inputs(a.workspace_root,c);corpus=build(a.workspace_root,c,native)
            corpus.update(operator_snapshot=SOURCE.name,native_receipt_sha256=sha256(receipt))
            path=a.output/'corpus.json';path.write_bytes(canonical_json(corpus));path.chmod(0o444)
            result.update(status='passed',corpus_sha256=sha256(path),cases=len(corpus['cases']),
                checked_boards=corpus['source_boards_checked'],checked_scores=corpus['source_scores_checked'])
        except BaseException as e:result.update(status='failed',error=repr(e));raise
        finally:
            result['finished_unix']=time.time();path=a.output/'result.json';path.write_bytes(canonical_json(result));path.chmod(0o444);print(json.dumps(result))
        return
    require(a.corpus is not None and sha256(a.corpus)==a.expected_corpus_sha256,'Corpus changed');corpus=read_json(a.corpus)
    if a.worker is not None:
        require(0<=a.worker<len(c['worker_cpus']),'Invalid worker index');worker(a,c,corpus);return
    require(a.protocol is not None and sha256(a.protocol)==a.expected_protocol_sha256,'Protocol missing or changed')
    protocol=read_json(a.protocol);require(protocol['snapshot']==SOURCE.name and protocol['phase']==a.phase
        and protocol['corpus_sha256']==a.expected_corpus_sha256 and protocol['registered_unix']<time.time()
        and protocol['maximum_attempts']==1 and a.output==a.workspace_root/protocol['output'], 'Protocol differs')
    start=time.time();report={'schema_version':1,'kind':'endgame_diagnostic_attempt','snapshot':SOURCE.name,'phase':a.phase,
        'protocol_sha256':a.expected_protocol_sha256,'corpus_sha256':a.expected_corpus_sha256,'started_unix':start,'workers':[],'status':'running'}
    def launch(rank):
        directory=a.output/f'worker-{rank}';command=['taskset','-c',','.join(map(str,c['worker_cpus'][rank])),sys.executable,'-B',str(Path(__file__).resolve()),
            '--workspace-root',str(a.workspace_root),'--phase',a.phase,'--output',str(directory),'--corpus',str(a.corpus),
            '--expected-corpus-sha256',a.expected_corpus_sha256,'--worker',str(rank)]
        before=time.time()
        try:
            with (a.output/f'worker-{rank}.log').open('w')as log:
                result=subprocess.run(command,env={**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                    stdout=log,stderr=subprocess.STDOUT,timeout=c['maximum_worker_seconds'])
            return {'rank':rank,'returncode':result.returncode,'seconds':time.time()-before}
        except BaseException as e:return {'rank':rank,'returncode':-1,'error':repr(e),'seconds':time.time()-before}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(c['worker_cpus']))as pool:
        report['workers']=list(pool.map(launch,range(len(c['worker_cpus']))))
    report.update(status='passed'if all(r['returncode']==0 for r in report['workers'])else'failed',finished_unix=time.time())
    verify(SOURCE);require(sha256(a.protocol)==a.expected_protocol_sha256,'Protocol changed')
    (a.output/'result.json').write_bytes(canonical_json(report))
    for path in a.output.iterdir():
        if path.is_file():path.chmod(0o444)
    print(json.dumps(report));require(report['status']=='passed','Diagnostic worker failed')


if __name__=='__main__':main()
