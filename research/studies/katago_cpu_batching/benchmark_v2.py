"""Bounded raw-teacher batching pilot, isolated from corpus producers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
    path.chmod(0o444)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    if sha(args.plan)!=args.plan_sha256:raise ValueError('Plan changed')
    plan=json.loads(args.plan.read_text())
    if plan['kind']!='katago_cpu_batching_pilot' or sha(Path(__file__))!=plan['operator_sha256']:
        raise ValueError('Wrong plan/operator')
    if sorted(os.sched_getaffinity(0))!=plan['cpus'] or len(plan['cpus'])!=2:
        raise ValueError('Use only the explicit auxiliary CPU allocation')
    if plan['concurrency_order']!=[8,16,32,32,16,8] or plan['queries']!=64 or plan['warmup_queries']!=8:
        raise ValueError('Unregistered pilot dimensions')
    directory=args.plan.parent/'run-002';directory.mkdir(exist_ok=False)
    environment=Path(plan['producer_environment'])
    producer=json.loads((environment/'snapshot.json').read_text())
    if producer['snapshot']!=plan['producer_snapshot']:raise ValueError('Producer source changed')
    for name in ('__init__.py','gtp.py','data/__init__.py','data/katago.py','data/label.py'):
        if sha(environment/'site-packages/flygo'/name)!=producer['source_sha256'][name]:
            raise ValueError('Teacher transport changed')
    sys.path.insert(0,str(environment/'site-packages'))
    from flygo.data.katago import AnalysisClient,query,model_path
    from flygo.data.label import parse_label
    from gozero.corpus_sequence_batches import Dataset
    from gozero.corpus_format import unpack_legal
    import numpy as np
    contract=json.loads(Path(plan['contract']).read_text())
    if hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()!=plan['contract_id']:
        raise ValueError('Producer contract differs')
    binary=Path(plan['storage_root'])/'artifacts'/contract['engine_sha256']/'katago'
    model=model_path(Path(plan['storage_root']),contract['teacher'])
    if sha(binary)!=plan['engine_sha256'] or sha(model)!=plan['teacher_sha256']:
        raise ValueError('Teacher engine/weights changed')
    dataset=Dataset(plan['dataset']['path'],plan['dataset']['manifest_sha256'])
    if dataset.size!=19 or dataset.manifest['target_teacher_sha256']!=plan['teacher_sha256']:
        raise ValueError('Wrong real-board fixture or teacher')
    candidates={}
    for shard,episode in dataset.indices['expert',0]:
        arrays=dataset.shards[shard]
        start,end=map(int,arrays['expert_offsets'][episode:episode+2])
        actions=arrays['actions'][start:end]
        for numerator in range(1,17):
            length=(end-start-1)*numerator//17
            if length<1:continue
            history=actions[:length].astype(int).tolist()
            identity=hashlib.sha256(canonical_json(history)).hexdigest()
            candidates[identity]=dict(actions=history,
                legal=np.flatnonzero(unpack_legal(arrays['legal'][start+length:start+length+1],19)[0]).tolist(),
                teacher_policy=arrays['policies'][start+length].copy(),
                teacher_value=float(arrays['values'][start+length]),
                game_id=bytes(arrays['games'][episode]['game_id']).decode())
    if len(candidates)<plan['queries']:raise ValueError('Too few distinct training prefixes')
    chosen=sorted(candidates)[:plan['queries']];positions=[candidates[k] for k in chosen]
    requests=[query(p['actions'],visits=1,size=19,komi=contract['komi']) for p in positions]
    selection=dict(training_only=True,queries=len(chosen),history_ids=chosen,
        histories=[dict(game_id=p['game_id'],actions=p['actions'],legal=p['legal']) for p in positions],
        history_lengths=[len(p['actions']) for p in positions])
    publish(directory/'selection.json',selection)
    reference_policy=reference_value=None;cases=[];started=time.monotonic()
    result=dict(kind='katago_cpu_batching_pilot_result',status='running',created=time.time(),
        plan_sha256=args.plan_sha256,selection_sha256=sha(directory/'selection.json'))
    try:
        for index,concurrency in enumerate(plan['concurrency_order']):
            if time.monotonic()-started>plan['max_seconds']:raise TimeoutError('Pilot time bound reached')
            output=directory/f'case-{index:02d}-c{concurrency}'
            cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
            with AnalysisClient(binary,model,output,plan['cpus'],concurrency=concurrency,
                    eigen_threads=2,max_pending=64,size=19) as client:
                identity=client.request(dict(action='query_models'))
                publish(output/'identity.json',identity)
                futures=[client.submit(q) for q in requests[:plan['warmup_queries']]]
                for f in futures:f.result(timeout=600)
                client.request(dict(action='clear_cache'))
                completed=[None]*len(requests);sent=[];futures=[];before=time.perf_counter()
                for i,q in enumerate(requests):
                    sent.append(time.perf_counter());f=client.submit(q)
                    f.add_done_callback(lambda _,i=i:completed.__setitem__(i,time.perf_counter()))
                    futures.append(f)
                responses=[f.result(timeout=600) for f in futures]
                elapsed=time.perf_counter()-before
                # EOF after all responses requests normal engine shutdown, so
                # its real NN row/batch counters can be retained.
                client.process.stdin.close();client.process.wait(timeout=30)
                if client.process.returncode!=0:raise ValueError('Teacher did not exit normally')
            cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
            labels=[parse_label(r,p['legal'],1+len(p['actions'])%2,size=19) for r,p in zip(responses,positions)]
            policy=np.asarray([r['raw_policy'] for r in labels]);value=np.asarray([r['raw_value'] for r in labels])
            if reference_policy is None:reference_policy=policy.copy();reference_value=value.copy()
            policy_delta=float(np.max(np.abs(policy-reference_policy)))
            value_delta=float(np.max(np.abs(value-reference_value)))
            teacher_policy=np.asarray([p['teacher_policy'] for p in positions]);teacher_value=np.asarray([p['teacher_value'] for p in positions])
            label_policy_delta=float(np.max(np.abs(policy-teacher_policy)))
            label_value_delta=float(np.max(np.abs(value-teacher_value)))
            if max(policy_delta,value_delta,label_policy_delta,label_value_delta)>plan['absolute_output_tolerance']:
                raise ValueError('Raw teacher outputs differ beyond registered tolerance')
            log=(output/'stderr.log').read_text();counts={}
            for key,pattern in [('rows',r'NN rows: ([0-9]+)'),('batches',r'NN batches: ([0-9]+)'),('average_batch',r'NN avg batch size: ([0-9.eE+\-]+)')]:
                values=re.findall(pattern,log)
                if len(values)!=1:raise ValueError('Missing or ambiguous actual NN counters')
                counts[key]=float(values[0]) if key=='average_batch' else int(values[0])
            expected_rows=plan['queries']+plan['warmup_queries']
            if counts['rows']!=expected_rows:raise ValueError('Unexpected NN cache reuse or additional evaluations')
            array_path=output/'outputs.npz';np.savez(array_path,policy=policy,value=value)
            row=dict(index=index,concurrency=concurrency,eigen_threads=2,queries=len(requests),
                seconds=elapsed,queries_per_second=len(requests)/elapsed,
                burst_response_latency_seconds=np.quantile(np.asarray(completed)-np.asarray(sent),[.5,.95,1]).tolist(),
                all_process_cpu_seconds=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
                nn_counters_including_warmup=counts,policy_max_abs_difference=policy_delta,
                value_max_abs_difference=value_delta,stored_label_policy_max_abs_difference=label_policy_delta,
                stored_label_value_max_abs_difference=label_value_delta,output_sha256=sha(array_path),
                configuration_sha256=sha(output/'analysis.cfg'),stderr_sha256=sha(output/'stderr.log'))
            publish(output/'result.json',row);cases.append(row)
            print(json.dumps(row),flush=True)
        means={str(c):float(np.exp(np.mean([np.log(r['queries_per_second']) for r in cases if r['concurrency']==c]))) for c in (8,16,32)}
        result.update(status='passed',geometric_mean_queries_per_second=means,
            relative_throughput_to_c8={key:value/means['8'] for key,value in means.items()})
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result.update(cases=cases,seconds=time.monotonic()-started,completed=time.time(),
            scope='Two auxiliary CPU cores; raw one-visit teacher inference on '
                '64 fixed training prefixes. Mirrored case order, warmup then '
                'cleared NN cache, actual NN counters and output/label checks. '
                'Burst latency includes queuing. This is a throughput pilot, '
                'not a live-producer change, 16-visit gameplay benchmark, full '
                'CPU-allocation result or model-learning experiment.')
        publish(directory/'result.json',result)
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],completed_cases=len(cases))),flush=True)


if __name__=='__main__':main()
