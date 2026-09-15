"""Start the registered short TPU job and arm its injector at launch, automatically."""
import argparse
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.pod import SSH_OPTIONS,load_hosts,pdsh_command,pdsh_environment
from gozero.snapshots import canonical_json,read_json,verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=read_json(args.protocol)
    if p['kind']!='automatically_armed_tpu_interruption' or p['orchestrator_sha256']!=sha256(Path(__file__)):
        raise ValueError('Unexpected protocol or orchestrator identity')
    if p['controller_sha256']!=sha256(ROOT/'ops/pod_run.py') or p['supervisor_sha256']!=sha256(ROOT/'packages/gozero/src/gozero/pod.py'):
        raise ValueError('Operational source changed')
    source=ROOT/'.gozero/snapshots'/p['snapshot_id'];verify(source)
    receipt=ROOT/'.gozero/native'/source.name/'receipt.json'
    if read_json(receipt)['binary_sha256']!=p['native_binary_sha256']:raise ValueError('Native identity differs')
    hosts=load_hosts(source/'ops/hosts.json');host=hosts[p['failure_host_rank']]
    if p['attempt_timeout_seconds']!=60 or p['prepare_timeout_seconds']!=180 or p['resume_turn']!=48:
        raise ValueError('Unexpected qualification bounds')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B',str(ROOT/'ops/pod_run.py'),'--snapshot',str(source),'--native-receipt',str(receipt),
             '--timeout','60','--prepare-timeout','180','--controller-cpus','32']
    report={'schema_version':1,'kind':p['kind'],'status':'failed','protocol_sha256':sha256(args.protocol),
            'orchestrator_sha256':sha256(Path(__file__)),'started_unix':time.time(),'command':command}
    launch=None;injector=None;pod=None
    with (args.output/'pod.stdout.log').open('x') as log,(args.output/'pod.stderr.log').open('x') as err,\
         (args.output/'injector.stdout.log').open('x') as inject_out,(args.output/'injector.stderr.log').open('x') as inject_err:
        try:
            pod=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,start_new_session=True)
            select=selectors.DefaultSelector();select.register(pod.stdout,selectors.EVENT_READ)
            deadline=time.monotonic()+360
            while True:
                if time.monotonic()>deadline:raise TimeoutError('Fault orchestration deadline expired')
                ready=select.select(.1)
                if not ready:
                    if pod.poll() is not None:break
                    continue
                line=pod.stdout.readline()
                if not line:break
                log.write(line);log.flush()
                event=json.loads(line)
                if event.get('kind')=='pod_attempt':
                    if launch is not None or event['snapshot_id']!=source.name:raise ValueError('Unexpected launch identity')
                    launch=event;report['attempt']=launch['attempt_id']
                    environment=ROOT/'.gozero/environments'/launch['runtime_key']/'bin/python'
                    argv=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','taskset','-c','100-103',str(environment),'-B',
                          str(source/'ops/inject_failure.py'),'--snapshot',str(source),'--attempt',launch['attempt'],
                          '--checkpoint-turn','48','--timeout','90']
                    inject_command=['ssh',*SSH_OPTIONS,host.ssh,shlex.join(argv)]
                    report['injection_command']=inject_command
                    injector=subprocess.Popen(inject_command,stdout=inject_out,stderr=inject_err,start_new_session=True)
                    print(json.dumps({'kind':'injector_armed','attempt':launch['attempt_id'],'host':host.rank}),flush=True)
            report['pod_returncode']=pod.wait(timeout=30)
            if injector is None:raise RuntimeError('No injector was armed')
            report['injector_returncode']=injector.wait(timeout=100)
            if report['injector_returncode']!=0:raise RuntimeError('Injection did not complete')
            attempt=Path(launch['attempt']);failure=read_json(attempt/f"rank-{host.rank}/injected_failure.json")
            if failure['status']!='sent' or failure['checkpoint_turn']!=48 or failure['snapshot_id']!=source.name:
                raise ValueError('Wrong injection receipt')
            ranks=[read_json(attempt/f'rank-{i}/result.json') for i in range(4)]
            if any(r['status']!='failed' or r['timed_out'] or not r['source_integrity'] for r in ranks):
                raise ValueError('Peers did not stop as required')
            if ranks[host.rank]['returncode']!=-9:raise ValueError('Injected trainer exit differs')
            report.update(status='passed',failure=failure,rank_exit_codes=[r['returncode'] for r in ranks],
                          peer_cancelled=[r['cancelled'] for r in ranks],pod_result_sha256=sha256(attempt/'result.json'))
        except BaseException as error:
            report['error']=repr(error)
            if launch is not None and pod is not None and pod.poll() is None:
                cancel=[sys.executable,str(source/'ops/cancel_host.py'),'--snapshot',str(source),'--attempt',launch['attempt']]
                try:subprocess.run(pdsh_command(hosts,cancel,20),env=pdsh_environment(os.environ),stdout=log,stderr=err,timeout=30)
                except Exception as cleanup:report['cleanup_error']=repr(cleanup)
            raise
        finally:
            for process in (injector,pod):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            report['finished_unix']=time.time();(args.output/'result.json').write_bytes(canonical_json(report))
            print(json.dumps(report),flush=True)


if __name__=='__main__':main()
