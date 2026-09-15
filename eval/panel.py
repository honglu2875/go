#!/usr/bin/env python3
"""Run independent frozen match specifications concurrently on disjoint CPU cores."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json,read_json,verify
from match import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--artifacts-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE)
    spec=args.spec.resolve()
    if not spec.is_relative_to(SOURCE): raise ValueError('Panel spec must be frozen')
    c=read_json(spec)
    if not 1<=c['workers']<=8 or not 1<=len(c['matches'])<=64: raise ValueError('Bounded panel required')
    used=set();names=set();inputs=[]
    for item in c['matches']:
        if not re.fullmatch('[a-z0-9_-]+',item['id']) or item['id'] in names: raise ValueError('Invalid match identity')
        names.add(item['id']);path=(SOURCE/item['spec']).resolve()
        if not path.is_relative_to(SOURCE): raise ValueError('Child specification must be frozen')
        child=read_json(path)
        cpus=child['candidate_cpus']+child['katago_cpus']
        if len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0): raise ValueError('Invalid child affinity')
        # Include hardware siblings: distinct Linux CPU numbers alone are not isolation.
        cores={Path(f'/sys/devices/system/cpu/cpu{x}/topology/thread_siblings_list').read_text().strip() for x in cpus}
        if used&cores: raise ValueError('Concurrent panel specifications share physical cores')
        used|=cores
        if not 0<child['game_timeout_seconds']<=600 or not 1<=len(child['openings'])<=16: raise ValueError('Unbounded child panel')
        inputs.append((item,path,child))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'kind':'parallel_katago_panel','status':'running','snapshot_id':SOURCE.name,
            'spec_sha256':sha256(spec),'started_unix':time.time(),'claims_go_strength_improvement':False,'matches':[]}
    def execute(value):
        item,path,child=value;directory=output/item['id']
        argv=[sys.executable,'-B',str(SOURCE/'eval/match.py'),'--spec',str(path),
              '--artifacts-root',str(args.artifacts_root.resolve()),'--output',str(directory)]
        timeout=2*len(child['openings'])*child['game_timeout_seconds']+60
        start=time.time();timed_out=False
        print(json.dumps({'kind':'panel_match_start','id':item['id']}),flush=True)
        with (output/(item['id']+'.log')).open('w') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;process.terminate()
                try:code=process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();code=process.wait()
        result_path=directory/'result.json'
        result=read_json(result_path) if result_path.exists() else {'status':'failed','error':'child result missing'}
        entry={'id':item['id'],'group':item['group'],'returncode':code,'timed_out':timed_out,
               'elapsed_seconds':time.time()-start,'status':'passed' if code==0 and not timed_out and result['status']=='passed' else 'failed',
               'result_sha256':sha256(result_path) if result_path.exists() else None,'summary':result.get('summary'),'games':result.get('games',[])}
        print(json.dumps({'kind':'panel_match_finished',**{k:v for k,v in entry.items() if k!='games'}}),flush=True)
        return entry
    try:
        with ThreadPoolExecutor(max_workers=c['workers']) as pool:
            for future in as_completed([pool.submit(execute,value) for value in inputs]):
                report['matches'].append(future.result())
                (output/'progress.json').write_bytes(canonical_json(report))
        report['matches'].sort(key=lambda x:x['id'])
        groups={}
        for entry in report['matches']:
            games=groups.setdefault(entry['group'],[])
            pair_offset=1+max((g['pair'] for g in games),default=-1)
            games.extend({**game,'pair':game['pair']+pair_offset,'panel_match':entry['id']} for game in entry['games'])
        report['summaries']={group:summarize(games) for group,games in groups.items()}
        report['status']='passed' if all(x['status']=='passed' for x in report['matches']) else 'failed'
        verify(SOURCE)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();(output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k!='matches'}),flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
