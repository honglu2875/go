"""Check checkpoint exports through real native search and the JAX GTP adapter."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.snapshots import canonical_json,read_json,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--artifacts-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();verify(SOURCE)
    spec=args.spec.resolve()
    if not spec.is_relative_to(SOURCE):raise ValueError('Qualification spec must be frozen')
    c=read_json(spec);root=args.artifacts_root.resolve();descriptors={}
    for name in ('legacy','final','intermediate'):
        path=(SOURCE/c[name]).resolve()
        if not path.is_relative_to(SOURCE):raise ValueError('Descriptors must be frozen')
        descriptors[name]=(path,read_json(path))
    if descriptors['legacy'][1]['model_export_sha256']!=descriptors['final'][1]['model_export_sha256']:
        raise ValueError('Final export bytes differ from original training export')
    if descriptors['intermediate'][1]['network_version']>=descriptors['final'][1]['network_version']:
        raise ValueError('Expected an earlier checkpoint')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'status':'running','kind':'checkpoint_export_gtp_parity','snapshot_id':SOURCE.name,
            'spec_sha256':sha256(spec),'descriptor_sha256':{n:sha256(x[0]) for n,x in descriptors.items()},
            'claims_go_strength':False,'moves':[]}
    environment={**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def command(name,index):
        path,d=descriptors[name];native=root/'.gozero/native'/d['training_snapshot']/'receipt.json'
        return ['taskset','-c',','.join(map(str,range(100+4*index,104+4*index))),sys.executable,'-B',
                str(SOURCE/'eval/learned_gtp.py'),'--candidate',str(path),'--native-receipt',str(native),
                '--artifacts-root',str(root),'--simulations','8','--cpuct','0']
    try:
        with ExitStack() as stack:
            clients={name:stack.enter_context(GTPClient(command(name,i),args.output/name,environment=environment))
                     for i,name in enumerate(('legacy','final','intermediate'))}
            for client in clients.values():
                client.command('boardsize 3');client.command('komi 0.5');client.command('clear_board')
            passes=0
            for ply in range(128):
                color='B' if ply%2==0 else 'W'
                move=clients['legacy'].command('genmove '+color)
                if move!=clients['final'].command('genmove '+color):raise ValueError('Export changed a chosen move')
                stats=[json.loads(clients[n].command('gozero-search-stats')) for n in ('legacy','final')]
                for stat in stats:stat.pop('seconds')
                if stats[0]!=stats[1]:raise ValueError('Export changed search values or work')
                boards=[clients[n].command('showboard') for n in ('legacy','final')]
                if boards[0]!=boards[1]:raise ValueError('Export changed the real board')
                report['moves'].append({'color':color,'move':move,'search':stats[0]})
                passes=passes+1 if move.lower()=='pass' else 0
                if passes==2:break
            else:raise ValueError('Parity game did not finish inside its qualification cap')
            scores=[clients[n].command('final_score') for n in ('legacy','final')]
            if scores[0]!=scores[1]:raise ValueError('Export changed terminal score')
            report['score']=scores[0];report['exact_final_game_and_search']=True
            report['intermediate_move']=clients['intermediate'].command('genmove B')
            report['intermediate_search']=json.loads(clients['intermediate'].command('gozero-search-stats'))
            if report['intermediate_search']['simulations']!=8:raise ValueError('Intermediate search budget differs')
        report['status']='passed';verify(SOURCE)
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        (args.output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k!='moves'}),flush=True)


if __name__=='__main__':main()
