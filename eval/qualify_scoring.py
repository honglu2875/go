#!/usr/bin/env python3
"""Frozen pointwise ownership differential plus real KataGo GTP scoring regressions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.gtp import GTPClient
from gozero.native import load_library
from gozero.snapshots import canonical_json,read_json,verify
from gozero.scoring import direct_area,score_string
from qualify_katago import kata_cells
from learned_gtp import action,vertex


def run(args):
    import numpy as np
    verify(SOURCE)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    root=args.artifacts_root.resolve();receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=SOURCE.name: raise ValueError('Native source differs')
    native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
    pinned=read_json(SOURCE/'eval/katago_build.json')
    external=root/pinned['source_path']/'cpp'
    relative=['game/board.cpp']+['core/'+x+'.cpp' for x in ('global','rand','hash','sha2','md5','timer','bsearch')]
    sources=[external/x for x in relative]
    inputs={str(p.relative_to(external)):sha256(p) for p in external.rglob('*') if p.is_file() and p.suffix in ('.h','.cpp') and p.parts[-2] in ('core','game')}
    oracle=output/'katago-area-oracle'
    cmd=['c++','-std=c++17','-O2','-pthread','-DCOMPILE_MAX_BOARD_LEN=25','-I',str(external),str(SOURCE/'eval/katago_area_oracle.cpp'),*map(str,sources),'-o',str(oracle)]
    report={'schema_version':1,'kind':'pass_alive_scoring_qualification','status':'running','snapshot_id':SOURCE.name,
            'native_sha256':receipt['binary_sha256'],'katago_commit':pinned['commit'],
            'oracle_external_source_sha256':inputs,'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
            'build_command':cmd,'started_unix':time.time(),'seed':27,'cases':0,'points':0,'raw_margin_differences':0,'gtp_games':[]}
    try:
        with (output/'build.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        report['oracle_sha256']=sha256(oracle)
        random=np.random.default_rng(27)
        cases=[]; expected=[]
        for size in (1,3,5,7,9,13,19,25):
            game=None; steps=0
            for _ in range(args.cases_per_size):
                if game is None or game.state()[2] or steps>=size*size*4:
                    game=native.Game(json.dumps({'size':size,'komi':7.5,'scoring':'pass_alive_area','history':2,'simulations':0,'cpuct':1.5,'max_search_edges':1000}))
                    steps=0
                for _ in range(4):
                    legal=game.legal()
                    if not legal: break
                    # Reach dense boards while retaining some early two-pass endings.
                    a=size*size if random.random()<0.02 else int(random.choice(legal))
                    game.play(game.state()[1],a); steps+=1
                _,_,_,score,stones=game.state()
                flat=''.join('.XO'[int(c)] for c in stones)
                owner=''.join(str(int(c)) for c in game.ownership())
                raw=direct_area('\n'.join(flat[y*size:(y+1)*size] for y in range(size)),7.5)
                if score!=raw: report['raw_margin_differences']+=1
                cases.append(f'{size} {flat}\n');expected.append(owner)
                report['cases']+=1;report['points']+=size*size
        payload=''.join(cases)
        (output/'cases.txt').write_text(payload)
        result=subprocess.run([str(oracle)],input=payload,capture_output=True,text=True,timeout=60)
        (output/'oracle.stderr').write_text(result.stderr)
        (output/'oracle.txt').write_text(result.stdout);(output/'native.txt').write_text('\n'.join(expected)+'\n')
        result.check_returncode()
        actual=result.stdout.splitlines()
        if len(actual)!=len(expected): raise AssertionError('Oracle case count differs')
        for i,(a,b) in enumerate(zip(actual,expected)):
            if a!=b: raise AssertionError(f'Pointwise ownership mismatch at case {i}: {cases[i].strip()} oracle={a} native={b}')
        report['pointwise_parity']=True
        # Replay the four actual candidate-vs-KataGo games which exposed this issue.
        matches=root/'runs/eval/bootstrap-3ab60224/result.json'
        report['regression_match_sha256']=sha256(matches)
        games=read_json(matches)['games']
        weights=read_json(SOURCE/'eval/katago_9x9.json')
        binary=root/pinned['binary_path'];model=root/weights['path']
        if sha256(binary)!=pinned['binary_sha256'] or sha256(model)!=weights['sha256']: raise ValueError('KataGo artifact hash mismatch')
        report['katago_binary_sha256']=sha256(binary);report['katago_weights_sha256']=sha256(model)
        report['katago_config_sha256']=sha256(SOURCE/'eval/katago_9x9_smoke.cfg')
        with GTPClient(['taskset','-c','64-71',str(binary),'gtp','-model',str(model),'-config',str(SOURCE/'eval/katago_9x9_smoke.cfg')],output/'katago') as kata:
            rules=json.loads(kata.command('kata-get-rules'))
            if rules['ko']!='POSITIONAL' or rules['scoring']!='AREA' or not rules['suicide']: raise ValueError('Rules differ')
            for i,saved in enumerate(games):
                kata.command('boardsize 9');kata.command('clear_board');kata.command('komi 7.5')
                game=native.Game(json.dumps({'size':9,'komi':7.5,'scoring':'pass_alive_area','history':2,'simulations':0,'cpuct':1.5,'max_search_edges':1000}))
                for ply,move in enumerate(saved['moves']):
                    game.play(1+ply%2,action(move['vertex'],9));kata.command('play '+move['color']+' '+move['vertex'])
                _,_,terminal,score,stones=game.state()
                if not terminal: raise AssertionError('Regression game is incomplete')
                diagram='\n'.join(''.join('.XO'[int(c)] for c in row) for row in stones.reshape(9,9))
                if kata_cells(kata.command('showboard'),9)!=diagram: raise AssertionError('KataGo terminal board differs')
                external_score=kata.command('final_score')
                entry={'game':i,'plies':len(saved['moves']),'native_score':score_string(score),'katago_score':external_score,'raw_score':score_string(direct_area(diagram,7.5))}
                report['gtp_games'].append(entry)
                if entry['native_score']!=external_score: raise AssertionError('KataGo final_score differs')
        report.update(status='passed',files_sha256={name:sha256(output/name) for name in ('cases.txt','oracle.txt','native.txt')})
        verify(SOURCE)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished_unix']=time.time();(output/'result.json').write_bytes(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k not in ('oracle_external_source_sha256','build_command')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cases-per-size',type=int,default=1024)
    args=parser.parse_args()
    if not 1<=args.cases_per_size<=16384: parser.error('Bounded case count required')
    run(args)
