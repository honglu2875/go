#!/usr/bin/env python3
"""Frozen CPU inference/GTP adapter for a pinned recipe-owned JAX model and Rust search."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.native import load_library
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import read_json,verify


COLUMNS='ABCDEFGHJKLMNOPQRSTUVWXYZ'
COMMANDS=('protocol_version','name','version','known_command','list_commands','boardsize','clear_board','komi','play','genmove','showboard','final_score','quit','gozero-search-stats')


def action(vertex,size):
    if vertex.lower()=='pass': return size*size
    match=re.fullmatch(r'([A-HJ-Z])([1-9][0-9]*)',vertex.upper())
    if not match: raise ValueError('invalid vertex')
    x=COLUMNS.index(match[1]); y=size-int(match[2])
    if not 0<=x<size or not 0<=y<size: raise ValueError('vertex outside board')
    return y*size+x


def vertex(a,size):
    return 'pass' if a==size*size else COLUMNS[a%size]+str(size-a//size)


class Engine:
    def __init__(self,args):
        import jax
        import jax.numpy as jnp
        import numpy as np
        self.jax=jax; self.np=np
        verify(SOURCE)
        candidate=read_json(args.candidate)
        identity=validate_candidate(args.artifacts_root,candidate)
        snapshot=identity['snapshot'];manifest=identity['manifest'];weights=identity['weights'];config=identity['config']
        specification=importlib.util.spec_from_file_location('frozen_go_model',snapshot/manifest['recipe']/'model.py')
        model=importlib.util.module_from_spec(specification);specification.loader.exec_module(model)
        self.history=config['actors']['history']; self.net=config['model']; self.size=config['actors']['size']; self.komi=config['actors']['komi']
        self.scoring=config['actors'].get('scoring','raw_area')
        self.fpu_reduction=config['actors'].get('fpu_reduction')
        self.score_utility=config['actors'].get('score_utility')
        self.gumbel=config['actors'].get('gumbel')
        if self.gumbel is not None: self.gumbel={**self.gumbel,'gumbel_scale':0.0}
        if not 1<=self.size<=25: raise ValueError('GTP board size must fit one-letter vertices')
        self.supported_size=self.size
        initialized=model.initialize(config['seed'],2*self.history+4,self.net)
        leaves,definition=jax.tree.flatten(initialized)
        with np.load(weights,allow_pickle=False) as data:
            if set(data.files)!={f'p_{i:04d}' for i in range(len(leaves))}: raise ValueError('parameter tree mismatch')
            saved=[]
            for i,expected in enumerate(leaves):
                p=data[f'p_{i:04d}']
                if p.shape!=expected.shape or str(p.dtype)!=str(expected.dtype) or not np.isfinite(p).all(): raise ValueError('invalid model array')
                saved.append(jnp.asarray(p))
        self.params=definition.unflatten(saved)
        self.forward=jax.jit(lambda x:model.apply(self.params,x,self.net)).lower(jnp.zeros((1,self.size,self.size,2*self.history+4),jnp.float32)).compile()
        receipt=read_json(args.native_receipt)
        if identity['native_receipt'] is None:
            if receipt['snapshot_id']!=snapshot.name: raise ValueError('evaluation native source differs from training source')
        elif args.native_receipt.resolve()!=identity['native_receipt']:
            raise ValueError('Evaluation native receipt differs from the declared training dependency')
        if identity['native_binary_sha256'] is not None and receipt['binary_sha256']!=identity['native_binary_sha256']:
            raise ValueError('Evaluation native binary differs from training binary')
        self.native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
        self.network=candidate['network_version']; self.simulations=args.simulations; self.cpuct=args.cpuct
        self.last_stats=None;self.reset()
        print(json.dumps({'kind':'engine_ready','candidate_sha256':sha256(args.candidate),'training_snapshot':snapshot.name,
                          'weights_sha256':candidate['model_export_sha256'],'native_sha256':receipt['binary_sha256'],
                          'backend':jax.default_backend(),'simulations_excluding_root':self.simulations,'cpuct':self.cpuct}),file=sys.stderr,flush=True)

    def reset(self):
        config={'size':self.size,'komi':self.komi,'history':self.history,'simulations':self.simulations,'cpuct':self.cpuct,'max_search_edges':1000000}
        if self.native.ABI_VERSION >= 2: config['scoring']=self.scoring
        elif self.scoring != 'raw_area': raise ValueError('scoring profile requires native interface version 2')
        # A source-qualified historical engine has no such field; only include
        # this extension when the training recipe explicitly selected it.
        if self.fpu_reduction is not None: config['fpu_reduction']=self.fpu_reduction
        if self.score_utility is not None: config['score_utility']=self.score_utility
        if self.gumbel is not None: config['gumbel']=self.gumbel
        self.game=self.native.Game(json.dumps(config,allow_nan=False));self.last_stats=None

    def evaluate_leaf(self,request,features):
        x=features.reshape(1,self.size,self.size,2*self.history+4)
        logits,values=self.jax.device_get(self.forward(x))
        return logits[0],float(values[0])

    def command(self,name,args):
        if name not in COMMANDS: raise ValueError('unknown command')
        expected={'known_command':1,'boardsize':1,'komi':1,'play':2,'genmove':1}.get(name,0)
        if len(args)!=expected: raise ValueError('wrong argument count')
        if name=='protocol_version': return '2'
        if name=='name': return 'gozero'
        if name=='version': return 'gozero-native-gumbel-v1' if self.gumbel is not None else 'az_reference-native-puct-v1'
        if name=='known_command': return str(args[0] in COMMANDS).lower()
        if name=='list_commands': return '\n'.join(COMMANDS)
        if name=='boardsize':
            size=int(args[0])
            if size!=self.supported_size: raise ValueError('checkpoint evaluation is pinned to its trained board size')
            self.size=size;self.reset();return ''
        if name=='clear_board': self.reset();return ''
        if name=='komi':
            value=float(args[0])
            if value!=self.komi: raise ValueError('checkpoint evaluation is pinned to its trained komi')
            return ''
        if name=='showboard':
            _,_,_,_,stones=self.game.state()
            return '\n'.join(''.join('.XO'[int(x)] for x in row) for row in stones.reshape(self.size,self.size))
        if name=='final_score':
            _,_,terminal,score,_=self.game.state()
            if not terminal: raise ValueError('final score requires two passes')
            return '0' if score==0 else ('W' if score>0 else 'B')+'+'+str(abs(score))
        if name=='gozero-search-stats': return json.dumps(self.last_stats,sort_keys=True)
        if name in ('play','genmove'):
            color={'b':1,'black':1,'w':2,'white':2}.get(args[0].lower())
            if color is None: raise ValueError('invalid color')
            if name=='play': self.game.play(color,action(args[1],self.size));return ''
            if color!=self.game.state()[1]: raise ValueError('wrong player')
            start=time.perf_counter();request,features=self.game.start(self.network)
            while request is not None:
                logits,value=self.evaluate_leaf(request,features)
                request,features=self.game.evaluate(request,self.np.ascontiguousarray(logits,dtype=self.np.float32),value)
            chosen,policy,value,simulations,neural,terminal=self.game.finish()
            self.game.play(color,chosen)
            self.last_stats={'seconds':time.perf_counter()-start,'simulations':simulations,'neural_evaluations':neural,'terminal_evaluations':terminal,'root_value':value}
            return vertex(chosen,self.size)
        return ''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--simulations',type=int,default=16)
    parser.add_argument('--cpuct',type=float,default=1.5)
    args=parser.parse_args()
    if os.environ.get('JAX_PLATFORMS','cpu')!='cpu': parser.error('This GTP inference adapter is CPU-only')
    os.environ['JAX_PLATFORMS']='cpu'
    serve(Engine(args))


def serve(engine):
    for line in sys.stdin:
        line=line.split('#',1)[0].strip()
        if not line: continue
        parts=line.split();identifier=parts.pop(0) if parts[0].isdigit() else ''
        try:
            if not parts: raise ValueError('missing command')
            response=engine.command(parts[0],parts[1:]);prefix='='
        except Exception as error:
            response=str(error).replace('\n',' ');prefix='?'
        sys.stdout.write(prefix+identifier+(' '+response if response else '')+'\n\n');sys.stdout.flush()
        if parts and parts[0]=='quit' and prefix=='=': break


if __name__=='__main__': main()
