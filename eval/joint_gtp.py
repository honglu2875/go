#!/usr/bin/env python3
"""Rust Go search evaluated by one pinned joint V7 policy/value owner."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.joint_artifacts import validate_descriptor
from gozero.joint_rpc import Client
from gozero.native import load_library
from gozero.snapshots import read_json,verify
from gozero.v7_state import digest_from_native
from learned_gtp import Engine as BaseEngine,serve


def contract(c):
    fields=set(('schema_version kind size komi scoring simulations cpuct fpu_reduction gumbel '
                'max_search_edges max_game_moves cache_positions native_receipt_sha256').split())
    if not isinstance(c,dict) or set(c)!=fields or c['schema_version']!=1 or c['kind']!='joint_v7_mcts_inference':
        raise ValueError('Invalid joint V7 search contract')
    for key,low,high in [('size',1,25),('simulations',0,1000000),('max_game_moves',2,1000000),
                         ('cache_positions',2,2048),('max_search_edges',c['size']**2+1,1000000)]:
        if type(c[key]) is not int or not low<=c[key]<=high:raise ValueError('Invalid '+key)
    if (type(c['cpuct']) not in (int,float) or not math.isfinite(c['cpuct']) or c['cpuct']<0
            or type(c['komi']) not in (int,float) or not math.isfinite(c['komi'])
            or c['scoring']!='pass_alive_area' or c['max_game_moves']+c['simulations']>=c['cache_positions']):
        raise ValueError('Search rules or complete-context capacity differ')
    if c['gumbel'] is not None and c['gumbel'].get('gumbel_scale')!=0:
        raise ValueError('Evaluation search must disable Gumbel sampling noise')
    return c


def identity(source,candidate,inference):
    return dict(source=source.name,candidate_sha256=sha256(candidate),inference_sha256=sha256(inference))


class Engine(BaseEngine):
    def __init__(self,args):
        verify(SOURCE);self.np=np
        for path in (args.candidate,args.inference_config):
            if not path.resolve().is_relative_to(SOURCE):
                raise ValueError('Candidate and search contract must be frozen with the adapter')
        candidate=validate_descriptor(read_json(args.candidate));c=contract(read_json(args.inference_config))
        inputs=candidate['input_contract']
        if (c['size']!=inputs['size'] or c['komi']!=inputs['komi']
                or c['cache_positions']>candidate['max_positions']):
            raise ValueError('Search differs from the trained input/context contract')
        if sha256(args.native_receipt)!=c['native_receipt_sha256']:
            raise ValueError('Native inference receipt differs')
        receipt=read_json(args.native_receipt)
        self.native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
        if getattr(self.native,'CAUSAL_GAME_ABI_VERSION',None)!=1:
            raise ValueError('Native pending-leaf complete histories are unavailable')
        self.rpc=Client(args.socket,identity(SOURCE,args.candidate,args.inference_config),timeout=120.)
        self.size,self.komi,self.scoring=(c[k] for k in ('size','komi','scoring'))
        self.supported_size=self.size;self.history=1
        self.simulations,self.cpuct=c['simulations'],c['cpuct']
        self.fpu_reduction,self.gumbel,self.score_utility=c['fpu_reduction'],c['gumbel'],None
        self.max_game_moves,self.max_search_edges=c['max_game_moves'],c['max_search_edges']
        self.cache_positions,self.network=c['cache_positions'],candidate['network_version']
        self.reset()
        print(json.dumps(dict(kind='engine_ready',adapter='joint_v7_single_policy_rpc',
                             **identity(SOURCE,args.candidate,args.inference_config),slot=self.rpc.slot,
                             native_binary_sha256=receipt['binary_sha256'],simulations=self.simulations)),
              file=sys.stderr,flush=True)

    def reset(self):
        self.game=self.native.Game(json.dumps(dict(size=self.size,komi=self.komi,scoring=self.scoring,
            history=1,simulations=self.simulations,cpuct=self.cpuct,fpu_reduction=self.fpu_reduction,
            gumbel=self.gumbel,max_search_edges=self.max_search_edges),allow_nan=False))
        self.plies=0;self.last_stats=None

    def evaluate_leaf(self,request,features):
        history=self.game.request_history(request).tolist()
        if not self.plies<=len(history)<=self.plies+self.simulations or len(history)>=self.cache_positions:
            raise ValueError('Native leaf exceeds the complete-history contract')
        digest=digest_from_native(features,history=history,size=self.size,komi=self.komi)
        response=self.rpc.predict(history,digest)
        return np.asarray(response['policy'],np.float32),float(response['value'])

    def command(self,name,args):
        if name=='version' and not args:return 'gozero-joint-v7-rpc-v1'
        if name in ('play','genmove'):
            if self.plies>=self.max_game_moves:raise ValueError('Registered game cap reached')
            response=super().command(name,args);self.plies+=1;return response
        return super().command(name,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','inference-config','native-receipt','socket'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();engine=Engine(args)
    try:serve(engine)
    finally:engine.rpc.close()


if __name__=='__main__':main()
