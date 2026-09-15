#!/usr/bin/env python3
"""Full-history causal policy/value evaluation of exact native MCTS leaves."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import read_json,verify
from learned_gtp import Engine as BoardEngine,serve


def inference_contract(c,model):
    fields='schema_version kind size komi history scoring simulations cpuct fpu_reduction gumbel max_search_edges max_game_moves'
    if not isinstance(c,dict) or set(c)!=set(fields.split()):raise ValueError('Unknown or missing causal inference fields')
    if c['schema_version']!=1 or c['kind']!='causal_mcts_inference':raise ValueError('Unknown causal inference contract')
    if c['size']!=model['size'] or c['komi']!=model['komi']:raise ValueError('Inference board or komi differs from training')
    for key,lo,hi in [('size',1,25),('history',1,64),('simulations',0,1000000),('max_search_edges',model['size']**2+1,1000000),('max_game_moves',2,1000000)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError('Invalid '+key)
    if type(c['cpuct']) not in (int,float) or not math.isfinite(c['cpuct']) or c['cpuct']<0:raise ValueError('Invalid cpuct')
    if c['scoring'] not in ('raw_area','pass_alive_area'):raise ValueError('Unknown scoring contract')
    # A fresh search can extend by at most one edge per simulation. Reserve its
    # full worst-case suffix before starting: no dropped history or forced pass.
    if c['max_game_moves']+c['simulations']>=model['max_tokens']:raise ValueError('Game cap plus worst-case search depth exceeds causal context')
    return c


class Engine(BoardEngine):
    def __init__(self,args):
        import jax
        import jax.numpy as jnp
        import numpy as np
        self.jax=jax;self.np=np
        verify(SOURCE)
        if jax.default_backend()!='cpu' or jax.process_count()!=1:raise ValueError('Causal GTP adapter is single-process CPU inference')
        candidate=read_json(args.candidate);identity=validate(args.artifacts_root,candidate)
        self.board_conditioned=candidate['kind']=='board_causal_history_policy'
        snapshot=identity['snapshot'];config=identity['config'];manifest=identity['manifest']
        infer=args.inference_config.resolve()
        if not infer.is_relative_to(SOURCE):raise ValueError('Causal inference contract must be frozen with the adapter')
        c=inference_contract(read_json(infer),config['model'])
        if c['simulations']!=args.simulations or c['cpuct']!=args.cpuct:raise ValueError('Command-line search budget differs from frozen causal inference')
        dataset=artifact(args.artifacts_root,str(Path(config['dataset']['path'])/'manifest.json'))
        if sha256(dataset)!=config['dataset']['manifest_sha256']:raise ValueError('Training dataset rule identity differs')
        dataset_manifest=read_json(dataset)
        if self.board_conditioned:
            if dataset_manifest['kind']!='board_causal_teacher_dataset':raise ValueError('Board model requires its exact observation overlay')
            rules=dataset_manifest['rules']
            base=dataset_manifest['parent_dataset'];base_manifest=artifact(args.artifacts_root,str(Path(base['path'])/'manifest.json'))
            if sha256(base_manifest)!=base['manifest_sha256']:raise ValueError('Underlying teacher dataset differs')
        else:rules=dataset_manifest['spec']
        if any(rules[k]!=c[k] for k in ('size','komi','scoring')):raise ValueError('Inference rules differ from teacher dataset')
        self.history=c['history'];self.net=config['model'];self.size=c['size'];self.komi=c['komi'];self.scoring=c['scoring']
        self.fpu_reduction=c['fpu_reduction'];self.score_utility=None;self.gumbel=c['gumbel']
        self.supported_size=self.size;self.max_game_moves=c['max_game_moves'];self.max_search_edges=c['max_search_edges']
        self.network=candidate['network_version'];self.simulations=c['simulations'];self.cpuct=c['cpuct']
        module_spec=importlib.util.spec_from_file_location('frozen_causal_model',snapshot/manifest['recipe']/'model.py')
        model=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(model)
        initial=model.initialize(config['seed'],self.net)
        leaves,definition=jax.tree.flatten(initial)
        schema=[{'path':jax.tree_util.keystr(path),'shape':list(a.shape),'dtype':str(a.dtype)} for path,a in jax.tree_util.tree_flatten_with_path(initial)[0]]
        if schema!=identity['model_schema']:raise ValueError('Causal parameter tree differs from trained checkpoint')
        self.params=definition.unflatten([jnp.asarray(identity['arrays'][f'p_{i:04d}']) for i in range(len(leaves))])
        receipt=read_json(args.native_receipt)
        # Offline causal training did not call a native environment. Inference
        # therefore declares and verifies a separate, explicit native identity.
        if receipt['snapshot_id']!=args.inference_native_snapshot:raise ValueError('Causal inference native source differs')
        verify(artifact(args.artifacts_root,'.gozero/snapshots/'+receipt['snapshot_id']))
        self.native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
        if getattr(self.native,'CAUSAL_GAME_ABI_VERSION',None)!=1:raise ValueError('Native pending-leaf history ABI is unavailable')
        def predict(params,tokens,lengths):
            h,_=model.prefill(params,tokens,lengths,self.net)
            last=h[0,lengths[0]]
            # Every search leaf uses the expert head, for both players. The
            # behavior head forecasts observations and never supplies MCTS priors.
            return last@params['play'],jnp.tanh(last@params['value'])[0]
        self.tokens=np.full((1,self.net['max_tokens']),self.size**2+2,np.int32);self.tokens[0,0]=self.size**2+1
        if self.board_conditioned:
            def predict_board(params,tokens,lengths,stones):
                h,_=model.prefill(params,tokens,lengths,self.net)
                last=h[0,lengths[0]][None,None,:]
                board=model.board_features(params,stones[:,None,:],(1+lengths%2)[:,None],self.net)
                logits,_,value=model.head_outputs(params,last,jnp.zeros((1,1,self.size**2+1),jnp.float32),board)
                return logits[0,0],value[0,0]
            dummy_stones=np.zeros((1,self.size**2),np.uint8)
            self.forward=jax.jit(predict_board).lower(self.params,self.tokens,np.zeros(1,np.int32),dummy_stones).compile()
            jax.block_until_ready(self.forward(self.params,self.tokens,np.zeros(1,np.int32),dummy_stones))
        else:
            self.forward=jax.jit(predict).lower(self.params,self.tokens,np.zeros(1,np.int32)).compile()
            jax.block_until_ready(self.forward(self.params,self.tokens,np.zeros(1,np.int32)))
        self.reset()
        print(json.dumps({'kind':'engine_ready','adapter':'board_causal_full_prefill' if self.board_conditioned else 'causal_full_prefill','candidate_sha256':sha256(args.candidate),
            'training_snapshot':snapshot.name,'model_code_sha256':identity['model_code_sha256'],
            'weights_sha256':candidate['model_export_sha256'],'native_snapshot':receipt['snapshot_id'],
            'native_sha256':receipt['binary_sha256'],'inference_config_sha256':sha256(infer),
            'backend':jax.default_backend(),'simulations_excluding_root':self.simulations,'cpuct':self.cpuct,
            'maximum_game_moves':self.max_game_moves,'context_tokens':self.net['max_tokens'],
            'board_mode':self.net.get('board_mode')}),file=sys.stderr,flush=True)

    def reset(self):
        c={'size':self.size,'komi':self.komi,'history':self.history,'scoring':self.scoring,
           'simulations':self.simulations,'cpuct':self.cpuct,'max_search_edges':self.max_search_edges,
           'fpu_reduction':self.fpu_reduction,'gumbel':self.gumbel}
        self.game=self.native.Game(json.dumps(c,allow_nan=False));self.last_stats=None;self.plies=0

    def evaluate_leaf(self,request,features):
        history=self.game.request_history(request)
        n=len(history)
        if n<self.plies or n>self.plies+self.simulations or n>=self.net['max_tokens']:
            raise ValueError('Pending history violates prevalidated context bound')
        self.tokens.fill(self.size**2+2);self.tokens[0,0]=self.size**2+1;self.tokens[0,1:n+1]=history
        if self.board_conditioned:
            f=features.reshape(self.size**2,2*self.history+4)
            if not self.np.all(f[:,-4]==float(n%2==0)):raise ValueError('Leaf board perspective differs from causal history')
            if not self.np.isin(f[:,:2],[0.,1.]).all() or self.np.any(f[:,:2].sum(axis=-1)>1):raise ValueError('Invalid native leaf stone planes')
            stones=self.np.where(f[:,0]>.5,1+n%2,self.np.where(f[:,1]>.5,2-n%2,0)).astype(self.np.uint8)[None,:]
            if self.net['board_mode']=='empty':stones.fill(0)
            logits,value=self.jax.device_get(self.forward(self.params,self.tokens,self.np.asarray([n],self.np.int32),stones))
        else:
            logits,value=self.jax.device_get(self.forward(self.params,self.tokens,self.np.asarray([n],self.np.int32)))
        return logits,float(value)

    def command(self,name,args):
        if name=='version' and not args:return 'gozero-board-causal-mcts-history-v1' if self.board_conditioned else 'gozero-causal-mcts-history-v1'
        if name in ('play','genmove'):
            if self.plies>=self.max_game_moves:raise ValueError('Registered causal game move cap reached')
            response=super().command(name,args);self.plies+=1;return response
        return super().command(name,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--simulations',type=int,required=True)
    parser.add_argument('--cpuct',type=float,required=True);parser.add_argument('--inference-config',type=Path,required=True)
    parser.add_argument('--inference-native-snapshot',required=True)
    args=parser.parse_args()
    if os.environ.get('JAX_PLATFORMS','cpu')!='cpu':parser.error('This GTP adapter is CPU-only')
    os.environ['JAX_PLATFORMS']='cpu';serve(Engine(args))


if __name__=='__main__':main()
