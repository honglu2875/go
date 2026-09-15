#!/usr/bin/env python3
"""Qualification only: run official PyTorch, exporting safe arrays for JAX checks."""
import argparse
import copy
from pathlib import Path
import json
import sys
import numpy as np

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import canonical_json, verify
from gozero.checkpoints import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,default=4)
    args=p.parse_args(); verify(SOURCE)
    sys.path.insert(0,str(args.official_source/'python'))
    import torch
    from katago.train import modelconfigs, model_pytorch
    torch.set_num_threads(4); torch.manual_seed(71342)
    c=copy.deepcopy(modelconfigs.config_of_name['b40c768nbt-fson-mish-rvglr-bnh'])
    c.update(trunk_num_channels=32,mid_num_channels=16,gpool_num_channels=4,p1_num_channels=8,g1_num_channels=8,
             v1_num_channels=8,sbv2_num_channels=8,v2_size=8)
    c['block_kind']=c['block_kind'][:args.layers]; c['intermediate_head_blocks']=args.layers
    model=model_pytorch.Model(c,pos_len=5); model.initialize(); model.train()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.endswith(('.gamma','.beta')): p.add_(torch.randn_like(p)*.03)
    rng=np.random.default_rng(98123)
    spatial=rng.integers(0,2,(5,22,5,5)).astype(np.float32)
    for i,n in enumerate((5,3,4,2,5)):
        spatial[i,:,n:,:]=0; spatial[i,:,:,n:]=0
        spatial[i,0]=0; spatial[i,0,:n,:n]=1
    glob=(rng.normal(size=(5,19))*.2).astype(np.float32)
    legal=np.concatenate([spatial[:,0].reshape(5,25)>0,np.ones((5,1),bool)],-1)
    target=rng.uniform(.01,1.,(5,26)).astype(np.float32)*legal; target/=target.sum(-1,keepdims=True)
    weights=np.asarray([.1,1.,.7,.3,1.5],np.float32)
    x=torch.tensor(spatial); g=torch.tensor(glob)
    outputs=model(x,g); logits=[o[0][:,0] for o in outputs]
    def ce(y): return -(torch.tensor(target)*torch.log_softmax(y,-1)).sum(-1)
    loss=((.2*ce(logits[0])+.8*ce(logits[1]))*torch.tensor(weights)).sum()/float(weights.sum())
    loss.backward()
    arrays={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            arrays['p.'+name]=p.detach().numpy(); arrays['g.'+name]=p.grad.detach().numpy()
    arrays.update(spatial=spatial,global_features=glob,target=target,weights=weights,
                  main=logits[0].detach().numpy(),helper=logits[1].detach().numpy(),loss=loss.detach().numpy())
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'reference.npz').open('xb') as f: np.savez(f,**arrays)
    meta={'kind':'official_katago_policy_arithmetic_reference','operator_snapshot':SOURCE.name,'config':c,
        'torch_version':torch.__version__,'numpy_version':np.__version__,
        'source_files':{name:sha256(args.official_source/name) for name in
            ('python/katago/train/model_pytorch.py','python/katago/train/modelconfigs.py','python/katago/train/trainloop_helpers.py')},
        'arrays_sha256':sha256(args.output/'reference.npz'),'seed':71342,'input_seed':98123,
        'loss':float(loss.detach()),'active_gradients':len([x for x in arrays if x.startswith('g.')])}
    (args.output/'manifest.json').write_bytes(canonical_json(meta))
    for path in args.output.iterdir(): path.chmod(0o444)
    print(json.dumps({'status':'passed','manifest_sha256':sha256(args.output/'manifest.json'),'loss':meta['loss']}),flush=True)
if __name__=='__main__': main()
