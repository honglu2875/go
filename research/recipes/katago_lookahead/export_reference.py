"""Execute original KataGo Lookahead statements with actual Torch parameters."""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REVISION='92ee95c0a4b25fec214da00951ab69e97e207729'
TRAIN_SHA='991cbfb2d5cea8180cd3daf818ef2a516aca048e0ae0eab68865e9a125967e9e'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    train=a.source/'python/train.py'
    if sha(train)!=TRAIN_SHA:raise ValueError('Pinned trainer source changed')
    tree=ast.parse(train.read_text())
    lines={'initialize':(1046,1047),'begin_subepoch':(1510,),
           'step':(1707,1708),'end_epoch':(1736,)}
    fragments={};locations={}
    for op,wanted in lines.items():
        nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.stmt) and n.lineno==line) for line in wanted]
        fragments[op]=compile(ast.Module(body=nodes,type_ignores=[]),str(train),'exec')
        locations[op]=[dict(line=n.lineno,end_line=n.end_lineno) for n in nodes]
    a.output.mkdir(parents=True,exist_ok=False)
    shapes={'bias':(4,),'conv':(2,3,3,3,4),'matrix':(7,5),'norm':(2,4)}
    rng=np.random.Generator(np.random.PCG64(91312741));arrays={}
    initial={name:rng.integers(-512,513,size=shape).astype(np.float32)/256 for name,shape in shapes.items()}
    delta=[{name:rng.integers(-32,33,size=shape).astype(np.float32)/1024 for name,shape in shapes.items()} for _ in range(23)]
    for i,(name,value) in enumerate(initial.items()):arrays['initial_'+str(i)]=value
    cases=[]
    for case,config in enumerate((dict(k=None,alpha=None),dict(k=1,alpha=.25),dict(k=3,alpha=.3),dict(k=6,alpha=.5))):
        names=list(shapes);params=[torch.nn.Parameter(torch.from_numpy(initial[name].copy())) for name in names]
        context=dict(torch=torch,logging=logging,lookahead_k=config['k'],lookahead_alpha=config['alpha'],lookahead_counter=0,
            optimizer=SimpleNamespace(param_groups=[dict(params=params[:2]),dict(params=params[2:])]))
        exec(fragments['initialize'],context)
        events=[];turn=0
        def record(op,deltas=None):
            number=len(events);prefix=f'c{case}_e{number}_'
            values={}
            for role,tensors in (('fast',params),('slow',[context['lookahead_cache'][p] for p in params] if config['k'] is not None else [])):
                keys={}
                for i,tensor in enumerate(tensors):
                    key=prefix+role+str(i);arrays[key]=tensor.detach().numpy().copy();keys[names[i]]=key
                values[role]=keys
            item=dict(op=op,expected=values,counter=context['lookahead_counter'],
                synchronized=bool(op=='step' and config['k'] is not None and context['lookahead_counter']==0))
            if deltas is not None:
                item['delta']={}
                for i,name in enumerate(names):
                    key=prefix+'delta'+str(i);arrays[key]=deltas[name];item['delta'][name]=key
            events.append(item)
        for lengths in ((5,7),(2,9)):
            for length in lengths:
                exec(fragments['begin_subepoch'],context);record('begin_subepoch')
                for _ in range(length):
                    d=delta[turn];turn+=1
                    with torch.no_grad():
                        for name,param in zip(names,params):param.data.add_(torch.from_numpy(d[name]))
                        exec(fragments['step'],context)
                    record('step',d)
            exec(fragments['end_epoch'],context);record('end_epoch')
        cases.append(dict(config=config,events=events,optimizer_steps=turn))
    data=a.output/'arrays.npz';np.savez_compressed(data,**arrays)
    result=dict(kind='original_katago_lookahead_reference',revision=REVISION,trainer_sha256=TRAIN_SHA,
        exporter_sha256=sha(Path(__file__)),torch_version=torch.__version__,fragments=locations,
        initial={name:'initial_'+str(i) for i,name in enumerate(shapes)},
        arrays=dict(path='arrays.npz',sha256=sha(data)),cases=cases,
        scope='Actual original Torch slow-weight/counter statements with predetermined fast-optimizer increments; no neural training or Muon parameter selection.')
    out=a.output/'manifest.json';out.write_text(json.dumps(result,indent=2)+'\n')
    data.chmod(0o444);out.chmod(0o444)
    print(json.dumps(dict(status='passed',cases=len(cases),events=sum(len(c['events']) for c in cases),sha256=sha(out))))


if __name__=='__main__':main()
