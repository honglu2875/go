"""Run the pinned official value head in its existing PyTorch reference env."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
REVISION='92ee95c0a4b25fec214da00951ab69e97e207729'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'.gozero/external/katago-source'/REVISION/('KataGo-'+REVISION)
    sys.path.insert(0,str(source/'python'))
    import torch
    from katago.train import modelconfigs,model_pytorch
    torch.set_num_threads(2);torch.manual_seed(92513)
    config=copy.deepcopy(modelconfigs.config_of_name['b40c768nbt-fson-mish-rvglr-bnh'])
    head=model_pytorch.ValueHead(768,256,256,256,8,config,'mish',19)
    head.initialize();head.train()
    with torch.no_grad():head.bias1.beta.add_(torch.randn_like(head.bias1.beta)*.03)
    rng=np.random.default_rng(81651);mask=np.zeros((5,1,19,19),np.float32)
    for i,size in enumerate((19,13,9,1,17)):mask[i,0,:size,:size]=1.
    x=(rng.normal(size=(5,768,19,19))*.35).astype(np.float32)*mask
    glob=(rng.normal(size=(5,19))*.2).astype(np.float32)
    weights=np.asarray([.1,0.,1.7,.5,1.],np.float32);target=np.asarray([-.85,-.1,.4,.9,-.7],np.float32)
    xt=torch.tensor(x,requires_grad=True);mt=torch.tensor(mask)
    outputs=head(xt,mt,mt.sum((2,3),keepdim=True),float(mt.sum()),torch.tensor(glob),None)
    logits=outputs[0];probability=torch.softmax(logits,-1);value=probability[:,0]-probability[:,1]
    loss=((value-torch.tensor(target))**2*torch.tensor(weights)).sum()/weights.sum();loss.backward()
    arrays={}
    for name,p in head.named_parameters():
        if p.grad is not None:
            arrays['p.'+name]=p.detach().numpy();arrays['g.'+name]=p.grad.detach().numpy()
    arrays.update(x=x,mask=mask,global_features=glob,weights=weights,target=target,
                  logits=logits.detach().numpy(),value=value.detach().numpy(),loss=loss.detach().numpy(),
                  input_gradient=xt.grad.detach().numpy())
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'arrays.npz').open('xb') as f:np.savez(f,**arrays)
    result=dict(kind='official_katago_value_branch_reference',status='passed',source_revision=REVISION,
        dimensions=dict(width=768,spatial_channels=256,hidden=256,output_logits=3),config=config,
        array_sha256=sha(a.output/'arrays.npz'),parameter_shapes={k[2:]:list(v.shape) for k,v in arrays.items() if k.startswith('p.')},
        torch_version=torch.__version__,numpy_version=np.__version__,loss=float(loss.detach()),
        source_sha256={name:sha(source/name) for name in ('python/katago/train/model_pytorch.py','python/katago/train/modelconfigs.py','python/katago/train/trainloop_helpers.py')},
        operator_sha256=sha(Path(__file__)),scope='Full-width value readout branch on five differently masked boards; no backbone or joint learner qualification')
    with (a.output/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for path in a.output.iterdir():path.chmod(0o444)
    print(json.dumps(dict(status='passed',manifest_sha256=sha(a.output/'manifest.json'),parameters=len(result['parameter_shapes']))),flush=True)


if __name__=='__main__':main()
