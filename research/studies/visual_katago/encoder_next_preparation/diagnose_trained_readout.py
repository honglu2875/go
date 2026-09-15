"""Read-only gradient probe of the audited trained C128 control."""
from pathlib import Path
import hashlib
import json
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path('/workspace/go')
SOURCE=ROOT/'.gozero/snapshots/96b6d19bc7377d6c02b777db74d9a4c93c0ea6a07d379de316c72be8dbf5f678'
sys.path[:0]=[str(SOURCE/'research/recipes/visual_token_encoder'),str(SOURCE/'packages/gozero/src')]
from gozero.snapshots import verify,read_json
from gozero import checkpoints
from gozero.checkpoint_archive import publish
from gozero.katago_sequence_batches import Dataset
import causal

verify(SOURCE)
audit_path=ROOT/'runs/capacity-c128-continuation-63b01359/1-3e-04-audit.json'
assert checkpoints.sha256(audit_path)=='56e27331918e98f28666f8948d3b0782de62535c51c69f4b8c8947559bc271d5'
audit=read_json(audit_path)
owner=read_json(ROOT/'runs'/audit['attempt']/'rank-0/artifacts/result.json')
training_source=ROOT/'.gozero/snapshots'/audit['training_snapshot'];verify(training_source)
cfg=read_json(training_source/'resolved_config.json')
path=Path(owner['latest_checkpoint']['path'])
state,arrays,_=checkpoints.read(path,expected_manifest_sha256=owner['latest_checkpoint']['manifest_sha256'],array_prefix='p_')
schema=causal.parameter_schema(cfg['model'])
assert state['model_schema']==schema and state['turn']==1024
assert len(arrays)==len(schema)
params={row['path']:jnp.asarray(arrays[f'p_{i:04d}']) for i,row in enumerate(schema)}
for row in schema:
    assert list(params[row['path']].shape)==row['shape'] and str(params[row['path']].dtype)==row['dtype']
del arrays
data=Dataset(cfg['dataset']['path'],cfg['dataset']['manifest_sha256'])
raw=data.batch([('expert',0,0)],positions=256)
target=jnp.asarray(raw['policies'][0,127]);legal=jnp.asarray(raw['legal'][0,127])
net={**cfg['model'],'dtype':'float32','attention_backend':'xla'}
cases=[];start=time.time()
for moves in [16,64,128]:
    begin=128-moves;spatial=jnp.asarray(raw['spatial'][:,begin:128])
    glob=jnp.asarray(raw['global_features'][:,begin:128]);actions=jnp.asarray(raw['actions'][:,begin:128]);counts=jnp.asarray([moves],jnp.int32)
    def objective(params,spatial):
        logits=causal.forward(params,spatial,glob,actions,counts,net)[0,-1]
        logp=jax.nn.log_softmax(jnp.where(legal,logits,-1e9))
        return -jnp.sum(target*logp)
    began=time.time();loss,grad=jax.jit(jax.value_and_grad(objective,argnums=1))(params,spatial)
    jax.block_until_ready((loss,grad));norm=np.sqrt(np.sum(np.asarray(grad,dtype=np.float64)**2,axis=(0,2,3,4)))
    row={'prefix_moves':moves,'cross_entropy':float(loss),'gradient_l2_by_frame':norm.tolist(),
         'fraction_of_summed_frame_norms_in_last_n_frames':{str(n):float(norm[-min(n,moves):].sum()/norm.sum()) for n in [1,2,4,8,16]},
         'norm_weighted_mean_lag_in_moves':float(np.dot(norm,np.arange(moves-1,-1,-1))/norm.sum()),'seconds':time.time()-began}
    cases.append(row);print(json.dumps({k:v for k,v in row.items() if k!='gradient_l2_by_frame'}),flush=True)
out=ROOT/'research/studies/visual_katago/encoder_next_preparation/trained_readout_gradient_context_diagnostic.json'
publish(out,{'kind':'trained_control_policy_gradient_frame_allocation','status':'passed','operator_snapshot':SOURCE.name,
    'training_snapshot':training_source.name,'audit_sha256':checkpoints.sha256(audit_path),
    'checkpoint_manifest_sha256':owner['latest_checkpoint']['manifest_sha256'],'checkpoint_path':str(path),
    'driver_sha256':checkpoints.sha256(Path(__file__)),'training_episode':[0,0],'common_endpoint':128,
    'execution_model':net,'cases':cases,'seconds':time.time()-start,
    'scope':'One fixed training example, not held-out evaluation. Audited1024-update C128 control with its original readout. Same context windows and endpoint as the initialization diagnostic. Float32CPU input-gradient probe; gradients are not attention probabilities or proof of causal feature reliance. No model update is performed.'})
print(out,checkpoints.sha256(out),flush=True)
