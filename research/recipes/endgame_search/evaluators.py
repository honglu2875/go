"""Frozen student and one explicitly identified teacher-mixture component."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from gozero.checkpoints import read as read_checkpoint,sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import read_json,verify
from causal_gtp import Engine as StudentEngine
from corpus import require


def student(root,source,c,receipt):
    inference=read_json(source/c['inference'])
    engine=StudentEngine(SimpleNamespace(candidate=source/c['candidate'],artifacts_root=root,
        native_receipt=receipt,inference_config=source/c['inference'],simulations=inference['simulations'],
        cpuct=inference['cpuct'],inference_native_snapshot=c['native_snapshot']))
    return engine,inference


def teacher(root,c,native):
    manifest=artifact(root,c['teacher_dataset_manifest'])
    require(sha256(manifest)==c['teacher_dataset_manifest_sha256'],'Teacher mixture manifest changed')
    spec=read_json(manifest)['spec']['shards'][c['teacher_shard_index']]
    source=artifact(root,'.gozero/snapshots/'+spec['training_snapshot']);identity=verify(source);config=read_json(source/'resolved_config.json')
    checkpoint=artifact(root,spec['checkpoint']);group_path=checkpoint.with_suffix('.group.json')
    require(sha256(group_path)==spec['group_sha256'],'Teacher checkpoint group changed')
    state,arrays,_=read_checkpoint(checkpoint,expected_manifest_sha256=spec['manifest_sha256'],array_prefix='p_');group=read_json(group_path)
    require(state['snapshot_id']==group['snapshot_id']==source.name and state['turn']==group['turn']==config['selfplay_turns']
        and group['rank_manifests'][state['jax_rank']]==spec['manifest_sha256'],'Teacher checkpoint lineage differs')
    module_spec=importlib.util.spec_from_file_location('diagnostic_teacher_model',source/identity['recipe']/'model.py')
    model=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(model)
    a=config['actors'];m=config['model'];history=a['history'];template=model.initialize(config['seed'],2*history+4,m)
    leaves,tree=jax.tree.flatten(template);require(set(arrays)=={f'p_{i:04d}'for i in range(len(leaves))},'Teacher parameter schema differs')
    for i,v in enumerate(leaves):require(v.shape==arrays[f'p_{i:04d}'].shape and str(v.dtype)==str(arrays[f'p_{i:04d}'].dtype),'Teacher array differs')
    params=tree.unflatten([jnp.asarray(arrays[f'p_{i:04d}'])for i in range(len(leaves))])
    forward=jax.jit(lambda x:model.apply(params,x,m)).lower(jnp.zeros((1,9,9,2*history+4),jnp.float32)).compile()
    jax.block_until_ready(forward(jnp.zeros((1,9,9,2*history+4),jnp.float32)))
    def evaluate(request,features):
        logits,values=jax.device_get(forward(features.reshape(1,9,9,2*history+4)))
        return logits[0],float(values[0])
    contract={k:a[k]for k in ('size','komi','scoring','history','simulations','cpuct','max_search_edges')}
    contract['gumbel']={**a['gumbel'],'gumbel_scale':0.};contract['fpu_reduction']=a.get('fpu_reduction')
    contract['score_utility']=a.get('score_utility')
    require(contract['size']==9 and contract['komi']==7.5 and contract['scoring']=='pass_alive_area'
        and contract['simulations']==16 and contract['cpuct']==0.
        and (a.get('score_utility')is None or a['score_utility']['factor']==0.),'Teacher inference contract differs')
    return SimpleNamespace(native=native,network=state['turn'],evaluate_leaf=evaluate,game=None,plies=0),contract,{
        'mixture_component':spec,'training_seed':config['seed'],'model_code_sha256':sha256(source/identity['recipe']/'model.py'),
        'parameter_elements_sha256':__import__('hashlib').sha256(b''.join(arrays[f'p_{i:04d}'].tobytes()for i in range(len(leaves)))).hexdigest()}
