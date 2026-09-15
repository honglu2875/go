"""Tie evaluation weights to a final training record or a committed checkpoint."""
from pathlib import Path
import numpy as np
from . import checkpoints
from .snapshots import read_json,verify


def artifact(root,path):
    root=Path(root).resolve();resolved=(root/path).resolve(strict=True)
    if not resolved.is_relative_to(root):raise ValueError('Artifact escapes its declared root')
    return resolved


def checkpoint_parameters(root,reference,source_id,network_version):
    if set(reference)!={'path','manifest_sha256','group_sha256'}:raise ValueError('Invalid checkpoint reference')
    path=artifact(root,reference['path']);group_path=path.parent/(path.name+'.group.json')
    if checkpoints.sha256(group_path)!=reference['group_sha256']:raise ValueError('Checkpoint group hash differs')
    group=read_json(group_path)
    state,arrays,_=checkpoints.read(path,expected_manifest_sha256=reference['manifest_sha256'],array_prefix='p_')
    if (group['schema_version']!=1 or state['schema_version']!=1
            or group['snapshot_id']!=source_id or state['snapshot_id']!=source_id
            or group['config_sha256']!=state['config_sha256'] or group['turn']!=state['turn']
            or group['world_size']!=state['world_size'] or group['updates']!=network_version
            or state['counters']['updates']!=network_version
            or len(group['rank_manifests'])!=state['world_size'] or not 0<=state['jax_rank']<state['world_size']
            or group['rank_manifests'][state['jax_rank']]!=reference['manifest_sha256']):
        raise ValueError('Checkpoint scientific identity differs')
    if not arrays or set(arrays)!={f'p_{i:04d}' for i in range(len(arrays))}:
        raise ValueError('Checkpoint parameter names differ')
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite checkpoint parameters')
    return state,arrays


def validate_candidate(root,candidate):
    root=Path(root).resolve();version=candidate['schema_version']
    if version not in (1,2):raise ValueError('Unsupported evaluation candidate schema')
    source=artifact(root,Path('.gozero/snapshots')/candidate['training_snapshot']);manifest=verify(source)
    if source.name!=candidate['training_snapshot']:raise ValueError('Training source identity differs')
    config=read_json(source/'resolved_config.json');weights=artifact(root,candidate['model_export_path'])
    if checkpoints.sha256(weights)!=candidate['model_export_sha256']:raise ValueError('Candidate weights hash differs')
    if type(candidate['network_version']) is not int or candidate['network_version']<0:raise ValueError('Invalid model version')
    native=None;turn=None;native_receipt=None
    if version==1:
        record_path=artifact(root,candidate['training_result_path'])
        if checkpoints.sha256(record_path)!=candidate['training_result_sha256']:raise ValueError('Training record hash differs')
        record=read_json(record_path)
        if (record['status']!='passed' or record['snapshot_id']!=source.name
                or record['counters']['updates']!=candidate['network_version']
                or record['model_export_sha256']!=candidate['model_export_sha256']):
            raise ValueError('Candidate final training identity differs')
        native=record.get('native',{}).get('binary_sha256');turn=record.get('turn')
        provenance=record
    else:
        state,parameters=checkpoint_parameters(root,candidate['checkpoint'],source.name,candidate['network_version'])
        if state['config_sha256']!=checkpoints.sha256(source/'resolved_config.json'):
            raise ValueError('Checkpoint configuration differs from source')
        with np.load(weights,allow_pickle=False) as data:
            if len(data.files)!=len(set(data.files)) or set(data.files)!=set(parameters):
                raise ValueError('Exported parameter tree differs from checkpoint')
            for key,value in parameters.items():
                saved=data[key]
                if saved.dtype!=value.dtype or saved.shape!=value.shape or not np.array_equal(saved,value):
                    raise ValueError('Exported weights differ from checkpoint: '+key)
        native=state['native_sha256'];turn=state['turn']
        provenance=state
    if 'initialization' in config:
        # A complete-state child keeps the actual parent binary as an explicit
        # dependency. Verify the fork contract instead of relabelling a build
        # with the child's source ID or accepting an arbitrary native override.
        from .checkpoint_forks import contract
        origin,_,native_receipt,receipt=contract(root,source,config)
        if (provenance.get('initialization')!=origin or native!=receipt['binary_sha256']
                or provenance.get('config_sha256')!=checkpoints.sha256(source/'resolved_config.json')):
            raise ValueError('Candidate initialization or declared native dependency differs')
    return {'snapshot':source,'manifest':manifest,'config':config,'weights':weights,
            'native_binary_sha256':native,'native_receipt':native_receipt,'checkpoint_turn':turn}
