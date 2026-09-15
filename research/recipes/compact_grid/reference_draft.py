"""Load a paired trained parent on host0 and broadcast only its policy weights."""
from pathlib import Path
import hashlib
import jax
import numpy as np
from jax.experimental import multihost_utils as mh
from gozero import checkpoints
from gozero.snapshots import verify,read_json,canonical_json


def load(spec,c,schema,host,replicated):
    root=Path('/workspace/go');params=None;receipt=None
    if host==0:
        audit=Path(spec['audit'])
        if checkpoints.sha256(audit)!=spec['audit_sha256']:raise ValueError('Parent audit changed')
        a=read_json(audit)
        if a['status']!='passed':raise ValueError('Parent training audit did not pass')
        source=root/'.gozero/snapshots'/a['training_snapshot'];verify(source);old=read_json(source/'resolved_config.json')
        if old['seed']!=c['seed'] or old['dataset']!=c['dataset']:raise ValueError('Parent seed or dataset differs')
        for path,wanted in a['input_files'].items():
            if checkpoints.sha256(root/path)!=wanted:raise ValueError('Parent learning evidence changed')
        report=read_json(root/'runs'/a['attempt']/'rank-0/artifacts/result.json')
        if report['model_schema']!=schema:raise ValueError('Auxiliary model must retain every parent parameter shape/path')
        checkpoint=Path(report['latest_checkpoint']['owner_checkpoint_path'])
        group_path=Path(report['latest_checkpoint']['path']).with_suffix('.group.json')
        if checkpoints.sha256(group_path)!=report['latest_checkpoint']['group_sha256']:raise ValueError('Parent checkpoint group changed')
        group=read_json(group_path)
        _,arrays,_=checkpoints.read(checkpoint,expected_manifest_sha256=group['host_manifests']['0'])
        params={item['path']:arrays['p_'+str(i).zfill(4)] for i,item in enumerate(schema)}
        if any(not np.isfinite(x).all() for x in params.values()):raise ValueError('Nonfinite reference parameter')
        h=hashlib.sha256()
        for i,item in enumerate(schema):
            value=params[item['path']];h.update(canonical_json(['p_'+str(i).zfill(4),list(value.shape),str(value.dtype)]));h.update(value.tobytes(order='C'))
        receipt={'audit_sha256':spec['audit_sha256'],'snapshot':source.name,'checkpoint':str(checkpoint),
            'parent_training_turn':report['turn'],
            'parent_initial_parameter_elements_sha256':report['initial_parameter_elements_sha256'],
            'checkpoint_manifest_sha256':checkpoints.sha256(checkpoint/'manifest.json'),
            'policy_parameter_elements_sha256':h.hexdigest(),'historical_full_validation':a['validation_curve'][-1]}
        del arrays
    # Other hosts need only static placeholder shapes; optimizer state is never broadcast.
    if params is None:params={x['path']:np.zeros(x['shape'],np.float32) for x in schema}
    shared=mh.broadcast_one_to_all(params,is_source=host==0)
    result=jax.tree.map(lambda x:jax.device_put(x,replicated),shared)
    jax.block_until_ready(result)
    return result,receipt
