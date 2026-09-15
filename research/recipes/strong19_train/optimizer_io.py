"""Bind flat parameter/moment checkpoints to exact source and configuration.

This module stores optimizer state. A complete trainer must additionally save
its sampling, augmentation, dataset, progress and rank state.
"""
import re
import numpy as np


def identity(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Expected an exact SHA-256 identity')
    return value


def flatten(params,state,*,configuration_sha256,source_sha256):
    identity(configuration_sha256);identity(source_sha256)
    if set(state)!={'first','second','step'} or set(params)!=set(state['first']) or set(params)!=set(state['second']):
        raise ValueError('Optimizer state coverage differs')
    step=np.asarray(state['step'])
    if step.shape!=() or step.dtype!=np.dtype('int32') or int(step)<0:raise ValueError('Invalid optimizer step')
    schema=[];arrays={'step':step}
    for i,name in enumerate(sorted(params)):
        p=np.asarray(params[name])
        if p.dtype!=np.float32:raise ValueError('Expected FP32 master parameters')
        schema.append(dict(path=name,shape=list(p.shape),dtype='float32'))
        for prefix,tree in [('p',params),('m',state['first']),('v',state['second'])]:
            value=np.asarray(tree[name])
            if value.shape!=p.shape or value.dtype!=p.dtype or not np.isfinite(value).all():
                raise ValueError('Invalid parameter or moment array')
            if prefix=='v' and np.any(value<0):raise ValueError('Negative second moment')
            arrays[f'{prefix}_{i:04d}']=value
    metadata=dict(kind='joint_adamw_optimizer_state',version=1,configuration_sha256=configuration_sha256,
                  source_sha256=source_sha256,parameter_schema=schema,step=int(step))
    return metadata,arrays


def restore(metadata,arrays,*,schema,configuration_sha256,source_sha256):
    identity(configuration_sha256);identity(source_sha256)
    if (metadata.get('kind')!='joint_adamw_optimizer_state' or metadata.get('version')!=1
            or metadata.get('configuration_sha256')!=configuration_sha256
            or metadata.get('source_sha256')!=source_sha256 or metadata.get('parameter_schema')!=schema):
        raise ValueError('Checkpoint source, configuration or schema differs')
    if len({row['path'] for row in schema})!=len(schema):raise ValueError('Duplicate parameters')
    expected={'step'}|{f'{prefix}_{i:04d}' for prefix in ('p','m','v') for i in range(len(schema))}
    if set(arrays)!=expected:raise ValueError('Checkpoint array coverage differs')
    trees={prefix:{row['path']:np.asarray(arrays[f'{prefix}_{i:04d}']) for i,row in enumerate(schema)} for prefix in ('p','m','v')}
    state=dict(first=trees['m'],second=trees['v'],step=np.asarray(arrays['step']))
    actual,_=flatten(trees['p'],state,configuration_sha256=configuration_sha256,source_sha256=source_sha256)
    if actual!=metadata:raise ValueError('Checkpoint metadata disagrees with arrays')
    return trees['p'],state
