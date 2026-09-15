"""Typed fast/slow parameter checkpoint fixture, independent of optimizer codec."""
import math
import re

import numpy as np


def flatten(params,state,config,*,source_sha256):
    if re.fullmatch('[0-9a-f]{64}',source_sha256 or '') is None:raise ValueError('Source identity required')
    if set(config)!={'k','alpha'}:raise ValueError('Lookahead configuration required')
    enabled=config['k'] is not None
    if enabled:
        if (type(config['k']) is not int or config['k']<1 or type(config['alpha']) not in (int,float)
                or not math.isfinite(config['alpha']) or not 0<config['alpha']<1):raise ValueError('Invalid Lookahead settings')
    elif config['alpha'] is not None:raise ValueError('Disabled alpha differs')
    if set(state)!={'slow','counter'} or set(state['slow'])!=(set(params) if enabled else set()):
        raise ValueError('State coverage differs')
    clock=np.asarray(state['counter'])
    if clock.dtype!=np.int32 or clock.shape!=() or not 0<=int(clock)<(config['k'] if enabled else 1):
        raise ValueError('Invalid Lookahead counter')
    arrays=dict(counter=clock);schema=[]
    for i,(name,value) in enumerate(sorted(params.items())):
        p=np.asarray(value)
        if p.dtype!=np.float32 or not np.isfinite(p).all():raise ValueError('Invalid fast parameter')
        schema.append(dict(path=name,shape=list(p.shape),dtype='float32'))
        arrays[f'p_{i:04d}']=p
        if enabled:
            s=np.asarray(state['slow'][name])
            if s.dtype!=np.float32 or s.shape!=p.shape or not np.isfinite(s).all():raise ValueError('Invalid slow parameter')
            arrays[f's_{i:04d}']=s
    if not schema:raise ValueError('Empty parameter state')
    return dict(kind='lookahead_fast_slow_fixture',version=1,source_sha256=source_sha256,
        config=config,schema=schema,counter=int(clock)),arrays


def restore(metadata,arrays,*,config,schema,source_sha256):
    if (metadata.get('kind')!='lookahead_fast_slow_fixture' or metadata.get('version')!=1
            or metadata.get('source_sha256')!=source_sha256 or metadata.get('config')!=config
            or metadata.get('schema')!=schema):raise ValueError('Lookahead identity/config/schema differs')
    params={};slow={};expected={'counter'}
    for i,row in enumerate(schema):
        key=f'p_{i:04d}';expected.add(key);params[row['path']]=arrays[key]
        if config['k'] is not None:
            key=f's_{i:04d}';expected.add(key);slow[row['path']]=arrays[key]
    if set(arrays)!=expected:raise ValueError('Lookahead checkpoint coverage differs')
    state=dict(slow=slow,counter=arrays['counter'])
    result,_=flatten(params,state,config,source_sha256=source_sha256)
    if result!=metadata:raise ValueError('Checkpoint metadata and arrays disagree')
    return params,state
