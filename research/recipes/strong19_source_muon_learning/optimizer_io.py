"""Compose qualified Muon and Lookahead codecs, retaining dynamic scalars."""
from types import SimpleNamespace
import numpy as np
import lookahead_state_io
import muon_groups
import muon_state_io


def flatten(params,state,*,configuration_sha256,source_sha256,lookahead_config):
    if set(state)!={'first','second','step','lookahead','hyperparameters'}:raise ValueError('Incomplete source optimizer state')
    fast={k:state[k] for k in ('first','second','step')}
    meta,arrays=muon_state_io.flatten(params,fast,muon_groups.cnn(params),
        configuration_sha256=configuration_sha256,source_sha256=source_sha256)
    slow,values=lookahead_state_io.flatten(params,state['lookahead'],lookahead_config,source_sha256=source_sha256)
    for key,value in values.items():
        if key.startswith('p_'):
            if not np.array_equal(value,arrays[key]):raise ValueError('Fast parameter codecs disagree')
        else:arrays[key]=value
    h=np.asarray(state['hyperparameters'])
    if h.shape!=(13,) or h.dtype!=np.float32 or not np.isfinite(h).all() or (h<0).any() or h[-1]<=0:
        raise ValueError('Invalid dynamic schedule scalars')
    arrays['hyperparameters']=h
    return dict(kind='joint_source_muon_lookahead_state',version=1,step=meta['step'],
        configuration_sha256=configuration_sha256,source_sha256=source_sha256,
        parameter_schema=meta['parameter_schema'],muon=meta,lookahead=slow),arrays


def restore(metadata,arrays,*,schema,configuration_sha256,source_sha256,lookahead_config):
    if (metadata.get('kind')!='joint_source_muon_lookahead_state' or metadata.get('version')!=1
            or metadata.get('configuration_sha256')!=configuration_sha256
            or metadata.get('source_sha256')!=source_sha256):raise ValueError('Source state identity differs')
    descriptions={r['path']:SimpleNamespace(ndim=len(r['shape']),dtype=np.dtype(r['dtype'])) for r in schema}
    specs=muon_groups.cnn(descriptions);expected=[dict(**r,**specs[r['path']]) for r in schema]
    names={'step'};slow_names={'counter'}
    for i,row in enumerate(expected):
        names.update((f'p_{i:04d}',f'm_{i:04d}'));slow_names.add(f'p_{i:04d}')
        if row['group'] in muon_state_io.ADAM_GROUPS:names.add(f'v_{i:04d}')
        if lookahead_config['k'] is not None:slow_names.add(f's_{i:04d}')
    if set(arrays)!=names|slow_names|{'hyperparameters'}:raise ValueError('Combined array coverage differs')
    params,state=muon_state_io.restore(metadata['muon'],{k:arrays[k] for k in names},expected_schema=expected,
        configuration_sha256=configuration_sha256,source_sha256=source_sha256)
    _,slow=lookahead_state_io.restore(metadata['lookahead'],{k:arrays[k] for k in slow_names},
        config=lookahead_config,schema=schema,source_sha256=source_sha256)
    state=dict(**state,lookahead=slow,hyperparameters=arrays['hyperparameters'])
    actual,_=flatten(params,state,configuration_sha256=configuration_sha256,source_sha256=source_sha256,
                      lookahead_config=lookahead_config)
    if actual!=metadata:raise ValueError('Combined metadata and arrays disagree')
    return params,state
