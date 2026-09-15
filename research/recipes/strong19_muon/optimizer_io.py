"""Joint-harness adapter for the typed Muon/auxiliary-Adam state codec."""
from types import SimpleNamespace

import numpy as np

import muon_groups
import muon_state_io


def flatten(params,state,*,configuration_sha256,source_sha256):
    return muon_state_io.flatten(params,state,muon_groups.cnn(params),
        configuration_sha256=configuration_sha256,source_sha256=source_sha256)


def restore(metadata,arrays,*,schema,configuration_sha256,source_sha256):
    # Classification uses names/ranks, without allocating model-sized dummy
    # tensors or transferring host checkpoint arrays to a JAX device.
    descriptions={r['path']:SimpleNamespace(ndim=len(r['shape']),dtype=np.dtype(r['dtype'])) for r in schema}
    specs=muon_groups.cnn(descriptions)
    expected=[dict(**row,**specs[row['path']]) for row in schema]
    return muon_state_io.restore(metadata,arrays,expected_schema=expected,
        configuration_sha256=configuration_sha256,source_sha256=source_sha256)
