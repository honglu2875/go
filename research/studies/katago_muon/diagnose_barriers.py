"""Isolate stage boundaries without changing the momentum equation."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from diagnose_polar import STUDY, difference, sha, trace
from pathlib import Path


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU diagnostic only')
    with np.load(STUDY/'polar-trace-reference-001.npz',allow_pickle=False) as z:
        ref = {k:z[k] for k in z.files}
    with np.load(STUDY/'reference-001/arrays.npz',allow_pickle=False) as z:
        gradient = jnp.asarray(z['g1.0'].transpose(0,4,3,1,2).copy().reshape(3,8,12))
    report = dict(kind='muon_stage_boundary_isolation',created=time.time(),
                  operator_sha256=sha(Path(__file__)),
                  trace_operator_sha256=sha(STUDY/'diagnose_polar.py'),variants=[])
    for stable, divide in [(False,False),(False,True),(True,False)]:
        options = dict(stable_lerp=stable,float_div=divide,barriers=True)
        actual = jax.device_get(jax.jit(lambda g: trace(g,**options))(gradient))
        rows = {k:difference(v,ref[k]) for k,v in actual.items()}
        report['variants'].append(dict(options=options,stages=rows))
        print(json.dumps(dict(options=options,differences={k:v for k,v in rows.items() if v['different']})),flush=True)
    path = STUDY/'polar-diagnostic-002.json'
    with path.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    path.chmod(0o444)
    print(json.dumps(dict(sha256=sha(path))),flush=True)


if __name__ == '__main__':
    main()
