"""Locate arithmetic differences against the retained first-leaf Torch trace."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/recipes/katago_muon'))
import muon


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return dict(different=int(np.count_nonzero(a != b)), total=int(a.size),
                maximum=float(np.max(np.abs(a-b))),
                relative=float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)))


def trace(g, *, stable_lerp=False, float_div=False, barriers=False):
    result = {}
    keep = lambda x: jax.lax.optimization_barrier(x) if barriers else x
    result['momentum'] = m = keep(.05 * g)
    result['nesterov'] = u = keep(m - .05 * (m-g) if stable_lerp else g + .95 * (m-g))
    result['cast'] = x = keep(u.astype(jnp.bfloat16))
    result['norm'] = norm = keep(jnp.sqrt(jnp.sum(x.astype(jnp.float32)**2, axis=(-2,-1), keepdims=True)).astype(jnp.bfloat16))
    denominator = keep(norm + jnp.bfloat16(1e-7))
    if float_div:
        x = keep((x.astype(jnp.float32)/denominator.astype(jnp.float32)).astype(jnp.bfloat16))
    else:
        x = keep(x / denominator)
    result['normalized'] = x
    mm = lambda a,b: keep(jnp.matmul(a,b,preferred_element_type=jnp.float32).astype(jnp.bfloat16))
    sc = lambda a,b: keep(muon.scalar_bf16(a,b))
    for i in range(5):
        result[f'A{i}'] = a = mm(x, jnp.swapaxes(x,-2,-1))
        result[f'cA{i}'] = ca = sc(a,2.0315)
        result[f'B{i}'] = b = keep(sc(a,-4.775) + mm(ca,a))
        result[f'ax{i}'] = ax = sc(x,3.4445)
        result[f'bx{i}'] = bx = mm(b,x)
        result[f'X{i}'] = x = keep(ax+bx)
    result['scaled'] = sc(x,.2*math.sqrt(12))
    return result


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU diagnostic only')
    with np.load(STUDY/'polar-trace-reference-001.npz',allow_pickle=False) as z:
        ref = {k:z[k] for k in z.files}
    with np.load(STUDY/'reference-001/arrays.npz',allow_pickle=False) as z:
        gradient = jnp.asarray(z['g1.0'].transpose(0,4,3,1,2).copy().reshape(3,8,12))
        original = z['u1.0'].transpose(0,4,3,1,2).copy().reshape(3,8,12)
    report = dict(kind='muon_first_difference_diagnostic', created=time.time(),
                  operator_sha256=sha(Path(__file__)), muon_sha256=sha(Path(muon.__file__)),
                  reference_trace_sha256=sha(STUDY/'polar-trace-reference-001.npz'),
                  trace_vs_original=difference(ref['scaled'],original), variants=[])
    variants = [dict(stable_lerp=False,float_div=False,barriers=False),
                dict(stable_lerp=False,float_div=True,barriers=False),
                dict(stable_lerp=True,float_div=True,barriers=False),
                dict(stable_lerp=True,float_div=True,barriers=True)]
    for options in variants:
        for mode in ('eager','jit_all','jit_final'):
            fn = lambda g: trace(g,**options)
            if mode == 'jit_all':
                fn = jax.jit(fn)
            elif mode == 'jit_final':
                fn = jax.jit(lambda g: dict(scaled=trace(g,**options)['scaled']))
            actual = jax.device_get(fn(gradient))
            rows = {k:difference(v,ref[k]) for k,v in actual.items()}
            row = dict(options=options,mode=mode,stages=rows)
            report['variants'].append(row)
            print(json.dumps(dict(options=options,mode=mode,
                                  differences={k:v for k,v in rows.items() if v['different']})),flush=True)
    path = STUDY/'polar-diagnostic-001.json'
    with path.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    path.chmod(0o444)
    print(json.dumps(dict(trace_vs_original=report['trace_vs_original'],sha256=sha(path))),flush=True)


if __name__ == '__main__':
    main()
