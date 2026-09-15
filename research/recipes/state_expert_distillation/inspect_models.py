#!/usr/bin/env python3
"""Record full expert parameter counts and CPU compiler estimates without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); verify(SOURCE); a.output.mkdir(parents=True, exist_ok=False)
    import jax
    import numpy as np
    import model
    report = {'schema_version': 1, 'kind': 'state_expert_model_inspection', 'status': 'failed', 'snapshot': SOURCE.name,
              'started_unix': time.time(), 'arms': {}, 'claims_speedup': False, 'claims_mfu': False,
              'scope': 'Untrained full-architecture CPU leaf compilation and parameter accounting; no throughput or training comparison.'}
    try:
        for mode in ('history', 'state'):
            path = Path(__file__).parent / f'pilot_41_{mode}.json'; config = read_json(path); c = config['model']
            params = model.initialize(config['seed'], c)
            schema = [{'path': jax.tree_util.keystr(p), 'shape': list(x.shape), 'dtype': str(x.dtype)}
                      for p, x in jax.tree_util.tree_flatten_with_path(params)[0]]
            digest = hashlib.sha256(canonical_json(schema))
            for x in jax.tree.leaves(params):
                digest.update(np.ascontiguousarray(np.asarray(x)).tobytes())
            tokens = np.full((1, c['max_tokens']), c['size'] ** 2 + 2, np.int32); tokens[0, 0] = c['size'] ** 2 + 1
            arguments = (params, tokens, np.zeros(1, np.int32), np.zeros((1, c['size'] ** 2), np.uint8),
                         np.ones((1, c['size'] ** 2 + 1), bool), np.zeros(1, bool))
            start = time.perf_counter()
            lower = jax.jit(lambda *xs: model.leaf_predictions(*xs, c)).lower(*arguments)
            hlo = a.output / (mode + '.hlo.txt'); hlo.write_text(lower.compiler_ir('hlo').as_hlo_text())
            executable = lower.compile(); elapsed = time.perf_counter() - start
            logits, value = jax.device_get(executable(*arguments))
            if logits.shape != (1, c['size'] ** 2 + 1) or value.shape != (1,) or not np.isfinite(logits).all() or not np.isfinite(value).all():
                raise ValueError('Invalid full-model leaf output')
            report['arms'][mode] = {'config_sha256': sha256(path), 'model': c, 'parameter_count': sum(x.size for x in jax.tree.leaves(params)),
                'parameter_arrays': len(schema), 'initial_parameter_elements_sha256': digest.hexdigest(),
                'cpu_compilation_seconds': elapsed, 'compiler_cost_estimate': executable.cost_analysis(),
                'hlo_sha256': sha256(hlo), 'finite_leaf_outputs': True, 'frozen_observer_not_counted': True}
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(report))
        verify(SOURCE); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
