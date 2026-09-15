"""Check initialization-only encoder probes without selecting a learning run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify


def main():
    output = STUDY / 'encoder-scale-preparation-001.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    source = ROOT / '.gozero/snapshots/23c9dfe60c0a4c1c376e74f634b568fd1c85f3b13a0c7fef9172859ad38a40fb'
    manifest = verify(source)
    recipe = source / manifest['recipe']
    sys.path.insert(0, str(recipe))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import compute_budget
    import policy_model
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU-only preparation')
    original = read_json(source / 'resolved_config.json')['model']
    budget = compute_budget.transformer(original)
    schema = policy_model.parameter_schema(original)
    cpu = copy.deepcopy(original)
    cpu.update(read_json(recipe / 'cpu_transformer_harness.json')['model'])
    cpu['encoder_layer_scale'] = original['encoder_layer_scale']
    cpu['first_pass_aux_weight'] = original['first_pass_aux_weight']
    seed = 91312427
    initial = policy_model.initialize(seed, cpu)
    small = jax.device_get(initial)
    generator = np.random.default_rng(79123)
    batch = dict(spatial=jnp.asarray(generator.normal(size=(2,4,9,9,22)), jnp.float32),
                 global_features=jnp.asarray(generator.normal(size=(2,4,19)), jnp.float32),
                 actions=jnp.asarray([[0,1,81,2],[3,81,4,5]], jnp.int32),
                 counts=jnp.asarray([4,3], jnp.int32),
                 legal=jnp.ones((2,4,82), jnp.bool_),
                 policies=jnp.full((2,4,82), 1/82, jnp.float32))
    def lower(config, parameters):
        function = jax.value_and_grad(lambda p,b:policy_model.losses(p,b,config), has_aux=True)
        lowered = jax.jit(function).lower(parameters, batch)
        assembly = lowered.compiler_ir('stablehlo').operation.get_asm(enable_debug_info=False)
        return lowered, assembly
    lowered, assembly = lower(cpu, initial)
    compiled = lowered.compile()
    allowed = {'encoder.blocks.gamma', 'encoder.attention.attn.gamma', 'encoder.attention.mlp.gamma'}
    cases = []
    for scale in (1e-6, 1e-3, 1e-2, 1e-1):
        config = copy.deepcopy(original)
        config['encoder_layer_scale'] = scale
        if policy_model.parameter_schema(config) != schema or compute_budget.transformer(config) != budget:
            raise ValueError('Initialization changed full-size shape or decoding work')
        fixture = copy.deepcopy(cpu)
        fixture['encoder_layer_scale'] = scale
        params = policy_model.initialize(seed, fixture)
        concrete = jax.device_get(params)
        changed = {name for name in small if not np.array_equal(concrete[name], small[name])}
        if changed != (set() if scale == original['encoder_layer_scale'] else allowed):
            raise ValueError('Initialization altered parameters outside residual scales')
        for name in allowed:
            if not np.all(concrete[name] == np.float32(scale)):
                raise ValueError('Residual initialization has the wrong value')
        _, candidate_assembly = lower(fixture, params)
        if candidate_assembly != assembly:
            raise ValueError('The small forward/backward program changed')
        (loss, metrics), gradients = jax.device_get(compiled(params, batch))
        if not all(np.isfinite(x).all() for x in jax.tree.leaves((loss, metrics, gradients))):
            raise ValueError('Nonfinite initialization fixture loss or gradient')
        cases.append(dict(scale=scale, full_size_schema_equal=True, complete_decoding_budget_equal=True,
            non_scale_parameter_arrays_exact=True, changed_arrays=sorted(changed),
            small_forward_backward_stablehlo_identical=True, loss=float(loss),
            gradient_norm=float(np.sqrt(sum(np.sum(np.asarray(x,np.float64)**2) for x in gradients.values())))))
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(kind='encoder_scale_initialization_preparation', status='passed', created=time.time(),
        parent_snapshot=source.name, operator_sha256=sha(Path(__file__)),
        source_files={name:sha(recipe/name) for name in ('encoder.py','causal.py','policy_model.py','compute_budget.py')},
        cpu_model=cpu, budget=budget, cases=cases, jax_version=jax.__version__,
        small_forward_backward_stablehlo_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
        seconds=time.monotonic()-started,
        scope='Numerical preparation only: full-size parameter/schema and complete cached decoding arithmetic, small float32 initialization isolation and identical forward/backward HLO with finite gradients. No scientific rate/scale selection, learning registration, new snapshot, TPU dispatch, or performance claim.')
    with output.open('xb') as stream:
        stream.write(canonical_json(result))
    output.chmod(0o444)
    print(json.dumps(dict(status='passed', sha256=sha(output), cases=len(cases),
                         parameters=budget['trainable_parameters'],
                         decoding_flops_per_move=budget['multiply_add_flops_per_move'],
                         seconds=result['seconds'])))


if __name__ == '__main__':
    main()
