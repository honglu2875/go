"""Trace the helper-free CNN and verify its unchanged historical main graph."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.jaxpr_cost import analyze
from gozero.snapshots import canonical_json, read_json, verify
import katago
import policy_model
import compute_budget
import training_arithmetic

NUMERICAL_FILES = ('katago.py', 'policy_model.py', 'policy_config.py',
                   'policy_optimizer.py', 'train_policy.py', 'train.py', 'compute_budget.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); baseline = verify(a.baseline)
    c = read_json(a.config); old = read_json(a.baseline / 'resolved_config.json')
    recipe = Path(__file__).parent; original = a.baseline / baseline['recipe']
    if c != old:
        raise ValueError('Full learning settings must exactly match the historical CNN')
    unchanged = [n for n in NUMERICAL_FILES if n != 'policy_model.py']
    if any((recipe/n).read_bytes() != (original/n).read_bytes() for n in unchanged):
        raise ValueError('A numerical file outside the objective changed')
    old_text = (original/'policy_model.py').read_text().split('\ndef losses(')[0]
    new_text = (recipe/'policy_model.py').read_text().split('\ndef losses(')[0]
    if old_text != new_text:
        raise ValueError('Model code outside losses changed')
    m = c['model']; s = jax.ShapeDtypeStruct
    p = jax.eval_shape(lambda: policy_model.initialize(c['seed'], m))
    graph = jax.make_jaxpr(lambda p,x,g: katago.forward(p,x,g,m))(
        p, s((128,9,9,22),jnp.float32), s((128,19),jnp.float32))
    traced = analyze(graph); analytical = compute_budget.cnn(m,9,128)
    if (traced['unaccounted_primitives'] or traced['counts']['multiply_add_flops']
            != analytical['multiply_add_flops_per_batch']):
        raise ValueError('Unqualified inference arithmetic')
    schema = policy_model.parameter_schema(m)
    stored = sum(x['elements'] for x in schema)
    inactive = sum(x['elements'] for x in schema if x['path'].startswith(('intermediate_', 'norm_intermediate_')))
    if stored != 232431872 or stored-inactive != analytical['inference_parameters']:
        raise ValueError('Unexpected active parameter allocation')
    training = {}
    for length in c['dataset']['buckets']:
        b = {'spatial':s((128,length,9,9,22),jnp.float32),
             'global_features':s((128,length,19),jnp.float32),
             'actions':s((128,length),jnp.int32), 'counts':s((128,),jnp.int32),
             'policies':s((128,length,82),jnp.float32), 'legal':s((128,length,82),jnp.bool_)}
        g = jax.make_jaxpr(jax.value_and_grad(lambda p,b: policy_model.losses(p,b,m)[0]))(p,b)
        training[str(length)] = training_arithmetic.count(g)
    result = {'status':'passed', 'kind':'cnn_main_cpu_qualification', 'config':c,
        'baseline_snapshot':a.baseline.name, 'unchanged_numerical_files':unchanged,
        'source_sha256':{n:hashlib.sha256((recipe/n).read_bytes()).hexdigest() for n in NUMERICAL_FILES},
        'stored_parameters':stored, 'active_parameters':stored-inactive, 'unused_helper_slots':inactive,
        'decode':{'analytical':analytical,'traced':traced}, 'training_arithmetic':training,
        'scope':'Logical inference and differentiated matrix arithmetic, including rematerialization. No achieved MFU claim; full TPU qualification is separate.'}
    with a.output.open('xb') as f: f.write(canonical_json(result))
    a.output.chmod(0o444)
    print(json.dumps({'status':'passed','active_parameters':stored-inactive,
        'unused_helper_slots':inactive,'output':str(a.output)}),flush=True)


if __name__ == '__main__': main()
