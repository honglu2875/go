"""Refresh the prepared C128 spatial-policy fallback against current sources."""
from pathlib import Path
import json
import sys
import jax

ROOT = Path('/workspace/go')
SOURCE = ROOT / '.gozero/snapshots/f3065dbdadab9d11ef6af8f10384b0ce7b11a3f88ec8947ee62c752d9fab3bb4'
RECIPE = SOURCE / 'research/recipes/visual_token_encoder'
sys.path[:0] = [str(RECIPE), str(SOURCE / 'packages/gozero/src')]
from gozero.snapshots import verify, read_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish
from gozero.jaxpr_cost import analyze
from qualify_budget import trace_transformer, abstract
import compute_budget
import katago

verify(SOURCE)
CNN = ROOT / '.gozero/snapshots/4eff9303401f25868248047c26418d1fc5d2390f9b6f7794242802b130fc7a89'
verify(CNN)
cnn = read_json(CNN / 'resolved_config.json')['model']
params = jax.eval_shape(lambda: katago.initialize(0, cnn))
graph = jax.make_jaxpr(lambda p, s, g: katago.forward(p, s, g, cnn))(
    params, abstract((128, 9, 9, 22)), abstract((128, 19)))
traced_cnn = analyze(graph)
analytic_cnn = compute_budget.cnn(cnn, 9, 128)
assert not traced_cnn['unaccounted_primitives']
assert traced_cnn['counts']['multiply_add_flops'] == analytic_cnn['multiply_add_flops_per_batch']
config = RECIPE / 'c128_d2_local.json'
model = read_json(config)['model']
cases = {}
for board in (9, 19):
    for past in (0, 32, 128, 256):
        cases[f'{board}x{board}-past{past}'] = trace_transformer(model, board, past)
        print(json.dumps({'board': board, 'past': past, 'status': 'passed'}), flush=True)
chosen = cases['9x9-past128']
ratios = {
    'parameters': chosen['analytic']['trainable_parameters'] / analytic_cnn['trainable_parameters'] - 1,
    'dense_multiply_add_flops': chosen['analytic']['multiply_add_flops_per_batch'] / analytic_cnn['multiply_add_flops_per_batch'] - 1,
    'unit_cost_floating_operations': chosen['traced']['floating_operations_unit_cost'] / traced_cnn['floating_operations_unit_cost'] - 1,
}
assert all(abs(value) <= .01 for value in ratios.values())
output = Path(__file__).with_name('c128_current_decode_budgets.json')
publish(output, {
    'kind': 'c128_spatial_policy_current_source_budget', 'status': 'passed',
    'snapshot': SOURCE.name, 'driver_sha256': sha256(Path(__file__)),
    'configuration': str(config.relative_to(SOURCE)), 'config_sha256': sha256(config),
    'model': model, 'cases': cases, 'relative_differences_at_reference': ratios,
    'numerical_sources': {name: sha256(RECIPE / name) for name in ('causal.py', 'compute_budget.py', 'observation_attention.py')},
    'scope': 'Prepared fallback only; no TPU job or learning selection. Includes the reused full encoder, pointwise correction and cached decoder. One-percent matching applies to 9x9, batch128, past128; other contexts are explicitly reported.',
})
print(json.dumps({'status': 'passed', 'sha256': sha256(output), 'ratios': ratios}), flush=True)
