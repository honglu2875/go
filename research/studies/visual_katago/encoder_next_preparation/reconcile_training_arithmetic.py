"""Reconcile existing, audited matrix counts with the realized study schedule."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/go')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / '.gozero/snapshots/f3065dbdadab9d11ef6af8f10384b0ce7b11a3f88ec8947ee62c752d9fab3bb4'
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import read_json, verify
from gozero.checkpoint_archive import publish
from gozero.checkpoints import sha256

COUNT_SOURCE = ROOT / '.gozero/snapshots/7010cbdff4c01fa19a4c2e5c7851a3cb233df42ffb2f52ac1112f0d02fe76c65'
CNN_SOURCE = ROOT / '.gozero/snapshots/4eff9303401f25868248047c26418d1fc5d2390f9b6f7794242802b130fc7a89'
OLD_CNN = ROOT / '.gozero/snapshots/625f75ca64b04da8b65e36e6ad8c83f42ee5674eb67637faabcf60efaf256997'
for source in (SOURCE, COUNT_SOURCE, CNN_SOURCE, OLD_CNN):
    verify(source)

cnn_path = ROOT / 'research/studies/visual_katago/training_arithmetic_verified_tiles.json'
transformer_path = ROOT / 'research/studies/visual_katago/encoder_large_preparation/bf16_boundary/training_arithmetic.json'
cnn = read_json(cnn_path)
transformer = read_json(transformer_path)
assert cnn['operator_snapshot'] == COUNT_SOURCE.name
assert cnn['baseline_snapshot'] == OLD_CNN.name
assert read_json(OLD_CNN / 'resolved_config.json')['model'] == read_json(CNN_SOURCE / 'resolved_config.json')['model']
assert read_json(CNN_SOURCE / 'resolved_config.json')['model']['rematerialize'] is True

count_recipe = COUNT_SOURCE / 'research/recipes/visual_token_linear'
cnn_recipe = CNN_SOURCE / 'research/recipes/visual_katago_lr'
assert sha256(count_recipe / 'katago.py') == sha256(cnn_recipe / 'katago.py')

# The counting operator extends the CNN adapter with a causal architecture
# branch. Removing precisely those branches must recover the executed CNN
# adapter, including its 0.2/0.8 main/helper objective and metric expressions.
class CnnOnly(ast.NodeTransformer):
    def visit_Import(self, node):
        return None if [n.name for n in node.names] == ['causal'] else node

    def visit_If(self, node):
        if ast.dump(node.test) == ast.dump(ast.parse("c['architecture']=='causal_visual_policy'", mode='eval').body):
            assert not node.orelse
            return None
        return self.generic_visit(node)

count_ast = CnnOnly().visit(ast.parse((count_recipe / 'policy_model.py').read_text()))
cnn_ast = ast.parse((cnn_recipe / 'policy_model.py').read_text())
assert ast.dump(count_ast) == ast.dump(cnn_ast)

schedule = {128: 875, 256: 142, 384: 7}
assert transformer['fixed_bucket_schedule'] == {str(k): v for k, v in schedule.items()}
cnn_total = sum(cnn['cases'][f'cnn-{length}-remat1']['total_matrix_flops'] * multiplicity
                for length, multiplicity in schedule.items())
totals = {'cnn': cnn_total, **transformer['total_matrix_flops']}
result = {
    'kind': 'encoder_cnn_realized_training_matrix_comparison',
    'status': 'passed',
    'driver_sha256': sha256(Path(__file__)),
    'inputs': {str(p.relative_to(ROOT)): sha256(p) for p in (cnn_path, transformer_path)},
    'cnn_training_snapshot': CNN_SOURCE.name,
    'cnn_counting_snapshot': COUNT_SOURCE.name,
    'cnn_model_identical_to_counted_baseline': True,
    'cnn_adapter_ast_identical_after_removing_unused_transformer_branch': True,
    'cnn_numerical_sources': {name: sha256(cnn_recipe / name) for name in ('katago.py', 'policy_model.py')},
    'fixed_bucket_schedule': schedule,
    'padded_board_positions': 128 * sum(k * v for k, v in schedule.items()),
    'actual_training_position_exposures': 11469333,
    'total_matrix_flops': totals,
    'relative_to_cnn': {k: v / cnn_total - 1 for k, v in totals.items()},
    'scope': 'Reuses independently counted differentiated graphs and the realized common 1024-update schedule. Includes CNN helper loss, encoder and decoder backward matrices, explicit rematerialization, and complete active Splash tiles for padded histories. FMA counts as two operations.',
    'interpretation': 'The study matches complete cached decoding arithmetic at its registered reference context, not total training arithmetic. C128 and the large rematerialized encoder use about 22% fewer counted training matrix operations than the CNN on these exact exposures. Matching inference cost does not imply matching training cost or time.',
    'exclusions': 'Elementwise/nonlinear operations, optimizer operations, SPMD communication, physical MXU layout, transfers and compilation. These are logical graph counts, not measured MFU or hardware FLOP counters.',
}
destination = OUT / 'cnn_training_arithmetic_reconciliation.json'
publish(destination, result)
print(json.dumps({'status': 'passed', 'output': str(destination), 'sha256': sha256(destination),
                  'total_matrix_flops': totals, 'relative_to_cnn': result['relative_to_cnn']}), flush=True)
