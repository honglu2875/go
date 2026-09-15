"""Full decode budgets and differentiated recomputation cost before TPU use."""
from pathlib import Path
import json
import sys
import jax
import jax.numpy as jnp

ROOT=Path('/workspace/go');OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'.gozero/snapshots/eb1cf83264188cd19a5bfeaaa1a5370ec5c54c932229fae8ea3f365de256ccde'
RECIPE=SOURCE/'research/recipes/visual_token_encoder'
sys.path[:0]=[str(RECIPE),str(SOURCE/'packages/gozero/src')]
from gozero.snapshots import verify,read_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish
from gozero.jaxpr_cost import analyze
from qualify_budget import trace_transformer,abstract
from training_arithmetic import count
import causal
import katago
import compute_budget
import policy_model

verify(SOURCE)
cnn_source=ROOT/'.gozero/snapshots/4eff9303401f25868248047c26418d1fc5d2390f9b6f7794242802b130fc7a89'
verify(cnn_source);cnn=read_json(cnn_source/'resolved_config.json')['model']
p=jax.eval_shape(lambda:katago.initialize(0,cnn))
graph=jax.make_jaxpr(lambda p,s,g:katago.forward(p,s,g,cnn))(p,abstract((128,9,9,22)),abstract((128,19)))
ca=analyze(graph);cb=compute_budget.cnn(cnn,9,128)
assert not ca['unaccounted_primitives'] and ca['counts']['multiply_add_flops']==cb['multiply_add_flops_per_batch']
rows=[]
for name in ['c768_d6_l30_causal_remat','c768_d6_l30_mean_remat','c768_d6_l30_local_remat','c768_d6_l30_mean_local_remat']:
    path=RECIPE/(name+'.json');model=read_json(path)['model']
    cases={f'{size}x{size}-past{past}':trace_transformer(model,size,past)
           for size in (9,19) for past in (0,32,128,256)}
    chosen=cases['9x9-past128'];without=trace_transformer({**model,'encoder_rematerialize':False},9,128)
    assert chosen['analytic']==without['analytic'] and chosen['traced']['counts']==without['traced']['counts']
    ratios={'parameters':chosen['analytic']['trainable_parameters']/cb['trainable_parameters']-1,
            'dense_multiply_add_flops':chosen['analytic']['multiply_add_flops_per_batch']/cb['multiply_add_flops_per_batch']-1,
            'unit_cost_floating_operations':chosen['traced']['floating_operations_unit_cost']/ca['floating_operations_unit_cost']-1}
    assert all(abs(x)<=.01 for x in ratios.values())
    rows.append({'config':name+'.json','config_sha256':sha256(path),'model':model,'cases':cases,'relative_differences_at_reference':ratios})
    print(json.dumps({'candidate':name,'decode_budget':'passed','ratios':ratios}),flush=True)
publish(OUT/'decode_budgets.json',{'kind':'large_encoder_rematerialization_decode_budgets','status':'passed',
        'snapshot':SOURCE.name,'driver_sha256':sha256(Path(__file__)),
        'numerical_sources':{n:sha256(RECIPE/n) for n in ['causal.py','compute_budget.py','observation_attention.py']},
        'candidates':rows,'scope':'Complete encoder and cached decoder accounting at eight board/history contexts per candidate. Rematerialization preserves inference arithmetic exactly. The1% matching criterion applies only to9x9,batch128,past128. No TPU qualification or learning yet.'})

control_source=ROOT/'.gozero/snapshots/d2a87e3bd45cdf7464e4be4bde0d9776ca0d450ffac283a832ec48f665465993'
verify(control_source);base=read_json(control_source/'resolved_config.json')['model']
large=read_json(RECIPE/'c768_d6_l30_causal_remat.json')['model']
models={'c128_control':base,'large_no_encoder_remat':{**large,'encoder_rematerialize':False},'large_encoder_remat':large}
cases={};totals={};schedule={128:875,256:142,384:7}
for label,model in models.items():
    total=0;params=jax.eval_shape(lambda:policy_model.initialize(0,model))
    for length,multiplicity in schedule.items():
        s=jax.ShapeDtypeStruct
        batch={'spatial':s((128,length,9,9,22),jnp.float32),'global_features':s((128,length,19),jnp.float32),
               'actions':s((128,length),jnp.int32),'counts':s((128,),jnp.int32),
               'policies':s((128,length,82),jnp.float32),'legal':s((128,length,82),jnp.bool_)}
        graph=jax.make_jaxpr(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,model)[0]))(params,batch)
        row=count(graph);cases[f'{label}-{length}']=row;total+=multiplicity*row['total_matrix_flops']
        print(json.dumps({'candidate':label,'bucket':length,'training_matrix_flops':row['total_matrix_flops']}),flush=True)
    totals[label]=total
for length in schedule:
    actual=cases[f'large_encoder_remat-{length}']['total_matrix_flops']-cases[f'large_no_encoder_remat-{length}']['total_matrix_flops']
    expected=2*128*length*81*(9*22*768+5*9*768*768)
    if actual!=expected:raise ValueError('Recomputation cost is not exactly one additional forward convolution per layer')
publish(OUT/'training_arithmetic.json',{'kind':'large_encoder_training_arithmetic_with_recomputation','status':'passed',
        'snapshot':SOURCE.name,'driver_sha256':sha256(Path(__file__)),'cases':cases,'fixed_bucket_schedule':schedule,
        'padded_board_positions':128*sum(length*n for length,n in schedule.items()),'actual_training_position_exposures':11469333,
        'total_matrix_flops':totals,'relative_large_to_control':totals['large_encoder_remat']/totals['c128_control']-1,
        'scope':'Full differentiated model matrices, explicit decoder and encoder rematerialization, qualified512-tile Splash work and padded complete histories. Additional convolutional recomputation is included. This is arithmetic accounting, not measured MFU.'})
print(json.dumps({'status':'passed','decode_budgets_sha256':sha256(OUT/'decode_budgets.json'),
                  'training_arithmetic_sha256':sha256(OUT/'training_arithmetic.json'),'training_totals':totals}),flush=True)
