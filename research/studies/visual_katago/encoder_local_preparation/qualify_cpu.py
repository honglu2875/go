"""Account for prepared encoder options without running or selecting learning."""
from pathlib import Path
import hashlib
import json
import sys
import jax
import jax.numpy as jnp

ROOT=Path('/workspace/go')
OUT=ROOT/'research/studies/visual_katago/encoder_local_preparation'
SOURCE=ROOT/'.gozero/snapshots/c5d5c4d393e234d1e724c803efb88bc6cd92854121200f9d54d96ecbf2ac1b42'
RECIPE=SOURCE/'research/recipes/visual_token_encoder'
sys.path[:0]=[str(RECIPE),str(SOURCE/'packages/gozero/src')]
from gozero.snapshots import verify,read_json
from gozero.checkpoints import sha256
from gozero.checkpoint_archive import publish
from gozero.jaxpr_cost import analyze
import causal
import katago
import policy_model
import compute_budget
from qualify_budget import trace_transformer,abstract
from training_arithmetic import count

verify(SOURCE)
baseline=ROOT/'.gozero/snapshots/4eff9303401f25868248047c26418d1fc5d2390f9b6f7794242802b130fc7a89'
verify(baseline)
cn=read_json(baseline/'resolved_config.json')['model']
cp=jax.eval_shape(lambda:katago.initialize(0,cn))
cg=jax.make_jaxpr(lambda p,s,g:katago.forward(p,s,g,cn))(cp,abstract((128,9,9,22)),abstract((128,19)))
ca=analyze(cg);cb=compute_budget.cnn(cn,9,128)
assert not ca['unaccounted_primitives'] and ca['counts']['multiply_add_flops']==cb['multiply_add_flops_per_batch']
rows=[]
names=['c128_d2_mean','c768_d6_l30_causal','c768_d6_l30_mean',
       'c128_d2_local','c128_d2_mean_local','c768_d6_l30_mean_local']
for name in names:
    path=RECIPE/(name+'.json');cfg=read_json(path);model=cfg['model']
    cases={f'{board}x{board}-past{past}':trace_transformer(model,board,past)
           for board in (9,19) for past in (0,32,128,256)}
    chosen=cases['9x9-past128']
    ratios={'parameters':chosen['analytic']['trainable_parameters']/cb['trainable_parameters']-1,
            'dense_multiply_add_flops':chosen['analytic']['multiply_add_flops_per_batch']/cb['multiply_add_flops_per_batch']-1,
            'unit_cost_floating_operations':chosen['traced']['floating_operations_unit_cost']/ca['floating_operations_unit_cost']-1}
    if any(abs(x)>.01 for x in ratios.values()):raise ValueError('Unmatched candidate: '+name)
    if model.get('policy_spatial_bias',False):
        parent=trace_transformer({k:v for k,v in model.items() if k!='policy_spatial_bias'},9,128)
        assert chosen['analytic']['trainable_parameters']-parent['analytic']['trainable_parameters']==causal.encoder_channels(model)+1
        assert chosen['analytic']['multiply_add_flops_per_batch']-parent['analytic']['multiply_add_flops_per_batch']==2*128*81*causal.encoder_channels(model)
    rows.append({'config':name+'.json','config_sha256':sha256(path),'model':model,'cases':cases,
                 'relative_differences_at_reference':ratios})
    print(json.dumps({'kind':'complete_decode_budget','candidate':name,'status':'passed','ratios':ratios}),flush=True)
result={'kind':'optional_spatial_policy_complete_decode_budgets','status':'passed','snapshot':SOURCE.name,
        'driver_sha256':sha256(Path(__file__)),'numerical_sources':{n:sha256(RECIPE/n) for n in ['causal.py','compute_budget.py','observation_attention.py']},
        'candidates':rows,'cnn':{'analytic':cb,'traced':ca},'learning_scheduled':False,
        'scope':'Prepared alternatives only; full encoder plus cached inference, including the optional pointwise policy correction and scalar operations. Analytic and traced matrix work agree. Parameters and arithmetic match CNN within1% only at9x9,batch128,past128; other contexts remain explicitly unmatched. TPU qualification remains required.'}
publish(OUT/'decode_budgets.json',result)

# Check differentiated matrices: the shared encoder must not be run twice.
training=[]
for name in ['c128_d2_mean_local','c768_d6_l30_mean_local']:
    model=read_json(RECIPE/(name+'.json'))['model']
    for length in (128,256,384):
        shape=jax.ShapeDtypeStruct
        batch={'spatial':shape((128,length,9,9,22),jnp.float32),
               'global_features':shape((128,length,19),jnp.float32),
               'actions':shape((128,length),jnp.int32),'counts':shape((128,),jnp.int32),
               'policies':shape((128,length,82),jnp.float32),'legal':shape((128,length,82),jnp.bool_)}
        work={}
        for local in (False,True):
            net={**model,'policy_spatial_bias':local}
            params=jax.eval_shape(lambda:policy_model.initialize(0,net))
            graph=jax.make_jaxpr(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,net)[0]))(params,batch)
            work[str(local)]=count(graph)
        expected=3*2*128*length*81*causal.encoder_channels(model)
        actual=work['True']['total_matrix_flops']-work['False']['total_matrix_flops']
        if actual!=expected or work['True']['matrix_flops']['convolutions']!=work['False']['matrix_flops']['convolutions']:
            raise ValueError('Local policy duplicates encoding or differentiated matrix work differs')
        training.append({'candidate':name,'bucket':length,'differentiated_extra_matrix_flops':actual,
                         'expected_three_projection_matrices':expected,'work':work})
        print(json.dumps({'kind':'training_graph_delta','candidate':name,'bucket':length,'status':'passed','extra_matrix_flops':actual}),flush=True)
publish(OUT/'training_graph_delta.json',{'kind':'spatial_policy_training_matrix_delta','status':'passed',
        'snapshot':SOURCE.name,'driver_sha256':sha256(Path(__file__)),'cases':training,
        'scope':'Logical differentiated training matrices with explicit rematerialization and512-tile Splash work. Adding the single policy correction leaves convolution work unchanged and adds exactly forward, input-gradient and weight-gradient projections. This is arithmetic, not a hardware utilization measurement.'})
print(json.dumps({'status':'passed','decode_budgets_sha256':sha256(OUT/'decode_budgets.json'),
                  'training_graph_delta_sha256':sha256(OUT/'training_graph_delta.json')}),flush=True)
