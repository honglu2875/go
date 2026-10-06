"""Bind qualified source lineage and freeze the CNN systems control."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]; STUDY=Path(__file__).resolve().parent
RECIPE=ROOT/'research/recipes/strong19_moe_comparison'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
from gozero.snapshots import canonical_json,freeze
from gozero.durable_files import atomic_json,sha256


def main():
    import joint,train_config
    parent=ROOT/'research/recipes/strong19_moe_temporal'
    evidence=ROOT/'research/studies/strong19_moe_temporal/cpu-qualification-002.json'
    cpu=json.loads(evidence.read_text());assert cpu['status']=='passed'
    for relative,digest in cpu['source_sha256'].items():
        assert sha256(ROOT/relative)==digest,relative
    inherited={}
    for source in parent.glob('*.py'):
        target=RECIPE/source.name
        if source.name=='qualify_system.py':
            expected=source.read_text().replace('mesh=mesh,skip_padding=True)',
                       "mesh=mesh,skip_padding=base['training'].get('skip_padding',False))")
            assert target.read_text()==expected
        else:assert source.read_bytes()==target.read_bytes(),source.name
        inherited[source.name]=dict(parent_sha256=sha256(source),clone_sha256=sha256(target))
    cnn_cpu=json.loads((STUDY/'cpu-ce-qualification-001.json').read_text())
    assert cnn_cpu['status']=='passed' and cnn_cpu['tests']==6
    for name,digest in cnn_cpu['sources'].items():assert sha256(RECIPE/name)==digest,name
    cnn=json.loads((STUDY/'cnn-config-001.json').read_text())
    temporal=json.loads((STUDY/'temporal-config-001.json').read_text())
    dense=json.loads((ROOT/'research/studies/strong19_recovery/flat-config-001.json').read_text())
    for config in (cnn,temporal):
        train_config.validate(config)
        for key in ('seed','dataset','learner','evaluation','eval_every','steps','value_model'):
            assert config[key]==dense[key],key
    old=json.loads((ROOT/'research/studies/strong19_joint/budget-qualification-001.json').read_text())
    assert old['status']=='passed' and cnn['model']==old['configs']['cnn']
    schema=joint.parameter_schema(cnn['model'],cnn['value_model'])
    assert schema==old['parameter_schemas']['cnn']
    assert sum(x['elements'] for x in schema)==233220870
    temporal_budget=json.loads((ROOT/'research/studies/strong19_moe_temporal/budget-001.json').read_text())
    temporal_schema=joint.parameter_schema(temporal['model'],temporal['value_model'])
    assert temporal_schema==temporal_budget['budgets']['temporal_experts']['schema']
    matrix=old['cnn_decode']['multiply_add_flops_per_move']
    budgets=dict(cnn=dict(parameters=233220870,decode_matrix_flops=matrix),
        **{name:dict(parameters=temporal_budget['budgets'][key]['parameters'],
                     decode_matrix_flops=temporal_budget['budgets'][key]['decode']['128']['total_matrix_flops'])
           for name,key in [('dense','dense'),('temporal','temporal_experts'),('all_experts','all_experts')]})
    for row in budgets.values():row['active_flop_ratio_to_dense']=row['decode_matrix_flops']/budgets['dense']['decode_matrix_flops']
    assert all(abs(x['active_flop_ratio_to_dense']-1)<.01 for x in budgets.values())
    atomic_json(STUDY/'source-budget-lineage-001.json',dict(status='passed',created=time.time(),
        files=inherited,budgets=budgets,cnn_schema=schema,temporal_schema=temporal_schema,
        inherited_temporal_cpu_sha256=sha256(evidence),cnn_cpu_sha256=sha256(STUDY/'cpu-ce-qualification-001.json'),
        original_cnn_budget_sha256=sha256(ROOT/'research/studies/strong19_joint/budget-qualification-001.json'),
        operator_sha256=sha256(Path(__file__)),
        scope='No learner-math changes. CNN schema/configuration equal the registered earlier control. Cached-move logical matrix work includes encoder, both normal temporal tokens, routers and heads; physical overhead is measured separately.'),replace=False)
    config=json.loads((ROOT/'research/studies/strong19_moe_temporal/system-proposal-config-001.json').read_text())
    config.update(reference_config=cnn,reference_config_sha256=hashlib.sha256(canonical_json(cnn)).hexdigest(),
        cases=[dict(name='cnn',moe=False,bucket=b,chunk_frames=16) for b in (512,768)])
    cfg=STUDY/'cnn-system-config-001.json';atomic_json(cfg,config,replace=False)
    snapshot=freeze(ROOT,RECIPE,cfg,ROOT/'.gozero/snapshots')
    prerequisites=[STUDY/'cpu-ce-qualification-001.json',STUDY/'source-budget-lineage-001.json',
        ROOT/'research/studies/strong19_moe_temporal/system-review-001.json']
    operators=[STUDY/name for name in ('PROTOCOL.md','prepare_system.py','execute_system.py','audit_qualification.py','review_system.py')]
    plan=dict(status='prepared',kind='matched_cnn_system_qualification_registration',created=time.time(),
        snapshot=snapshot.name,config_sha256=sha256(snapshot/'resolved_config.json'),
        must_follow_successful_attempt='pod-20261004T143934Z-cd561c20',timeout_seconds=1800,
        output_directory='cnn-system-stage-001',
        prerequisites={str(p.relative_to(ROOT)):sha256(p) for p in prerequisites},
        operators={str(p.relative_to(ROOT)):sha256(p) for p in operators},
        scope='Two fresh CNN updates at each current-corpus sequence bucket. Existing signed-target CE, AdamW, BN-free model and 16-frame chunks. No learning curve or checkpoint payload.')
    output=STUDY/'cnn-system-registration-001.json';atomic_json(output,plan,replace=False)
    print(json.dumps(dict(status='prepared',snapshot=snapshot.name,sha256=sha256(output),budgets=budgets)))


if __name__=='__main__':main()
