"""Freeze the qualified connector-only comparison and its immutable operators."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
RECIPE=ROOT/'research/recipes/strong19_attention_pool'
sys.path[:0]=[str(RECIPE),str(ROOT/'packages/gozero/src')]
from gozero.snapshots import canonical_json,freeze,verify


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def publish(path,value):
    with path.open('xb') as f:f.write(canonical_json(value))
    path.chmod(0o444)


def main():
    from train_config import validate
    cpu=read(STUDY/'cpu-qualification-002.json')
    assert cpu['status']=='passed'
    for name,digest in cpu['source_sha256'].items():assert sha(RECIPE/name)==digest,name
    assert cpu['operator_sha256']==sha(STUDY/'qualify_cpu.py')
    recovery=read(STUDY/'harness-qualification-001.json')
    assert recovery['status']=='passed'
    decode=read(STUDY/'decode-qualification-001.json')
    assert decode['status']=='passed'
    for name,digest in decode['source_sha256'].items():assert sha(RECIPE/name)==digest,name
    stage_audit=read(STUDY/'stage-audit-qualification-001.json')
    assert stage_audit['status']=='passed' and stage_audit['steps']==2
    assert stage_audit['operator_sha256']==sha(STUDY/'audit_run.py')
    old=ROOT/'research/studies/strong19_long_pair'
    prior=read(old/'registration-002.json')
    assert sha(old/'draw-replay-001.json')==prior['draw_sha256']
    # Explicit execution accounting for both opening and cached appends. The
    # deployed append computes both queries over its entire declared extent,
    # including the masked score for a query's future key.
    budgets={}
    for arm in ('flat','attention'):
        c=validate(read(STUDY/(arm+'-config-001.json')));m=c['model']
        import encoder
        d,w,L=m['width'],m['encoder_width'],m['layers'];kv=m['kv_heads']*(d//m['heads'])
        matrix_weights=2*d*d+2*d*kv+3*d*m['mlp_hidden']
        policy=2*d*362+2*361*w+2*d*64+2*361*w*64+2*361*64
        value=2*d*256+2*256*3
        decodes={}
        for positions in (1,2,64,128,256,512,768,1536):
            queries=1 if positions==1 else 2
            extent=2*positions-1
            parts=dict(encoder=encoder.flops(m,19),temporal_dense=2*queries*L*matrix_weights,
                temporal_attention=4*queries*extent*d*L,policy=policy,value=value)
            decodes[str(positions)]={**parts,'total_matrix_flops':sum(parts.values()),
                'attention_positions':positions,'executed_key_extent':extent,'queries':queries}
        q=cpu['budgets'][arm]
        budgets[arm]=dict(parameters=q['parameters'],encoder_parameters=q['encoder_parameters'],
            connector_parameters=q['connector_parameters'],decodes=decodes)
    delta=dict(parameters=budgets['attention']['parameters']-budgets['flat']['parameters'],
        parameter_ratio=budgets['attention']['parameters']/budgets['flat']['parameters'],
        decode_ratios={k:budgets['attention']['decodes'][k]['total_matrix_flops']/v['total_matrix_flops']
            for k,v in budgets['flat']['decodes'].items()})
    assert abs(delta['parameter_ratio']-1)<.001
    assert all(abs(v-1)<.001 for v in delta['decode_ratios'].values())
    publish(STUDY/'budget-001.json',dict(status='passed',arms=budgets,delta=delta,
        convention='Multiply-add=2; complete encoder, connector, executed temporal attention, policy and value. Elementwise/cache costs excluded and documented.',
        supersedes='CPU qualification provisional causal nonzero-attention count; this budget counts all executed masked attention entries.'))
    snapshots={}
    for arm in ('flat','attention'):
        snapshot=freeze(ROOT,RECIPE,STUDY/(arm+'-config-001.json'),ROOT/'.gozero/snapshots')
        verify(snapshot)
        snapshots[arm]=dict(snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),
            parameters=budgets[arm]['parameters'])
    # Verify both arms use identical source payloads, differing only in config.
    a,b=[read(ROOT/'.gozero/snapshots'/snapshots[k]['snapshot']/'manifest.json')['files'] for k in ('flat','attention')]
    assert {k:v for k,v in a.items() if k!='resolved_config.json'}=={k:v for k,v in b.items() if k!='resolved_config.json'}
    files=('execute_pair.py','execute_run.py','audit_run.py','replicate.py','observation.py','prepare.py','PROTOCOL.md')
    operators={str((STUDY/name).relative_to(ROOT)):sha(STUDY/name) for name in files}
    prerequisite_paths=[STUDY/'cpu-qualification-002.json',STUDY/'harness-qualification-001.json',
        STUDY/'decode-qualification-001.json',STUDY/'stage-audit-qualification-001.json',STUDY/'budget-001.json',
        ROOT/'research/studies/strong19_throughput/VARIED_001.json']
    prerequisites={str(p.relative_to(ROOT)):sha(p) for p in prerequisite_paths}
    r=dict(status='prepared',kind='joint19_attention_pool_pair',created=time.time(),arms=snapshots,
        steps=256,schedule_steps=512,order=['attention','flat'],operators=operators,prerequisites=prerequisites,
        draw_reference=str((old/'draw-replay-001.json').relative_to(ROOT)),draw_sha256=prior['draw_sha256'],
        validation_population_sha256=prior['validation_population_sha256'],probe_population_sha256=prior['probe_population_sha256'],
        audit_python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),
        paired_positions=sum(x['positions'] for x in read(old/'draw-replay-001.json')['draws'][:256]),
        checkpoint_bytes_per_arm={k:12*v['parameters']+4 for k,v in budgets.items()},
        primary='Fixed endpoint policy KL with family KL and last-three-validation support; all curves retained.',
        horizon_policy='256-update midpoint per arm of the same immutable 512-step schedule. No adaptive endpoint or extra queue. Continue only in a separately recorded stage.')
    publish(STUDY/'registration-001.json',r)
    print(json.dumps(dict(status='prepared',arms=snapshots,positions=r['paired_positions'],budget_delta=delta)),flush=True)


if __name__=='__main__':main()
