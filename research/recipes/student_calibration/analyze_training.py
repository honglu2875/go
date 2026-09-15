#!/usr/bin/env python3
"""Audit both fixed board-input training arms and their registered prediction gate."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import read as checkpoint_read,sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify

REGISTRATION='ee1e35a16ea0830034101a5203df2c338b4429b890716405401276c8c98f9cbd'


def require(value,message):
    if not value:raise ValueError(message)


def audit_arm(root,protocol,mode):
    arm=protocol['arms'][mode];descriptor_path=SOURCE/arm['descriptor'];descriptor=read_json(descriptor_path)
    trained=validate(root,descriptor);source=trained['snapshot'];config=trained['config']
    require(source.name==arm['snapshot_id'] and sha256(source/'resolved_config.json')==arm['config_sha256'],'Training arm source/config differs')
    result_path=artifact(root,descriptor['training_result_path']);attempt=result_path.parents[2]
    pod=read_json(attempt/'result.json')
    require(pod['status']=='passed' and pod['snapshot_id']==source.name and pod['start_unix_time']>protocol['registered_unix'] and pod['elapsed_seconds']<protocol['maximum_training_seconds_per_arm'],'Attempt failed, preceded registration or exceeded budget')
    results=[];ranks=[];digests=[];manifests=[]
    for host in range(4):
        artifacts=attempt/f'rank-{host}/artifacts';result=read_json(artifacts/'result.json')
        cp=artifacts/'checkpoints'/f'turn-{protocol["steps"]:09d}';group=read_json(cp.with_suffix('.group.json'))
        rank=result['jax_rank'];state,arrays,_=checkpoint_read(cp,expected_manifest_sha256=group['rank_manifests'][rank],array_prefix='p_')
        require(result['status']=='passed' and result['training_complete'] and result['kind']=='board_causal_expert_behavior_distillation' and result['snapshot_id']==source.name and result['turn']==protocol['steps'] and result['world_size']==4 and result['board_mode']==mode,'Rank did not complete its declared model condition')
        require(state['snapshot_id']==source.name and state['config_sha256']==arm['config_sha256']==group['config_sha256'] and state['turn']==group['turn']==protocol['steps'] and state['jax_rank']==rank and state['world_size']==group['world_size']==4 and state['dataset_manifest_sha256']==protocol['dataset_manifest_sha256'],'Checkpoint scientific identity differs')
        require(sha256(cp/'arrays.npz')==group['replicated_arrays_sha256'] and sha256(artifacts/'model_export.npz')==result['model_export_sha256']==descriptor['model_export_sha256'],'Replicated checkpoint/export bytes differ')
        require(set(arrays)==set(trained['arrays']) and all(a.dtype==trained['arrays'][k].dtype and a.shape==trained['arrays'][k].shape and a.tobytes()==trained['arrays'][k].tobytes() for k,a in arrays.items()),'Rank export parameters differ from committed evaluation model')
        for key in ('updates','expert_token_exposures','behavior_token_exposures'):
            require(state['counters'][key]==result['counters'][key],'Checkpoint work counters differ')
        results.append(result);ranks.append(rank);digests.append(sha256(cp.with_suffix('.group.json')));manifests.append(sha256(artifacts/'result.json'))
    require(set(ranks)==set(range(4)) and len(set(digests))==1,'Rank coverage or group manifest differs')
    primary=results[0]
    for result in results:
        for key in ('initial_parameter_elements_sha256','parameter_count','initial_validation','validation_history','test'):
            require(result[key]==primary[key],'Replicated model or prediction metrics differ')
        for key in ('updates','expert_token_exposures','behavior_token_exposures'):
            require(result['counters'][key]==primary['counters'][key],'Replicated global exposure counter differs')
    require(primary['validation_history'][-1]['turn']==protocol['steps'],'Fixed final validation checkpoint missing')
    return {'mode':mode,'attempt':attempt.name,'snapshot':source.name,'descriptor_sha256':sha256(descriptor_path),
        'model_export_sha256':descriptor['model_export_sha256'],'pod_result_sha256':sha256(attempt/'result.json'),
        'rank_results_sha256':manifests,'checkpoint_group_sha256':digests[0],
        'initial_parameter_elements_sha256':primary['initial_parameter_elements_sha256'],'parameter_count':primary['parameter_count'],
        'global_exposures':{k:primary['counters'][k] for k in ('updates','expert_token_exposures','behavior_token_exposures')},
        'initial_validation':primary['initial_validation'],'final_validation':primary['validation_history'][-1]['validation'],'test':primary['test'],
        'attempt_seconds':pod['elapsed_seconds'],'recorded_attempt_chip_hours':pod['reserved_chip_hours'],
        'maximum_learner_seconds':max(r['counters']['learner_seconds'] for r in results),
        'maximum_segment_seconds':max(r['elapsed_segment_seconds'] for r in results)},config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(SOURCE);root=args.workspace_root.resolve();path=artifact(root,'research/studies/board_state_distillation/pilot_spec.json')
    require(sha256(path)==REGISTRATION,'Study registration differs');protocol=read_json(path)
    rows={};configs={}
    for mode in ('empty','exact'):rows[mode],configs[mode]=audit_arm(root,protocol,mode)
    expected=copy.deepcopy(configs['empty']);expected['model']['board_mode']='exact'
    require(expected==configs['exact'],'Training intervention changed other configuration')
    for name in ('model.py','train.py','config.py'):
        require(sha256(root/'.gozero/snapshots'/rows['empty']['snapshot']/'research/recipes/board_state_distillation'/name)==sha256(root/'.gozero/snapshots'/rows['exact']['snapshot']/'research/recipes/board_state_distillation'/name),'Scientific implementation changed between arms')
    for key in ('parameter_count','initial_parameter_elements_sha256','global_exposures'):
        require(rows['empty'][key]==rows['exact'][key],'Initialization, parameter tree or data exposure differs')
    a=rows['empty']['final_validation'];b=rows['exact']['final_validation'];gate=protocol['primary_prediction_criterion']
    metrics={'expert_validation_kl_ratio':b['expert_kl']/a['expert_kl'],'expert_validation_illegal_probability_ratio':b['illegal_probability']/a['illegal_probability'],'behavior_validation_nll_increase':b['behavior_loss']-a['behavior_loss']}
    passed=metrics['expert_validation_kl_ratio']<=gate['expert_validation_kl_ratio_at_most'] and metrics['expert_validation_illegal_probability_ratio']<=gate['expert_validation_illegal_probability_ratio_at_most'] and metrics['behavior_validation_nll_increase']<=gate['behavior_validation_nll_increase_at_most']
    result={'schema_version':1,'kind':'board_input_training_audit','status':'passed','analysis_snapshot':SOURCE.name,'protocol_sha256':REGISTRATION,
        'arms':rows,'comparison':metrics,'registered_prediction_criterion_met':passed,
        'recorded_pilot_attempt_chip_hours':sum(r['recorded_attempt_chip_hours'] for r in rows.values()),
        'global_exposure_note':'Counters are already replicated global totals; do not sum them across hosts.',
        'external_benchmark_required':True,'claims_strength_improvement':False,'claims_rl_sample_efficiency':False,'claims_mfu':False,'limitations':protocol['limitations']}
    verify(SOURCE);require(sha256(path)==REGISTRATION,'Registration changed during audit');args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:f.write(canonical_json(result))
    args.output.chmod(0o444);print(sha256(args.output))


if __name__=='__main__':main()
