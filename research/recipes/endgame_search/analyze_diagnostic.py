#!/usr/bin/env python3
"""Audit raw searches and reconstruct Gumbel value transformations independently."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(SOURCE/'packages/gozero/src'),str(SOURCE/'eval')]
from gozero.checkpoints import sha256,read as checkpoint_read
from gozero.causal_artifacts import validate as validate_student
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify
from corpus import qualified_cases,require


def transformed(d,rescale):
    legal=[i for i,x in enumerate(d['log_priors'])if x is not None];mass=weighted=0.
    for a in legal:
        if d['visits'][a]:
            p=max(d['priors'][a],float(np.finfo(np.float32).tiny));mass+=p
            weighted+=p*d['value_sums'][a]/d['visits'][a]
    mixed=(d['network_value']+d['root_visits']*(weighted/mass if mass else 0.))/(d['root_visits']+1)
    q=[d['value_sums'][a]/d['visits'][a]if d['visits'][a]else mixed for a in legal]
    low=min(q+([mixed]if len(legal)<82 else []));high=max(q+([mixed]if len(legal)<82 else []))
    largest=max(d['visits']);scale=(50.+largest)*float(np.float32(.1))
    values=[d['log_priors'][a]+scale*((v-low)/max(high-low,1e-8)if rescale else v)for a,v in zip(legal,q)]
    weights=[math.exp(x-max(values))for x in values];total=sum(weights)
    policy=np.zeros(82,np.float32)
    for a,p in zip(legal,weights):policy[a]=p/total
    candidates=[(max(x,-1e9),-a,a)for a,x in zip(legal,values)if d['visits'][a]==largest]
    return {'policy':policy,'chosen':max(candidates)[2],'mixed_value':mixed,'value_range':high-low,
        'values':{a:v for a,v in zip(legal,q)},'transformed_logits':{a:v for a,v in zip(legal,values)}}


def audit_search(case,row):
    d=row['inspection'];leaves=row['leaves'];legal=case['legal'];root=leaves[0]
    require(len(leaves)==d['neural_evaluations']and d['neural_evaluations']+d['terminal_evaluations']==17
        and sum(d['visits'])==d['root_visits']==d['completed_simulations']==16,'Search budget differs')
    require(root['is_root']and root['suffix']==[]and root['value']==d['network_value'],'Root neural evidence differs')
    require([a for a,p in enumerate(d['log_priors'])if p is not None]==legal,'Search legal actions differ')
    logits=root['logits'];maximum=max(logits[a]for a in legal);expected=np.zeros(82,np.float32)
    weights=[math.exp(logits[a]-maximum)for a in legal];mass=sum(weights)
    for a,w in zip(legal,weights):
        require(d['log_priors'][a]==logits[a]-maximum,'Raw legal log prior differs');expected[a]=w/mass
    np.testing.assert_array_equal(expected,np.asarray(d['priors'],np.float32))
    for leaf in leaves:
        require(len(leaf['logits'])==82 and np.isfinite(leaf['logits']).all()and math.isfinite(leaf['value'])
            and -1<=leaf['value']<=1 and all(type(a)is int and 0<=a<=81 for a in leaf['suffix']),'Invalid retained neural leaf')
    for a in range(82):
        n=d['visits'][a]
        if n:require(d['action_values'][a]==float(np.float32(d['value_sums'][a]/n)),'Root child value sum differs')
        else:require(d['action_values'][a]is None,'Unvisited child has a value')
        if a not in legal:require(d['value_sums'][a]is None and d['policy'][a]==0 and n==0,'Illegal action has mass')
    reconstructed=transformed(d,row['rescale_values'])
    error=float(np.max(np.abs(reconstructed['policy']-np.asarray(d['policy'],np.float32))))
    require(error<=2e-7 and reconstructed['chosen']==d['chosen'],'Independent Gumbel reconstruction differs')
    if d['terminal_child_values'][81]is not None:
        value=case['pass_terminal_value'];require(value is not None and -d['terminal_child_values'][81]==value
            and d['action_values'][81]==value and d['value_sums'][81]==value*d['visits'][81],'Terminal-pass backup differs')
    return reconstructed,error


def summarize(rows):
    output={}
    for category in ('after_pass','control'):
        selected=[r for r in rows if r['category']==category];summary={'positions':len(selected)}
        for model in ('student','teacher'):
            counts=Counter();ranges=[];raw=[];pass_priors=[]
            for row in selected:
                on=row[model+'_on'];off=row[model+'_off'];counts['on_passes']+=on['chosen']==81;counts['off_passes']+=off['chosen']==81
                counts['changed_action']+=on['chosen']!=off['chosen'];counts['play_to_pass']+=on['chosen']!=81 and off['chosen']==81
                counts['pass_to_play']+=on['chosen']==81 and off['chosen']!=81;counts['pass_visited_on']+=on['pass_visits']>0
                counts['pass_top16_prior']+=on['pass_prior_rank']<=16
                if row['pass_terminal_value']is not None:
                    label='winning'if row['pass_terminal_value']>0 else'losing'if row['pass_terminal_value']<0 else'draw'
                    counts[label+'_native_pass_positions']+=1
                    counts[label+'_on_passes']+=on['chosen']==81;counts[label+'_off_passes']+=off['chosen']==81
                counts['on_evidence_unscaled_changes_action']+=on['unscaled_choice_using_same_evidence']!=on['chosen']
                counts['on_evidence_unscaled_selects_pass']+=on['unscaled_choice_using_same_evidence']==81
                ranges.append(on['value_range']);raw.append(on['network_value']);pass_priors.append(on['pass_prior'])
            summary[model]={**dict(counts),'value_range_quantiles':np.quantile(ranges,[0,.25,.5,.75,1]).tolist()if ranges else[],
                'network_value_quantiles':np.quantile(raw,[0,.25,.5,.75,1]).tolist()if raw else[],
                'pass_prior_quantiles':np.quantile(pass_priors,[0,.25,.5,.75,1]).tolist()if pass_priors else[]}
        output[category]=summary
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
    root=a.workspace_root.resolve();require(sha256(a.protocol)==a.expected_protocol_sha256,'Registration changed');protocol=read_json(a.protocol)
    execution=artifact(root,'.gozero/snapshots/'+protocol['snapshot']);verify(execution);c=read_json(execution/'resolved_config.json')
    require(sha256(execution/'resolved_config.json')==protocol['config_sha256'],'Configuration changed')
    attempt=artifact(root,a.attempt);result=read_json(attempt/'result.json');require(a.attempt==protocol['output']and result['status']=='passed'
        and result['snapshot']==execution.name and result['phase']==protocol['phase']and result['protocol_sha256']==a.expected_protocol_sha256
        and protocol['registered_unix']<result['started_unix']<result['finished_unix']
        and result['finished_unix']-result['started_unix']<protocol['maximum_seconds'],'Attempt identity, status or budget differs')
    cp=artifact(root,protocol['corpus']);require(sha256(cp)==protocol['corpus_sha256']==result['corpus_sha256'],'Corpus changed');corpus=read_json(cp)
    for path,digest in corpus['files_sha256'].items():require(sha256(artifact(root,path))==digest,'Underlying match evidence changed')
    cases=qualified_cases(corpus,c['qualification_positions'])if result['phase']=='qualify'else corpus['cases'];by_id={r['case_id']:r for r in cases}
    require(len(cases)==protocol['positions']and len(by_id)==len(cases),'Registered population differs')
    require({r['rank']for r in result['workers']}==set(range(4))and all(r['returncode']==0 for r in result['workers']),'Worker coverage differs')
    rows=[];seen=set();max_policy_error=0.;workers=[];leaf_count=0;misses=[];teacher_ids=[]
    for rank in range(4):
        directory=attempt/f'worker-{rank}';path=directory/'result.json';d=read_json(path)
        require(d['status']=='passed'and d['operator_snapshot']==execution.name and d['worker']==rank
            and d['config_sha256']==protocol['config_sha256']and d['corpus_sha256']==protocol['corpus_sha256']
            and d['native_receipt_sha256']==protocol['native_receipt_sha256']
            and d['case_ids']==[r['case_id']for r in cases[rank::4]]and not d['recorded_decision_mismatches'],'Worker identity or assignment differs')
        require(d['candidate_sha256']==sha256(execution/c['candidate'])and d['inference_sha256']==sha256(execution/c['inference']),'Loaded student identity differs')
        native=artifact(root,'.gozero/native/'+c['native_snapshot']+'/receipt.json');require(sha256(native)==d['native_receipt_sha256']
            and read_json(native)==d['native']and sha256(native.parent/d['native']['filename'])==d['native']['binary_sha256'],'Native identity changed')
        require(sha256(directory/'cases.jsonl.gz')==d['cases_sha256'],'Raw case file changed');teacher_ids.append(d['teacher_identity'])
        workers.append({'rank':rank,'result_sha256':sha256(path),'cases_sha256':d['cases_sha256']})
        with gzip.open(directory/'cases.jsonl.gz','rt')as stream:
            for line in stream:
                record=json.loads(line);case=record['case'];key=case['case_id'];require(key in by_id and key not in seen,'Unexpected or duplicate case');seen.add(key)
                require({k:v for k,v in case.items()if k not in ('pass_terminal_value','pass_terminal_white_score')}==by_id[key],'Retained case differs from corpus')
                searches=record['searches'];require(set(searches)=={'student_on','student_off','teacher_on','teacher_off'},'Search arms differ')
                summary={k:case[k]for k in ('case_id','category','panel','ply','game_status','pass_terminal_value','pass_terminal_white_score')}
                for name,search in searches.items():
                    require(search['rescale_values']==name.endswith('_on'),'Search setting differs');transformed_values,error=audit_search(case,search)
                    max_policy_error=max(max_policy_error,error);leaf_count+=len(search['leaves']);stats=search['inspection']
                    same_evidence=transformed(stats,False);prior=stats['priors'][81]
                    summary[name]={'chosen':stats['chosen'],'network_value':stats['network_value'],'root_value':stats['root_value'],
                        'pass_prior':prior,'pass_prior_rank':1+sum((stats['priors'][v],-v)>(prior,-81)for v in case['legal']if v!=81),
                        'pass_policy':stats['policy'][81],'pass_visits':stats['visits'][81],'pass_action_value':stats['action_values'][81],
                        'value_range':transformed_values['value_range'],'unscaled_choice_using_same_evidence':same_evidence['chosen'],
                        'neural_evaluations':stats['neural_evaluations'],'terminal_evaluations':stats['terminal_evaluations']}
                for name in ('student','teacher'):
                    predicted={}
                    for arm in (name+'_on',name+'_off'):
                        for leaf in searches[arm]['leaves']:
                            suffix=tuple(leaf['suffix']);value=(leaf['features_sha256'],leaf['logits'],leaf['value'])
                            require(leaf['reused_prediction']==(suffix in predicted),'Memoization receipt differs')
                            if suffix in predicted:require(predicted[suffix]==value,'Same history changed prediction')
                            predicted[suffix]=value
                old=case['recorded_search'];reproduced=searches['student_on']['inspection']
                same=(reproduced['chosen']==case['recorded_action']and np.float32(reproduced['root_value'])==np.float32(old['root_value'])
                    and reproduced['neural_evaluations']==old['neural_evaluations']and reproduced['terminal_evaluations']==old['terminal_evaluations']
                    and reproduced['completed_simulations']==old['simulations'])
                require(record['recorded_decision_exact']==bool(same),'Recorded reproduction flag differs')
                if not same:misses.append(key)
                rows.append(summary)
    require(seen==set(by_id)and not misses and all(t==teacher_ids[0]for t in teacher_ids),'Population, reproduction or teacher identity differs')
    validate_student(root,read_json(execution/c['candidate']))
    mixture=artifact(root,c['teacher_dataset_manifest']);require(sha256(mixture)==c['teacher_dataset_manifest_sha256'],'Teacher mixture changed')
    teacher=teacher_ids[0];spec=read_json(mixture)['spec']['shards'][c['teacher_shard_index']]
    require(teacher['mixture_component']==spec,'Teacher component differs')
    ts=artifact(root,'.gozero/snapshots/'+spec['training_snapshot']);tm=verify(ts);tc=read_json(ts/'resolved_config.json')
    checkpoint=artifact(root,spec['checkpoint']);require(sha256(checkpoint.with_suffix('.group.json'))==spec['group_sha256'],'Teacher group changed')
    state,arrays,_=checkpoint_read(checkpoint,expected_manifest_sha256=spec['manifest_sha256'],array_prefix='p_')
    elements=hashlib.sha256(b''.join(arrays[f'p_{i:04d}'].tobytes()for i in range(len(arrays)))).hexdigest()
    require(elements==teacher['parameter_elements_sha256']and teacher['training_seed']==tc['seed']
        and teacher['model_code_sha256']==sha256(ts/tm['recipe']/'model.py'),'Teacher model identity changed')
    student_gumbel=read_json(execution/c['inference'])['gumbel'];teacher_gumbel=tc['actors']['gumbel']
    for settings in (student_gumbel,teacher_gumbel):
        require(settings['max_considered_actions']==16 and settings['maxvisit_init']==50.
            and settings['value_scale']==.1 and settings['rescale_values'],'Unexpected value-transform constants')
    report={'schema_version':1,'kind':'endgame_value_rescaling_audit','status':'passed','analysis_snapshot':SOURCE.name,
        'execution_snapshot':execution.name,'protocol_sha256':a.expected_protocol_sha256,'attempt':a.attempt,
        'attempt_result_sha256':sha256(attempt/'result.json'),'corpus_sha256':protocol['corpus_sha256'],'workers':workers,
        'positions':len(rows),'raw_neural_leaf_rows_checked':leaf_count,'recorded_decisions_exact':True,
        'maximum_reconstructed_policy_error':max_policy_error,'teacher_identity':teacher_ids[0],
        'summary':summarize(rows),'positions_summary':sorted(rows,key=lambda r:r['case_id']),
        'claims_go_strength':False,'claims_sample_efficiency':False,'claims_mfu':False,
        'limitations':['Post hoc retained positions, including previous qualification cases; no fresh match outcomes.',
            'Counterfactual pass scores use the native pass-alive scorer. The corpus separately verifies original boards and completed scores against retained real KataGo transcripts.',
            'One explicitly identified CNN component represents part of a four-model teacher mixture.',
            'More passes can concede losses; pass frequency and changed moves are not strength metrics.',
            'Different value transforms can explore different leaves. The report also applies unscaled values to the same recorded on-setting evidence.',
            'No model training, new KataGo games, throughput promotion or production run occurred.']}
    verify(SOURCE);require(sha256(a.protocol)==a.expected_protocol_sha256,'Protocol changed during audit')
    with a.output.open('xb')as f:f.write(canonical_json(report))
    a.output.chmod(0o444);print(sha256(a.output));print(json.dumps({'positions':len(rows),'max_policy_error':max_policy_error,'summary':report['summary']}))


if __name__=='__main__':main()
