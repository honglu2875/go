#!/usr/bin/env python3
"""Audit the registered first causal-MCTS games against pinned real KataGo."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.evaluation_inputs import verify as verify_inputs
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify
from analyze_prefetch_learning import audit_games
from match import summarize

REGISTRATION='fd7bf662d0e524e5d86a7fed3a1c9fe53c5711a7fda88591cb7331e58911fb67'


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(SOURCE);root=args.artifacts_root.resolve()
    protocol_path=artifact(root,'research/studies/causal_mcts/spec.json')
    require(sha256(protocol_path)==REGISTRATION,'Study registration differs');protocol=read_json(protocol_path)
    source=artifact(root,'.gozero/snapshots/'+protocol['operator_snapshot']);verify(source);verify_inputs(source,protocol['input_closure'])
    panel_spec=read_json(source/'eval/causal_student/panel.json');output=artifact(root,protocol['output']);panel=read_json(output/'result.json')
    require(panel['snapshot_id']==source.name and panel['spec_sha256']==sha256(source/'eval/causal_student/panel.json'),'Panel identity differs')
    require(panel['status']=='passed' and protocol['registered_unix']<panel['started_unix']<=panel['finished_unix'] and panel['finished_unix']-panel['started_unix']<protocol['maximum_seconds'],'Panel failed, preceded registration or exceeded budget')
    require(len(panel['matches'])==len(panel_spec['matches']) and {x['id'] for x in panel['matches']}=={x['id'] for x in panel_spec['matches']},'Panel coverage differs')
    groups={};files={};all_games=[]
    for declaration in panel_spec['matches']:
        entry=next(x for x in panel['matches'] if x['id']==declaration['id']);directory=output/declaration['id'];c=read_json(source/declaration['spec'])
        result=read_json(directory/'result.json');descriptor=read_json(source/c['candidate']);trained=validate(root,descriptor)
        receipt_path=artifact(root,'.gozero/native/'+c['inference_native_snapshot']+'/receipt.json');native=read_json(receipt_path)
        inference=read_json(source/c['causal_inference']);weights=read_json(source/c['katago_weights']);kata=read_json(source/'eval/katago_build.json')
        require(entry['returncode']==0 and not entry['timed_out'] and entry['status']=='passed' and result['status']=='passed' and sha256(directory/'result.json')==entry['result_sha256'],'Child attempt failed or changed')
        require(result['snapshot_id']==source.name and result['spec_sha256']==sha256(source/declaration['spec']) and read_json(directory/'resolved_spec.json')==c,'Child specification differs')
        require(result['games']==entry['games'] and result['candidate']==descriptor and result['candidate_sha256']==sha256(source/c['candidate']),'Child model or game records differ')
        require(result['candidate_adapter']=='eval/causal_gtp.py' and result['candidate_inference_config_sha256']==sha256(source/c['causal_inference']),'Inference adapter/configuration differs')
        require(result['native_receipt_sha256']==protocol['native_receipt_sha256']==sha256(receipt_path) and native['snapshot_id']==c['inference_native_snapshot']==result['inference_native_snapshot'] and native['binary_sha256']==c['inference_native_binary_sha256']==sha256(receipt_path.parent/native['filename']),'Native identity differs')
        require(result['katago_binary_sha256']==kata['binary_sha256']==sha256(artifact(root,kata['binary_path'])) and result['katago_weights_sha256']==weights['sha256']==sha256(artifact(root,weights['path'])),'External KataGo artifact differs')
        require(result['katago_weights_descriptor_sha256']==sha256(source/c['katago_weights']) and result['base_katago_config_sha256']==sha256(source/c['katago_config']),'External configuration differs')
        audit_games(result['games'],directory,c,direct=False)
        require(result['summary']==entry['summary']==summarize(result['games']),'Game summary differs')
        for game in result['games']:
            folder=directory/f'pair-{game["pair"]:03d}-{game["candidate_color"]}'
            require(game['status']=='completed' and game['candidate_version']=='gozero-causal-mcts-history-v1','Causal game did not complete')
            stderr=folder/'candidate/stderr.log'
            ready=[json.loads(line) for line in stderr.read_text().splitlines() if line.startswith('{') and json.loads(line).get('kind')=='engine_ready']
            require(len(ready)==1,'Engine readiness identity absent or duplicated');ready=ready[0]
            require(ready['adapter']=='causal_full_prefill' and ready['training_snapshot']==descriptor['training_snapshot'] and ready['weights_sha256']==descriptor['model_export_sha256'] and ready['model_code_sha256']==trained['model_code_sha256'] and ready['native_snapshot']==native['snapshot_id'] and ready['native_sha256']==native['binary_sha256'] and ready['inference_config_sha256']==sha256(source/c['causal_inference']),'Actual loaded engine identity differs')
            require(ready['simulations_excluding_root']==c['candidate_simulations_excluding_root']==inference['simulations'] and ready['maximum_game_moves']==c['max_game_moves']==inference['max_game_moves'] and ready['maximum_game_moves']+ready['simulations_excluding_root']<ready['context_tokens'],'Actual inference context or search bound differs')
            for path in folder.rglob('*'):
                if path.is_file():files[str(path.relative_to(output))]=sha256(path)
        groups[declaration['group']]={'summary':result['summary'],'result_sha256':sha256(directory/'result.json'),
            'checked_boards':sum(g['checked_positions'] for g in result['games']),
            'search_totals':{k:sum(g['candidate_search_totals'][k] for g in result['games']) for k in ('simulations','neural_evaluations','terminal_evaluations')}}
        all_games.extend(result['games'])
    require(len(all_games)==protocol['scheduled_games'],'Scheduled game count differs')
    result={'schema_version':1,'kind':'causal_mcts_integration_audit','status':'passed','operator_snapshot':SOURCE.name,
        'protocol_sha256':sha256(protocol_path),'panel_result_sha256':sha256(output/'result.json'),
        'primary_integration_criterion_met':True,'scheduled_games':len(all_games),'completed_games':len(all_games),
        'checked_boards':sum(g['checked_positions'] for g in all_games),'checked_final_scores':len(all_games),
        'elapsed_seconds':panel['finished_unix']-panel['started_unix'],'groups':groups,'raw_files':dict(sorted(files.items())),
        'claims_strength_improvement':False,'claims_sample_efficiency':False,'claims_throughput':False,'limitations':protocol['limitations']}
    verify(SOURCE);require(sha256(protocol_path)==REGISTRATION,'Study registration changed during audit');args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(canonical_json(result))
    args.output.chmod(0o444);print(sha256(args.output))


if __name__=='__main__':main()
