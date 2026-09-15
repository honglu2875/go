#!/usr/bin/env python3
"""Analyze the registered component pilot, retaining incomplete external matches."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json,read_json,verify


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(ROOT);root=args.workspace_root.resolve()
    protocol=read_json(args.protocol);training=read_json(args.training_report);panel=read_json(args.panel);recovery=read_json(args.recovery)
    if sha256(args.protocol)!=training['pilot_spec_sha256']: raise ValueError('Study protocol identity differs')
    report={'schema_version':1,'analysis_snapshot':ROOT.name,'kind':'ownership_component_pilot_analysis',
            'status':'analyzed','claims_strength_improvement':False,'claims_sample_efficiency_improvement':False,'claims_mfu':False,
            'input_sha256':{name:sha256(getattr(args,name)) for name in ('protocol','training_report','panel','recovery')},
            'training':training['arms'],'evaluation_status':panel['status'],'evaluation':panel['summaries'],
            'recovery':recovery,'final_replay_diagnostics':{}}
    for arm,entry in training['arms'].items():
        run=root/'runs'/entry['attempt']
        if sha256(run/'result.json')!=entry['pod_result_sha256'] or entry['snapshot_id']!=protocol[arm+'_snapshot']:
            raise ValueError('Training identity differs')
        verify(root/'.gozero/snapshots'/entry['snapshot_id'])
        supports=[];peaks=[]
        for host in range(4):
            result=read_json(run/f'rank-{host}/artifacts/result.json')
            if result['status']!='passed' or result['model_export_sha256']!=entry['model_export_sha256']:
                raise ValueError('Incomplete or inconsistent replicated training')
            checkpoint=run/f'rank-{host}/artifacts/checkpoints/turn-000008192'
            if sha256(checkpoint/'manifest.json')!=result['latest_checkpoint']['manifest_sha256']:
                raise ValueError('Final checkpoint manifest differs')
            manifest=read_json(checkpoint/'manifest.json')
            if sha256(checkpoint/'arrays.npz')!=manifest['files']['arrays.npz']['sha256']: raise ValueError('Replay integrity differs')
            with np.load(checkpoint/'arrays.npz',allow_pickle=False) as data:
                pi=data['replay_pi'];meta=data['replay_meta']
                if not np.all(meta[:,3]==16): raise ValueError('Replay search budget differs')
                supports.append(np.count_nonzero(pi,axis=1));peaks.append(np.max(pi,axis=1))
        support=np.concatenate(supports);peak=np.concatenate(peaks)
        report['final_replay_diagnostics'][arm]={'rows':len(support),'mean_visited_root_actions':float(np.mean(support)),
            'fraction_with_16_visited_root_actions':float(np.mean(support==16)),'mean_largest_visit_policy_mass':float(np.mean(peak)),
            'scope':'Last replay windows only, from different self-play distributions. This is a search-allocation diagnostic, not a held-out strength or value-accuracy metric.'}
    games=[]
    for entry in panel['matches']:
        path=args.panel.parent/entry['id']/'result.json'
        if sha256(path)!=entry['result_sha256']: raise ValueError('Child match identity differs')
        child=read_json(path);arm=entry['group'].split('-')[0]
        if child['candidate']['model_export_sha256']!=training['arms'][arm]['model_export_sha256']:
            raise ValueError('Evaluation candidate differs from trained model')
        for game in child['games']:
            if game['checked_positions']!=len(game['moves']): raise ValueError('Not every played board was checked')
            if game['status']=='completed' and not game['score_margin_agrees_with_katago']:
                raise ValueError('Completed game adjudication differs')
        games.extend(child['games'])
    report['observed_games']={'scheduled':len(games),'completed':sum(g['status']=='completed' for g in games),
        'move_limit_truncations':sum(g['status']=='truncated' for g in games),'failures':sum(g['status']=='failed' for g in games),
        'candidate_wins':sum(g.get('candidate_points')==1 for g in games),'checked_boards':sum(g['checked_positions'] for g in games)}
    report['decision']='Do not promote ownership supervision from this pilot. Completed games are at the zero-win floor and incomplete pairs invalidate the planned strength screen. Neither superiority nor equivalence is established.'
    report['limitations']=['One initialization and two opening pairs per opponent level.','Outcomes of move-limit games are unknown and are not treated as draws or losses.',
        'Training losses are measured on each arm\'s evolving replay distribution.','Ownership is one KataGo component; the control is not a full KataGo reproduction.',
        'Recovery overlapped CPU evaluation on separate physical cores; recovery timings are not a systems comparison. No faults were injected.']
    report['followup']='Test FPU/search allocation at 16 simulations and larger independent-game batches; add a stronger learning control and a calibrated opponent ladder before scaling. Confirm successful candidates with new seeds and larger panels.'
    verify(ROOT)
    with args.output.open('xb') as f:f.write(canonical_json(report))
    print(json.dumps({k:v for k,v in report.items() if k in ('status','decision','observed_games','final_replay_diagnostics')}),flush=True)


if __name__=='__main__':main()
