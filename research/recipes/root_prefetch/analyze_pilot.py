#!/usr/bin/env python3
"""Analyze the registered extended Gumbel versus PUCT study."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def training(root, attempt, expected_source):
    import numpy as np
    directory = root / 'runs' / attempt
    pod = read_json(directory / 'result.json')
    if pod['status'] != 'passed' or pod['snapshot_id'] != expected_source:
        raise ValueError('Training attempt identity or status differs')
    source = root / '.gozero/snapshots' / expected_source
    manifest = verify(source)
    config = read_json(source / 'resolved_config.json')
    ranks = [read_json(directory / f'rank-{host}/artifacts/result.json') for host in range(4)]
    if any(r['status'] != 'passed' or r['snapshot_id'] != expected_source or r['turn'] != config['selfplay_turns'] for r in ranks):
        raise ValueError('Incomplete training')
    if len({r['model_export_sha256'] for r in ranks}) != 1 or len({r['counters']['updates'] for r in ranks}) != 1:
        raise ValueError('Replicated learner differs')
    supports = []; peaks = []; pass_mass = []; outcomes = []
    for host, result in enumerate(ranks):
        artifacts = directory / f'rank-{host}/artifacts'
        if sha256(artifacts / 'model_export.npz') != result['model_export_sha256']:
            raise ValueError('Export identity differs')
        checkpoint = artifacts / 'checkpoints' / f"turn-{result['turn']:09d}"
        if sha256(checkpoint / 'manifest.json') != result['latest_checkpoint']['manifest_sha256']:
            raise ValueError('Checkpoint manifest differs')
        saved = read_json(checkpoint / 'manifest.json')
        if sha256(checkpoint / 'arrays.npz') != saved['files']['arrays.npz']['sha256']:
            raise ValueError('Replay integrity differs')
        with np.load(checkpoint / 'arrays.npz', allow_pickle=False) as arrays:
            pi = arrays['replay_pi']; meta = arrays['replay_meta']
            if not np.all(meta[:, 3] == config['actors']['simulations']):
                raise ValueError('Replay search budget differs')
            supports.append(np.count_nonzero(pi, axis=1)); peaks.append(np.max(pi, axis=1))
            pass_mass.append(pi[:, -1]); outcomes.append(arrays['replay_z'])
    support = np.concatenate(supports); peak = np.concatenate(peaks)
    return {
        'attempt': attempt, 'snapshot_id': expected_source, 'pod_result_sha256': sha256(directory / 'result.json'),
        'rank_result_sha256': [sha256(directory / f'rank-{host}/artifacts/result.json') for host in range(4)],
        'model_export_sha256': ranks[0]['model_export_sha256'],
        'model_implementation_sha256': sha256(source / manifest['recipe'] / 'model.py'),
        'trainer_implementation_sha256': sha256(source / manifest['recipe'] / 'train.py'),
        'global': {k: sum(r['counters'][k] for r in ranks) for k in ('real_moves', 'completed_games', 'truncated_games', 'eligible_rows', 'active_neural_evaluations', 'neural_slots')},
        'global_updates': ranks[0]['counters']['updates'], 'max_training_seconds': max(r['elapsed_segment_seconds'] for r in ranks),
        'max_inference_seconds': max(r['counters']['inference_seconds'] for r in ranks),
        'max_native_seconds': max(r['counters']['native_seconds'] for r in ranks),
        'host_average_cpu_cores': [r['process_cpu_segment_seconds'] / r['elapsed_segment_seconds'] for r in ranks],
        'recorded_attempt_chip_hours': pod['reserved_chip_hours'], 'last_metrics': ranks[0]['last_metrics'],
        'final_replay_diagnostics': {'rows': len(support), 'mean_policy_support': float(support.mean()),
            'mean_largest_policy_mass': float(peak.mean()), 'mean_pass_policy_mass': float(np.concatenate(pass_mass).mean()),
            'mean_side_to_move_outcome': float(np.concatenate(outcomes).mean()),
            'scope': 'Final replay windows from different self-play distributions; not held-out accuracy or strength.'},
    }, config


REGISTRATIONS={
    'ad4452db13420501430d67f9f5a36e7d77662587eb1ef7374556695f694b61ec':('eval/gumbel_search',True),
    '0f76baaefa1cba51150f09a0144b13ded02d0f9e7439f2c26141f93da86d6493':('eval/gumbel_replication_28',False),
}
KATA_WEIGHTS='a1298ce1adc1dad7bd868ca962b2384cc8388ed373a00e6bae1114fa6f9e2d61'
KATA_BINARY='1ae1ed2108caa025bba853634b7f1aad3a29baa0ba27b4a298759ca6b172d104'


def game_checks(games):
    for game in games:
        if game['checked_positions']!=len(game['moves']):raise ValueError('Board checks incomplete')
        if game['status']=='completed':
            if not game['score_margin_agrees_with_katago']:raise ValueError('Completed score differs')
            expected=0.5 if game['score']=='0' else float(game['score'][0]==game['candidate_color'])
            if game['candidate_points']!=expected:raise ValueError('Reported outcome differs from score')
        elif game.get('candidate_points') is not None:raise ValueError('Incomplete game has assigned outcome')
        if game.get('integrity_failure'):raise ValueError('Engine integrity failure')


def absolute(root,path,model_hash,arm,protocol,evaluation_directory='eval/gumbel_search'):
    panel=read_json(path);source=root/'.gozero/snapshots'/panel['snapshot_id'];verify(source)
    panel_spec=source/evaluation_directory/f'{arm}_panel.json'
    if sha256(panel_spec)!=panel['spec_sha256']:raise ValueError('Absolute panel identity differs')
    declared=read_json(panel_spec)['matches']
    if len(panel['matches'])!=4 or {m['id'] for m in panel['matches']}!={m['id'] for m in declared}:
        raise ValueError('Registered absolute pairs differ')
    games=[];profiles=[]
    for item in panel['matches']:
        child_path=path.parent/item['id']/'result.json'
        if sha256(child_path)!=item['result_sha256']:raise ValueError('Absolute child identity differs')
        child=read_json(child_path)
        if child['candidate']['model_export_sha256']!=model_hash or len(child['games'])!=2:
            raise ValueError('Absolute model or color pair differs')
        spec_path=source/next(m['spec'] for m in declared if m['id']==item['id'])
        if sha256(spec_path)!=child['spec_sha256']:raise ValueError('Absolute frozen spec differs')
        spec=read_json(spec_path);p=protocol['absolute_metric']
        required={'candidate_simulations_excluding_root':p['candidate_simulations'],
                  'candidate_cpuct':0. if arm=='candidate' else 1.5,'size':9,'komi':7.5,
                  'max_game_moves':p['max_game_moves'],'game_timeout_seconds':p['game_timeout_seconds']}
        if any(spec[k]!=v for k,v in required.items()) or child['candidate_scoring_profile']!='pass_alive_area':
            raise ValueError('Absolute budgets or rules differ')
        if child['katago_weights_sha256']!=KATA_WEIGHTS or child['katago_binary_sha256']!=KATA_BINARY:
            raise ValueError('Absolute KataGo artifacts differ')
        if sha256(source/spec['katago_config'])!=child['base_katago_config_sha256']:
            raise ValueError('Absolute KataGo configuration differs')
        if {g['candidate_color'] for g in child['games']}!={'B','W'} or any(g['opening']!=spec['openings'][0] for g in child['games']):
            raise ValueError('Absolute opening or colors differ')
        profiles.append((spec['katago_max_visits'],json.dumps(spec['openings'])))
        games.extend(child['games'])
    expected=sorted((visits,json.dumps([opening])) for visits in (1,16) for opening in ([],['C3','G7']))
    if sorted(profiles)!=expected:raise ValueError('Absolute visits or openings differ')
    game_checks(games)
    return {'path':str(path),'sha256':sha256(path),'status':panel['status'],'scheduled':8,
            'completed':sum(g['status']=='completed' for g in games),'truncated':sum(g['status']=='truncated' for g in games),
            'failed':sum(g['status']=='failed' for g in games),'wins':sum(g.get('candidate_points')==1 for g in games),
            'losses':sum(g.get('candidate_points')==0 for g in games),'checked_boards':sum(g['checked_positions'] for g in games),
            'elapsed_seconds':panel['finished_unix']-panel['started_unix']}


def relative(root,path,arms,protocol,evaluation_directory='eval/gumbel_search'):
    panel=read_json(path);source=root/'.gozero/snapshots'/panel['snapshot_id'];verify(source)
    spec_path=source/evaluation_directory/'selfmatch_panel.json';spec=read_json(spec_path)
    if sha256(spec_path)!=panel['spec_sha256']:raise ValueError('Relative panel spec differs')
    p=protocol['primary_metric']
    for key in ('openings','candidate_simulations','opponent_simulations','candidate_cpuct','opponent_cpuct','size','komi','max_game_moves','game_timeout_seconds'):
        if spec[key]!=p[key]:raise ValueError('Relative registered '+key+' differs')
    for side,arm in [('candidate','candidate'),('opponent','control')]:
        if panel['models'][side]['model_export_sha256']!=arms[arm]['model_export_sha256']:
            raise ValueError('Relative checkpoint differs')
        if sha256(source/spec[side])!=panel[side+'_descriptor_sha256']:
            raise ValueError('Relative descriptor identity differs')
    if panel['referee_binary_sha256']!=KATA_BINARY or panel['referee_weights_sha256']!=KATA_WEIGHTS:
        raise ValueError('Relative referee artifacts differ')
    if sha256(source/spec['referee_config'])!=panel['referee_config_sha256']:raise ValueError('Relative referee config differs')
    games=panel['games'];seen=set();pairs={}
    for g in games:
        identity=(g['pair'],g['candidate_color'])
        if identity in seen or not 0<=g['pair']<32 or g['candidate_color'] not in ('B','W'):
            raise ValueError('Relative game identity differs')
        seen.add(identity)
        if g['opening']!=p['openings'][g['pair']]:raise ValueError('Relative opening differs')
        child=path.parent/f"pair-{g['pair']:03d}-{g['candidate_color']}"/'result.json'
        if read_json(child)!=g:raise ValueError('Relative child record differs')
        for move in g['moves']:
            if not move['opening'] and move['search']['simulations']!=16:raise ValueError('Relative actual search work differs')
        pairs.setdefault(g['pair'],[]).append(g)
    game_checks(games)
    complete=[sum(g['candidate_points'] for g in pair)/2 for pair in pairs.values()
              if len(pair)==2 and all(g['status']=='completed' for g in pair)]
    score=sum(complete)/len(complete) if complete else None
    radius=math.sqrt(math.log(40)/(2*len(complete))) if complete else None
    interval=[max(0.,score-radius),min(1.,score+radius)] if complete else None
    qualified=len(games)==64 and len(complete)==32 and panel['status']=='passed'
    promising=qualified and score>=0.65 and interval[0]>0.5
    return {'path':str(path),'sha256':sha256(path),'status':panel['status'],'scheduled':64,'recorded':len(games),
            'complete_pairs':len(complete),'paired_score':score,'paired_hoeffding_95_interval':interval,
            'wins':sum(g.get('candidate_points')==1 for g in games),'losses':sum(g.get('candidate_points')==0 for g in games),
            'draws':sum(g.get('candidate_points')==0.5 for g in games),'truncated':sum(g['status']=='truncated' for g in games),
            'failed':sum(g['status']=='failed' for g in games),'checked_boards':sum(g['checked_positions'] for g in games),
            'qualified':qualified,'registered_promising_flag':promising,
            'elapsed_seconds':panel['finished_unix']-panel['started_unix'],
            'uncertainty_scope':'Conservative bounded-score interval across opening pairs. Incomplete pairs invalidate qualification; no Elo or independent-seed claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    for arm in ('control','candidate'):
        p.add_argument('--'+arm+'-attempt',required=True);p.add_argument('--'+arm+'-panel',type=Path,required=True)
    p.add_argument('--relative-panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(ROOT);root=args.workspace_root.resolve();protocol=read_json(args.protocol)
    report={'schema_version':1,'kind':'gumbel_extended_pilot_analysis','analysis_snapshot':ROOT.name,'status':'failed',
            'protocol_sha256':sha256(args.protocol),'arms':{},'absolute':{},'claims_strength_improvement':False,
            'claims_sample_efficiency_improvement':False,'claims_mfu':False,'production_promotion':False}
    try:
        if sha256(args.protocol) not in REGISTRATIONS:raise ValueError('Registration changed')
        evaluation_directory,discovery=REGISTRATIONS[sha256(args.protocol)]
        report['phase']='discovery' if discovery else 'independent_seed_replication'
        configs=[]
        for arm in ('control','candidate'):
            result,c=training(root,getattr(args,arm+'_attempt'),protocol[arm+'_snapshot'])
            report['arms'][arm]=result;configs.append(c)
            if (result['global']['real_moves']!=protocol['budget']['global_real_moves_per_arm'] or c['selfplay_turns']!=32768
                    or c['seed']!=protocol['budget']['seed'] or c['actors']['seed']!=protocol['budget']['seed']):
                raise ValueError('Training budget differs')
            report['absolute'][arm]=absolute(root,getattr(args,arm+'_panel'),result['model_export_sha256'],arm,protocol,evaluation_directory)
        changes={'cpuct':(1.5,0.),'fpu_reduction':(0.2,None),'dirichlet_fraction':(0.25,0.),
                 'temperature_early':(1.,0.),'temperature_moves':(30,0),
                 'gumbel':(None,{'max_considered_actions':16,'value_scale':0.1,'maxvisit_init':50.,'rescale_values':True,'gumbel_scale':1.})}
        for key,expected in changes.items():
            for c,value in zip(configs,expected):
                if c['actors'].pop(key)!=value:raise ValueError('Registered search setting differs: '+key)
        if configs[0]!=configs[1]:raise ValueError('Training differs outside registered search bundle')
        a,b=(report['arms'][x] for x in ('control','candidate'))
        for key in ('model_implementation_sha256','trainer_implementation_sha256'):
            if a[key]!=b[key]:raise ValueError('Training implementation differs')
        report['relative']=relative(root,args.relative_panel,report['arms'],protocol,evaluation_directory)
        total_eval=sum(x['elapsed_seconds'] for x in report['absolute'].values())+report['relative']['elapsed_seconds']
        report['total_eval_seconds']=total_eval
        report['eval_budget_respected']=total_eval<=protocol['budget']['maximum_total_eval_seconds']
        if not report['eval_budget_respected']:report['relative']['registered_promising_flag']=False
        report['eligible_rows_ratio']=b['global']['eligible_rows']/a['global']['eligible_rows']
        report['truncated_games_ratio']=b['global']['truncated_games']/a['global']['truncated_games'] if a['global']['truncated_games'] else None
        report['charged_training_attempt_chip_hours']=a['recorded_attempt_chip_hours']+b['recorded_attempt_chip_hours']
        if discovery:
            regression=root/'runs/qualification/gumbel-resource-preflight-ad4452db/control-regression-semantic.json'
            report['control_at8192_regression']={'path':str(regression),'sha256':sha256(regression),'result':read_json(regression)}
            if report['control_at8192_regression']['result']['status']!='passed':raise ValueError('PUCT regression check failed')
        if report['relative']['registered_promising_flag']:
            report['decision']=('Promising exploratory search bundle under the registered panel; independent seeds and larger anchors required.'
                if discovery else 'The independent seed supports the registered relative advantage over this fixed PUCT control; stronger absolute anchors and fair search tuning remain required.')
        else:
            report['decision']='No promotion: the registered complete-panel promising criterion was not met.'
        if not discovery:
            report['replication_supported']=report['relative']['registered_promising_flag']
        report['limitations']=protocol['limitations']+[
            'Control absolute evaluation overlapped candidate training on separate physical cores; timings are diagnostics, not isolated speed comparisons.',
            'Gumbel targets are dense completed-Q policies, so policy support is not a visited-action count.',
            'Different self-play distributions and warmup times produce different row counts and learner updates at equal real moves.',
            'Unassigned incomplete games invalidate the primary qualification even when completed games favor an arm.',
            'Attempt chip-hours exclude reserved engineering and idle time.']
        report['status']='analyzed';verify(ROOT)
    except Exception as error:report['error']=repr(error);raise
    finally:
        with args.output.open('xb') as stream:stream.write(canonical_json(report))
        print(json.dumps({k:v for k,v in report.items() if k not in ('arms','absolute')}),flush=True)


if __name__=='__main__':main()
