"""Exercise the complete joint source optimizer and exact fresh continuation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import read_json,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    sha=checkpoints.sha256
    if args.output.exists():raise FileExistsError(args.output)
    if sha(args.plan)!=args.plan_sha256:raise ValueError('Plan changed')
    plan=read_json(args.plan)
    if sha(Path(__file__))!=plan['operator_sha256']:raise ValueError('Operator changed')
    audit=ROOT/plan['recovery_audit']['path']
    if sha(audit)!=plan['recovery_audit']['sha256']:raise ValueError('Recovery audit changed')
    spec=importlib.util.spec_from_file_location('source_recovery',audit);harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    snapshot=ROOT/'.gozero/snapshots'/plan['snapshot_id'];manifest=verify(snapshot);c=read_json(snapshot/'resolved_config.json')
    if (c['training']['purpose']!='qualification' or c['training']['optimizer']!='muon_source_runtime'
            or c['steps']!=11 or c['dataset']['manifest_sha256']!=plan['dataset_sha256']):
        raise ValueError('Wrong source optimizer execution fixture')
    folder=ROOT/plan['run_directory']
    if folder.exists():raise FileExistsError(folder)
    started=time.monotonic();result=dict(kind='joint_source_muon_neural_continuation_qualification',
        created=time.time(),status='running',plan_sha256=args.plan_sha256,snapshot=snapshot.name)
    try:
        full=harness.run_stage(args.python,snapshot,folder/'full')
        prefix=harness.run_stage(args.python,snapshot,folder/'prefix',stop=6)
        resumed=harness.run_stage(args.python,snapshot,folder/'resumed',resume=Path(prefix['latest_checkpoint']['path']))
        for record in (full,prefix,resumed):
            if record['optimizer_family']!='source_muon_aux_adam_lookahead':raise ValueError('Wrong optimizer family')
        result['recovery']=harness.audit_model(folder,snapshot,full,prefix,resumed)
        if full['latest_checkpoint']['manifest_sha256']!=resumed['latest_checkpoint']['manifest_sha256']:
            raise ValueError('Complete checkpoint bytes differ after continuation')
        sys.path.insert(0,str(snapshot/manifest['recipe']))
        import optimizer_io
        import source_runtime
        import numpy as np
        def state(record):
            cp=record['latest_checkpoint']
            return checkpoints.read(Path(cp['path']),expected_manifest_sha256=cp['manifest_sha256'])[:2]
        boundary,arrays=state(prefix);final,final_arrays=state(full)
        if int(arrays['counter'])!=2 or boundary['source_runtime']['batch_in_subepoch']!=2:
            raise ValueError('Restart must occur immediately before Lookahead sync')
        if not any(not np.array_equal(arrays[k],arrays['s_'+k[2:]]) for k in arrays if k.startswith('p_')):
            raise ValueError('Restart did not preserve distinct fast/slow weights')
        if any(not np.array_equal(final_arrays[k],final_arrays['s_'+k[2:]]) for k in final_arrays if k.startswith('p_')):
            raise ValueError('Final epoch did not flush slow weights')
        rows=harness.rows(folder/'full/artifacts/metrics.jsonl')
        transitions={key:[r['turn'] for r in rows if r[key]] for key in
            ('lookahead_synchronized','source_norm_snapshot','source_epoch_flush','source_subepoch_entry','source_epoch_entry')}
        expected=dict(lookahead_synchronized=[3,7,10],source_norm_snapshot=[3,6,10],
            source_epoch_flush=[7,11],source_subepoch_entry=[1,5,8],source_epoch_entry=[1,8])
        if transitions!=expected:raise ValueError('Neural update transition cadence differs')
        config=c['learner']['source_runtime']
        source_runtime.validate_state(final['source_runtime'],config,c['steps'],positions=final['counters']['expert_positions'])
        # Replay schedule state from logged positions and pre-update norms,
        # without using the saved final state as a continuation starting point.
        # Original-source AST checks separately qualify this controller's math.
        clock=source_runtime.initialize(config,c['steps'],final['source_runtime']['baseline'])
        for row in rows:
            clock,entry=source_runtime.before_step(clock,config,c['steps'])
            for metric,value in [('normal_learning_rate',clock['settings']['rates']['normal']),
                    ('normal_weight_decay',clock['settings']['decays']['normal']),
                    ('source_sum_gradient_clip_cap',clock['settings']['source_sum_gradient_clip_cap'])]:
                if row[metric]!=float(np.float32(value)):raise ValueError('Applied device setting disagrees with schedule replay')
            captured=dict(input=row['source_pre_norm_input'],normal=row['source_pre_norm_normal']) if row['source_norm_snapshot'] else None
            clock,exit=source_runtime.after_step(clock,config,c['steps'],positions=int(row['positions']),pre_update_norms=captured)
            if clock['samples']!=row['source_samples'] or clock['refresh_count']!=row['source_refresh_count']:
                raise ValueError('Logged source counters disagree with replay')
        if clock!=final['source_runtime']:raise ValueError('Final host schedule differs from record replay')
        if final['source_runtime']['samples']!=config['initial_samples']+sum(int(r['positions']) for r in rows):
            raise ValueError('Source sample progress differs from actual exposures')
        schema=[{key:r[key] for key in ('path','shape','dtype')} for r in final['model_schema']]
        p,s=optimizer_io.restore(final['optimizer_metadata'],final_arrays,schema=schema,
            configuration_sha256=final['config_sha256'],source_sha256=snapshot.name,lookahead_config=config['lookahead'])
        if not np.array_equal(s['hyperparameters'],source_runtime.vector(final['source_runtime']['settings'])):
            raise ValueError('Saved host/device source settings disagree')
        result.update(status='passed',transitions=transitions,prefix_counter=int(arrays['counter']),
            final_counter=int(final_arrays['counter']),source_runtime=final['source_runtime'],
            preserved_fast_slow_boundary=True,final_epoch_flush_exact=True,all_applied_settings_replayed=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result.update(seconds=time.monotonic()-started,
            scope='Complete small CNN policy/value training on real19 histories;11 updates versus fresh6+5, full fast/slow/moment/scalar and host schedule state, all RNGs/counters/diagnostics. Explicit execution-only batch/schedule fixture; no scientific19 result or full-size TPU qualification.')
        with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        args.output.chmod(0o444)
        print(json.dumps(dict(status=result['status'],seconds=result['seconds'],sha256=sha(args.output))),flush=True)


if __name__=='__main__':main()
