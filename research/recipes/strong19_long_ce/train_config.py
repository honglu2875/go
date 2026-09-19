"""Explicit joint harness configuration; scientific registration is separate."""
import math

import joint
import learner
import policy_config


def validate(c):
    base_fields=set('schema_version kind platform expected_processes expected_devices seed model dataset steps checkpoint_every eval_every log_every learner evaluation'.split())
    fields=base_fields|{'training','value_model'}
    optional={'checkpoint_temporary','checkpoint_temporary_uncompressed'}
    if not fields<=set(c) or set(c)-fields-optional or c['kind']!='fixed_joint_learning':
        raise ValueError('Unknown joint training fields')
    # Preserve the qualified common loader, batching and checkpoint contract.
    base={k:v for k,v in c.items() if k in base_fields|optional}
    base['kind']='fixed_policy_learning';policy_config.validate(base)
    joint.validate_value_config(c['value_model'])
    t=c['training']
    if set(t)!={'purpose','optimizer','value_weight','path','chunk_frames','value_objective'}:
        raise ValueError('Explicit joint objective and runtime path required')
    if t['purpose'] not in ('qualification','learning') or t['optimizer']!='adamw':
        raise ValueError('Unsupported training purpose or optimizer')
    if (t['value_objective']!='signed_target_cross_entropy' or t['path']!='bounded'
            or c['model']['architecture'] not in ('causal_visual_policy','katago_nested_policy')):
        raise ValueError('The paired study requires bounded CNN/transformer signed-target cross-entropy')
    if type(t['value_weight']) not in (int,float) or not math.isfinite(t['value_weight']) or t['value_weight']<=0:
        raise ValueError('Joint training requires an explicit positive value weight')
    if t['path'] not in ('bounded','materialized') or type(t['chunk_frames']) is not int or not 1<=t['chunk_frames']<=512:
        raise ValueError('Invalid joint memory path')
    if c['evaluation']['run_test'] or not c['evaluation'].get('training_probe_games'):
        raise ValueError('Selection requires closed test targets and an explicit training probe')
    if c['model']['architecture']=='causal_visual_policy':
        import compact
        compact.validate(c['model'],t['chunk_frames'])
    opt={k:v for k,v in c['learner'].items() if k not in ('games_per_host','augmentation')}
    learner.validate_optimizer({**opt,'horizon_steps':c['steps']})
    return c
