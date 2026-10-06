"""Explicit joint harness configuration; scientific registration is separate."""
import math

import joint
import learner
import policy_config


def validate(c):
    base_fields=set('schema_version kind platform expected_processes expected_devices seed model dataset steps checkpoint_every eval_every log_every learner evaluation'.split())
    fields=base_fields|{'training','value_model'}
    optional={'checkpoint_temporary','checkpoint_temporary_uncompressed','checkpoint_minimum_free_bytes','checkpoint_disk','batch_replay'}
    if not fields<=set(c) or set(c)-fields-optional or c['kind']!='fixed_joint_learning':
        raise ValueError('Unknown joint training fields')
    if 'checkpoint_minimum_free_bytes' in c and (not c.get('checkpoint_temporary')
            or type(c['checkpoint_minimum_free_bytes']) is not int
            or c['checkpoint_minimum_free_bytes'] < 32*(1<<30)):
        raise ValueError('RAM checkpoints require an explicit safe space reserve')
    # Preserve the qualified common loader, batching and checkpoint contract.
    base={k:v for k,v in c.items() if k in base_fields|optional and k not in ('checkpoint_minimum_free_bytes','checkpoint_disk','batch_replay')}
    base['kind']='fixed_policy_learning';policy_config.validate(base)
    if 'checkpoint_disk' in c:
        from pathlib import Path
        d=c['checkpoint_disk']
        if (set(d)!={'peer','root','keep','floor_bytes','peer_floor_bytes'}
                or d['peer'] not in (None,1,2,3) or not Path(d['root']).is_absolute()
                or type(d['keep']) is not int or d['keep']<2
                or any(type(d[k]) is not int or d[k]<1<<30 for k in ('floor_bytes','peer_floor_bytes'))
                or any(k in c for k in ('checkpoint_temporary','checkpoint_archive'))
                or (c['platform']=='tpu' and d['peer'] is None)):
            raise ValueError('Invalid disk checkpoint contract')
    elif c['training']['purpose']=='learning':
        raise ValueError('Recovered learning runs require durable full-state checkpoints')
    import batch_replay
    batch_replay.validate(c)
    joint.validate_value_config(c['value_model'])
    t=c['training']
    if set(t)-{'skip_padding'}!={'purpose','optimizer','value_weight','path','chunk_frames','value_objective'}:
        raise ValueError('Explicit joint objective and runtime path required')
    if type(t.get('skip_padding',False)) is not bool:raise ValueError('skip_padding must be boolean')
    if t.get('skip_padding',False) and c['model']['architecture']!='causal_visual_policy':
        raise ValueError('Only transformer padding skipping is qualified')
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
