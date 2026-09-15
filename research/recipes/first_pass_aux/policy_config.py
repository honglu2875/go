"""Explicit sequential fixed-data policy learning contract."""
import math
from pathlib import Path
import katago
import causal

def validate(c):
    fields=set('schema_version kind platform expected_processes expected_devices seed model dataset steps checkpoint_every eval_every log_every learner evaluation'.split())
    if not fields <= set(c) or set(c)-fields-{'profile_decode', 'checkpoint_archive', 'checkpoint_temporary','draft_reference'} or c['schema_version']!=1 or c['kind']!='fixed_policy_learning':
        raise ValueError('Unknown policy learning fields')
    (causal.validate if c['model']['architecture']=='causal_visual_policy' else katago.validate)(c['model'])
    if type(c.get('profile_decode',False)) is not bool: raise ValueError('Invalid decode profiling flag')
    if 'draft_reference' in c:
        ref=c['draft_reference']
        if set(ref)!={'audit','audit_sha256'} or not Path(ref['audit']).is_absolute() or len(ref['audit_sha256'])!=64 or not c['model'].get('first_pass_aux_weight',0):raise ValueError('Invalid pinned draft reference')
    if c['platform'] not in ('cpu','tpu'): raise ValueError('Unsupported platform')
    if c['platform']=='tpu' and c['model'].get('first_pass_aux_weight',0) and 'draft_reference' not in c:raise ValueError('Select the pinned paired draft reference before any auxiliary TPU launch')
    if 'checkpoint_temporary' in c:
        if (c['checkpoint_temporary'] is not True or c['platform']!='tpu' or 'checkpoint_archive' in c
                or c['checkpoint_every']!=c['steps']):raise ValueError('Invalid temporary final-checkpoint contract')
    if 'checkpoint_archive' in c:
        import re
        a = c['checkpoint_archive']
        if (c['platform'] != 'tpu' or set(a) != {'host', 'reservation'} or type(a['host']) is not int
                or a['host'] not in (1,2,3) or not re.fullmatch(r'\.gozero/checkpoint-reservations/[a-z0-9-]+', a['reservation'])
                or c['checkpoint_every'] != c['steps']):
            raise ValueError('Invalid single-final-checkpoint archive contract')
    def integer(obj,key,lo,hi):
        if type(obj[key]) is not int or not lo<=obj[key]<=hi: raise ValueError('Invalid '+key)
    for key,lo,hi in [('expected_processes',1,4),('expected_devices',1,16),('seed',0,2**32-1000000),
                      ('steps',1,1000000),('checkpoint_every',1,1000000),('eval_every',1,1000000),('log_every',1,1000000)]:
        integer(c,key,lo,hi)
    if c['expected_devices']%c['expected_processes']: raise ValueError('Uneven topology')
    d=c['dataset']
    if set(d)!=set('path manifest_sha256 buckets bucket_probabilities warmup_buckets'.split()) or not Path(d['path']).is_absolute():
        raise ValueError('Explicit pinned dataset required')
    if len(d['manifest_sha256'])!=64 or any(v not in '0123456789abcdef' for v in d['manifest_sha256']):
        raise ValueError('Invalid dataset digest')
    if not d['buckets'] or d['buckets']!=sorted(set(d['buckets'])) or any(type(v) is not int or not 1<=v<=c['model']['max_positions'] for v in d['buckets']):
        raise ValueError('Invalid history buckets')
    if len(d['bucket_probabilities'])!=len(d['buckets']) or any(not math.isfinite(v) or v<=0 for v in d['bucket_probabilities']) or not math.isclose(sum(d['bucket_probabilities']),1.,abs_tol=1e-8):
        raise ValueError('Invalid bucket distribution')
    if any(v not in d['buckets'] for v in d['warmup_buckets']) or len(d['warmup_buckets'])>c['steps']:
        raise ValueError('Invalid compilation warmup draws')
    l=c['learner']
    if set(l)!=set('games_per_host learning_rate end_learning_rate warmup_steps beta1 beta2 epsilon weight_decay max_grad_norm augmentation'.split()):
        raise ValueError('Unknown policy optimizer fields')
    integer(l,'games_per_host',1,128); integer(l,'warmup_steps',0,c['steps'])
    if l['games_per_host']%(c['expected_devices']//c['expected_processes']): raise ValueError('Uneven device batch')
    for key,lo,hi in [('learning_rate',0.,.1),('end_learning_rate',0.,.1),('beta1',0.,1.),('beta2',0.,1.),
                      ('epsilon',0.,1.),('weight_decay',0.,1.),('max_grad_norm',0.,1000.)]:
        if not isinstance(l[key],(float,int)) or not math.isfinite(l[key]) or not lo<=l[key]<=hi:
            raise ValueError('Invalid optimizer '+key)
    if min(l['learning_rate'],l['epsilon'],l['max_grad_norm'])<=0 or max(l['beta1'],l['beta2'])>=1:
        raise ValueError('Invalid optimizer boundary')
    if l['augmentation'] not in ('none','d4'): raise ValueError('Invalid augmentation')
    e=c['evaluation']
    if set(e)!=set('games_per_bucket run_test'.split()) or type(e['run_test']) is not bool:
        raise ValueError('Invalid evaluation contract')
    integer(e,'games_per_bucket',1,10000)
    return c
