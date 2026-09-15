"""Explicit, bounded historical-SGD control configuration."""
import math
import katago


def validate(c):
    fields = {'schema_version', 'kind', 'platform', 'expected_processes', 'expected_devices', 'seed',
              'model', 'dataset', 'learner', 'evaluation', 'sampling', 'profile_decode'}
    if set(c) != fields or c['schema_version'] != 1 or c['kind'] != 'fixed_policy_position_learning':
        raise ValueError('Unknown SGD configuration fields')
    katago.validate(c['model'])
    if c['platform'] not in ('cpu', 'tpu') or c['profile_decode'] is not False:
        raise ValueError('Invalid platform/profile')
    def integer(value, low, high):
        if type(value) is not int or not low <= value <= high: raise ValueError('Invalid integer')
    integer(c['expected_processes'], 1, 4); integer(c['expected_devices'], 1, 16)
    integer(c['seed'], 0, 2**32 - 1000000)
    if c['expected_devices'] % c['expected_processes']: raise ValueError('Uneven topology')
    if set(c['dataset']) != {'path', 'manifest_sha256', 'buckets'}: raise ValueError('Invalid dataset fields')
    l = c['learner']
    if set(l) != {'batch_size', 'updates_per_dispatch', 'momentum', 'nesterov', 'l2_coefficient',
                 'per_sample_lr', 'warmup_per_sample_lr', 'warmup_positions', 'gradient_clipping'}:
        raise ValueError('Invalid SGD fields')
    integer(l['batch_size'], 1, 4096); integer(l['updates_per_dispatch'], 1, 64)
    integer(l['warmup_positions'], 0, 2**30)
    if l['batch_size'] % c['expected_devices']: raise ValueError('Uneven device batch')
    if l['nesterov'] is not True or l['gradient_clipping'] is not None:
        raise ValueError('This recipe implements unclipped Nesterov SGD')
    for key, low, high in [('momentum', 0., 1.), ('l2_coefficient', 0., .1),
                          ('per_sample_lr', 0., .001), ('warmup_per_sample_lr', 0., .001)]:
        if type(l[key]) not in (float, int) or not math.isfinite(l[key]) or not low <= l[key] < high:
            raise ValueError('Invalid ' + key)
    if min(l['per_sample_lr'], l['warmup_per_sample_lr']) <= 0: raise ValueError('Zero learning rate')
    if set(c['sampling']) != {'reference_steps', 'position_exposures', 'shuffle_seed'}:
        raise ValueError('Invalid sampling contract')
    integer(c['sampling']['reference_steps'], 1, 1024)
    integer(c['sampling']['position_exposures'], 1, 2**30)
    integer(c['sampling']['shuffle_seed'], 0, 2**32-1)
    if set(c['evaluation']) != {'games_per_bucket', 'run_test', 'every_reference_turns'} or c['evaluation']['run_test'] is not False:
        raise ValueError('The test split must remain closed')
    integer(c['evaluation']['games_per_bucket'], 1, 10000)
    integer(c['evaluation']['every_reference_turns'], 1, 1024)
    return c
