"""Recipe-owned strict configuration validation; all scientific choices are explicit."""
import math


def fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(label + ' has missing or unknown fields')


def integer(value, minimum=0, maximum=2**31-1):
    return type(value) is int and minimum <= value <= maximum


def real(value, minimum=0, maximum=float('inf')):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def validate(c):
    fields(c, 'schema_version platform expected_processes expected_devices seed selfplay_turns checkpoint_every log_every actors model learner', 'config')
    if c['schema_version'] != 1 or c['platform'] not in ('cpu', 'tpu'):
        raise ValueError('Unsupported recipe schema or platform')
    for key in ('expected_processes', 'expected_devices', 'selfplay_turns', 'checkpoint_every', 'log_every'):
        if not integer(c[key], 1): raise ValueError('Invalid '+key)
    if not integer(c['seed'], 0, 2**32-3): raise ValueError('Invalid seed')
    if c['platform'] == 'cpu' and c['expected_processes'] != 1: raise ValueError('CPU recipe uses one process')
    n = c['model']; l = c['learner']; a = c['actors']
    fields(n, 'width blocks groups value_hidden dtype', 'model')
    if not all(integer(n[k], 1, 4096) for k in ('width','groups','value_hidden')) or not integer(n['blocks'], 0, 1024) or n['width']%n['groups'] or n['dtype'] not in ('float32','bfloat16'):
        raise ValueError('Invalid model configuration')
    fields(l, 'batch_size replay_capacity warmup_rows learning_rate momentum l2 max_grad_norm augment_symmetries updates_per_turn ownership_weight', 'learner')
    if not real(l['ownership_weight'],0,10): raise ValueError('Invalid ownership coefficient')
    if not all(integer(l[k], 1) for k in ('batch_size', 'replay_capacity', 'warmup_rows', 'updates_per_turn')) or l['warmup_rows']>l['replay_capacity']:
        raise ValueError('Invalid replay or update count')
    if not real(l['learning_rate'], 1e-12, 10) or not real(l['momentum'], 0, 1) or not real(l['l2'], 0, 10) or not real(l['max_grad_norm'], 1e-12) or type(l['augment_symmetries']) is not bool:
        raise ValueError('Invalid optimization configuration')
    # Rust performs complete actor validation. These checks precede Python allocation.
    fields(a, 'size komi scoring history games workers worker_cpus simulations cpuct fpu_reduction max_search_edges max_game_moves dirichlet_alpha dirichlet_fraction temperature_early temperature_late temperature_moves seed actor_offset', 'actors')
    if a['scoring'] not in ('raw_area','pass_alive_area'): raise ValueError('Invalid scoring profile')
    if a['fpu_reduction'] is not None and not real(a['fpu_reduction'],0,2): raise ValueError('Invalid FPU reduction')
    for k in ('size', 'history', 'games', 'workers', 'max_game_moves'):
        if not integer(a[k], 1): raise ValueError('Invalid actor '+k)
    if a['size']>26 or a['history']>64: raise ValueError('Recipe SGF and history limits exceeded')
    if a['actor_offset'] != 0: raise ValueError('Recipe allocates actor offsets by JAX rank')
    local_devices=c['expected_devices']//c['expected_processes']
    if not local_devices or c['expected_devices']%c['expected_processes'] or a['games']%local_devices or l['batch_size']%local_devices:
        raise ValueError('Local inference and learner batches must divide local devices')
    feature_width=a['size']**2*(2*a['history']+4)
    estimate=4*(feature_width+2*a['size']**2+1)*(l['replay_capacity']+a['games']*a['max_game_moves'])
    if estimate>64*2**30: raise ValueError('Replay and unfinished row estimate exceeds 64 GiB per host')
    return c
