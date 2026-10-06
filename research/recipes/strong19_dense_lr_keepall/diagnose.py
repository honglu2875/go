"""Separate conditional lowering from omitted padding, using one executable.

This is a diagnostic, not a replacement acceptance threshold for screen 001.
Only small reports are persisted; full reference states stay in host memory.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from benchmark import publish


def validate(c):
    if c['kind'] != 'joint19_runtime_diagnostic' or c['platform'] != 'tpu':
        raise ValueError('Wrong diagnostic')
    base = c['reference_config']
    if hashlib.sha256(canonical_json(base)).hexdigest() != c['reference_config_sha256']:
        raise ValueError('Scientific configuration changed')
    from train_config import validate as validate_training
    validate_training(base)
    if (c['updates'] != 2 or c['draw']['bucket'] != 512 or c['chunk_frames'] != 8
            or c['variants'] != ['original_dense', 'conditional_dense', 'conditional_skip']
            or c['padding_control_requires_exact_equality'] is not True
            or base['steps'] != 512 or base['learner']['games_per_host'] != 32):
        raise ValueError('Diagnostic scope changed')
    return base


def leaves(state):
    params, opt = state
    for part, values in [('params', params), ('first', opt['first']), ('second', opt['second'])]:
        for name, value in sorted(values.items()):
            yield part, name, value
    yield 'counter', 'step', opt['step']


def state_hash(state):
    import numpy as np
    h = hashlib.sha256()
    for part, name, value in leaves(state):
        h.update(canonical_json([part, name, list(value.shape), str(value.dtype)]))
        h.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    return h.hexdigest()


def compare(reference, actual, c):
    """Report every leaf and aggregate by state part and semantic model group."""
    import numpy as np
    import adamw
    rows, aggregates = [], {}
    for (part, name, x), (other_part, other_name, y) in zip(leaves(reference), leaves(actual), strict=True):
        if (part, name, x.shape, x.dtype) != (other_part, other_name, y.shape, y.dtype):
            raise ValueError('State schema changed')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite state')
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        delta = xf - yf
        raw = dict(elements=x.size, different=int(np.count_nonzero(x != y)),
                   outside_screen001_tolerance=int(np.count_nonzero(
                       np.abs(delta) > c['screen001_state_atol'] + c['screen001_state_rtol'] * np.abs(xf))),
                   squared_error=float(np.sum(delta * delta)), reference_squared=float(np.sum(xf * xf)),
                   actual_squared=float(np.sum(yf * yf)), dot=float(np.sum(xf * yf)),
                   max_abs=float(np.max(np.abs(delta))))
        rows.append(dict(part=part, name=name, **raw))
        for key in (part, part + '/' + adamw.group(name), 'all'):
            total = aggregates.setdefault(key, {k: 0 for k in raw})
            for field, value in raw.items():
                total[field] = max(total[field], value) if field == 'max_abs' else total[field] + value
    def summarize(raw):
        denominator = math.sqrt(raw['reference_squared'] * raw['actual_squared'])
        return dict(**raw, relative_l2=math.sqrt(raw['squared_error'] / max(raw['reference_squared'], 1e-30)),
                    cosine=raw['dot'] / denominator if denominator else None,
                    exactly_equal=raw['different'] == 0)
    return dict(exactly_equal=aggregates['all']['different'] == 0,
                groups={k: summarize(v) for k, v in aggregates.items()},
                leaves=[{**r, **summarize(r)} for r in rows])


def run(args, c, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset, augment
    import adamw
    import joint
    import learner
    base = validate(c)
    rank = jax.process_index()
    if jax.process_count() != 4 or len(jax.devices()) != 16 or any(d.platform != 'tpu' for d in jax.devices()):
        raise ValueError('Unexpected topology')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    def host(tree):
        return jax.tree.map(lambda x: np.array(x.addressable_shards[0].data, copy=True), tree)
    def emit(kind, **values):
        row = dict(kind=kind, time=time.time(), jax_rank=rank, **values)
        with (args.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
        print(json.dumps(row, allow_nan=False), flush=True)
    data = Dataset(base['dataset']['path'], base['dataset']['manifest_sha256'])
    bucket = c['draw']['bucket']
    pool = data.bucket_entries(base['dataset']['buckets'])['expert', bucket]
    rng = np.random.Generator(np.random.PCG64(base['seed'] + 1 + 104729 * rank))
    aug_rng = np.random.Generator(np.random.PCG64(base['seed'] + 400003 + 104729 * rank))
    entries = [pool[int(i)] for i in rng.integers(len(pool), size=32)]
    symmetries = aug_rng.integers(0, 8, len(entries))
    entries_sha = hashlib.sha256(canonical_json(entries)).hexdigest()
    expected = c['draw']['ranks'][rank]
    if entries_sha != expected['local_entries_sha256'] or symmetries.tolist() != expected['local_symmetries']:
        raise ValueError('Registered draw changed')
    local = augment(data.batch(entries, positions=bucket), symmetries)
    positions = int(np.asarray(mh.process_allgather(local['counts'])).sum())
    if positions != c['draw']['positions']:
        raise ValueError('Wrong exposure count')
    # Both masks are data arguments, never closed-over compilation constants.
    local['encoder_counts'] = np.full_like(local['counts'], bucket)
    dense_batch = {k: jax.make_array_from_process_local_data(batched, v) for k, v in local.items()}
    skip_batch = {**dense_batch, 'encoder_counts': dense_batch['counts']}
    jax.block_until_ready((dense_batch, skip_batch))
    report.update(jax_rank=rank, host_rank=int(os.environ['GOZERO_HOST_RANK']),
                  reference_config_sha256=c['reference_config_sha256'],
                  parameter_count=sum(r['elements'] for r in joint.parameter_schema(base['model'], base['value_model'])),
                  draw=dict(bucket=bucket, local_entries_sha256=entries_sha,
                            local_symmetries=symmetries.tolist(), global_positions=positions), cases=[])
    if report['parameter_count'] != 232011540:
        raise ValueError('Model changed')
    opt = {k: v for k, v in base['learner'].items() if k not in ('games_per_host', 'augmentation')}
    opt['horizon_steps'] = base['steps']
    def initialize():
        p = joint.initialize(base['seed'], base['model'], base['value_model'])
        return p, adamw.initialize(p)
    initialize = jax.jit(initialize, out_shardings=replicated)
    references, metrics_reference = {}, {}
    initial_hash = None
    conditional_executable = None
    conditional_compile = None
    for name in c['variants']:
        mh.sync_global_devices(name + '-start')
        emit('case_start', variant=name)
        params, state = initialize()
        jax.block_until_ready((params, state))
        observed_hash = state_hash(host((params, state)))
        if initial_hash is None:
            initial_hash = observed_hash
        if initial_hash != observed_hash:
            raise ValueError('Initial state changed')
        batch = skip_batch if name == 'conditional_skip' else dense_batch
        if name == 'conditional_skip':
            executable = conditional_executable
            compile_record = {**conditional_compile, 'reused_executable': True, 'compile_seconds': 0.}
        else:
            update = learner.step(base['model'], opt, value_weight=base['training']['value_weight'],
                                  chunk_frames=c['chunk_frames'], mesh=mesh, skip_padding=name != 'original_dense')
            start = time.perf_counter()
            lowered = jax.jit(update, donate_argnums=(0, 1)).lower(params, state, batch)
            executable = lowered.compile()
            seconds = time.perf_counter() - start
            memory = executable.memory_analysis()
            sizes = {k: int(getattr(memory, k)) for k in (
                'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            peak = sizes['argument_size_in_bytes'] + sizes['output_size_in_bytes'] + sizes['temp_size_in_bytes'] - sizes['alias_size_in_bytes']
            # JAX with x64 disabled truncates int64 collectives. Gather a ceiling
            # in KiB as int32 and convert back on the host; never gather raw bytes.
            ceiling_kib = np.asarray((peak + 1023) // 1024, np.int32)
            all_peak = [int(v) * 1024 for v in np.asarray(mh.process_allgather(ceiling_kib)).reshape(-1)]
            if max(all_peak) > c['compiled_memory_limit_bytes']:
                raise ValueError('Collective compiled-memory limit exceeded')
            compile_record = dict(compile_seconds=seconds, memory_bytes=sizes, estimated_peak_bytes=peak,
                                  all_rank_peak_upper_bytes=all_peak, reused_executable=False,
                                  hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest())
            if name == 'conditional_dense':
                conditional_executable, conditional_compile = executable, compile_record
        case = dict(variant=name, initial_state_sha256=observed_hash, **compile_record, updates=[])
        emit('compiled', variant=name, **compile_record)
        mh.sync_global_devices(name + '-compiled')
        for update_index in range(1, c['updates'] + 1):
            start = time.perf_counter()
            params, state, metrics = executable(params, state, batch)
            jax.block_until_ready((params, state, metrics))
            seconds = time.perf_counter() - start
            scalars = {k: float(v) for k, v in host(metrics).items()}
            if not all(math.isfinite(v) for v in scalars.values()) or scalars['accepted'] != 1:
                raise ValueError('Nonfinite/rejected update')
            saved = host((params, state))
            row = dict(update=update_index, seconds=seconds, metrics=scalars, comparisons={})
            for reference_name in ('original_dense', 'conditional_dense'):
                if reference_name == name:
                    break
                comparison = compare(references[reference_name, update_index], saved, c)
                metric_diff = {k: scalars[k] - v for k, v in metrics_reference[reference_name, update_index].items()}
                comparison.update(metric_differences=metric_diff, metrics_exactly_equal=all(v == 0 for v in metric_diff.values()))
                publish(args.output / f'{name}-vs-{reference_name}-update-{update_index}.json', comparison)
                row['comparisons'][reference_name] = {k: comparison[k] for k in ('exactly_equal', 'metrics_exactly_equal', 'groups')}
            if name != 'conditional_skip':
                references[name, update_index] = saved
                metrics_reference[name, update_index] = scalars
            case['updates'].append(row)
            emit('update', variant=name, update=update_index, seconds=seconds,
                 equality={k: dict(state=v['exactly_equal'], metrics=v['metrics_exactly_equal']) for k, v in row['comparisons'].items()})
            del saved
        publish(args.output / f'{name}.json', case)
        report['cases'].append(case)
        del params, state, metrics
        if name == 'original_dense':
            del executable, lowered, update
            jax.clear_caches()
        gc.collect()
    controls = report['cases'][-1]['updates']
    exact = all(r['comparisons']['conditional_dense']['exactly_equal'] and
                r['comparisons']['conditional_dense']['metrics_exactly_equal'] for r in controls)
    report.update(status='passed' if exact else 'failed', padding_control_exact=exact,
                  optimization_qualified=False, previous_screen_status='failed')
    verify(SOURCE)
    mh.sync_global_devices('runtime-diagnostic-complete')
    if not exact:
        raise ValueError('Skipping changes results within the same executable')


def entry(args):
    c = read_json(args.config)
    validate(c)
    verify(SOURCE)
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')):
        raise ValueError('Unfrozen config')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / 'resolved_config.json', c)
    report = dict(kind=c['kind'], snapshot_id=SOURCE.name, status='running', started=time.time(),
                  optimization_qualified=False, claims_learning_or_strength=False)
    import jax
    initialized = False
    try:
        jax.distributed.initialize(initialization_timeout=90)
        initialized = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished'] = time.time()
        publish(args.output / 'result.json', report)
        if initialized:
            jax.distributed.shutdown()
