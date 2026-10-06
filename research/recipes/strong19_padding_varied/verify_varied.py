"""Varied-batch qualification following the isolated compiler-rounding control.

Three paths use identical states and draws. The two conditional paths share
compiled executables and must remain exactly equal. Original-compiler drift is
checked independently at meaningful state-group scale. The failed coordinate
gate from screen 001 is still reported and is never relabeled as passed.
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
from diagnose import compare, state_hash


def validate(c):
    base = c['reference_config']
    from train_config import validate as validate_training
    validate_training(base)
    if (c['kind'] != 'joint19_padding_varied' or c['platform'] != 'tpu'
            or hashlib.sha256(canonical_json(base)).hexdigest() != c['reference_config_sha256']
            or c['draw_turns'] != [3, 4, 9, 13] or c['chunk_frames'] != 8
            or c['variants'] != ['original_dense', 'conditional_dense', 'conditional_skip']
            or not c['padding_control_requires_exact_equality']
            or base['steps'] != 512 or base['learner']['games_per_host'] != 32
            or [d['turn'] for d in c['replay_prefix']] != list(range(1, 14))):
        raise ValueError('Qualification scope changed')
    return base


def original_gate(comparison, metrics, reference_metrics, c):
    failures = []
    for key, actual in metrics.items():
        if not math.isclose(actual, reference_metrics[key], rel_tol=c['original_metric_rtol'], abs_tol=c['original_metric_atol']):
            failures.append('metric/' + key)
    for key, values in comparison['groups'].items():
        part = key.split('/')[0]
        if part in c['original_group_relative_l2_limits']:
            if values['relative_l2'] > c['original_group_relative_l2_limits'][part]:
                failures.append('relative_l2/' + key)
            if part == 'first' and values['cosine'] is not None and values['cosine'] < c['original_first_moment_min_cosine']:
                failures.append('cosine/' + key)
        elif part == 'counter' and not values['exactly_equal']:
            failures.append('counter')
    return dict(status='passed' if not failures else 'failed', failures=failures,
                screen001_coordinate_gate_passed=comparison['groups']['all']['outside_screen001_tolerance'] == 0)


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
    pools = data.bucket_entries(base['dataset']['buckets'])
    rng = np.random.Generator(np.random.PCG64(base['seed'] + 1 + 104729 * rank))
    aug_rng = np.random.Generator(np.random.PCG64(base['seed'] + 400003 + 104729 * rank))
    batches, draws = {}, []
    # Replay metadata draws even for skipped turns so selected batches exactly
    # match the pre-existing immutable registration. No target-driven selection.
    for draw in c['replay_prefix']:
        bucket, turn = draw['bucket'], draw['turn']
        pool = pools['expert', bucket]
        entries = [pool[int(i)] for i in rng.integers(len(pool), size=32)]
        symmetries = aug_rng.integers(0, 8, len(entries))
        entries_sha = hashlib.sha256(canonical_json(entries)).hexdigest()
        expected = draw['ranks'][rank]
        if entries_sha != expected['local_entries_sha256'] or symmetries.tolist() != expected['local_symmetries']:
            raise ValueError('Replay changed')
        if turn not in c['draw_turns']:
            continue
        local = augment(data.batch(entries, positions=bucket), symmetries)
        positions = int(np.asarray(mh.process_allgather(local['counts'])).sum())
        if positions != draw['positions']:
            raise ValueError('Wrong exposure count')
        batches[turn] = local
        draws.append(dict(turn=turn, bucket=bucket, local_entries_sha256=entries_sha,
                          local_symmetries=symmetries.tolist(), global_positions=positions))
    report.update(jax_rank=rank, host_rank=int(os.environ['GOZERO_HOST_RANK']), draws=draws,
                  reference_config_sha256=c['reference_config_sha256'],
                  parameter_count=sum(r['elements'] for r in joint.parameter_schema(base['model'], base['value_model'])), cases=[])
    if report['parameter_count'] != 232011540:
        raise ValueError('Model changed')
    opt = {k: v for k, v in base['learner'].items() if k not in ('games_per_host', 'augmentation')}
    opt['horizon_steps'] = base['steps']
    def initialize():
        p = joint.initialize(base['seed'], base['model'], base['value_model'])
        return p, adamw.initialize(p)
    initialize = jax.jit(initialize, out_shardings=replicated)
    references, metric_references = {}, {}
    initial_hash = None
    conditional_executables, conditional_records = {}, {}
    all_gates = []
    for name in c['variants']:
        mh.sync_global_devices(name + '-start')
        emit('case_start', variant=name)
        params, state = initialize()
        jax.block_until_ready((params, state))
        observed = state_hash(host((params, state)))
        if initial_hash is None:
            initial_hash = observed
        if observed != initial_hash:
            raise ValueError('Initialization differs')
        executables = conditional_executables if name == 'conditional_skip' else {}
        compile_records = conditional_records if name == 'conditional_skip' else {}
        case = dict(variant=name, initial_state_sha256=observed, compilations={}, updates=[])
        for index, draw in enumerate(draws, 1):
            bucket, turn = draw['bucket'], draw['turn']
            local = batches[turn]
            encoder_counts = local['counts'] if name == 'conditional_skip' else np.full_like(local['counts'], bucket)
            batch = {k: jax.make_array_from_process_local_data(batched, v)
                     for k, v in {**local, 'encoder_counts': encoder_counts}.items()}
            jax.block_until_ready(batch)
            if bucket not in executables:
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
                all_peak = [int(v) * 1024 for v in np.asarray(mh.process_allgather(
                    np.asarray((peak + 1023) // 1024, np.int32))).reshape(-1)]
                if max(all_peak) > c['compiled_memory_limit_bytes']:
                    raise ValueError('Collective compiled-memory limit exceeded')
                record = dict(compile_seconds=seconds, memory_bytes=sizes, estimated_peak_bytes=peak,
                              all_rank_peak_upper_bytes=all_peak,
                              hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest())
                executables[bucket], compile_records[bucket] = executable, record
                emit('compiled', variant=name, bucket=bucket, **record)
                del lowered, update, executable
            case['compilations'][str(bucket)] = {**compile_records[bucket], 'reused_from_conditional_dense': name == 'conditional_skip'}
            mh.sync_global_devices(f'{name}-draw-{turn}')
            start = time.perf_counter()
            params, state, metrics = executables[bucket](params, state, batch)
            jax.block_until_ready((params, state, metrics))
            seconds = time.perf_counter() - start
            scalars = {k: float(v) for k, v in host(metrics).items()}
            if not all(math.isfinite(v) for v in scalars.values()) or scalars['accepted'] != 1:
                raise ValueError('Nonfinite/rejected update')
            del batch, metrics
            saved = host((params, state))
            row = dict(update=index, draw_turn=turn, bucket=bucket, seconds=seconds, metrics=scalars, comparisons={})
            for reference_name in ('original_dense', 'conditional_dense'):
                if reference_name == name:
                    break
                comparison = compare(references[reference_name, index], saved, c)
                ref_metrics = metric_references[reference_name, index]
                differences = {k: scalars[k] - v for k, v in ref_metrics.items()}
                comparison.update(metric_differences=differences, metrics_exactly_equal=all(v == 0 for v in differences.values()))
                if reference_name == 'original_dense':
                    gate = original_gate(comparison, scalars, ref_metrics, c)
                else:
                    gate = dict(status='passed' if comparison['exactly_equal'] and comparison['metrics_exactly_equal'] else 'failed')
                comparison['gate'] = gate
                all_gates.append(gate['status'] == 'passed')
                publish(args.output / f'{name}-vs-{reference_name}-update-{index}.json', comparison)
                row['comparisons'][reference_name] = {k: comparison[k] for k in ('gate', 'exactly_equal', 'metrics_exactly_equal', 'groups')}
            if name != 'conditional_skip':
                references[name, index], metric_references[name, index] = saved, scalars
            case['updates'].append(row)
            emit('update', variant=name, update=index, draw_turn=turn, bucket=bucket, seconds=seconds,
                 gates={k: v['gate'] for k, v in row['comparisons'].items()})
            del saved
        publish(args.output / f'{name}.json', case)
        report['cases'].append(case)
        del params, state
        if name == 'conditional_dense':
            conditional_executables, conditional_records = executables, compile_records
        elif name == 'original_dense':
            executables.clear()
            jax.clear_caches()
        gc.collect()
    report.update(status='passed' if all(all_gates) else 'failed', previous_screen_status='failed',
                  scope='Numerical and systems qualification only; no learning-quality, strength or MFU claim.')
    verify(SOURCE)
    mh.sync_global_devices('varied-padding-complete')
    if report['status'] != 'passed':
        raise ValueError('Varied-batch runtime gate failed')


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
                  claims_learning_or_strength=False)
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
