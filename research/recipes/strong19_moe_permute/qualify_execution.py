"""Bounded execution-variant check on the complete sparse learner.

Reference optimizer states stay in host RAM. Only scalar reports are persisted.
This entry point never resumes, saves learned weights, or changes a live run.
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
from qualify_kernels import publish
from diagnose import compare, state_hash


def validate(c):
    import train_config
    base = c['reference_config']
    train_config.validate(base)
    if (c['kind'] != 'moe_execution_qualification' or c['platform'] != 'tpu'
            or hashlib.sha256(canonical_json(base)).hexdigest() != c['reference_config_sha256']
            or c['updates_per_case'] != 2 or c['buckets'] != [512, 768]
            or c['variants'] != ['original8', 'candidate8']
            or c['compiled_memory_limit_bytes'] != 31 * (1 << 30)
            or base['model']['moe'].get('remat_activations', False)
            or base['training']['chunk_frames'] != 8
            or base['learner']['games_per_host'] != 32):
        raise ValueError('Execution qualification scope changed')
    if (set(c['candidate_moe_overrides']) != {'tiling','permutation_vjp'}
            or c['group_relative_l2_limits'] != {'params':1e-4,'first':1e-3,'second':5e-3}
            or c['first_moment_min_cosine'] != .99999
            or c['metric_rtol'] != .005 or c['metric_atol'] != 1e-4
            or base['model']['moe'].get('permutation_vjp',False)):
        raise ValueError('Execution candidate or numerical limits changed')
    candidate={**base,'model':{**base['model'],'moe':{**base['model']['moe'],**c['candidate_moe_overrides']}}}
    train_config.validate(candidate)
    current={'tiling':base['model']['moe']['tiling'],
             'permutation_vjp':base['model']['moe'].get('permutation_vjp',False)}
    if c['candidate_moe_overrides']==current:
        raise ValueError('Candidate is the reference')
    return base


def gate(comparison, metrics, reference, name, c):
    failures = []
    exact_metrics = all(metrics[k] == v for k, v in reference.items())
    for key, value in reference.items():
        if not math.isclose(metrics[key], value,
                            rel_tol=c['metric_rtol'], abs_tol=c['metric_atol']):
            failures.append('metric/' + key)
    for key, values in comparison['groups'].items():
        part = key.split('/')[0]
        if part in c['group_relative_l2_limits']:
            if values['relative_l2'] > c['group_relative_l2_limits'][part]:
                failures.append('relative_l2/' + key)
            if (part == 'first' and values['cosine'] is not None
                    and values['cosine'] < c['first_moment_min_cosine']):
                failures.append('cosine/' + key)
        elif part == 'counter' and not values['exactly_equal']:
            failures.append('counter')
    return dict(status='passed' if not failures else 'failed', failures=failures,
                state_exactly_equal=comparison['exactly_equal'], metrics_exactly_equal=exact_metrics,
                coordinate_gate_passed=comparison['groups']['all']['outside_screen001_tolerance'] == 0)


def run(args, c, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gozero.corpus_sequence_batches import Dataset, augment
    import joint, learner, adamw
    base = validate(c)
    if jax.process_count() != 4 or len(jax.devices()) != 16 or any(d.platform != 'tpu' for d in jax.devices()):
        raise ValueError('Topology changed')
    rank = jax.process_index()
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))

    def host(tree):
        return jax.tree.map(lambda x: np.array(x.addressable_shards[0].data, copy=True), tree)

    def emit(kind, **values):
        row = dict(kind=kind, time=time.time(), jax_rank=rank, **values)
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
        print(json.dumps(row, allow_nan=False), flush=True)

    # At most two 5.052 GB reference states plus one comparison state per host.
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    enough = int(memory['MemAvailable'] > 32 * (1 << 30))
    if not np.asarray(mh.process_allgather(np.asarray(enough, np.int32))).all():
        raise MemoryError('Insufficient host RAM for full optimizer comparisons')
    data = Dataset(base['dataset']['path'], base['dataset']['manifest_sha256'])
    pools = data.bucket_entries(base['dataset']['buckets'])
    rng = np.random.Generator(np.random.PCG64(base['seed'] + 1 + 104729 * rank))
    augmentation = np.random.Generator(np.random.PCG64(base['seed'] + 400003 + 104729 * rank))
    batches = {}
    for draw in c['draws']:
        pool = pools['expert', draw['bucket']]
        entries = [pool[int(i)] for i in rng.integers(len(pool), size=32)]
        symmetries = augmentation.integers(0, 8, len(entries))
        expected = draw['ranks'][rank]
        if (hashlib.sha256(canonical_json(entries)).hexdigest() != expected['local_entries_sha256']
                or symmetries.tolist() != expected['local_symmetries']):
            raise ValueError('Registered draw changed')
        local = augment(data.batch(entries, positions=draw['bucket']), symmetries)
        if int(np.asarray(mh.process_allgather(local['counts'])).sum()) != draw['positions']:
            raise ValueError('Exposure count changed')
        batches[draw['bucket']] = local
    parameters = sum(row['elements'] for row in joint.parameter_schema(base['model'], base['value_model']))
    if parameters != 420991764:
        raise ValueError('Sparse model changed')
    report.update(jax_rank=rank, host_rank=int(os.environ['GOZERO_HOST_RANK']),
                  jax_version=jax.__version__, parameters=parameters, cases=[])
    opt = {k: v for k, v in base['learner'].items() if k not in ('games_per_host', 'augmentation')}
    opt['horizon_steps'] = base['steps']
    all_gates = []
    for bucket in c['buckets']:
        references, reference_metrics = {}, {}
        initial_hash = None
        for name in c['variants']:
            chunk = 8
            overrides = {} if name == 'original8' else c['candidate_moe_overrides']
            model = {**base['model'], 'moe': {**base['model']['moe'], **overrides}}
            mh.sync_global_devices(f'{name}-{bucket}-start')
            emit('case_start', name=name, bucket=bucket, chunk_frames=chunk)
            def initialize():
                p = joint.initialize(base['seed'], model, base['value_model'])
                return p, adamw.initialize(p)
            init = jax.jit(initialize, out_shardings=replicated)
            params, state = init(); jax.block_until_ready((params, state))
            observed_hash = state_hash(host((params, state)))
            if initial_hash is None:
                initial_hash = observed_hash
            if observed_hash != initial_hash:
                raise ValueError('Initial state changed')
            batch = {k: jax.make_array_from_process_local_data(batched, v) for k, v in batches[bucket].items()}
            jax.block_until_ready(batch)
            update = learner.step(model, opt, value_weight=base['training']['value_weight'],
                                  chunk_frames=chunk, mesh=mesh, skip_padding=True)
            started = time.perf_counter()
            lowered = jax.jit(update, donate_argnums=(0, 1)).lower(params, state, batch)
            executable = lowered.compile()
            seconds = time.perf_counter() - started
            memory = executable.memory_analysis()
            sizes = {k: int(getattr(memory, k)) for k in (
                'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            peak = sizes['argument_size_in_bytes'] + sizes['output_size_in_bytes'] + sizes['temp_size_in_bytes'] - sizes['alias_size_in_bytes']
            peaks = [1024 * int(v) for v in np.asarray(mh.process_allgather(np.asarray((peak + 1023) // 1024, np.int32))).reshape(-1)]
            row = dict(name=name, bucket=bucket, chunk_frames=chunk, initial_state_sha256=initial_hash,
                       memory_bytes=sizes, all_rank_peak_upper_bytes=peaks, compile_seconds=seconds,
                       hlo_sha256=hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(), updates=[])
            emit('case_compiled', name=name, bucket=bucket, peak_bytes=max(peaks), compile_seconds=seconds)
            if max(peaks) > c['compiled_memory_limit_bytes']:
                if name == 'original8':
                    raise MemoryError('Reference exceeds the memory guard; no comparisons are possible')
                row['status'] = 'rejected_memory'; report['cases'].append(row)
                publish(args.output / f'{name}-{bucket}.json', row)
                # No candidate is executed beyond the predeclared allocation guard.
                all_gates.append(False)
            else:
                for step in range(1, c['updates_per_case'] + 1):
                    mh.sync_global_devices(f'{name}-{bucket}-step-{step}')
                    started = time.perf_counter()
                    params, state, metrics = executable(params, state, batch)
                    jax.block_until_ready((params, state, metrics))
                    seconds = time.perf_counter() - started
                    scalars = {k: float(v) for k, v in host(metrics).items()}
                    if (scalars['accepted'] != 1 or scalars['moe_dropped_tokens'] != 0
                            or not all(math.isfinite(v) for v in scalars.values())):
                        raise ValueError('Nonfinite/rejected update or dropped token')
                    saved = host((params, state))
                    observed = dict(step=step, seconds=seconds, metrics=scalars)
                    if name == 'original8':
                        references[step], reference_metrics[step] = saved, scalars
                    else:
                        comparison = compare(references[step], saved, c)
                        accepted = gate(comparison, scalars, reference_metrics[step], name, c)
                        comparison.update(gate=accepted, metric_differences={k: scalars[k] - v for k, v in reference_metrics[step].items()})
                        publish(args.output / f'{name}-{bucket}-step-{step}-comparison.json', comparison)
                        observed['comparison'] = dict(gate=accepted, groups=comparison['groups'])
                        all_gates.append(accepted['status'] == 'passed')
                        del comparison
                    row['updates'].append(observed)
                    emit('update', name=name, bucket=bucket, step=step, seconds=seconds,
                         gate=observed.get('comparison', {}).get('gate'))
                    del saved, metrics
                row['status'] = 'passed' if all(r.get('comparison', {}).get('gate', {}).get('status', 'passed') == 'passed' for r in row['updates']) else 'failed_numerics'
                report['cases'].append(row); publish(args.output / f'{name}-{bucket}.json', row)
            del params, state, batch, executable, lowered, update, init
            jax.clear_caches(); gc.collect()
        del references, reference_metrics
        gc.collect()
    report['status'] = 'passed' if all(all_gates) else 'failed'
    verify(SOURCE); mh.sync_global_devices('execution-qualification-complete')
    if report['status'] != 'passed':
        raise ValueError('One or more predeclared memory/numerical gates failed')


def entry(args):
    c = read_json(args.config)
    validate(c); verify(SOURCE)
    if c != read_json(SOURCE / 'resolved_config.json'):
        raise ValueError('Unfrozen configuration')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / 'resolved_config.json', c)
    report = dict(kind=c['kind'], snapshot_id=SOURCE.name, status='running', started=time.time(),
                  claims_learning=False, claims_strength=False,
                  scope='Two repeated real-batch updates per fresh-state variant and bucket; full optimizer comparison, memory guard and measured latency. No learned checkpoint retained.')
    import jax
    initialized = False
    try:
        jax.distributed.initialize(initialization_timeout=90); initialized = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report['finished'] = time.time(); publish(args.output / 'result.json', report)
        if initialized:
            jax.distributed.shutdown()
