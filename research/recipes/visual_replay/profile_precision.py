"""Matched full-model gradients isolate mixed-dtype matmul precision overhead."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset


def run(c, output, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import model
    if jax.process_count() != c['expected_processes'] or len(jax.devices()) != c['expected_devices']:
        raise ValueError('Precision comparison topology differs')
    mesh = Mesh(np.asarray(jax.devices()), ('data',)); replicated = NamedSharding(mesh, P()); batched = NamedSharding(mesh, P('data'))
    net = c['model']; model.validate_model(net)
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    all_entries = data.bucket_entries(c['dataset']['buckets']); bucket = c['dataset']['selected_bucket']
    random = np.random.default_rng(c['seed'] + jax.process_index())
    entries = [all_entries[role, bucket][int(i)] for role in ('expert', 'behavior')
               for i in random.integers(len(all_entries[role, bucket]), size=c['dataset']['games_per_role'])]
    raw = data.batch(entries, positions=bucket)
    batch = {k: jax.device_put(v, replicated) if k == 'loss_weights' else jax.make_array_from_process_local_data(batched, v)
             for k, v in raw.items()}
    specs = {k: P() if k == 'loss_weights' else P('data') for k in batch}
    params = jax.jit(lambda: model.initialize(c['seed'], net), out_shardings=replicated)()
    jax.block_until_ready(params)
    report.update(parameter_count=sum(s['elements'] for s in model.parameter_schema(net)),
        global_sequences=len(entries) * jax.process_count(), bucket=bucket, jax_rank=jax.process_index(),
        local_entries_sha256=hashlib.sha256(canonical_json(entries)).hexdigest(),
        dtype=net['dtype'], gradient_input_observation='JAX transposes bf16 operands with a float32 output cotangent. Precision controls these mixed backward dots.')
    compiled = {}; comparison = {}; report['compilation'] = {}
    for precision in ('highest', 'default'):
        selected = {**net, 'linear_precision': precision}
        def loss(p, b):
            return model.losses(p, b, selected, axis_name='data')[0]
        objective = jax.shard_map(loss, mesh=mesh, in_specs=(P(), specs), out_specs=P(), check_vma=False)
        started = time.perf_counter()
        lowered = jax.jit(jax.value_and_grad(objective)).lower(params, batch)
        fn = lowered.compile(); compiled[precision] = fn
        report['compilation'][precision] = {'seconds': time.perf_counter() - started,
            'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
            'cost_analysis': fn.cost_analysis()}
        print(json.dumps({'kind': 'precision_compiled', 'precision': precision, 'rank': jax.process_index()}), flush=True)
        comparison[precision] = fn(params, batch); jax.block_until_ready(comparison[precision])
    def numerical(a, b):
        dot = sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        norm_a = sum(jnp.sum(x * x) for x in jax.tree.leaves(a))
        norm_b = sum(jnp.sum(x * x) for x in jax.tree.leaves(b))
        error = sum(jnp.sum((x - y) ** 2) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(x - y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))]))
        return {'cosine': dot / jnp.sqrt(norm_a * norm_b), 'relative_l2': jnp.sqrt(error / norm_a), 'maximum_absolute_error': maximum}
    def host_scalar(tree):
        return jax.tree.map(lambda a: float(np.asarray(a.addressable_shards[0].data)), tree)
    report['gradient_comparison'] = host_scalar(jax.jit(numerical)(comparison['highest'][1], comparison['default'][1]))
    report['losses'] = {k: host_scalar(v[0]) for k, v in comparison.items()}
    del comparison
    report['timing'] = []
    for index, order in enumerate(c['orders']):
        times = {}
        for precision in order:
            mh.sync_global_devices(f'precision-{index}-{precision}')
            before = time.perf_counter(); result = compiled[precision](params, batch)
            jax.block_until_ready(result); times[precision] = time.perf_counter() - before
            del result
        report['timing'].append(times)
        print(json.dumps({'kind': 'precision_times', 'rank': jax.process_index(), 'repetition': index, **times}), flush=True)
    report['device_memory_stats'] = [d.memory_stats() for d in jax.local_devices()]
    report['status'] = 'passed'; verify(SOURCE); mh.sync_global_devices('precision-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices seed model dataset orders'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_precision_profile'
            or c['platform'] != 'tpu' or not 1 <= len(c['orders']) <= 7
            or any(sorted(order) != ['default', 'highest'] for order in c['orders'])
            or c['model']['dtype'] != 'bfloat16'
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))
            or args.resume is not None or args.stop_after_turn is not None):
        raise ValueError('Invalid frozen precision profile')
    os.environ['JAX_PLATFORMS'] = 'tpu'; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
              'claims_learning_equivalence': False, 'claims_mfu': False}
    started = time.time(); distributed = False
    try:
        import jax
        jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(c, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        (args.output / 'result.json').write_bytes(canonical_json(report))
        if distributed:
            jax.distributed.shutdown()
