"""Post-hoc diagnosis on generated games excluded from both continuation arms."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_artifacts import validate as validate_candidate
from gozero.visual_sequence_batches import Dataset


def entries(data, receipt_path, expected_receipt_sha256, buckets):
    if sha256(receipt_path) != expected_receipt_sha256: raise ValueError('Collection receipt differs')
    receipt = read_json(receipt_path)
    if receipt['status'] != 'passed' or receipt['visual_manifest_sha256'] != sha256(data.directory / 'manifest.json'):
        raise ValueError('Dataset and collection receipt differ')
    holdouts = {g['id']: g for g in receipt['raw_game_index'] if g['terminal'] and not g['train']}
    if len(holdouts) != receipt['collection_holdout_terminal_games']: raise ValueError('Holdout identity collision')
    result = {b: [] for b in buckets}; selected = []
    for shard, arrays in enumerate(data.shards):
        for episode, identity in enumerate(arrays['expert_game_ids']):
            identity = int(identity)
            if identity not in holdouts: continue
            begin, end = map(int, arrays['expert_offsets'][episode:episode + 2])
            if bool(arrays['expert_training_eligible'][episode]) or end - begin != holdouts[identity]['moves']:
                raise ValueError('A collection holdout entered training or changed length')
            bucket = next(b for b in buckets if b >= end - begin)
            result[bucket].append(('expert', shard, episode)); selected.append({'id': identity, 'moves': end - begin, 'bucket': bucket})
    if len({r['id'] for r in selected}) != len(selected): raise ValueError('Duplicate selected holdout')
    return result, selected, len(holdouts)


def run(c, output, report):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import metrics
    import model
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if (jax.process_count() != 4 or len(jax.devices()) != 16 or any(x.platform != 'tpu' for x in jax.devices())):
        raise ValueError('Collection evaluation requires the registered four-host topology')
    root = SOURCE.parents[2]; data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'], rank=host, world=4)
    groups, selected, total = entries(data, artifact(root, c['collection_receipt']), c['collection_receipt_sha256'], c['buckets'])
    candidate_path = artifact(SOURCE, c['candidate'])
    if sha256(candidate_path) != c['candidate_sha256']: raise ValueError('Checkpoint differs')
    checked = validate_candidate(root, read_json(candidate_path)); net = checked['config']['model']
    if checked['rules'] != data.manifest['rules'] or checked['model_code_sha256'] != sha256(Path(__file__).with_name('model.py')):
        raise ValueError('Model implementation or rules differ')
    definition = jax.tree.structure(jax.eval_shape(lambda: model.initialize(0, net)))
    params = definition.unflatten([checked['arrays'][f'p_{i:04d}'] for i in range(len(checked['model_schema']))]); del checked
    mesh = Mesh(np.asarray(jax.local_devices()), ('data',)); rep = NamedSharding(mesh, P()); bat = NamedSharding(mesh, P('data'))
    params = jax.device_put(params, rep); raw = {}; batches = []; compilation = {}
    for bucket in c['buckets']:
        all_entries = groups[bucket]; width = c['sequences_per_host']
        padded = [None] * width
        template = data.batch(padded, positions=bucket)
        specs = {k: P() if k == 'loss_weights' else P('data') for k in template}
        def put(batch): return {k: jax.device_put(v, rep if k == 'loss_weights' else bat) for k, v in batch.items()}
        fn = jax.jit(jax.shard_map(lambda p, b: metrics.totals(p, b, net, tuple(c['exit_depths']), axis_name='data'),
            mesh=mesh, in_specs=(P(), specs), out_specs=P(), check_vma=False))
        start = time.perf_counter(); lowered = fn.lower(params, put(template)); executable = lowered.compile()
        compilation[str(bucket)] = {'seconds': time.perf_counter() - start, 'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest()}
        for index in range(0, len(all_entries), width):
            chosen = all_entries[index:index + width]; chosen += [None] * (width - len(chosen))
            batch = data.batch(chosen, positions=bucket)
            totals = {k: float(v) for k, v in jax.device_get(executable(params, put(batch))).items()}
            if not np.isfinite(list(totals.values())).all(): raise ValueError('Nonfinite diagnostic metrics')
            for k, v in totals.items(): raw[k] = raw.get(k, 0.) + v
            batches.append({'bucket': bucket, 'index': index, 'totals': totals})
        print(json.dumps({'kind': 'collection_bucket', 'host': host, 'bucket': bucket, 'games': len(all_entries)}), flush=True)
    if raw['expert_count'] != sum(x['moves'] for x in selected) or raw['behavior_count'] != 0:
        raise ValueError('Complete held-out expert coverage differs')
    report.update(status='passed', host_rank=host, jax_rank=jax.process_index(), candidate_sha256=c['candidate_sha256'],
        global_holdout_games=total, selected_games=selected, raw_totals=raw, averages=metrics.averages(raw), batches=batches,
        compilation=compilation, parameter_count=sum(x.size for x in jax.tree.leaves(params)))
    verify(SOURCE); mh.sync_global_devices('collection-diagnostic-complete')


def entry(args):
    c = read_json(args.config); verify(SOURCE)
    fields = 'schema_version kind platform expected_processes expected_devices candidate candidate_sha256 dataset collection_receipt collection_receipt_sha256 buckets sequences_per_host exit_depths'
    if (set(c) != set(fields.split()) or c['schema_version'] != 1 or c['kind'] != 'visual_collection_diagnostic'
            or c['platform'] != 'tpu' or c['expected_processes'] != 4 or c['expected_devices'] != 16
            or c['buckets'] != [128, 256, 384] or c['sequences_per_host'] != 32 or c['exit_depths'] != [3, 6]
            or canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json'))
            or args.resume is not None or args.stop_after_turn is not None): raise ValueError('Invalid collection diagnostic configuration')
    os.environ['JAX_PLATFORMS'] = 'tpu'; args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
        'scope': 'Post-hoc complete collection-holdout evaluation after observing the fixed online endpoint. These games were never eligible for training in either arm. No new model selection, RL speedup or playing-strength claim.'}
    started = time.time(); distributed = False
    try:
        import jax
        jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(c, args.output, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time()); (args.output / 'result.json').write_bytes(canonical_json(report))
        if distributed: jax.distributed.shutdown()
