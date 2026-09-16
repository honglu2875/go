"""Owned joint policy/value harness over fixed complete expert histories.

The shared checkpoint transport follows the recovery-qualified policy learner.
The model/update functions are cloned from the qualified joint prototype.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from gozero.corpus_sequence_batches import Dataset, augment
from train_config import validate


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path); checkpoints._sync_directory(path.parent)


def run(args, c, report):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import adamw as optimizer
    import learner as joint_learner
    import joint as model
    import optimizer_io
    import evaluation
    import host_metrics
    rank, world = jax.process_index(), jax.process_count()
    host = int(os.environ.get('GOZERO_HOST_RANK', '0'))
    if (world != c['expected_processes'] or len(jax.devices()) != c['expected_devices']
            or any(d.platform != c['platform'] for d in jax.devices())):
        raise ValueError('Learning topology differs from frozen contract')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    replicated, batched = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    def replica(tree):
        return jax.tree.map(lambda x: np.asarray(x.addressable_shards[0].data), tree)
    def gather_json(value):
        raw = canonical_json(value)
        if len(raw) > 32760:
            raise ValueError('Checkpoint collective metadata exceeds bound')
        buffer = np.zeros(32768, np.uint8); buffer[:8] = np.frombuffer(len(raw).to_bytes(8, 'little'), np.uint8)
        buffer[8:8+len(raw)] = np.frombuffer(raw, np.uint8)
        rows = np.asarray(mh.process_allgather(buffer)).reshape(world, 32768)
        return [json.loads(bytes(row[8:8+int.from_bytes(bytes(row[:8]), 'little')])) for row in rows]
    def digest_arrays(arrays):
        h = hashlib.sha256()
        for key in sorted(arrays):
            a = np.asarray(arrays[key]); h.update(canonical_json([key, list(a.shape), str(a.dtype)])); h.update(a.tobytes(order='C'))
        return h.hexdigest()
    def global_batch(batch):
        return {k:jax.make_array_from_process_local_data(batched,v) for k,v in batch.items()}

    net, opt = c['model'], c['learner']
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    if data.manifest.get('qualification_only') and c['training']['purpose'] != 'qualification':
        raise ValueError('An execution fixture cannot be used as a learning population')
    buckets = c['dataset']['buckets']; by_bucket = data.bucket_entries(buckets)
    probe = None
    if c['evaluation'].get('training_probe_games'):
        import training_probe
        probe = training_probe.select({b:by_bucket['expert',b] for b in buckets}, c['evaluation']['training_probe_games'])
        report['training_probe_selection'] = {'requested_games':c['evaluation']['training_probe_games'],
            'games_by_bucket':{str(b):len(entries) for b,entries in probe.items()},
            'selection':'Hamilton proportional allocation across length buckets; SHA256 game ordering; independent of training RNG'}
    if data.size != net['max_board_size'] or data.time > net['max_positions']:
        raise ValueError('Dataset exceeds exact model context')
    # Bucket choice and uniform episodes within each role/bucket are explicit;
    # both arms get identical draws and exposures. Rare long games must not be
    # silently oversampled simply to exercise a compilation bucket.
    if any(not by_bucket[role, b] for role in ('expert',) for b in buckets):
        raise ValueError('Every bucket needs expert training episodes')
    config_sha = hashlib.sha256(canonical_json(c)).hexdigest()
    mapping = gather_json({'host': host, 'jax_rank': rank})
    if sorted(x['host'] for x in mapping) != list(range(world)):
        raise ValueError('Host/JAX mapping is not a bijection')
    schema = model.parameter_schema(net,c['value_model'])
    report.update(host_rank=host, jax_rank=rank, world_size=world, host_jax_mapping=mapping,
                  parameter_count=sum(x['elements'] for x in schema), model_schema=schema,
                  config_sha256=config_sha, jax_version=jax.__version__,
                  teacher_cost={'source_release':data.manifest['source_release'],'teacher_sha256':data.manifest['target_teacher_sha256'],'incremental_generation_in_this_run':0},
                  dataset_manifest_sha256=c['dataset']['manifest_sha256'],
                  global_sequences_per_batch=world * opt['games_per_host'],
                  sequences_per_device=opt['games_per_host'] // len(jax.local_devices()),
                  training_population={f'{role}-{b}': len(v) for (role, b), v in by_bucket.items()},
                  sampling={'bucket_probabilities': c['dataset'].get('bucket_probabilities', [1 / len(buckets)] * len(buckets)),
                            'initial_bucket_sequence': c['dataset'].get('warmup_buckets', []),
                            'within_bucket': 'Uniform complete episodes per role, with replacement.'})
    print(json.dumps({'kind': 'visual_learning_initialize', 'parameters': report['parameter_count'], 'host': host}), flush=True)
    params = jax.jit(lambda: model.initialize(c['seed'], net,c['value_model']), out_shardings=replicated)()
    state = jax.jit(optimizer.initialize, out_shardings=replicated)(params)
    jax.block_until_ready((params, state))
    initial = {f'p_{i:04d}': x for i, x in enumerate(jax.tree.leaves(replica(params)))}
    report['initial_parameter_elements_sha256'] = digest_arrays(initial)
    del initial
    random = np.random.Generator(np.random.PCG64(c['seed'] + 1 + 104729 * rank))
    bucket_random = np.random.Generator(np.random.PCG64(c['seed'] + 9143))
    augmentation_random = np.random.Generator(np.random.PCG64(c['seed'] + 400003 + 104729 * rank))
    turn = 0; initial_turn = 0; fork_lineage = None
    counters = {'updates': 0, 'expert_positions': 0, 'padded_position_slots': 0}
    timing = {k: 0. for k in ['sampling_seconds', 'learning_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'compilation_seconds']}
    history=[];training_history=[];overfit_history=[]
    if args.resume is not None:
        group = read_json(args.resume.with_suffix('.group.json'))
        if (group['kind'] != 'visual_replicated_checkpoint_group' or group['snapshot_id'] != SOURCE.name
                or group['config_sha256'] != config_sha or group['host_jax_mapping'] != mapping):
            raise ValueError('Resume source, configuration or topology differs')
        identities = gather_json(checkpoints.sha256(args.resume.with_suffix('.group.json')))
        if len(set(identities)) != 1:
            raise ValueError('Checkpoint group differs across ranks')
        saved, local_arrays, _ = checkpoints.read(args.resume, expected_manifest_sha256=group['host_manifests'][str(host)])
        if (saved['host_rank'] != host or saved['jax_rank'] != rank or saved['model_schema'] != schema
                or saved['snapshot_id'] != SOURCE.name or saved['dataset_manifest_sha256'] != c['dataset']['manifest_sha256']):
            raise ValueError('Rank scientific state differs')
        owner = Path(group['owner_checkpoint_path'])
        owner_state, arrays, _ = checkpoints.read(owner, expected_manifest_sha256=group['host_manifests']['0'])
        if owner_state['turn'] != saved['turn'] or digest_arrays(arrays) != group['replicated_arrays_elements_sha256']:
            raise ValueError('Shared model/optimizer checkpoint differs')
        if host!=0 and local_arrays:raise ValueError('Nonowner checkpoint contains arrays')
        if owner_state['optimizer_metadata']!=saved['optimizer_metadata']:
            raise ValueError('Optimizer metadata differs across ranks')
        expected_schema=[{key:item[key] for key in ('path','shape','dtype')} for item in schema]
        restored,restored_state=optimizer_io.restore(saved['optimizer_metadata'],arrays,schema=expected_schema,
            configuration_sha256=config_sha,source_sha256=SOURCE.name)
        if int(restored_state['step'])!=saved['turn']:raise ValueError('Optimizer step differs from rank progress')
        params=jax.tree.map(lambda x:jax.device_put(x,replicated),restored)
        state=jax.tree.map(lambda x:jax.device_put(x,replicated),restored_state)
        history=saved['validation_history'];training_history=saved['training_probe_history']
        overfit_history=saved['overfit_history']
        for rows in (history,training_history):
            turns=[row['turn'] for row in rows]
            if turns!=sorted(set(turns)) or any(t<0 or t>saved['turn'] for t in turns):
                raise ValueError('Checkpoint diagnostic history differs')
        turn = saved['turn']; counters = saved['counters']
        random.bit_generator.state = saved['numpy_rng']; bucket_random.bit_generator.state = saved['bucket_rng']
        augmentation_random.bit_generator.state = saved['augmentation_rng']
        initial_turn = saved.get('initial_turn', 0); fork_lineage = saved.get('fork_lineage')
        report['initial_parameter_elements_sha256'] = saved.get('initial_parameter_elements_sha256', report['initial_parameter_elements_sha256'])
        report['resume_group_sha256'] = identities[0]
        del arrays, local_arrays
    stop = c['steps'] if args.stop_after_turn is None else args.stop_after_turn
    if not turn < stop <= c['steps']:
        raise ValueError('Stop must advance the saved state within its frozen schedule')
    optimizer_config={k:v for k,v in opt.items() if k not in ('games_per_host','augmentation')}
    optimizer_config['horizon_steps']=c['steps']
    def specs(batch):return {k:P('data') for k in batch}
    compiled_steps, compiled_evaluations = {}, {}
    def get_step(batch):
        bucket = batch['actions'].shape[1]
        if bucket not in compiled_steps:
            start = time.perf_counter()
            update=joint_learner.step(net,optimizer_config,value_weight=c['training']['value_weight'],
                path=c['training']['path'],chunk_frames=c['training']['chunk_frames'],mesh=mesh)
            print(json.dumps({'kind': 'visual_compile_update', 'bucket': bucket, 'host': host}), flush=True)
            lowered = jax.jit(update, donate_argnums=(0, 1)).lower(params, state, batch)
            executable = lowered.compile(); compiled_steps[bucket] = executable
            memory = executable.memory_analysis()
            report.setdefault('compiled_updates', {})[str(bucket)] = {
                'compile_seconds': time.perf_counter() - start,
                'compiler_cost_estimate': executable.cost_analysis(),
                'hlo_sha256': hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),
                'memory_bytes': {k: int(getattr(memory, k)) for k in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']}}
            timing['compilation_seconds'] += time.perf_counter() - start
        return compiled_steps[bucket]
    def evaluate(split,*,entries_by_bucket=None):
        start=time.perf_counter();sums={};all_ids=[]
        selected_buckets=data.bucket_entries(buckets,split=split)
        chosen_by_bucket={bucket:(entries_by_bucket[bucket] if entries_by_bucket is not None else
            selected_buckets['expert',bucket][:c['evaluation']['games_per_bucket']]) for bucket in buckets}
        population=[entry for bucket in buckets for entry in chosen_by_bucket[bucket]]
        if not population:raise ValueError('Evaluation population is empty')
        family_weights=data.evaluation_family_weights(population);batch_size=opt['games_per_host']
        for bucket in buckets:
            entries=chosen_by_bucket[bucket];all_ids.extend(entries);local=entries[rank::world]
            for begin in range(0,math.ceil(len(entries)/world),batch_size):
                chosen=local[begin:begin+batch_size]
                batch=global_batch(data.batch(chosen+[None]*(batch_size-len(chosen)),positions=bucket,family_weights=family_weights))
                if bucket not in compiled_evaluations:
                    start_compile=time.perf_counter()
                    fn=jax.shard_map(lambda p,b:evaluation.totals(p,b,net,chunk_frames=c['training']['chunk_frames'],axis_name='data'),
                        mesh=mesh,in_specs=(P(),specs(batch)),out_specs=P(),check_vma=False)
                    print(json.dumps(dict(kind='joint_compile_evaluation',bucket=bucket,host=host)),flush=True)
                    compiled_evaluations[bucket]=jax.jit(fn).lower(params,batch).compile()
                    timing['compilation_seconds']+=time.perf_counter()-start_compile
                totals={k:float(v) for k,v in replica(compiled_evaluations[bucket](params,batch)).items()}
                if not all(math.isfinite(v) for v in totals.values()):raise FloatingPointError('Nonfinite joint evaluation')
                for k,v in totals.items():sums[k]=sums.get(k,0.)+v
        if sums['expert_count']!=sums['value_count']:raise ValueError('Policy/value population differs')
        result=dict(turn=turn,split=split,episode_ids_sha256=hashlib.sha256(canonical_json(all_ids)).hexdigest(),
            raw_totals=sums,metrics=host_metrics.averages(sums))
        timing['evaluation_seconds']+=time.perf_counter()-start
        kind='joint_training_probe' if entries_by_bucket is not None else 'joint_heldout'
        with (args.output/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(kind=kind,**result),sort_keys=True)+'\n');stream.flush()
        print(json.dumps(dict(kind=kind,host=host,**result)),flush=True)
        return result
    def save():
        start = time.perf_counter()
        optimizer_metadata,arrays=optimizer_io.flatten(replica(params),replica(state),
            configuration_sha256=config_sha,source_sha256=SOURCE.name)
        if optimizer_metadata['step']!=turn:raise ValueError('Optimizer and sampler progress differ')
        digest = digest_arrays(arrays)
        if len(set(gather_json(digest))) != 1:
            raise ValueError('Replicated parameters or optimizer diverged')
        path = args.output / 'checkpoints' / f'turn-{turn:09d}'
        saved = {'schema_version': 1, 'kind': 'visual_replicated_rank_state', 'snapshot_id': SOURCE.name,
                 'config_sha256': config_sha, 'host_rank': host, 'jax_rank': rank, 'turn': turn,
                 'dataset_manifest_sha256': c['dataset']['manifest_sha256'], 'model_schema': schema,
                 'numpy_rng': random.bit_generator.state, 'bucket_rng': bucket_random.bit_generator.state,
                 'augmentation_rng': augmentation_random.bit_generator.state,
                 'initial_turn': initial_turn, 'fork_lineage': fork_lineage,
                 'initial_parameter_elements_sha256': report['initial_parameter_elements_sha256'],
                 'counters': counters, 'owns_replicated_arrays': host == 0,
                 'optimizer_metadata':optimizer_metadata,'validation_history':history,
                 'training_probe_history':training_history,'overfit_history':overfit_history}
        logical_path = path; archive_record = None; temporary_record = None
        if host == 0 and c.get('checkpoint_temporary',False):
            from gozero import checkpoint_stage
            path, identity, temporary_record = checkpoint_stage.write(path, state=saved, arrays=arrays, actors='{}',
                compress=not c.get('checkpoint_temporary_uncompressed',False),minimum_free_bytes=64*(1<<30))
        elif host == 0 and 'checkpoint_archive' in c:
            from gozero import checkpoint_archive
            path, identity, archive_record = checkpoint_archive.write(path, state=saved, arrays=arrays,
                actors='{}', archive=c['checkpoint_archive'], python=sys.executable)
        else:
            identity = checkpoints.write(path, state=saved, arrays=arrays if host == 0 else {}, actors='{}', compress=True)
        del arrays
        records = gather_json({'host': host, 'manifest_sha256': identity, 'path': str(path)})
        group = {'schema_version': 1, 'kind': 'visual_replicated_checkpoint_group', 'snapshot_id': SOURCE.name,
                 'config_sha256': config_sha, 'turn': turn, 'host_jax_mapping': mapping,
                 'host_manifests': {str(r['host']): r['manifest_sha256'] for r in records},
                 'owner_checkpoint_path': next(r['path'] for r in records if r['host'] == 0),
                 'replicated_arrays_elements_sha256': digest}
        publish(path.with_suffix('.group.json'), group)
        if temporary_record is not None:
            from gozero.ram_checkpoints import seal
            seal([path.with_suffix('.group.json')])
        if path != logical_path:
            publish(logical_path.with_suffix('.group.json'), group)
        report['latest_checkpoint'] = {'path': str(path), 'manifest_sha256': identity,
            'group_sha256': checkpoints.sha256(path.with_suffix('.group.json')),
            'replicated_arrays_elements_sha256': digest, 'owner_checkpoint_path': group['owner_checkpoint_path']}
        if archive_record is not None:
            report['latest_checkpoint']['archive'] = archive_record
        if temporary_record is not None:
            report['latest_checkpoint']['temporary'] = temporary_record
        timing['checkpoint_seconds'] += time.perf_counter() - start
    segment_start = time.perf_counter()
    report['initial_turn'] = initial_turn; report['fork_lineage'] = fork_lineage
    if args.resume is None:
        report['initial_validation'] = evaluate(1)
        history.append(report['initial_validation'])
        if probe is not None:
            training_history.append(evaluate(0,entries_by_bucket=probe))
    with (args.output / 'metrics.jsonl').open('x') as log:
        while turn < stop:
            start = time.perf_counter()
            warmup_buckets = c['dataset'].get('warmup_buckets', [])
            if turn - initial_turn < len(warmup_buckets):
                bucket = warmup_buckets[turn - initial_turn]
            elif 'bucket_probabilities' in c['dataset']:
                bucket = int(bucket_random.choice(buckets, p=c['dataset']['bucket_probabilities']))
            else:
                bucket = buckets[int(bucket_random.integers(len(buckets)))]
            entries = [by_bucket[role, bucket][int(i)] for role in ('expert',)
                       for i in random.integers(len(by_bucket[role, bucket]), size=opt['games_per_host'])]
            raw = data.batch(entries, positions=bucket)
            symmetries = augmentation_random.integers(0, 8, len(entries)) if opt.get('augmentation', 'none') == 'd4' else np.zeros(len(entries), np.int64)
            if opt.get('augmentation', 'none') == 'd4':
                raw = augment(raw, symmetries)
            batch = global_batch(raw)
            timing['sampling_seconds'] += time.perf_counter() - start
            step = get_step(batch); start = time.perf_counter()
            params, state, metrics = step(params, state, batch)
            observed = {k: float(v) for k, v in replica(metrics).items()}
            timing['learning_seconds'] += time.perf_counter() - start
            if not all(math.isfinite(v) for v in observed.values()) or not observed['accepted']:
                raise FloatingPointError('Rejected or nonfinite visual learner update')
            turn += 1; counters['updates'] = turn
            for role in ('expert',):
                counters[role + '_positions'] += int(observed['positions'])
            counters['padded_position_slots'] += world * len(entries) * bucket
            record = {'turn': turn, 'bucket': bucket, 'local_entries_sha256': hashlib.sha256(canonical_json(entries)).hexdigest(),
                      'local_symmetries': symmetries.tolist(),
                      'cumulative_learning_seconds': timing['learning_seconds'],
                      'cumulative_sampling_seconds': timing['sampling_seconds'], **observed}
            log.write(json.dumps(record, sort_keys=True) + '\n'); log.flush()
            if turn % c['log_every'] == 0:
                print(json.dumps({'kind': 'visual_update', 'host': host, **record}), flush=True)
            if turn % c['eval_every'] == 0 or turn == c['steps']:
                history.append(evaluate(1))
                if probe is not None:
                    training_history.append(evaluate(0,entries_by_bucket=probe))
                    paired_validation=history
                    for metric in ('expert_kl','family_kl','value_mse','value_family_mse'):
                        observation=training_probe.overfit_observation(paired_validation,training_history,metric=metric)
                        overfit_history.append({'turn':turn,**observation})
                        if observation['sustained']:
                            print(json.dumps({'kind':'overfit_observation','host':host,'turn':turn,**observation}),flush=True)
            if turn % c['checkpoint_every'] == 0 or turn == stop:
                save()
    if turn == c['steps'] and c['evaluation']['run_test']:
        report['test'] = evaluate(2)
    # The owner checkpoint already contains exact parameters; avoid a redundant
    # near-gigabyte export. A separately pinned inference descriptor can select
    # its p_ arrays through the standard non-pickle checkpoint reader.
    report.update(status='passed', turn=turn, training_complete=turn == c['steps'],
                  counters=counters, segment_timing=timing, validation_history=history,
                  training_probe_history=training_history, overfit_observations=overfit_history,
                  last_metrics=observed, segment_elapsed_seconds=time.perf_counter() - segment_start,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  device_memory_stats=[d.memory_stats() for d in jax.local_devices()])
    verify(SOURCE); mh.sync_global_devices('visual-learning-complete')


def entry(args):
    c = validate(read_json(args.config)); verify(SOURCE)
    if canonical_json(c) != canonical_json(read_json(SOURCE / 'resolved_config.json')):
        raise ValueError('Training configuration is not frozen')
    os.environ['JAX_PLATFORMS'] = c['platform']
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    publish(args.output / 'resolved_config.json', c)
    report = {'schema_version': 1, 'kind': c['kind'], 'snapshot_id': SOURCE.name, 'status': 'running',
              'claims_go_strength': False, 'claims_rl_sample_efficiency': False,
              'input_kind': 'Pinned complete expert histories and exact V7 features; raw policy and signed player-to-move value targets.',
              'training_purpose':c['training']['purpose'],'optimizer_family':'adamw'}
    started = time.time(); distributed = False
    try:
        import jax
        if c['platform'] == 'tpu':
            jax.distributed.initialize(initialization_timeout=90); distributed = True
        run(args, c, report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        report.update(started_unix=started, finished_unix=time.time())
        publish(args.output / 'result.json', report)
        if distributed:
            jax.distributed.shutdown()
