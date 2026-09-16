"""Audit a complete joint learner across ranks, including optimizer and draws.

This accepts explicit qualification or learning purpose. The launch controller
must separately verify process closure and the registered experiment identity.
No model execution or test target access is needed by this auditor.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify

TIMINGS = {'cumulative_learning_seconds', 'cumulative_sampling_seconds'}
LOCAL = {'local_entries_sha256', 'local_symmetries'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(snapshot, directories, purpose, stop_turn, prefix_directories):
    import numpy as np
    from gozero.corpus_sequence_batches import Dataset
    manifest = verify(snapshot)
    c = read_json(snapshot / 'resolved_config.json')
    digest = hashlib.sha256(canonical_json(c)).hexdigest()
    require(c['kind'] == 'fixed_joint_learning' and c['training']['purpose'] == purpose
            and not c['evaluation']['run_test'], 'Wrong learner purpose or test scope')
    sys.path.insert(0, str(snapshot / manifest['recipe']))
    import joint
    import optimizer_io
    source = c['training']['optimizer'] == 'muon_source_runtime'
    require(source or c['training']['optimizer'] == 'adamw', 'Unsupported optimizer')
    runtime = None
    if source:
        import source_runtime
        runtime = source_runtime
    for name in ('checkpoints.py', 'corpus_sequence_batches.py', 'corpus_format.py', 'katago_sequence_batches.py'):
        relative = 'packages/gozero/src/gozero/' + name
        require(checkpoints.sha256(ROOT / relative) == manifest['files'][relative]['sha256'], 'Audit dependency changed')
    helper = ROOT / 'research/studies/strong19_source_muon/distributed_recovery.py'
    spec = importlib.util.spec_from_file_location('tensor_audit_reference', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(checkpoints.sha256(helper) == '1f3059d88f3bc248356c17673ee1706ec8511456fecc97084a2d734563b86afd', 'Tensor auditor changed')
    schema = joint.parameter_schema(c['model'], c['value_model'])
    world = c['expected_processes']; horizon = stop_turn
    require(not source and 0 < horizon <= c['steps'] and horizon % c['eval_every'] == 0, 'Expected an AdamW prefix ending on a validation boundary')
    require(len(directories) == world, 'Missing rank artifacts')
    reports = {}; states = {}; metrics = {}; groups = {}; inputs = {}; tensors = None
    require(len(prefix_directories) == world, 'Missing prefix ranks')
    prefixes = {}
    for directory in prefix_directories:
        parent = read_json(directory / 'result.json')
        require(parent['host_rank'] not in prefixes and parent['status'] == 'passed'
            and parent['snapshot_id'] == snapshot.name and 0 < parent['turn'] < horizon, 'Invalid prefix')
        prefixes[parent['host_rank']] = (directory, parent)
    def record(path):
        value = checkpoints.sha256(path)
        require(str(path) not in inputs or inputs[str(path)] == value, 'Artifact changed during audit')
        inputs[str(path)] = value
        return value
    for directory in directories:
        directory = directory.resolve()
        for name in ('result.json', 'resolved_config.json', 'metrics.jsonl', 'evaluations.jsonl'):
            record(directory / name)
        r = read_json(directory / 'result.json'); host = r['host_rank']
        require(host not in reports and host in range(world), 'Repeated or invalid rank')
        require(read_json(directory / 'resolved_config.json') == c and r['config_sha256'] == digest
                and r['snapshot_id'] == snapshot.name and r['world_size'] == world
                and r['status'] == 'passed' and r['training_complete'] == (horizon == c['steps']) and r['turn'] == horizon
                and r['kind'] == c['kind'] and r['training_purpose'] == purpose
                and r['optimizer_family'] == ('source_muon_aux_adam_lookahead' if source else 'adamw')
                and r['initial_turn'] == 0 and r['fork_lineage'] is None
                and r['dataset_manifest_sha256'] == c['dataset']['manifest_sha256']
                and r['model_schema'] == schema, 'Report identity or completion differs')
        mapping = r['host_jax_mapping']
        require(len(mapping) == world and sorted(x['host'] for x in mapping) == list(range(world))
                and sorted(x['jax_rank'] for x in mapping) == list(range(world))
                and r['jax_rank'] == next(x['jax_rank'] for x in mapping if x['host'] == host), 'Rank mapping differs')
        cp = r['latest_checkpoint']; path = Path(cp['path']); gp = path.with_suffix('.group.json')
        require(record(gp) == cp['group_sha256'], 'Checkpoint group changed')
        group = read_json(gp)
        require(group['kind'] == 'visual_replicated_checkpoint_group' and group['snapshot_id'] == snapshot.name
                and group['config_sha256'] == digest and group['turn'] == horizon
                and group['host_jax_mapping'] == mapping
                and set(group['host_manifests']) == {str(h) for h in range(world)}
                and group['host_manifests'][str(host)] == cp['manifest_sha256']
                and group['replicated_arrays_elements_sha256'] == cp['replicated_arrays_elements_sha256'], 'Checkpoint group differs')
        saved, arrays, actors = checkpoints.read(path, expected_manifest_sha256=cp['manifest_sha256'])
        for member in path.iterdir():
            record(member)
        require(actors == '{}' and saved['host_rank'] == host and saved['jax_rank'] == r['jax_rank']
                and saved['owns_replicated_arrays'] == (host == 0) and saved['snapshot_id'] == snapshot.name
                and saved['config_sha256'] == digest and saved['model_schema'] == schema
                and saved['dataset_manifest_sha256'] == c['dataset']['manifest_sha256']
                and saved['turn'] == horizon and saved['initial_turn'] == 0 and saved['fork_lineage'] is None
                and saved['initial_parameter_elements_sha256'] == r['initial_parameter_elements_sha256'], 'Saved identity differs')
        for key in ('counters', 'validation_history', 'training_probe_history') + (('source_runtime',) if source else ()):
            require(saved[key] == r[key], 'Saved diagnostics differ: ' + key)
        require(saved['overfit_history'] == r['overfit_observations'], 'Saved overfit diagnostics differ')
        if source:
            runtime.validate_state(saved['source_runtime'], c['learner']['source_runtime'], horizon,
                                   positions=saved['counters']['expert_positions'])
        if host == 0:
            require(str(path) == group['owner_checkpoint_path'] == cp['owner_checkpoint_path'], 'Wrong checkpoint owner')
            tensors = module.tensor_identity(arrays)
            require(tensors['sha256'] == group['replicated_arrays_elements_sha256'], 'Actual tensor hash differs')
            kwargs = dict(schema=[{k: x[k] for k in ('path', 'shape', 'dtype')} for x in schema],
                          configuration_sha256=digest, source_sha256=snapshot.name)
            if source:
                kwargs['lookahead_config'] = c['learner']['source_runtime']['lookahead']
            params, opt = optimizer_io.restore(saved['optimizer_metadata'], arrays, **kwargs)
            require(int(opt['step']) == horizon, 'Optimizer step differs')
            if source:
                require(np.array_equal(opt['hyperparameters'], runtime.vector(saved['source_runtime']['settings'])), 'Schedule scalar state differs')
                tensors['lookahead_counter'] = int(opt['lookahead']['counter'])
                tensors['fast_equals_slow'] = all(np.array_equal(params[k], opt['lookahead']['slow'][k]) for k in params)
            del params, opt
        else:
            require(not arrays, 'Nonowner stored replicated arrays')
        del arrays
        prefix_directory, prefix = prefixes[host]
        record(prefix_directory / 'result.json'); record(prefix_directory / 'metrics.jsonl')
        require(r['resume_group_sha256'] == prefix['latest_checkpoint']['group_sha256']
                and prefix['initial_parameter_elements_sha256'] == r['initial_parameter_elements_sha256']
                and prefix['host_jax_mapping'] == r['host_jax_mapping'], 'Resume lineage differs')
        for key in ('validation_history', 'training_probe_history'):
            require(r[key][:len(prefix[key])] == prefix[key], 'Prefix evaluation history changed')
        earlier = [json.loads(x) for x in (prefix_directory / 'metrics.jsonl').read_text().splitlines()]
        later = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines()]
        require([r['turn'] for r in earlier] == list(range(1, prefix['turn']+1))
                and [r['turn'] for r in later] == list(range(prefix['turn']+1, horizon+1)), 'Segment update boundaries differ')
        rows = earlier + later
        require([x['turn'] for x in rows] == list(range(1, horizon+1)), 'Missing/repeated updates')
        require(all(x['accepted'] == 1 and x.get('eligible', 1) == 1 for x in rows), 'Rejected updates')
        require(all(math.isfinite(v) for x in rows for v in x.values() if type(v) in (int, float)), 'Nonfinite metric')
        reports[host], states[host], groups[host], metrics[host] = r, saved, group, rows
    common = ('counters', 'optimizer_metadata', 'validation_history', 'training_probe_history',
              'overfit_history', 'initial_parameter_elements_sha256') + (('source_runtime',) if source else ())
    trim = lambda row: {k: v for k, v in row.items() if k not in TIMINGS | LOCAL}
    for host in range(world):
        require(groups[host] == groups[0] and all(states[host][k] == states[0][k] for k in common), 'Replicated state differs')
        require(list(map(trim, metrics[host])) == list(map(trim, metrics[0])), 'All-rank update metrics differ')
    data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
    entries = data.bucket_entries(c['dataset']['buckets'])
    randoms = {host: [np.random.Generator(np.random.PCG64(c['seed'] + n)) for n in
                     (1+104729*r['jax_rank'], 9143, 400003+104729*r['jax_rank'])] for host, r in reports.items()}
    positions = slots = 0; source_state = None; counter = 0
    if source:
        rc = c['learner']['source_runtime']
        source_state = runtime.initialize(rc, horizon, states[0]['source_runtime']['baseline'])
    for index in range(horizon):
        count = 0
        for host, (rng, bucket_rng, symmetry_rng) in randoms.items():
            warm = c['dataset']['warmup_buckets']
            bucket = warm[index] if index < len(warm) else int(bucket_rng.choice(c['dataset']['buckets'], p=c['dataset']['bucket_probabilities']))
            pool = entries['expert', bucket]
            chosen = [pool[int(i)] for i in rng.integers(len(pool), size=c['learner']['games_per_host'])]
            symmetries = symmetry_rng.integers(0, 8, len(chosen)).tolist() if c['learner']['augmentation'] == 'd4' else [0]*len(chosen)
            row = metrics[host][index]
            require(row['bucket'] == bucket and row['local_entries_sha256'] == hashlib.sha256(canonical_json(chosen)).hexdigest()
                    and row['local_symmetries'] == symmetries, 'Independent draw/D4 replay differs')
            count += sum(int(data.game_info(e)['length']) for e in chosen); slots += len(chosen)*bucket
        positions += count; row = metrics[0][index]
        require(row['positions'] == count, 'Actual loss-position count differs')
        if source:
            source_state, before = runtime.before_step(source_state, rc, horizon)
            if before['subepoch_entry']:
                counter = 0
            for key, value in dict(normal_learning_rate=source_state['settings']['rates']['normal'],
                normal_weight_decay=source_state['settings']['decays']['normal'],
                source_sum_gradient_clip_cap=source_state['settings']['source_sum_gradient_clip_cap'],
                schedule_reference_batch_positions=rc['reference_batch_positions']).items():
                require(row[key] == float(np.float32(value)), 'Applied source scalar differs: ' + key)
            require(row['source_epoch_entry'] == before['epoch_entry'] and row['source_subepoch_entry'] == before['subepoch_entry'], 'Source epoch entry differs')
            norms = {k: row['source_pre_norm_'+k] for k in ('input', 'normal')} if before['is_print_batch'] else None
            source_state, after = runtime.after_step(source_state, rc, horizon, positions=count, pre_update_norms=norms)
            counter += int(rc['lookahead']['k'] is not None)
            sync = rc['lookahead']['k'] is not None and counter == rc['lookahead']['k']
            if sync:
                counter = 0
            require(row['lookahead_counter'] == counter and row['lookahead_synchronized'] == sync
                    and row['source_norm_snapshot'] == after['norm_snapshot'] and row['source_epoch_flush'] == after['epoch_end']
                    and row['source_samples'] == source_state['samples'] and row['source_refresh_count'] == source_state['refresh_count'], 'Source schedule events differ')
    for host, state in states.items():
        require(state['counters'] == dict(updates=horizon, expert_positions=positions, padded_position_slots=slots), 'Exposure counters differ')
        for key, rng in zip(('numpy_rng', 'bucket_rng', 'augmentation_rng'), randoms[host]):
            require(state[key] == rng.bit_generator.state, 'Final sampler state differs')
        if source:
            require(state['source_runtime'] == source_state and tensors['lookahead_counter'] == counter, 'Complete source clock differs')
    turns = {0, horizon, *range(c['eval_every'], horizon+1, c['eval_every'])}
    if source:
        end = 0
        for epoch in rc['epochs']:
            end += sum(epoch); turns.add(end)
    for key in ('validation_history', 'training_probe_history'):
        history = reports[0][key]
        require([r['turn'] for r in history] == sorted(turns) and len({r['episode_ids_sha256'] for r in history}) == 1, 'Diagnostic observations or population changed')
        for row in history:
            require(row['raw_totals']['expert_count'] == row['raw_totals']['value_count'] > 0
                    and all(math.isfinite(v) for v in row['metrics'].values()), 'Invalid joint evaluation')
    require(len(reports[0]['overfit_observations']) == 4*(len(turns)-1), 'Missing policy/value overfit observations')
    return dict(kind='complete_joint_continuation_audit', status='passed', snapshot=snapshot.name, purpose=purpose,
        configuration_sha256=digest, dataset_manifest_sha256=c['dataset']['manifest_sha256'], world_size=world,
        steps=horizon, parameters=sum(x['elements'] for x in schema), checkpoint=tensors,
        all_rank_metrics_and_saved_state_verified=True, independently_replayed_rank_updates=world*horizon,
        positions=positions, padded_position_slots=slots, initial_parameters_sha256=states[0]['initial_parameter_elements_sha256'],
        validation_history=reports[0]['validation_history'], training_probe_history=reports[0]['training_probe_history'],
        overfit_observations=reports[0]['overfit_observations'], inputs=inputs,
        scope='Complete state, actual payloads, all-rank observations, sampling and source-optimizer clock validation. Process closure and paired experiment registration are separate controller checks. No playing-strength claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--artifacts', required=True, nargs='+', type=Path)
    p.add_argument('--purpose', choices=('qualification', 'learning'), required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stop-turn', required=True, type=int)
    p.add_argument('--prefix-artifacts', required=True, nargs='+', type=Path)
    args = p.parse_args(); require(not args.output.exists(), 'Output already exists')
    os.environ['JAX_PLATFORMS'] = 'cpu'; started = time.monotonic()
    result = audit(args.snapshot.resolve(), args.artifacts, args.purpose, args.stop_turn, args.prefix_artifacts)
    result.update(created=time.time(), seconds=time.monotonic()-started, operator_sha256=checkpoints.sha256(Path(__file__)))
    with args.output.open('xb') as stream:
        stream.write(canonical_json(result))
    print(json.dumps({k: result[k] for k in ('status', 'world_size', 'steps', 'parameters', 'positions', 'seconds')}))


if __name__ == '__main__':
    main()
