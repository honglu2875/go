"""Independent exposure-multiset, schedule, checkpoint and comparison audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero import checkpoints
from gozero.katago_sequence_batches import Dataset
import position_stream
import katago
import compute_budget
from train_sgd import array_digest, publish


def load_reference(root, path, digest):
    if checkpoints.sha256(path) != digest: raise ValueError('Reference audit changed')
    audit = read_json(path)
    if audit['status'] != 'passed' or audit['steps'] != 1024: raise ValueError('Incomplete reference')
    for name, wanted in audit['input_files'].items():
        if checkpoints.sha256(root/name) != wanted: raise ValueError('Reference observation changed')
    attempt = root/'runs'/audit['attempt']
    if checkpoints.sha256(attempt/'result.json') != audit['closed_result_sha256']: raise ValueError('Reference attempt changed')
    reports = [read_json(attempt/f'rank-{host}/artifacts/result.json') for host in range(4)]
    logs = {r['jax_rank']: [json.loads(x) for x in (attempt/f'rank-{host}/artifacts/metrics.jsonl').read_text().splitlines()]
            for host, r in enumerate(reports)}
    return audit, reports, logs


def qualify_exposures(data, c, reference, original_logs):
    tape, evidence, draws = position_stream.build(data, reference, c['sampling']['shuffle_seed'], c['sampling']['position_exposures'])
    for rank, rows in enumerate(draws):
        for row, observed in zip(rows, original_logs[rank]):
            if any(row[k] != observed[k] for k in row): raise ValueError('Reconstructed reference draw differs')
    # Independently build a position/symmetry histogram, traversing rank-major
    # rather than the stream generator's step-major ordering. This detects
    # lost/duplicated positions even if the final total or tape hash looks right.
    lengths = [len(a['expert_actions']) for a in data.shards]
    bases = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    expected = np.zeros(sum(lengths)*8, np.int32)
    allowed = np.zeros(sum(lengths), bool)
    for shard, episode in data.indices['expert', 0]:
        begin, end = map(int, data.shards[shard]['expert_offsets'][episode:episode+2])
        allowed[bases[shard]+begin:bases[shard]+end] = True
    entries = data.bucket_entries(reference['dataset']['buckets'])
    for rank in range(reference['expected_processes']):
        rng = np.random.Generator(np.random.PCG64(reference['seed']+1+104729*rank))
        for observed in original_logs[rank]:
            pool = entries['expert', observed['bucket']]
            chosen = rng.integers(len(pool), size=reference['learner']['games_per_host'])
            for index, code in zip(chosen, observed['local_symmetries']):
                _, shard, episode = pool[int(index)]
                begin, end = map(int, data.shards[shard]['expert_offsets'][episode:episode+2])
                expected[(bases[shard]+np.arange(begin,end))*8+code] += 1
    flat_positions = bases[tape[:,0]] + tape[:,1]
    if not np.all(allowed[flat_positions]): raise ValueError('Validation/test position entered training')
    actual = np.bincount(flat_positions*8+tape[:,2], minlength=len(expected))
    if not np.array_equal(actual, expected): raise ValueError('Shuffled training multiset differs')
    if int(actual.sum()) != c['sampling']['position_exposures']: raise ValueError('Wrong exposure count')
    evidence.update(exact_reference_position_and_symmetry_multiset=True, all_training_positions=True,
                    multiset_counts_sha256=position_stream.digest(expected),
                    independent_reference_draws_checked=sum(map(len, draws)))
    return evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--reference-audit', type=Path, required=True)
    p.add_argument('--reference-audit-sha256', required=True)
    p.add_argument('--attempt', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--preflight-sha256')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    c = read_json(SOURCE/'resolved_config.json')
    reference = read_json(Path(__file__).with_name('reference_config.json'))
    audit, refs, original_logs = load_reference(root, a.reference_audit, a.reference_audit_sha256)
    if reference != read_json(root/'.gozero/snapshots'/audit['training_snapshot']/'resolved_config.json'):
        raise ValueError('Wrong reference source/configuration')
    expected_model = {**reference['model'], 'microbatch':16}
    if c['model'] != expected_model or c['seed'] != reference['seed']:
        raise ValueError('Model or initialization controls changed')
    for name in ('katago.py', 'policy_model.py', 'compute_budget.py'):
        old = root/'.gozero/snapshots'/audit['training_snapshot']/'research/recipes/visual_katago_lr'/name
        if checkpoints.sha256(old) != checkpoints.sha256(Path(__file__).with_name(name)):
            raise ValueError('CNN implementation changed')
    schema = katago.parameter_schema(c['model'])
    if schema != refs[0]['model_schema']: raise ValueError('Parameter schema changed')
    if compute_budget.cnn(c['model'],9,128) != compute_budget.cnn(reference['model'],9,128):
        raise ValueError('Logical decode budget changed')
    if a.attempt is None:
        data = Dataset(c['dataset']['path'], c['dataset']['manifest_sha256'])
        evidence = qualify_exposures(data, c, reference, original_logs)
        publish(a.output, {'kind':'sgd_exposure_and_architecture_preflight','status':'passed',
            'snapshot':SOURCE.name,'reference_audit_sha256':a.reference_audit_sha256,
            'config_sha256':checkpoints.sha256(SOURCE/'resolved_config.json'), 'exposure':evidence,
            'parameter_count':sum(x['elements'] for x in schema), 'logical_decode_budget_unchanged':True,
            'initial_parameter_elements_sha256':refs[0]['initial_parameter_elements_sha256']})
        print(json.dumps({'status':'passed','preflight':str(a.output),'sha256':checkpoints.sha256(a.output),'positions':evidence['positions']}),flush=True)
        return
    if not a.preflight or checkpoints.sha256(a.preflight) != a.preflight_sha256: raise ValueError('Pinned preflight required')
    preflight = read_json(a.preflight)
    if preflight['status'] != 'passed' or preflight['config_sha256'] != checkpoints.sha256(SOURCE/'resolved_config.json'):
        raise ValueError('Preflight differs')
    attempt = a.attempt.resolve(); closed = read_json(attempt/'result.json')
    if closed['status'] != 'passed' or closed['snapshot_id'] != SOURCE.name or closed.get('resume_attempt'):
        raise ValueError('Expected a complete uninterrupted registered run')
    count = c['sampling']['position_exposures']; batch = c['learner']['batch_size']; steps = math.ceil(count/batch)
    reports = []; files = {}; log_digest = None; owner_blocks = None; owner_rows = None
    for host in range(c['expected_processes']):
        folder = attempt/f'rank-{host}/artifacts'; r = read_json(folder/'result.json')
        if r['status'] != 'passed' or not r['training_complete'] or r['step'] != steps or r['initial_step'] != 0:
            raise ValueError('Incomplete SGD learning')
        if r['config_sha256'] != checkpoints.sha256(SOURCE/'resolved_config.json'):
            raise ValueError('SGD configuration changed')
        for k, v in r['exposure'].items():
            if preflight['exposure'][k] != v: raise ValueError('Training tape changed')
        if r['initial_parameter_elements_sha256'] != preflight['initial_parameter_elements_sha256']:
            raise ValueError('Initial parameters changed')
        raw = (folder/'metrics.jsonl').read_bytes()
        if host == 0:
            log_digest = hashlib.sha256(raw).hexdigest(); rows = []; owner_blocks = []
            for line in raw.splitlines(keepends=True):
                block = json.loads(line)
                if block['first_step'] != len(rows)+1 or block['active_updates'] != len(block['rows']): raise ValueError('Missing/repeated SGD updates')
                rows.extend(block['rows'])
                owner_blocks.append({'first_step':block['first_step'], 'active_updates':block['active_updates'],
                                     'metrics_sha256':hashlib.sha256(line).hexdigest()})
            owner_rows = rows
        else:
            if [json.loads(x) for x in raw.splitlines()] != owner_blocks: raise ValueError('Replicated block metrics differ')
            rows = owner_rows
        if r['training_metrics_sha256'] != log_digest: raise ValueError('Replicated SGD metrics differ')
        if len(rows) != steps: raise ValueError('Wrong SGD update count')
        for i, row in enumerate(rows):
            n = min(batch, count-i*batch)
            lr = c['learner']['warmup_per_sample_lr'] if i*batch <= c['learner']['warmup_positions'] else c['learner']['per_sample_lr']
            if (not row['accepted'] or not row['active'] or int(row['update']) != i+1 or int(row['expert_positions']) != n
                    or int(row['samples_before']) != i*batch or int(row['samples_after']) != min((i+1)*batch,count)
                    or row['clip_scale'] != 1. or not math.isclose(row['per_sample_learning_rate'],lr,rel_tol=1e-6)
                    or not math.isclose(row['learning_rate'], n*lr,rel_tol=1e-6)
                    or any(not math.isfinite(v) for v in row.values())):
                raise ValueError('Invalid finite-update or historical schedule record')
        curve = [r['initial_validation'], *r['validation_history']]
        if [x['step'] for x in curve] != [0,*[x['sgd_step'] for x in r['evaluation_milestones']]]:
            raise ValueError('Validation cadence changed')
        for x in curve:
            if (x['episode_ids_sha256'] != audit['validation_curve'][0]['episode_ids_sha256']
                    or x['metrics']['expert_count'] != audit['validation_curve'][0]['metrics']['expert_count']):
                raise ValueError('Validation population changed')
        if abs(curve[0]['metrics']['expert_kl']-audit['validation_curve'][0]['metrics']['expert_kl']) > 1e-5:
            raise ValueError('Initial validation differs beyond the microbatch execution tolerance')
        for name in ('result.json','metrics.jsonl'): files[str((folder/name).relative_to(root))] = checkpoints.sha256(folder/name)
        reports.append(r)
    group_hashes = [checkpoints.sha256(Path(r['latest_checkpoint']['path']).with_suffix('.group.json')) for r in reports]
    if len(set(group_hashes)) != 1: raise ValueError('Checkpoint groups differ')
    group = read_json(Path(reports[0]['latest_checkpoint']['path']).with_suffix('.group.json'))
    for host, r in enumerate(reports):
        path = Path(r['latest_checkpoint']['path'])
        saved, arrays, _ = checkpoints.read(path, expected_manifest_sha256=group['host_manifests'][str(host)])
        if saved['samples'] != count or saved['step'] != steps: raise ValueError('Checkpoint counters differ')
        if host == 0:
            if set(arrays) != {f'{k}_{i:04d}' for k in ('p','m') for i in range(len(schema))} or any(not np.isfinite(x).all() for x in arrays.values()):
                raise ValueError('Checkpoint parameter/momentum coverage differs')
            if array_digest(arrays) != group['replicated_arrays_elements_sha256']: raise ValueError('Checkpoint content differs')
        elif arrays: raise ValueError('Unexpected non-owner arrays')
        del arrays
    publish(a.output, {'kind':'position_sgd_learning_audit','status':'passed','operator_snapshot':SOURCE.name,
        'attempt':attempt.name,'closed_result_sha256':checkpoints.sha256(attempt/'result.json'),
        'preflight_sha256':a.preflight_sha256,'reference_audit_sha256':a.reference_audit_sha256,
        'position_exposures':count,'steps':steps,'batch_size':batch,'last_batch_positions':count-(steps-1)*batch,
        'same_initial_parameters':True,'same_position_and_symmetry_multiset':True,
        'all_update_schedules_audited':True,'all_rank_metrics_equal':True,'validation_curve':[reports[0]['initial_validation'],*reports[0]['validation_history']],
        'timing':reports[0]['segment_timing'],'reserved_chip_hours':closed['reserved_chip_hours'],
        'checkpoint_group_sha256':group_hashes[0], 'input_files':files,
        'scope':'One historical SGD recipe versus tuned AdamW on the same exposure multiset and CNN. Batch size, order, optimizer, regularization and schedule differ; training FLOPs and update counts are not held equal. No Go/RL strength claim.'})
    print(json.dumps({'status':'passed','audit':str(a.output),'sha256':checkpoints.sha256(a.output)}),flush=True)

if __name__ == '__main__': main()
