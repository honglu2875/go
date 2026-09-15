"""Compare an audited spatial-query refinement with its pinned bilinear parent."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--candidate-audit', type=Path, required=True)
    p.add_argument('--candidate-audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve()
    for path, digest in ((a.registration, a.registration_sha256),
                         (a.candidate_audit, a.candidate_audit_sha256)):
        if sha(path) != digest:
            raise ValueError('Pinned input changed: '+str(path))
    reg, candidate = map(read, (a.registration, a.candidate_audit))
    if reg['intervention']!='single_spatial_query_refinement_rank128':
        raise ValueError('Wrong registered intervention')
    candidate_source = root/'.gozero/snapshots'/candidate['training_snapshot']
    # Use the frozen candidate's verifier, not a potentially changed workspace library.
    sys.path.insert(0, str(candidate_source/'packages/gozero/src'))
    from gozero.snapshots import verify
    verify(candidate_source)
    candidate_config = read(candidate_source/'resolved_config.json')
    seed = candidate_config['seed']; index = reg['training']['seeds'].index(seed)
    stage = ('first_seed', 'second_seed')[index]
    if candidate_source.name != reg['snapshots'][stage]:
        raise ValueError('Unregistered candidate snapshot')
    reference = reg['references'][index]; reference_path = root/reference['path']
    if sha(reference_path) != reference['sha256']:
        raise ValueError('Parent audit changed')
    audits = (read(reference_path), candidate)
    configs, reports, draws, arms = [], [], [], []
    for label, audit, audit_path in zip(('Bilinear rank 64 parent', 'Spatial query refinement'), audits, (reference_path,a.candidate_audit)):
        if audit['status'] != 'passed':
            raise ValueError('Unaudited run')
        attempt = root/'runs'/audit['attempt']
        if sha(attempt/'result.json') != audit['closed_result_sha256']:
            raise ValueError('Closed attempt changed')
        for name, digest in audit['input_files'].items():
            if sha(root/name) != digest:
                raise ValueError('Audited evidence changed')
        source = root/'.gozero/snapshots'/audit['training_snapshot']; verify(source)
        config = read(source/'resolved_config.json'); configs.append(config)
        rank_draws = {}
        for host in range(4):
            report = read(attempt/f'rank-{host}/artifacts/result.json')
            rows = [json.loads(line) for line in (attempt/f'rank-{host}/artifacts/metrics.jsonl').read_text().splitlines()]
            rank_draws[report['jax_rank']] = [{key:row[key] for key in
                ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions')} for row in rows]
        draws.append(rank_draws)
        report = read(attempt/'rank-0/artifacts/result.json'); reports.append(report)
        arms.append(dict(label=label, attempt=attempt.name, snapshot=source.name,
            audit_path=str(audit_path), audit_sha256=sha(audit_path), parameter_count=report['parameter_count'],
            validation_curve=audit['validation_curve'], training_probe_curve=audit.get('training_probe_curve',[]),
            overfit_observations=audit.get('overfit_observations',[]),
            initial_decode_median_ms=1000*statistics.median(report['decode_profile']['host_dispatch_latency_seconds']),
            trained_decode_median_ms=1000*statistics.median(report['trained_decode_profile']['host_dispatch_latency_seconds']),
            checkpoint=report['latest_checkpoint'], timing=audit['timing'], positions=audit['expert_positions']))
    for key in ('dataset','seed','steps','learner','eval_every','expected_processes','expected_devices'):
        if configs[0][key] != configs[1][key]:
            raise ValueError('Unpaired setting: '+key)
    left_eval = dict(configs[0]['evaluation']); right_eval = dict(configs[1]['evaluation'])
    right_eval.pop('training_probe_games')
    if left_eval != right_eval or draws[0] != draws[1]:
        raise ValueError('Held-out population or exact training draws differ')
    left_model, right_model = map(lambda c:dict(c['model']), configs)
    if right_model.pop('policy_refinement_dim') != 128 or left_model != right_model:
        raise ValueError('Unexpected model intervention')
    if reports[0]['initial_parameter_elements_sha256'] != reports[1]['initial_without_refinement_parameter_elements_sha256']:
        raise ValueError('Common initialized weights differ')
    if not all(r['initial_context_query_zero'] for r in reports):
        raise ValueError('Initial readout correction is nonzero')
    if not reports[1]['initial_refinement_output_zero']:
        raise ValueError('Initial refinement is nonzero')
    curves = [arm['validation_curve'] for arm in arms]
    if len(curves[0]) != len(curves[1]):
        raise ValueError('Validation schedule differs')
    differences = []
    for left, right in zip(*curves):
        if (left['turn'],left['episode_ids_sha256'],left['metrics']['expert_count']) != (
            right['turn'],right['episode_ids_sha256'],right['metrics']['expert_count']):
            raise ValueError('Validation population differs')
        differences.append(dict(turn=left['turn'], parent_kl=left['metrics']['expert_kl'],
            candidate_kl=right['metrics']['expert_kl'], difference=right['metrics']['expert_kl']-left['metrics']['expert_kl']))
    if abs(differences[0]['difference']) > 1e-6:
        raise ValueError('Initial full-policy predictions differ')
    gain = 1-differences[-1]['candidate_kl']/differences[-1]['parent_kl']
    tail_gain = 1-statistics.mean(x['candidate_kl'] for x in differences[-3:])/statistics.mean(x['parent_kl'] for x in differences[-3:])
    out = dict(status='passed', kind='paired_query_refinement_contrast', created_unix=time.time(), seed=seed,
        registration_sha256=a.registration_sha256,
        operator_sha256=sha(Path(__file__)), arms=arms, paired_curve=differences,
        relative_endpoint_kl_improvement=gain, relative_last_three_mean_kl_improvement=tail_gain,
        replication_eligible=gain>=reg['decisions']['first_seed_replication_min_relative_endpoint_gain'],
        all_training_draws_identical=True, all_validation_populations_identical=True,
        common_initialized_weights_identical=True,
        runtime_scope='Absolute measured latencies; parent used a different CPU allocation, so these are not contemporaneous matched runtime controls.',
        scope='One paired fixed-data seed; no test results, Go-strength, MFU or RL-speed claims.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(out,stream,indent=2);stream.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({key:out[key] for key in ('status','seed','relative_endpoint_kl_improvement','relative_last_three_mean_kl_improvement','replication_eligible')}))


if __name__ == '__main__':
    main()
