#!/usr/bin/env python3
"""Analyze the bounded four-host 9x9 root-prefetch qualification."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'ops'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from compare_resume import compare as compare_resume
from qualify_cpu import load, compare

REGISTRATION = '31f8762c4743a2580c69b2374fd77914713d33ef1a5413c5bbfd67c2b0fde4a4'


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'protocol', 'off', 'queued', 'resumed', 'wide', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(SOURCE)
    if checkpoints.sha256(args.protocol) != REGISTRATION:
        raise ValueError('TPU registration changed')
    protocol = read_json(args.protocol)
    root = args.workspace_root.resolve()
    report = {'schema_version': 1, 'kind': 'root_prefetch_tpu_qualification_result',
              'analysis_snapshot': SOURCE.name, 'protocol_sha256': REGISTRATION,
              'status': 'failed', 'claims_throughput': False, 'claims_mfu': False,
              'claims_strength': False, 'claims_sample_efficiency': False,
              'attempts': {}, 'hosts': []}
    try:
        if checkpoints.sha256(root / 'research/studies/root_prefetch/qualification_result.json') != protocol['cpu_qualification_sha256']:
            raise ValueError('CPU prerequisite differs')
        previous_end = 0
        for mode in ('off', 'queued', 'resumed', 'wide'):
            directory = getattr(args, mode).resolve()
            pod = read_json(directory / 'result.json')
            expected_source = protocol['sources']['queued' if mode == 'resumed' else mode]
            if (pod['status'] != 'passed' or pod['snapshot_id'] != expected_source
                    or pod['start_unix_time'] < max(previous_end, protocol['registered_unix'])
                    or pod['timeout_seconds'] != protocol['budget']['timeout_seconds']
                    or pod['prepare_timeout_seconds'] != protocol['budget']['prepare_timeout_seconds']):
                raise ValueError('Attempt identity, registration order or budget differs')
            if mode == 'resumed' and (pod['resume_attempt'] != args.queued.name or pod['resume_turn'] != 128):
                raise ValueError('Resume attempt identity differs')
            previous_end = pod['end_unix_time']
            ranks = [read_json(directory / f'rank-{host}/artifacts/result.json') for host in range(4)]
            if (len({r['model_export_sha256'] for r in ranks}) != 1
                    or len({r['counters']['updates'] for r in ranks}) != 1
                    or {r['jax_rank'] for r in ranks} != set(range(4))):
                raise ValueError('Replicated learner differs')
            report['attempts'][mode] = {'path': str(directory), 'snapshot_id': expected_source,
                'result_sha256': checkpoints.sha256(directory / 'result.json'),
                'attempt_chip_hours': pod['reserved_chip_hours'],
                'max_training_segment_seconds': max(r['elapsed_segment_seconds'] for r in ranks),
                'max_inference_seconds': max(r['counters']['inference_seconds'] for r in ranks),
                'model_export_sha256': ranks[0]['model_export_sha256'],
                'global_counters': {k: sum(r['counters'][k] for r in ranks) for k in (
                    'real_moves', 'completed_games', 'truncated_games', 'eligible_rows',
                    'active_neural_evaluations', 'neural_batches', 'neural_slots',
                    'inference_syncs', 'prefetch_neural_evaluations', 'prefetch_hits')}}
        for host in range(4):
            relative = Path(f'rank-{host}/artifacts')
            runs = {mode: load(root, getattr(args, mode).resolve() / relative, source, final_turn=256)
                    for mode, source in protocol['sources'].items()}
            for mode, run in runs.items():
                config = run['config']
                if config['root_prefetch'] != {'mode': mode, 'limit': 16} or config['selfplay_turns'] != 256:
                    raise ValueError('Registered execution settings differ')
                if run['result']['counters']['real_moves'] != 32768:
                    raise ValueError('Registered per-host real-move budget differs')
                group = read_json(run['directory'] / 'checkpoints/turn-000000256.group.json')
                if (group['rank_manifests'][run['result']['jax_rank']] != run['result']['latest_checkpoint']['manifest_sha256']
                        or checkpoints.sha256(run['directory'] / 'checkpoints/turn-000000256.group.json') != run['result']['latest_checkpoint']['group_sha256']):
                    raise ValueError('Rank checkpoint group differs')
            comparison = compare(runs['off'], runs['queued'])
            resumed = compare_resume(args.queued.resolve() / relative, args.resumed.resolve() / relative, 128, 256)
            off, wide = runs['off'], runs['wide']
            changes = [k for k, v in off['arrays'].items() if k not in wide['arrays']
                       or v.shape != wide['arrays'][k].shape or not np.array_equal(v, wide['arrays'][k])]
            parameter_error = max(float(np.max(np.abs(v.astype(np.float64) - wide['arrays'][k].astype(np.float64))))
                                  for k, v in off['arrays'].items() if k.startswith('p_'))
            report['hosts'].append({'host': host, 'queued_equivalence': comparison,
                'queued_recovery': resumed, 'wide_different_arrays': changes,
                'wide_max_final_parameter_difference': parameter_error,
                'wide_scope': 'Valid completed training state; final parameter differences include diverged self-play and learning, not isolated forward error.'})
        report['total_attempt_chip_hours'] = sum(a['attempt_chip_hours'] for a in report['attempts'].values())
        for mode, attempt in report['attempts'].items():
            counts = attempt['global_counters']
            attempt['unused_prefetched_evaluations'] = counts['prefetch_neural_evaluations'] - counts['prefetch_hits']
            if attempt['unused_prefetched_evaluations'] < 0:
                raise ValueError('Prefetch hits exceed evaluated outputs')
        report['limitations'] = protocol['limitations']
        report['decision'] = 'Queued full-state equivalence and recovery qualified on four TPU hosts. Register counterbalanced timing; wide numerical and learning validation remain separate.'
        report['status'] = 'passed'
        verify(SOURCE)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
