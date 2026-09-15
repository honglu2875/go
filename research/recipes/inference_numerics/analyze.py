#!/usr/bin/env python3
"""Audit the registered fixed-input TPU inference probe and its retained failure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, read_json, verify
from train import differences

PROTOCOL = '9d0223aa48aca2b1862b78a958e1218c14f236e8afa392b08dc1fe59a6282996'
AMENDMENT = '841c76b9870f81df15a0ac855f05c3e64e7609fa81957597395eddaa00596728'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'protocol', 'amendment', 'attempt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT)
    if checkpoints.sha256(args.protocol) != PROTOCOL or checkpoints.sha256(args.amendment) != AMENDMENT:
        raise ValueError('Registered protocol or amendment differs')
    protocol, amendment = read_json(args.protocol), read_json(args.amendment)
    root, directory = args.workspace_root.resolve(), args.attempt.resolve()
    result = {'schema_version': 1, 'kind': 'fixed_input_inference_analysis', 'status': 'failed',
              'analysis_snapshot': ROOT.name, 'protocol_sha256': PROTOCOL, 'amendment_sha256': AMENDMENT,
              'claims_rollout_speedup': False, 'claims_go_strength': False, 'claims_mfu': False,
              'production_promotion': False}
    try:
        pod = read_json(directory / 'result.json')
        source = root / '.gozero/snapshots' / amendment['new_source_snapshot']
        manifest = verify(source)
        config = read_json(source / 'resolved_config.json')
        if (pod['status'] != 'passed' or pod['snapshot_id'] != source.name
                or pod['start_unix_time'] < amendment['registered_unix']
                or pod['timeout_seconds'] != amendment['budget']['timeout_seconds']):
            raise ValueError('Probe execution identity or budget differs')
        for name, expected in amendment['identical_scientific_inputs_sha256'].items():
            if checkpoints.sha256(source / name) != expected:
                raise ValueError('Scientific input changed: ' + name)
        if checkpoints.sha256(source / manifest['recipe'] / 'train.py') != checkpoints.sha256(Path(__file__).with_name('train.py')):
            raise ValueError('Analysis numerical routine differs from executed probe')
        retained = root / amendment['failed_attempt']['path']
        if checkpoints.sha256(retained) != amendment['failed_attempt']['sha256']:
            raise ValueError('Retained preparation failure differs')
        failed = read_json(retained)
        if failed['status'] != 'failed' or failed['last_phase'] != 'prepare':
            raise ValueError('Original attempt contained unregistered computation')
        cpu = root / protocol['cpu_qualification']['path']
        if checkpoints.sha256(cpu) != protocol['cpu_qualification']['sha256'] or read_json(cpu)['status'] != 'passed':
            raise ValueError('CPU prerequisite differs')
        ranks = []
        raw_comparisons = {mode: [] for mode in ('queued', 'wide')}
        expected_order = [(rep, mode) for rep, order in enumerate(config['orders']) for mode in order]
        for host in range(4):
            artifacts = directory / f'rank-{host}/artifacts'
            path = artifacts / 'result.json'
            record = read_json(path)
            if (record['status'] != 'passed' or record['snapshot_id'] != source.name
                    or record['host_rank'] != host or record['world_size'] != 4
                    or not record['no_training_performed']):
                raise ValueError('Rank completion or identity differs')
            ref = config['input_checkpoints'][host]
            if any(record['input'][k] != v for k, v in ref.items()):
                raise ValueError('Input checkpoint reference differs')
            checkpoint = root / ref['path']
            if checkpoints.sha256(checkpoint.with_suffix('.group.json')) != ref['group_sha256']:
                raise ValueError('Input checkpoint group changed')
            state, arrays, _ = checkpoints.read(checkpoint, expected_manifest_sha256=ref['manifest_sha256'])
            parameters = hashlib.sha256()
            for key in sorted(k for k in arrays if k.startswith('p_')):
                parameters.update(arrays[key].tobytes())
            if parameters.hexdigest() != record['input']['parameters_sha256']:
                raise ValueError('Evaluated model parameters differ')
            count = config['small_batch'] * config['branches']
            indices = np.random.Generator(np.random.PCG64(config['input_seed'] + host)).choice(state['replay_count'], count, replace=False)
            if indices.tolist() != record['input']['sample_indices']:
                raise ValueError('Sampled replay indices differ')
            archive = artifacts / 'predictions.npz'
            if checkpoints.sha256(archive) != record['predictions_sha256']:
                raise ValueError('Prediction archive differs')
            with np.load(archive, allow_pickle=False) as saved:
                features = saved['features']
                if (not np.array_equal(features, arrays['replay_x'][indices])
                        or not np.array_equal(saved['sample_indices'], indices)
                        or hashlib.sha256(features.tobytes()).hexdigest() != record['input']['features_sha256']):
                    raise ValueError('Predictions were not evaluated on registered observations')
                baseline = (saved['serial_logits'], saved['serial_values'])
                for mode in raw_comparisons:
                    actual = (saved[mode + '_logits'], saved[mode + '_values'])
                    observed = differences(baseline, actual, features)
                    if observed != record['comparisons'][mode]:
                        raise ValueError('Reported prediction differences do not reproduce')
                    raw_comparisons[mode].append(observed)
            del arrays
            for name in ('small', 'wide'):
                if checkpoints.sha256(artifacts / (name + '.hlo.txt')) != record['hlo_sha256'][name]:
                    raise ValueError('Compiled HLO identity differs')
            if [(t['repetition'], t['mode']) for t in record['timings']] != expected_order:
                raise ValueError('Registered timing order differs')
            for timing in record['timings']:
                mode, iterations = timing['mode'], config['timed_iterations']
                expected = {'iterations': iterations, 'active_neural_evaluations': count * iterations,
                            'padded_slots': count * iterations,
                            'forward_dispatches': iterations * (1 if mode == 'wide' else config['branches']),
                            'host_fetches': iterations * (config['branches'] if mode == 'serial' else 1)}
                if (any(timing[k] != v for k, v in expected.items())
                        or not np.isfinite(timing['elapsed_seconds']) or timing['elapsed_seconds'] <= 0
                        or not np.isfinite(timing['process_cpu_seconds']) or timing['process_cpu_seconds'] < 0):
                    raise ValueError('Timing work accounting differs')
            ranks.append({'host_rank': host, 'jax_rank': record['jax_rank'], 'result_sha256': checkpoints.sha256(path),
                          'predictions_sha256': record['predictions_sha256'], 'input': record['input'],
                          'compile_seconds': record['compile_seconds'], 'timings': record['timings']})
        if {r['jax_rank'] for r in ranks} != set(range(4)):
            raise ValueError('Incomplete TPU rank coverage')
        if not all(r['all_outputs_exact'] for r in raw_comparisons['queued']):
            raise ValueError('Queued predictions differ from serial')
        comparisons = {}
        for mode, values in raw_comparisons.items():
            comparisons[mode] = {'all_outputs_exact': all(v['all_outputs_exact'] for v in values),
                                 'positions': sum(v['positions'] for v in values),
                                 'legal_top_one_disagreements': sum(v['legal_top_one_disagreements'] for v in values),
                                 'by_host': values}
        timings = []
        for repetition in range(len(config['orders'])):
            seconds = {mode: max(next(t['elapsed_seconds'] for t in rank['timings']
                                     if t['repetition'] == repetition and t['mode'] == mode) for rank in ranks)
                       for mode in ('serial', 'queued', 'wide')}
            timings.append({'repetition': repetition, 'slowest_host_seconds': seconds,
                            'serial_over_queued': seconds['serial'] / seconds['queued'],
                            'serial_over_wide': seconds['serial'] / seconds['wide']})
        result.update(status='analyzed', attempt=directory.name, source_snapshot=source.name,
                      pod_result_sha256=checkpoints.sha256(directory / 'result.json'), ranks=ranks,
                      comparisons=comparisons, timing_repetitions=timings,
                      median_serial_over_queued=float(np.median([t['serial_over_queued'] for t in timings])),
                      median_serial_over_wide=float(np.median([t['serial_over_wide'] for t in timings])),
                      successful_attempt_chip_hours=pod['reserved_chip_hours'],
                      retained_failed_attempt_chip_hours=failed['reserved_chip_hours'],
                      total_attempt_chip_hours=pod['reserved_chip_hours'] + failed['reserved_chip_hours'],
                      limitations=protocol['limitations'])
        verify(ROOT)
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        with args.output.open('xb') as stream:
            stream.write(canonical_json(result))
        args.output.chmod(0o444)
        print(json.dumps({k: v for k, v in result.items() if k not in ('ranks', 'comparisons')}), flush=True)


if __name__ == '__main__':
    main()
