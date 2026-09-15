"""Report compiler-accounted arithmetic rates from one-chip raw TPU traces."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--inspection', type=Path, required=True)
    p.add_argument('--inspection-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device-kind', required=True, help='Exact expected device type in the trace; configure for the target environment')
    a = p.parse_args(); verify(SOURCE)
    if a.output.exists() or sha256(a.inspection) != a.inspection_sha256: raise ValueError('Inspection differs or output exists')
    data = read_json(a.inspection)
    if sha256(Path(data['trace'])) != data['trace_sha256']: raise ValueError('Raw trace changed')
    devices = [p for p in data['planes'] if p['name'].startswith('/device:TPU:')]
    if len(devices) != 2: raise ValueError('Expected the two tensor-core planes of the configured single-chip trace')
    result = []
    for plane in devices:
        stats = dict(plane['plane_stats']); programs = []
        if stats['has_megacore'] != 1 or stats['device_type_string'] != a.device_kind: raise ValueError('Unexpected trace device')
        for line in plane['lines']:
            if line.get('name') in ('XLA Ops', 'XLA Modules') and line['occurrence_values'] != {'1': line['events']}:
                raise ValueError('Aggregated events need explicit occurrence accounting')
        for identifier, program in plane['program_summaries'].items():
            if program['module_occurrences'] != 16 or len(program['module_names']) != 1: raise ValueError('Unexpected profile repetition coverage')
            flops = program['leaf_operation_model_flops']; duration = program['summed_module_duration_ps']
            rate = flops / duration
            programs.append({'program_id': identifier, 'name': program['module_names'][0], 'occurrences': 16,
                'summed_module_seconds': duration * 1e-12, 'compiler_leaf_model_flops': flops,
                'compiler_model_flops_per_occurrence': flops / 16,
                'excluded_container_model_flops': program['container_operation_flops_excluded'],
                'compiler_accounted_teraflops_per_second': rate,
                'fraction_of_recorded_plane_peak': rate / stats['peak_teraflops_per_second']})
        result.append({'plane': plane['name'], 'core_details': stats['core_details'],
            'peak_teraflops_per_second': stats['peak_teraflops_per_second'], 'programs': programs})
    report = {'schema_version': 1, 'kind': 'visual_raw_xplane_kernel_arithmetic_accounting', 'status': 'passed',
        'operator_snapshot': SOURCE.name, 'inspection_sha256': a.inspection_sha256, 'trace_sha256': data['trace_sha256'],
        'devices': result, 'claims_end_to_end_mfu': False, 'hardware_counter_timeseries_available': False,
        'method': 'Sum per-occurrence compiler model_flops on XLA Ops, excluding while/call/conditional containers. Divide by enclosing module time and the same plane peak. All observed events have occurrence count one; fused HLO arithmetic remains included once.',
        'scope': 'One profiled chip; 16 supplied eight-move block calls and 16 supplied eight-step sequential calls. This is compiler-accounted dense arithmetic during traced kernels. Padding and masked attention remain in the numerator; host dispatch, drafting, Go transitions, queueing and acceptance are excluded.',
        'interpretation': 'This ratio is an arithmetic-rate diagnostic, not a measured MXU busy counter or end-to-end useful-model MFU. It must not be extrapolated to the complete pod or used as speculative rollout throughput.',
        'sources': ['https://raw.githubusercontent.com/openxla/xla/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto',
                    'https://raw.githubusercontent.com/openxla/xprof/master/xprof/utils/op_metrics_db_utils.cc',
                    'https://raw.githubusercontent.com/openxla/xprof/master/xprof/convert/xplane_to_op_metrics_db.cc']}
    a.output.write_bytes(canonical_json(report))
    print(canonical_json({'status': 'passed', 'devices': result}).decode().strip())


if __name__ == '__main__': main()
