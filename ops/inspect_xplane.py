"""Bounded protobuf-wire inspection of raw XPlane traces without TensorFlow.

Reads the public XSpace/XPlane schema. Unknown wire fields are skipped; this
operator reports metadata and event statistics, not inferred hardware utilization.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import re


def varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('Truncated protobuf integer')
        byte = data[position]; position += 1; value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise ValueError('Overlong protobuf integer')


def fields(data):
    position = 0
    while position < len(data):
        tag, position = varint(data, position); field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ValueError('Invalid protobuf field zero')
        if wire == 0:
            value, position = varint(data, position)
        else:
            if wire == 2:
                length, position = varint(data, position)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError('Unsupported protobuf wire type')
            if position + length > len(data):
                raise ValueError('Truncated protobuf field')
            value = data[position:position + length]; position += length
        yield field, wire, value


def string(value):
    return bytes(value).decode('utf-8', errors='replace')


def metadata(entry):
    values = {key: value for key, _, value in fields(entry)}
    return int(values.get(1, 0)), list(fields(values[2]))


def statistic(entry, names):
    values = {key: value for key, _, value in fields(entry)}; key = names.get(values.get(1, 0), '?')
    if 2 in values:
        value = struct.unpack('<d', values[2])[0]
    elif 3 in values:
        value = values[3]
    elif 4 in values:
        value = values[4] if values[4] < 2**63 else values[4] - 2**64
    elif 5 in values:
        value = string(values[5])[:500]
    elif 6 in values:
        value = {'bytes': len(values[6])}
        if key == 'core_details' and len(values[6]) <= 256:
            value['hex'] = bytes(values[6]).hex()
    elif 7 in values:
        value = names.get(values[7], '?ref-' + str(values[7]))
    else:
        value = None
    return key, value


def inspect_plane(data, scan_events, line_pattern):
    top = list(fields(data)); name = next((string(v) for k, _, v in top if k == 2), '')
    stat_names, event_names, constant_stats = {}, {}, {}
    for key, _, value in top:
        if key in (4, 5):
            identifier, raw = metadata(value); entry = {k: v for k, _, v in raw}
            (event_names if key == 4 else stat_names)[identifier] = string(entry.get(2, b''))
            if key == 4:
                constant_stats[identifier] = [v for k, _, v in raw if k == 5]
    constant_stats = {k: dict(statistic(v, stat_names) for v in values) for k, values in constant_stats.items()}
    selected_names = {k: v for k, v in stat_names.items() if any(word in v.lower() for word in
        ('flop', 'util', 'mxu', 'bandwidth', 'cycle', 'stall', 'counter', 'clock', 'peak', 'bytes'))}
    result = {'name': name, 'event_metadata_entries': len(event_names), 'stat_metadata_entries': len(stat_names),
        'stat_names': list(stat_names.values()), 'plane_stats': [statistic(v, stat_names) for k, _, v in top if k == 6],
        'selected_stat_names': selected_names, 'lines': [], 'selected_event_stats': []}
    counts = Counter(); durations = Counter(); stat_counts = Counter(); total = 0; programs = {}
    for key, _, value in top:
        if key != 3:
            continue
        line = {}; events = []
        for tag, _, item in fields(value):
            if tag in (1, 3, 9, 10):
                line[str(tag)] = item
            elif tag in (2, 11):
                line['name' if tag == 2 else 'display_name'] = string(item)
            elif tag == 4:
                events.append(item)
        line['events'] = len(events); total += len(events); result['lines'].append(line)
        if not scan_events or not re.search(line_pattern, line.get('name', '')):
            continue
        line_counts = Counter(); line_durations = Counter(); totals = Counter(); first = None; last = None
        occurrence_values = Counter()
        for event in events:
            identifier = 0; duration = 0; offset = 0; stats = {}; occurrences = 1
            for tag, _, item in fields(event):
                if tag == 1:
                    identifier = item
                elif tag == 2:
                    offset = item
                elif tag == 3:
                    duration = item
                elif tag == 4:
                    stat = statistic(item, stat_names); stats[stat[0]] = stat[1]; stat_counts[stat[0]] += 1
                elif tag == 5:
                    occurrences = max(1, item)
            occurrence_values[occurrences] += 1
            stats = {**constant_stats.get(identifier, {}), **stats}
            label = event_names.get(identifier, str(identifier)); counts[label] += 1; durations[label] += duration
            program = str(stats.get('program_id', '?'))
            if line.get('name') == 'XLA Modules':
                match = re.search(r'\(([0-9]+)\)$', label)
                if match:
                    program = match.group(1)
            if program != '?':
                summary = programs.setdefault(program, {'module_names': [], 'module_occurrences': 0,
                    'summed_module_duration_ps': 0, 'leaf_operation_flops': 0, 'leaf_operation_model_flops': 0,
                    'container_operation_flops_excluded': 0, 'category_duration_ps': Counter(),
                    'category_model_flops': Counter(), 'category_event_count': Counter()})
                if line.get('name') == 'XLA Modules':
                    if label not in summary['module_names']:
                        summary['module_names'].append(label)
                    summary['module_occurrences'] += 1; summary['summed_module_duration_ps'] += duration
                elif line.get('name') == 'XLA Ops':
                    category = stats.get('hlo_category', '?')
                    summary['category_duration_ps'][category] += duration
                    summary['category_model_flops'][category] += stats.get('model_flops', 0)
                    summary['category_event_count'][category] += 1
                    if category.lower() in ('while', 'call', 'conditional'):
                        summary['container_operation_flops_excluded'] += stats.get('model_flops', 0)
                    else:
                        summary['leaf_operation_flops'] += stats.get('flops', 0)
                        summary['leaf_operation_model_flops'] += stats.get('model_flops', 0)
            line_counts[label] += 1; line_durations[label] += duration
            first = offset if first is None else min(first, offset); last = offset + duration if last is None else max(last, offset + duration)
            for metric in ('flops', 'model_flops', 'bytes_accessed', 'raw_bytes_accessed'):
                if isinstance(stats.get(metric), (float, int)):
                    totals[metric] += stats[metric]
            if any(k in selected_names.values() for k in stats) and len(result['selected_event_stats']) < 100:
                result['selected_event_stats'].append({'name': label, 'duration_ps': duration, 'stats': stats})
        line.update(first_event_offset_ps=first, last_event_end_ps=last, additive_stat_totals=totals, occurrence_values=occurrence_values,
                    common_events=line_counts.most_common(30), largest_summed_durations_ps=line_durations.most_common(30))
    result.update(total_events=total, common_events=counts.most_common(50), event_stat_counts=stat_counts,
                  largest_summed_event_durations_ps=durations.most_common(50), program_summaries=programs)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scan-events', action='store_true')
    p.add_argument('--line-pattern', default='XLA Modules|XLA Ops|Steps|_counters_'); a = p.parse_args()
    if a.output.exists() or a.trace.stat().st_size > 2**31:
        raise ValueError('Output exists or trace exceeds the two-GiB inspection bound')
    data = memoryview(a.trace.read_bytes()); planes = []; messages = []
    for key, _, value in fields(data):
        if key == 1:
            planes.append(inspect_plane(value, a.scan_events, a.line_pattern))
        elif key in (2, 3, 4):
            messages.append((key, string(value)))
    result = {'schema_version': 1, 'kind': 'raw_xplane_schema_inspection', 'trace': str(a.trace.resolve()),
        'trace_sha256': hashlib.sha256(data).hexdigest(), 'operator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'schema': 'https://raw.githubusercontent.com/openxla/xla/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto',
        'scanned_events': a.scan_events, 'line_pattern': a.line_pattern, 'messages': messages, 'planes': planes, 'claims_mfu': False,
        'limitations': 'Summed event durations can overlap and are not utilization. Stat names alone do not establish performance-counter coverage.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    print(json.dumps({'planes': [{'name': p['name'], 'events': p['total_events'], 'stats': p['selected_stat_names']} for p in planes], 'messages': messages}))


if __name__ == '__main__':
    main()
