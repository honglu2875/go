#!/usr/bin/env python3
"""Stream large Chrome TPU traces into a bounded, hash-pinned schema summary."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


def events(path):
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt') as stream:
        buffer = ''; position = 0
        while True:
            more = stream.read(1024 * 1024)
            if not more:
                raise ValueError('Trace event array header missing')
            buffer += more
            found = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if found:
                position = found.end(); break
            if len(buffer) > 16 * 1024 * 1024 or not buffer:
                raise ValueError('Trace event array header missing')
        while True:
            while position < len(buffer) and buffer[position] in ' \n\r\t,':
                position += 1
            if position < len(buffer) and buffer[position] == ']':
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(1024 * 1024)
                if not more:
                    raise ValueError('Truncated trace event array')
                buffer = buffer[position:] + more; position = 0
                if len(buffer) > 64 * 1024 * 1024:
                    raise ValueError('Trace event exceeds bounded parser buffer')
                continue
            position = end
            yield value
            if position > 2 * 1024 * 1024:
                buffer = buffer[position:]; position = 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    phases, names, counters, tracks, samples, selected, regions = Counter(), Counter(), {}, {}, {}, [], []
    for event in events(a.trace):
        phase, name = event.get('ph', '?'), event.get('name', '')
        phases[phase] += 1; names[name[:160]] += 1
        if len(samples.setdefault(phase, [])) < 6:
            samples[phase].append(event)
        if phase == 'M':
            tracks[f"{event.get('pid')}:{event.get('tid')}:{name}"] = event.get('args')
        if phase == 'C':
            record = counters.setdefault(name, {'events': 0, 'examples': []})
            record['events'] += 1
            if len(record['examples']) < 4:
                record['examples'].append(event)
        if name.startswith('visual_supplied_path_'):
            regions.append(event)
        if len(selected) < 100 and any(x in name.lower() for x in ['mxu', 'utilization', 'flop', 'bandwidth', 'stall', 'cycle']):
            selected.append(event)
    with a.trace.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    result = {'schema_version': 1, 'kind': 'compute_trace_schema_inspection', 'trace': str(a.trace.resolve()),
              'operator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'trace_sha256': digest, 'phases': phases, 'common_names': names.most_common(40),
              'counter_types': counters, 'tracks': tracks, 'phase_examples': samples,
              'selected_compute_examples': selected, 'marked_regions': regions,
              'claims_mfu': False, 'scope': 'Schema inspection only; instruction names or timeline occupancy do not establish utilization.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    print(json.dumps({'events': sum(phases.values()), 'phases': phases, 'counters': list(counters), 'regions': regions, 'output': str(a.output)}))


if __name__ == '__main__':
    main()
