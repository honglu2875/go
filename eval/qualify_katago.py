#!/usr/bin/env python3
"""Run official KataGo self-play through the independent Rust rules endpoint.

This qualifies real search/weights/transport/rules; it is not a strength benchmark
of a learned gozero model. Execute this file from a verified source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.gtp import GTPClient
from gozero.snapshots import verify


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def kata_cells(text: str, size: int) -> str:
    rows = {}
    for line in text.splitlines():
        match = re.match(r'^\s*(\d+)\s+(.+)$', line)
        if match:
            cells = ''.join(c for c in match.group(2) if c in '.XO')
            if len(cells) == size:
                rows[int(match.group(1))] = cells
    if set(rows) != set(range(1, size+1)):
        raise ValueError('Could not parse the complete KataGo board: '+text)
    return '\n'.join(rows[row] for row in range(size, 0, -1))


def sgf_vertex(vertex: str, size: int):
    if vertex.lower() == 'pass':
        return ''
    col = ord(vertex[0].upper())-ord('A')
    col -= vertex[0].upper() > 'I'
    row = size-int(vertex[1:])
    if not (0 <= col < size and 0 <= row < size):
        raise ValueError('Invalid SGF move')
    return chr(ord('a')+col)+chr(ord('a')+row)


def run(args):
    snapshot = args.snapshot.resolve()
    if SOURCE != snapshot:
        raise ValueError('Execute eval/qualify_katago.py from the specified frozen snapshot')
    verify(snapshot)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'schema_version': 1, 'kind': 'katago_rules_qualification', 'claims_go_strength': False,
              'snapshot_id': snapshot.name, 'status': 'running', 'board_size': 9, 'komi': 7.5,
              'moves': [], 'started_unix': time.time()}
    try:
        engine = json.loads((snapshot/'eval/katago_build.json').read_text())
        weights = json.loads((snapshot/'eval/katago_9x9.json').read_text())
        root = args.artifacts_root.resolve()
        binary = root/engine['binary_path']
        model = root/weights['path']
        for path, expected in [(binary, engine['binary_sha256']), (model, weights['sha256'])]:
            if digest(path) != expected:
                raise ValueError('Artifact hash mismatch: '+str(path))
        config = snapshot/'eval/katago_9x9_smoke.cfg'
        result.update(engine_sha256=engine['binary_sha256'], model_sha256=weights['sha256'],
                      config_sha256=digest(config))
        # Build exactly this snapshot, without modifying its source or lockfile.
        environment = {**os.environ, 'RUSTUP_HOME': str(root/'.gozero/rustup'),
                       'CARGO_HOME': str(root/'.gozero/cargo')}
        cargo = root/'.gozero/cargo/bin/cargo'
        argv = [str(cargo), 'build', '--locked', '--release', '-p', 'go-gtp',
                '--manifest-path', str(snapshot/'Cargo.toml'), '--target-dir', str(output/'native')]
        result['rust_build_argv'] = argv
        with (output/'rust-build.log').open('w') as log:
            subprocess.run(argv, env=environment, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        rust = output/'native/release/go-gtp'
        result['rust_binary_sha256'] = digest(rust)
        result['rust_version'] = subprocess.check_output([str(root/'.gozero/cargo/bin/rustc'), '--version'],
                                                        env=environment, cwd=snapshot, text=True).strip()
        kata_argv = ['taskset', '-c', ','.join(str(i) for i in sorted(os.sched_getaffinity(0))[:8]),
                     str(binary), 'gtp', '-model', str(model), '-config', str(config)]
        deadline = time.monotonic()+args.timeout
        def command(client, text):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Qualification game deadline expired')
            return client.command(text, timeout=min(remaining, 60))
        with GTPClient(kata_argv, output/'katago') as kata, GTPClient([str(rust)], output/'rules') as rules:
            result['kata_name'] = command(kata, 'name')
            result['kata_version'] = command(kata, 'version')
            for client in (kata, rules):
                command(client, 'boardsize 9'); command(client, 'clear_board'); command(client, 'komi 7.5')
            rules_config = json.loads(command(kata, 'kata-get-rules'))
            result['katago_rules'] = rules_config
            if rules_config['ko'] != 'POSITIONAL' or rules_config['scoring'] != 'AREA' or not rules_config['suicide']:
                raise ValueError('KataGo rules differ from the required profile')
            passes = 0
            for ply in range(324):
                color = 'B' if ply % 2 == 0 else 'W'
                before = time.monotonic()
                move = command(kata, 'genmove '+color).strip()
                if move.lower() == 'resign':
                    raise ValueError('Resignation is disabled in this qualification')
                command(rules, 'play '+color+' '+move)
                actual = command(rules, 'showboard')
                expected = kata_cells(command(kata, 'showboard'), 9)
                if actual != expected:
                    raise AssertionError('Board mismatch at ply %d\n%s\n%s' % (ply, actual, expected))
                result['moves'].append({'color': color, 'vertex': move, 'seconds': time.monotonic()-before})
                passes = passes+1 if move.lower() == 'pass' else 0
                if passes == 2:
                    result['rust_score'] = command(rules, 'final_score')
                    result['kata_score'] = command(kata, 'final_score')
                    if result['rust_score'] != result['kata_score']:
                        raise AssertionError('Strict terminal score differs: '+str((result['rust_score'], result['kata_score'])))
                    result['status'] = 'passed'
                    break
            else:
                result['status'] = 'truncated'
        verify(snapshot)
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        result['finished_unix'] = time.time()
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        re_prop = 'RE[%s]' % result['rust_score'] if result['status'] == 'passed' else ''
        sgf = '(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RU[Tromp-Taylor]AP[gozero:qualification]'+re_prop
        sgf += ''.join(';'+m['color']+'['+sgf_vertex(m['vertex'], 9)+']' for m in result['moves'])+')\n'
        (output/'game.sgf').write_text(sgf)
        print(json.dumps({k:v for k,v in result.items() if k not in ('moves', 'rust_build_argv')}, sort_keys=True), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=240)
    raise SystemExit(run(parser.parse_args()))
