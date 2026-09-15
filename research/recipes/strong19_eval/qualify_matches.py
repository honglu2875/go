"""Serve small trained models while a separate process plays real KataGo games."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
sys.path.insert(0, str(SOURCE / 'eval'))
from gozero import joint_artifacts
from gozero.checkpoints import sha256
from gozero.joint_rpc import Server
from gozero.snapshots import canonical_json, read_json, verify
from gozero.v7_replay import Replay
from gozero.v7_state import row_digest
from joint_gtp import contract, identity
import inference
import joint


def run(args, c):
    started = time.monotonic()
    root = args.workspace_root.resolve()
    verify(SOURCE)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='trained_joint_katago_cpu_qualification', status='running', created=time.time(), snapshot=SOURCE.name)
    runner = oracle = server = pool = None
    try:
        if jax.default_backend() != 'cpu' or jax.device_count() != 4 or jax.process_count() != 1:
            raise ValueError('Require four simulated CPU devices in one process')
        candidate_path, search_path, spec_path = (SOURCE / c[key] for key in ('candidate', 'search', 'match'))
        if any(not p.resolve().is_relative_to(SOURCE) for p in (candidate_path, search_path, spec_path)):
            raise ValueError('Candidate/search/match must be frozen with the recipe')
        descriptor = read_json(candidate_path)
        state, params = joint_artifacts.load_candidate(root, descriptor)
        if state['training_purpose'] != 'qualification':
            raise ValueError('CPU execution fixture expected')
        net = state['model']
        joint_artifacts.compatible(state, Path(__file__).parent, schema=joint.parameter_schema(net, state['value_model']))
        search = contract(read_json(search_path))
        spec = read_json(spec_path)
        if any(spec[key] != c[key] for key in ('candidate', 'search')) or spec['purpose'] != 'qualification':
            raise ValueError('Match and service contracts differ')
        inputs = state['input_contract']
        size, komi = inputs['size'], inputs['komi']
        if search['size'] != size or search['komi'] != komi or search['cache_positions'] > net['max_positions']:
            raise ValueError('Search input/context differs')
        evidence = root / c['feature_equivalence']['path']
        if sha256(evidence) != c['feature_equivalence']['sha256']:
            raise ValueError('Feature equivalence changed')
        feature = root / c['feature_binary']['path']
        feature_sha = c['feature_binary']['sha256']
        joint_artifacts.compatible_features(state, feature_sha, evidence=read_json(evidence))
        runner = inference.Runner(params, net, Replay(feature, feature_sha, size=size, komi=komi),
                                  slots=4, network_version=state['network_version'], max_block=4)
        oracle = Replay(feature, feature_sha, size=size, komi=komi)
        forward = jax.jit(lambda p, b: joint.forward(p, b, net))
        comparisons, maximum, audited = [], {}, set()
        audit_lengths = set(c['full_prediction_audit_plies'])

        class AuditedRunner:
            slots = 4

            def __init__(self):
                self.size = size

            def score(self, requests):
                actual = runner.score(requests)
                for slot, (history, digest) in requests.items():
                    if len(history) not in audit_lengths or digest in audited:
                        continue
                    row = oracle([history])[0]
                    if row_digest(row, history, size=size, komi=komi) != digest:
                        raise ValueError('Native leaf digest differs from independent V7 replay')
                    cnn = net['architecture'] == 'katago_nested_policy'
                    count = 1 if cnn else len(history) + 1
                    length = min(net['max_positions'], 1 << (count - 1).bit_length())
                    if count > length:
                        raise ValueError('Audit history exceeds model context')
                    batch = dict(spatial=np.zeros((1, length, size, size, 22), np.float32),
                                 global_features=np.zeros((1, length, 19), np.float32),
                                 actions=np.zeros((1, length), np.int32), counts=np.array([count], np.int32))
                    batch['spatial'][0, :count] = row['spatial'][-1:] if cnn else row['spatial']
                    batch['global_features'][0, :count] = row['global_features'][-1:] if cnn else row['global_features']
                    if not cnn:
                        batch['actions'][0, :len(history)] = history
                    full = jax.device_get(forward(params, jax.tree.map(jnp.asarray, batch)))
                    for key in ('policy', 'value_logits', 'value'):
                        expected, observed = full[key][0, count - 1], actual[slot][key]
                        maximum[key] = max(maximum.get(key, 0.), float(np.max(np.abs(observed - expected))))
                        np.testing.assert_allclose(observed, expected, atol=5e-5, rtol=5e-4)
                    comparisons.append(dict(slot=slot, plies=len(history), state_digest=digest))
                    audited.add(digest)
                return actual

        socket = Path('/tmp') / ('gozero-match-' + hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:20] + '.sock')
        server = Server(socket, identity(SOURCE, candidate_path, search_path), AuditedRunner(),
                        gather_seconds=0., request_timeout=120.)

        def panel():
            argv = [sys.executable, '-B', str(SOURCE / 'eval/joint_match.py'), '--spec', str(spec_path),
                    '--artifacts-root', str(root), '--output', str(args.output / 'panel'), '--socket', str(socket)]
            with (args.output / 'panel.log').open('x') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
                try:
                    return child.wait(timeout=c['maximum_seconds'])
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise

        pool = ThreadPoolExecutor(1)
        future = pool.submit(panel)
        deadline = time.monotonic() + c['maximum_seconds'] + 30
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError('KataGo CPU qualification exceeded its bound')
            server.pump()
        code = future.result()
        path = args.output / 'panel/result.json'
        result = read_json(path)
        if code or result['status'] != 'passed' or not comparisons:
            raise ValueError('KataGo panel execution failed or model predictions were not checked')
        report.update(status='passed', architecture=net['architecture'], panel_sha256=sha256(path),
                      summary=result['summary'], complete_panel=result['complete_panel'],
                      checked_positions=sum(g['checked_positions'] for g in result['games']),
                      comparisons=comparisons, maximum_absolute_errors=maximum, runner_stats=runner.stats,
                      batches=server.records, compilation=runner.compilation,
                      scope='Small trained models against real pinned KataGo with paired colors, board/score checks and sampled full-vs-cached predictions. This tests interoperability, not architecture strength or TPU speed.')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if server is not None:
            server.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if runner is not None:
            runner.close()
        if oracle is not None:
            oracle.close()
        report['seconds'] = time.monotonic() - started
        path = args.output / 'result.json'
        with path.open('xb') as stream:
            stream.write(canonical_json(report))
        path.chmod(0o444)
        print(json.dumps(dict(status=report['status'], seconds=report['seconds'], sha256=sha256(path))), flush=True)
