"""Real-engine qualification of continuous admission, restart and both thread splits."""
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    source = Path(__file__).resolve().parent
    d = json.loads((source/'deployment.json').read_text())
    root, env = Path(d['root']), Path(d['environment'])
    sys.path.insert(0, str(env/'site-packages'))
    from flygo.data.corpus import audit_game, atomic_json, identity
    from flygo.data.generate import run_worker
    previous = json.loads((source.parent/'katago_corpus/qualification-result.json').read_text())
    old = json.loads((root/'environments'/previous['producer_snapshot']/'snapshot.json').read_text())
    new = json.loads((env/'snapshot.json').read_text())
    assert previous['status'] == 'passed' and previous['board_size'] == 19
    assert new['native_sha256'] == old['native_sha256'] == previous['native_sha256']
    for name in ('go.py', 'data/katago.py', 'data/label.py', 'storage.py', 'runtime.py'):
        assert old['source_sha256'][name] == new['source_sha256'][name] == sha(env/'site-packages/flygo'/name)
    for name, digest in new['source_sha256'].items():
        assert sha(env/'site-packages/flygo'/name) == digest
    result_path = source/'qualification-result-001.json'
    assert not result_path.exists(), 'Never overwrite qualification evidence'
    qroot = root/'continuous-v2-qualification-001'
    qroot.mkdir(exist_ok=False)
    (qroot/'artifacts').symlink_to(root/'artifacts', target_is_directory=True)
    contract = {**json.loads(Path(d['contract']).read_text()), 'max_moves': 4,
                'qualification_only': 'bounded truncated games, excluded from production'}
    common = dict(storage_root=str(qroot), host_index=0, concurrent_games=8,
                  producer_snapshot=d['snapshot'], contract=contract,
                  storage_limits=dict(files_cap=24*(1 << 30), free_files_floor=64*(1 << 30),
                                      available_memory_floor=96*(1 << 30)))
    started = time.time()
    cases = []
    for opponent_index, teacher_threads, opponent_threads in ((0, 7, 1), (7, 4, 4)):
        run_id = f'continuous-v2-opponent-{opponent_index:02d}'
        # Distinct worker indices prevent identity collisions between these two
        # qualification strata; production uses exactly one worker per stratum.
        workers = [dict(cpus=list(range(8)), opponent_index=i,
                        teacher_threads=teacher_threads, opponent_threads=opponent_threads)
                   for i in range(opponent_index+1)]
        config = {**common, 'run_id': run_id, 'workers': workers}
        config_path = qroot/(run_id+'.json'); atomic_json(config_path, config)
        case_started = time.time()
        run_worker(config_path, opponent_index, max_games=10)
        output = qroot/'runs'/run_id/f'worker-{opponent_index:02d}'
        first = json.loads((output/'status.json').read_text())
        assert first['state'] == 'stopped' and first['stream_result'] == dict(submitted=10, completed=10)
        assert first['next_sequence'] == first['completed_games'] == 10 and first['positions'] == 40
        # Reopen the same worker and verify persisted sequence, immutable prior
        # publications and accumulated counters through the actual implementation.
        directory = qroot/'corpora'/identity(contract)/'host-0'/f'worker-{opponent_index:02d}'
        before = {p.name: sha(p) for p in directory.glob('*.npz')}
        run_worker(config_path, opponent_index, max_games=2)
        final = json.loads((output/'status.json').read_text())
        assert final['state'] == 'stopped' and final['stream_result'] == dict(submitted=2, completed=2)
        assert final['next_sequence'] == final['completed_games'] == 12 and final['positions'] == 48
        assert final['in_flight_games'] == final['in_flight_positions'] == 0
        assert list((output/'session-history').glob('*.json'))
        assert all(sha(directory/name) == digest for name, digest in before.items())
        records = [audit_game(p) for p in sorted(directory.glob('*.npz'))]
        assert len(records) == 12 and all(r['rows'] == 4 and not r['terminal'] for r in records)
        assert all(not p.stat().st_mode & 0o222 for p in directory.glob('*.npz'))
        for role, threads in (('teacher', teacher_threads), ('opponent', opponent_threads)):
            assert f'numEigenThreadsPerModel = {threads}\n' in (output/role/'analysis.cfg').read_text()
        identities = json.loads((output/'model-identities.json').read_text())
        assert identities['teacher']['models'][0]['internalName'] == 'kata1-tf3-b11c768-s11500M-d6163M'
        cases.append(dict(opponent_index=opponent_index, teacher_threads=teacher_threads,
                          opponent_threads=opponent_threads, games=12, positions=48,
                          restart_preserved_publications=True, next_sequence=12,
                          elapsed_seconds=time.time()-case_started, state='stopped'))
        print(json.dumps(dict(kind='case_passed', **cases[-1])), flush=True)
    result = dict(status='passed', board_size=19, producer_snapshot=d['snapshot'],
                  native_sha256=new['native_sha256'], source_sha256=new['source_sha256'],
                  operator_sha256=sha(Path(__file__)), prior_model_gate_sha256=sha(source.parent/'katago_corpus/qualification-result.json'),
                  cases=cases, games=24, positions=96, elapsed_seconds=time.time()-started,
                  started=started, completed=time.time(), qualification_contract_isolated=True,
                  cpu_scope='Shared generation CPUs 0-7 for this bounded qualification; exclude interval from throughput comparisons')
    with result_path.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    result_path.chmod(0o444)
    print(json.dumps(dict(status='passed', output=str(result_path), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
