"""Stage frozen source, prepare locked environments, run one bounded pod attempt."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--workspace-root', type=Path,
                        help='Local artifact root; required when invoking a frozen controller')
    parser.add_argument('--remote-root', default='/workspace/go')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--prepare-timeout', type=int, default=600)
    parser.add_argument('--controller-cpus', type=int, default=16)
    parser.add_argument('--controller-cpu-list', help='Explicit comma-separated CPUs on every host')
    parser.add_argument('--chips', type=int, default=16)
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--resume-attempt')
    parser.add_argument('--resume-turn', type=int)
    parser.add_argument('--stop-after-turn', type=int)
    args = parser.parse_args()
    controller_source = Path(__file__).resolve().parents[1]
    root = (args.workspace_root or controller_source).resolve()
    sys.path.insert(0, str(controller_source / 'packages/gozero/src'))
    from gozero.pod import SSH_OPTIONS, load_hosts, pdsh_command, pdsh_environment, supervise
    from gozero.snapshots import canonical_json, verify
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.remote_root) or '..' in Path(args.remote_root).parts:
        parser.error('Use an absolute remote path without shell metacharacters or parent traversal')
    if min(args.timeout, args.prepare_timeout, args.controller_cpus, args.chips) <= 0:
        parser.error('Timeouts, controller CPU count, and chip count must be positive')
    if args.controller_cpu_list is not None:
        from run_host import select_controller_cpus
        try:
            requested = select_controller_cpus(range(65536), args.controller_cpus, args.controller_cpu_list)
        except ValueError as error:
            parser.error(str(error))
    snapshot = args.snapshot.resolve()
    manifest = verify(snapshot)
    if (root.is_relative_to(snapshot)
            or root.is_relative_to(controller_source) and (controller_source / 'manifest.json').is_file()):
        parser.error('Artifact root must be outside frozen source; provide --workspace-root')
    if not root.is_dir():
        parser.error('Workspace root must be an existing directory')
    if (args.resume_attempt is None) != (args.resume_turn is None):
        parser.error('--resume-attempt and --resume-turn must be provided together')
    if args.resume_attempt is not None:
        if not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}', args.resume_attempt) or args.resume_turn < 1:
            parser.error('Invalid resume attempt or turn')
        previous = json.loads((root / 'runs' / args.resume_attempt / 'launch.json').read_text())
        if previous['snapshot_id'] != manifest['snapshot_id']:
            parser.error('Resume must use the original source and resolved configuration')
    if args.stop_after_turn is not None and args.stop_after_turn < 1:
        parser.error('Stop turn must be positive')
    native = None
    recipe = json.loads((snapshot / manifest['recipe'] / 'recipe.json').read_text())
    if recipe.get('requires_native') and args.native_receipt is None:
        parser.error('This recipe requires --native-receipt from a qualified snapshot build')
    if args.native_receipt is not None:
        args.native_receipt = args.native_receipt.resolve()
        native = json.loads(args.native_receipt.read_text())
        if native['snapshot_id'] != manifest['snapshot_id'] or native['filename'] != 'lib_gozero_native.so':
            parser.error('Native receipt source or filename differs from the launch contract')
        with (args.native_receipt.parent / native['filename']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != native['binary_sha256']:
                parser.error('Native binary hash mismatch')
    hosts = load_hosts(snapshot / 'ops/hosts.json')
    uv = Path(shutil.which('uv') or '/home/go-user/.local/bin/uv').resolve()
    runtime_key = hashlib.sha256((snapshot / 'uv.lock').read_bytes() + (snapshot / '.python-version').read_bytes() + b'\ntpu\n' + uv.read_bytes()).hexdigest()
    remote = Path(args.remote_root)
    remote_snapshot = remote / '.gozero/snapshots' / manifest['snapshot_id']
    remote_native = remote / '.gozero/native' / manifest['snapshot_id']
    remote_uv = remote / '.gozero/tools' / ('uv-' + hashlib.sha256(uv.read_bytes()).hexdigest())
    environment = remote / '.gozero/environments' / runtime_key
    attempt_id = datetime.datetime.now(datetime.timezone.utc).strftime('pod-%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    attempt = root / 'runs' / attempt_id
    remote_attempt = remote / 'runs' / attempt_id
    attempt.mkdir(parents=True)
    start = time.time()
    launch = {'schema_version': 1, 'attempt_id': attempt_id, 'snapshot_id': manifest['snapshot_id'],
              'runtime_key': runtime_key, 'hosts': [h.ssh for h in hosts], 'start_unix_time': start,
              'timeout_seconds': args.timeout, 'prepare_timeout_seconds': args.prepare_timeout,
              'controller_cpus': args.controller_cpus, 'reserved_chips': args.chips}
    if args.controller_cpu_list is not None:
        launch['controller_cpu_list'] = requested
    peer_cancel = (snapshot / 'ops/cancel_host.py').is_file()
    launch.update(peer_cancel_protocol='attempt-token-v1' if peer_cancel else 'legacy-bounded-timeout',
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  supervisor_sha256=hashlib.sha256((controller_source/'packages/gozero/src/gozero/pod.py').read_bytes()).hexdigest())
    if native is not None:
        launch['native'] = native
    launch.update(resume_attempt=args.resume_attempt, resume_turn=args.resume_turn, stop_after_turn=args.stop_after_turn)
    (attempt / 'launch.json').write_bytes(canonical_json(launch))
    print(json.dumps({'kind': 'pod_attempt', 'attempt': str(attempt), **launch}), flush=True)
    def stage(host):
        commands = [
            ['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['mkdir', '-p', str(remote_snapshot), str(remote_uv.parent)])],
            ['rsync', '-a', '--checksum', '-e', shlex.join(['ssh', *SSH_OPTIONS]), str(snapshot) + '/', host.ssh + ':' + str(remote_snapshot) + '/'],
            ['rsync', '-a', '--checksum', '-e', shlex.join(['ssh', *SSH_OPTIONS]), str(uv), host.ssh + ':' + str(remote_uv)],
        ]
        if native is not None:
            commands.extend([
                ['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['mkdir', '-p', str(remote_native)])],
                ['rsync', '-a', '--checksum', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
                 str(args.native_receipt), str(args.native_receipt.parent / native['filename']), host.ssh + ':' + str(remote_native) + '/'],
            ])
        with (attempt / ('stage-rank-%d.log' % host.rank)).open('w') as log:
            for command in commands:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    phase = 'stage'
    returncode = 1
    failure = None
    try:
        config = json.loads((snapshot / 'resolved_config.json').read_text())
        candidate_name = config.get('candidate') or (config.get('fork') or {}).get('candidate')
        if isinstance(candidate_name, str):
            candidate = snapshot / candidate_name
            if not candidate.resolve().is_relative_to(snapshot):
                raise ValueError('Inference candidate must be inside frozen source')
            descriptor = json.loads(candidate.read_text())
            if descriptor.get('kind') == 'visual_causal_checkpoint':
                phase = 'candidate_inputs'
                if remote != root:
                    raise ValueError('Visual candidate paths require identical local and remote artifact roots')
                argv = [sys.executable, '-B', str(controller_source / 'ops/stage_visual_inputs.py'),
                    '--workspace-root', str(root), '--kind', 'candidate', '--manifest', str(candidate),
                    '--expected-sha256', hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    '--output', str(attempt / 'candidate_inputs')]
                with (attempt / 'candidate_inputs.log').open('w') as log:
                    subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        dataset = config.get('dataset')
        if (isinstance(dataset, dict) and 'path' in dataset and 'manifest_sha256' in dataset
                and Path(dataset['path']).is_relative_to('/dev/shm/gozero-datasets')):
            phase = 'ram_dataset_inputs'
            argv = [sys.executable, '-B', str(controller_source / 'ops/stage_ram_corpus.py'),
                '--dataset', dataset['path'], '--expected-sha256', dataset['manifest_sha256'],
                '--output', str(attempt/'dataset_inputs')]
            with (attempt/'dataset_inputs.log').open('w') as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        elif isinstance(dataset, dict) and 'path' in dataset and 'manifest_sha256' in dataset:
            from gozero.model_artifacts import artifact
            dataset_manifest = artifact(root, str(Path(dataset['path']) / 'manifest.json'))
            if json.loads(dataset_manifest.read_text()).get('kind') in ('visual_causal_teacher_dataset', 'katago_v7_expert_input_overlay'):
                phase = 'dataset_inputs'
                if remote != root:
                    raise ValueError('Visual dataset paths require identical local and remote roots')
                argv = [sys.executable, '-B', str(controller_source / 'ops/stage_visual_inputs.py'),
                    '--workspace-root', str(root), '--kind', 'dataset', '--manifest', str(dataset_manifest),
                    '--expected-sha256', dataset['manifest_sha256'], '--output', str(attempt / 'dataset_inputs')]
                with (attempt / 'dataset_inputs.log').open('w') as log:
                    subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        if isinstance(config.get('native_receipt'), str):
            from gozero.model_artifacts import artifact
            phase = 'native_inputs'
            if remote != root:
                raise ValueError('Pinned native inputs require identical local and remote roots')
            receipt_path = artifact(root, config['native_receipt'])
            receipt = json.loads(receipt_path.read_text())
            source = artifact(root, '.gozero/snapshots/' + receipt['snapshot_id']); native_source = verify(source)
            binary = artifact(root, str(receipt_path.parent / receipt['filename']))
            with binary.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['binary_sha256']:
                    raise ValueError('Pinned inference native binary changed')
            paths = [receipt_path, binary, source / 'manifest.json', *[source / name for name in native_source['files']]]
            assets = {}
            for path in paths:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                assets[str(path.relative_to(root))] = {'sha256': digest, 'bytes': path.stat().st_size}
            asset_manifest = attempt / 'native_assets.json'
            asset_manifest.write_bytes(canonical_json({'schema_version': 1, 'kind': 'visual_inference_assets',
                'purpose': 'Pinned native library, receipt and complete attested source before SPMD startup.', 'files': assets}))
            argv = [sys.executable, '-B', str(controller_source / 'ops/stage_visual_inputs.py'),
                '--workspace-root', str(root), '--kind', 'assets', '--manifest', str(asset_manifest),
                '--expected-sha256', hashlib.sha256(asset_manifest.read_bytes()).hexdigest(),
                '--output', str(attempt / 'native_inputs')]
            with (attempt / 'native_inputs.log').open('w') as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        if config.get('kind') == 'visual_inference_service' and config.get('mode') == 'matches':
            from gozero.model_artifacts import artifact
            phase = 'match_inputs'
            if remote != root:
                raise ValueError('Visual benchmark assets require identical local and remote roots')
            assets = {}
            def add_asset(name, expected):
                path = artifact(root, name)
                if str(path.relative_to(root)) in assets:
                    if assets[str(path.relative_to(root))]['sha256'] != expected:
                        raise ValueError('Conflicting benchmark asset identity: ' + name)
                    return
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError('Benchmark asset identity differs: ' + name)
                assets[str(path.relative_to(root))] = {'sha256': actual, 'bytes': path.stat().st_size}
            engine = json.loads((snapshot / 'eval/katago_build.json').read_text())
            add_asset(engine['binary_path'], engine['binary_sha256'])
            for names in config['matches_by_host'].values():
                for name in names:
                    specification = json.loads(artifact(snapshot, name).read_text())
                    weights = json.loads(artifact(snapshot, specification.get('katago_weights', 'eval/katago_9x9.json')).read_text())
                    add_asset(weights['path'], weights['sha256'])
            asset_manifest = attempt / 'match_assets.json'
            asset_manifest.write_bytes(canonical_json({'schema_version': 1, 'kind': 'visual_inference_assets',
                'purpose': 'Complete pinned external benchmark executable and weight closure before SPMD startup.', 'files': assets}))
            argv = [sys.executable, '-B', str(controller_source / 'ops/stage_visual_inputs.py'),
                '--workspace-root', str(root), '--kind', 'assets', '--manifest', str(asset_manifest),
                '--expected-sha256', hashlib.sha256(asset_manifest.read_bytes()).hexdigest(),
                '--output', str(attempt / 'match_inputs')]
            with (attempt / 'match_inputs.log').open('w') as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        if args.resume_attempt is not None:
            if config.get('kind') == 'fixed_joint_learning':
                phase = 'resume_inputs'
                if args.controller_cpu_list is None:
                    raise ValueError('Joint resume requires explicit staging CPU placement')
                if remote != root:
                    raise ValueError('Joint checkpoint paths require identical artifact roots')
                group = root / 'runs' / args.resume_attempt / 'rank-0/artifacts/checkpoints' / ('turn-%09d.group.json' % args.resume_turn)
                argv = [sys.executable, '-B', str(snapshot / 'ops/stage_joint_checkpoint.py'),
                    '--workspace-root', str(root), '--manifest', str(group),
                    '--expected-sha256', hashlib.sha256(group.read_bytes()).hexdigest(),
                    '--cpu-list', args.controller_cpu_list,
                    '--output', str(attempt / 'resume_inputs')]
                with (attempt / 'resume_inputs.log').open('w') as log:
                    subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
            if config.get('kind') in ('visual_causal_distillation', 'fixed_policy_learning'):
                phase = 'resume_inputs'
                if remote != root:
                    raise ValueError('Shared visual checkpoint paths require identical local and remote artifact roots')
                group = root / 'runs' / args.resume_attempt / 'rank-0/artifacts/checkpoints' / ('turn-%09d.group.json' % args.resume_turn)
                saved = json.loads(group.read_text())
                if (saved.get('kind') != 'visual_replicated_checkpoint_group'
                        or saved.get('snapshot_id') != manifest['snapshot_id']):
                    raise ValueError('Resume checkpoint group does not match the frozen visual run')
                # Every rank reads the shared owner's arrays. Hash-check and
                # stage that entire closure before any distributed initialize,
                # so a missing local file cannot strand healthy peers in JAX.
                argv = [sys.executable, '-B', str(snapshot / 'ops/stage_visual_inputs.py'),
                    '--workspace-root', str(root), '--kind', 'checkpoint', '--manifest', str(group),
                    '--expected-sha256', hashlib.sha256(group.read_bytes()).hexdigest(),
                    '--output', str(attempt / 'resume_inputs')]
                with (attempt / 'resume_inputs.log').open('w') as log:
                    subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.prepare_timeout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            futures = [executor.submit(stage, host) for host in hosts]
            for future in futures:
                future.result()
        # This managed interpreter was observed on every supplied host.
        bootstrap_python = '/home/go-user/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12'
        prepare = [bootstrap_python, str(remote_snapshot / 'ops/prepare_host.py'), '--snapshot', str(remote_snapshot),
                   '--environment', str(environment), '--uv', str(remote_uv), '--cache', str(remote / '.gozero/cache/uv')]
        execute = [bootstrap_python, str(remote_snapshot / 'ops/run_host.py'), '--snapshot', str(remote_snapshot),
                   '--environment', str(environment), '--attempt', str(remote_attempt), '--timeout', str(args.timeout),
                   '--controller-cpus', str(args.controller_cpus)]
        if args.controller_cpu_list is not None:
            execute.extend(['--controller-cpu-list', args.controller_cpu_list])
        if native is not None:
            execute.extend(['--native-receipt', str(remote_native / 'receipt.json')])
        if args.resume_attempt is not None:
            execute.extend(['--resume-attempt', str(remote / 'runs' / args.resume_attempt), '--resume-turn', str(args.resume_turn)])
        if args.stop_after_turn is not None:
            execute.extend(['--stop-after-turn', str(args.stop_after_turn)])
        for phase, command, limit in [('prepare', prepare, args.prepare_timeout), ('execute', execute, args.timeout + 30)]:
            print(json.dumps({'kind': 'pod_phase', 'phase': phase}), flush=True)
            if phase == 'execute' and peer_cancel:
                def cancel(reason):
                    argv = [bootstrap_python, str(remote_snapshot/'ops/cancel_host.py'),
                            '--snapshot', str(remote_snapshot), '--attempt', str(remote_attempt)]
                    with (attempt/'cancel.stdout.log').open('w') as stdout, (attempt/'cancel.stderr.log').open('w') as stderr:
                        done = subprocess.run(pdsh_command(hosts, argv, 20), env=pdsh_environment(os.environ),
                                              stdout=stdout, stderr=stderr, timeout=30)
                    return {'reason':reason,'returncode':done.returncode,'status':'delivered' if done.returncode==0 else 'failed'}
                observed = supervise([(host.rank,pdsh_command((host,),command,limit)) for host in hosts],
                    directory=attempt/'rank-launchers',environment=pdsh_environment(os.environ),
                    timeout_seconds=limit+20,cancel=cancel)
                (attempt/'supervisor.json').write_bytes(canonical_json(observed))
                if observed['status']!='passed':raise RuntimeError('SPMD rank supervision failed: '+str(observed['reason']))
                continue
            with (attempt / (phase + '.stdout.log')).open('w') as stdout, (attempt / (phase + '.stderr.log')).open('w') as stderr:
                result = subprocess.run(pdsh_command(hosts, command, limit), env=pdsh_environment(os.environ),
                                        stdout=stdout, stderr=stderr, timeout=limit + 20)
            if result.returncode:
                raise RuntimeError('%s failed with status %d; see attempt logs' % (phase, result.returncode))
        returncode = 0
    except Exception as error:
        failure = repr(error)
    finally:
        # Collection is useful on failed attempts too. No remote paths are deleted.
        for host in hosts:
            if host.hostname == socket.gethostname().split('.')[0] and remote_attempt == attempt:
                continue
            try:
                collected = subprocess.run(['rsync', '-a', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
                                            host.ssh + ':' + str(remote_attempt / ('rank-%d' % host.rank)), str(attempt) + '/'],
                                           capture_output=True, text=True, timeout=45)
                collection_error = collected.stderr if collected.returncode else None
            except subprocess.SubprocessError as error:
                collection_error = repr(error)
            if collection_error:
                (attempt / ('collect-rank-%d.error.log' % host.rank)).write_text(collection_error)
                returncode = 1
        for host in hosts:
            rank_result = attempt / ('rank-%d' % host.rank) / 'result.json'
            try:
                observed = json.loads(rank_result.read_text())
                valid = (observed['rank'] == host.rank and observed['world_size'] == len(hosts)
                         and observed['snapshot_id'] == manifest['snapshot_id'] and observed['status'] == 'passed'
                         and observed['returncode'] == 0 and observed['source_integrity'] and not observed['timed_out'])
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                returncode = 1
        end = time.time()
        result = {**launch, 'end_unix_time': end, 'elapsed_seconds': end-start,
                  'reserved_chip_hours': args.chips*(end-start)/3600, 'last_phase': phase,
                  'status': 'passed' if returncode == 0 else 'failed', 'error': failure}
        (attempt / 'result.json').write_bytes(canonical_json(result))
        print(json.dumps(result), flush=True)
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
