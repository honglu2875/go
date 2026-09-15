import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from gozero.pod import Host, load_hosts, pdsh_command, pdsh_environment, supervise
from gozero.snapshots import freeze, verify


class PodTests(unittest.TestCase):
    def test_remote_argv_is_quoted_without_shell_interpolation(self):
        words = ['python3', '-c', 'print("$HOME `whoami` ; $(id)")', "a b'c"]
        command = pdsh_command((Host(0, 'user@host-0'),), words, 30)
        self.assertEqual(shlex.split(command[-1]), words)
        self.assertIn('-S', command)
        environment = pdsh_environment({'PDSH_SSH_ARGS': 'unsafe override'})
        self.assertNotIn('PDSH_SSH_ARGS', environment)
        self.assertIn('StrictHostKeyChecking=yes', environment['PDSH_SSH_ARGS_APPEND'])

    def test_duplicate_and_noncontiguous_hosts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'hosts.json'
            for rows in ([{'rank': 0, 'ssh': 'user@a'}, {'rank': 1, 'ssh': 'other@a'}],
                         [{'rank': 1, 'ssh': 'user@a'}], [{'rank': 0, 'ssh': 'user@a;id'}]):
                path.write_text(json.dumps({'schema_version': 1, 'coordinator_rank': 0, 'hosts': rows}))
                with self.assertRaises(ValueError):
                    load_hosts(path)

    def fixture(self, directory, source):
        project = Path(__file__).resolve().parents[1]
        repo = directory / 'repo'
        recipe = repo / 'research/recipes/fixture'
        recipe.mkdir(parents=True)
        (recipe / 'train.py').write_text(source)
        (recipe / 'recipe.json').write_text('{"id":"fixture"}')
        (recipe / 'smoke.json').write_text('{}')
        shutil.copytree(project / 'packages/gozero', repo / 'packages/gozero', ignore=shutil.ignore_patterns('__pycache__'))
        (repo / 'ops').mkdir()
        shutil.copyfile(project / 'ops/run_host.py', repo / 'ops/run_host.py')
        shutil.copyfile(project / 'ops/pod_run.py', repo / 'ops/pod_run.py')
        (repo / 'ops/hosts.json').write_text(json.dumps({'schema_version': 1, 'coordinator_rank': 0,
            'hosts': [{'rank': 0, 'ssh': getpass.getuser() + '@' + socket.gethostname().split('.')[0]}]}))
        (repo / 'uv.lock').write_text('version = 1\n')
        snapshot = freeze(repo, recipe, recipe / 'smoke.json', directory / 'snapshots')
        environment = directory / 'env'
        (environment / 'bin').mkdir(parents=True)
        (environment / 'bin/python').symlink_to(sys.executable)
        (environment / 'gozero-runtime.json').write_text(json.dumps({
            'uv_lock_sha256': hashlib.sha256((snapshot / 'uv.lock').read_bytes()).hexdigest()}))
        return snapshot, environment

    def test_frozen_controller_rejects_artifacts_inside_either_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            controller, _ = self.fixture(directory / 'controller', 'print("controller fixture")\n')
            target, _ = self.fixture(directory / 'target', 'print("target fixture")\n')
            base = [sys.executable, '-B', str(controller / 'ops/pod_run.py')]
            for arguments in (['--snapshot', str(controller)],
                              ['--snapshot', str(target)],
                              ['--snapshot', str(target), '--workspace-root', str(target)],
                              ['--snapshot', str(target), '--workspace-root', str(controller)]):
                result = subprocess.run(base + arguments, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Artifact root must be outside frozen source', result.stderr)
                self.assertFalse((controller / 'runs').exists())
                self.assertFalse((target / 'runs').exists())
                verify(controller)
                verify(target)

    def test_frozen_rank_execution_records_result_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = ('import argparse, pathlib, os\n'
                      'p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--output");a=p.parse_args()\n'
                      'out=pathlib.Path(a.output);out.mkdir();(out/"identity").write_text(os.environ["GOZERO_SNAPSHOT_ID"])\n')
            snapshot, environment = self.fixture(directory, source)
            attempt = directory / 'attempt'
            command = [sys.executable, str(snapshot / 'ops/run_host.py'), '--snapshot', str(snapshot),
                       '--environment', str(environment), '--attempt', str(attempt), '--timeout', '5']
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((attempt / 'rank-0/result.json').read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertEqual((attempt / 'rank-0/artifacts/identity').read_text(), snapshot.name)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(json.loads((attempt / 'rank-0/result.json').read_text()), result)

    def test_timeout_terminates_descendants_but_preserves_unrelated_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = ('import argparse,pathlib,subprocess,sys,time\n'
                      'p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--output");a=p.parse_args()\n'
                      'out=pathlib.Path(a.output);out.mkdir()\n'
                      'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                      '(out/"descendant").write_text(str(child.pid))\ntime.sleep(60)\n')
            snapshot, environment = self.fixture(directory, source)
            attempt = directory / 'attempt'
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
            try:
                result = subprocess.run([sys.executable, str(snapshot / 'ops/run_host.py'), '--snapshot', str(snapshot),
                    '--environment', str(environment), '--attempt', str(attempt), '--timeout', '0.5'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                recorded = json.loads((attempt / 'rank-0/result.json').read_text())
                self.assertTrue(recorded['timed_out'])
                self.assertEqual(recorded['status'], 'failed')
                descendant = int((attempt / 'rank-0/artifacts/descendant').read_text())
                for _ in range(20):
                    try:
                        state = Path('/proc/%d/stat' % descendant).read_text().split(') ', 1)[1].split()[0]
                    except FileNotFoundError:
                        state = None
                    if state in (None, 'Z'):
                        break
                    time.sleep(0.05)
                self.assertIn(state, (None, 'Z'))
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_published_scientific_failure_stops_a_hung_distributed_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = ('import argparse,pathlib,json,os,time\n'
                      'p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--output");a=p.parse_args()\n'
                      'out=pathlib.Path(a.output);out.mkdir()\n'
                      '(out/"result.json").write_text(json.dumps({"schema_version":1,"snapshot_id":os.environ["GOZERO_SNAPSHOT_ID"],"status":"failed","error":"missing shared file"}))\n'
                      'time.sleep(60)\n')
            snapshot, environment = self.fixture(directory, source); attempt = directory / 'attempt'
            started = time.monotonic()
            done = subprocess.run([sys.executable, str(snapshot / 'ops/run_host.py'), '--snapshot', str(snapshot),
                '--environment', str(environment), '--attempt', str(attempt), '--timeout', '30'], capture_output=True, text=True, timeout=8)
            self.assertNotEqual(done.returncode, 0); self.assertLess(time.monotonic() - started, 5)
            recorded = json.loads((attempt / 'rank-0/result.json').read_text())
            self.assertTrue(recorded['reported_scientific_failure']); self.assertFalse(recorded['timed_out'])
            self.assertFalse(recorded['cancelled'])
            self.assertEqual(json.loads((attempt / 'rank-0/artifacts/result.json').read_text())['error'], 'missing shared file')

    def test_first_rank_failure_cancels_a_running_peer_and_its_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            source=('import argparse,pathlib,subprocess,sys,time\n'
                    'p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--output");a=p.parse_args()\n'
                    'out=pathlib.Path(a.output);out.mkdir()\n'
                    'child=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])\n'
                    '(out/"descendant").write_text(str(child.pid))\ntime.sleep(60)\n')
            snapshot,environment=self.fixture(directory,source);attempt=directory/'peer'
            peer=[sys.executable,str(snapshot/'ops/run_host.py'),'--snapshot',str(snapshot),
                  '--environment',str(environment),'--attempt',str(attempt),'--timeout','30']
            ready=attempt/'rank-0/artifacts/descendant'
            failure=[sys.executable,'-c','import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nwhile not p.exists():time.sleep(.02)\ntime.sleep(.15)\nsys.exit(7)',str(ready)]
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            requests=[]
            def cancel(reason):
                requests.append(reason)
                (attempt/'cancel.json').write_text(json.dumps({'schema_version':1,'kind':'cancel_attempt',
                    'snapshot_id':snapshot.name,'attempt_id':attempt.name}))
                return {'status':'delivered'}
            try:
                start=time.monotonic()
                result=supervise([(0,peer),(1,failure)],directory=directory/'launchers',environment=os.environ,
                                 timeout_seconds=10,cancel=cancel,grace_seconds=5)
                self.assertEqual(result['status'],'failed');self.assertEqual(result['returncodes']['1'],7)
                self.assertEqual(len(requests),1);self.assertLess(time.monotonic()-start,5)
                recorded=json.loads((attempt/'rank-0/result.json').read_text())
                self.assertTrue(recorded['cancelled']);self.assertFalse(recorded['timed_out'])
                pid=int(ready.read_text())
                for _ in range(20):
                    try:state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
                    except FileNotFoundError:state=None
                    if state in (None,'Z'):break
                    time.sleep(.05)
                self.assertIn(state,(None,'Z'));self.assertIsNone(unrelated.poll())
            finally:unrelated.terminate();unrelated.wait(timeout=5)

    def test_pending_attempt_cancellation_prevents_training_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            source='raise RuntimeError("trainer must not start")\n'
            snapshot,environment=self.fixture(directory,source);attempt=directory/'cancelled';attempt.mkdir()
            (attempt/'cancel.json').write_text(json.dumps({'schema_version':1,'kind':'cancel_attempt',
                'snapshot_id':snapshot.name,'attempt_id':attempt.name}))
            result=subprocess.run([sys.executable,str(snapshot/'ops/run_host.py'),'--snapshot',str(snapshot),
                '--environment',str(environment),'--attempt',str(attempt),'--timeout','5'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            recorded=json.loads((attempt/'rank-0/result.json').read_text())
            self.assertTrue(recorded['cancelled']);self.assertIsNone(recorded['returncode'])
            self.assertFalse((attempt/'rank-0/process.json').exists())


if __name__ == '__main__':
    unittest.main()
