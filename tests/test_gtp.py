import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from gozero.gtp import GTPClient, GTPCommandError, GTPError


SERVER = '''
import sys,time
for line in sys.stdin:
    ident,command=line.strip().split(' ',1)
    if command=='quit':
        print('='+ident+'\\n',flush=True);break
    if command=='reject':
        print('?'+ident+' illegal move\\n',flush=True)
    elif command=='multiline':
        for part in ('# comment\\n\\n=',ident+' first\\n','second\\n\\n'):
            sys.stdout.write(part);sys.stdout.flush();time.sleep(0.01)
    elif command=='wrong_id':
        print('=999 wrong\\n',flush=True)
    elif command=='die':
        break
    elif command=='hang':
        time.sleep(60)
    else:
        print('='+ident+' '+command+'\\n',flush=True)
'''


class GTPTests(unittest.TestCase):
    def test_fragmented_multiline_error_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'engine'
            with GTPClient([sys.executable, '-u', '-c', SERVER], output) as client:
                self.assertEqual(client.command('multiline'), 'first\nsecond')
                with self.assertRaises(GTPCommandError):
                    client.command('reject')
                self.assertEqual(client.command('name'), 'name')
            records = [json.loads(line) for line in (output / 'gtp.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['returncode'], 0)
            self.assertEqual(len([r for r in records if r['kind'] == 'response']), 4)

    def test_bad_ids_and_early_eof_poison_connection(self):
        for command in ('wrong_id', 'die'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temp:
                with GTPClient([sys.executable, '-u', '-c', SERVER], Path(temp) / 'engine') as client:
                    with self.assertRaises(GTPError):
                        client.command(command)
                    with self.assertRaises(GTPError):
                        client.command('name')

    def test_timeout_is_bounded_and_child_is_stopped(self):
        with tempfile.TemporaryDirectory() as temp:
            client = GTPClient([sys.executable, '-u', '-c', SERVER], Path(temp) / 'engine')
            try:
                with self.assertRaises(TimeoutError):
                    client.command('hang', timeout=0.1)
            finally:
                client.close()
            self.assertIsNotNone(client.process.poll())

    def test_command_injection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with GTPClient([sys.executable, '-u', '-c', SERVER], Path(temp) / 'engine') as client:
                with self.assertRaises(ValueError):
                    client.command('name\nquit')
                self.assertEqual(client.command('version'), 'version')

    def test_deadline_also_bounds_a_full_stdin_pipe(self):
        with tempfile.TemporaryDirectory() as temp:
            client = GTPClient([sys.executable, '-c', 'import time;time.sleep(60)'], Path(temp) / 'engine')
            try:
                start = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, 'write deadline'):
                    client.command('x' * (2*1024*1024), timeout=0.1)
                self.assertLess(time.monotonic()-start, 1.0)
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
