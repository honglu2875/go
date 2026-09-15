"""Reject cross-run evidence and stop requests before any scheduling mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import continue_comparison as queue


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class QueueGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.enterContext(patch.object(queue,'ROOT',self.root));self.enterContext(patch.object(queue,'STUDY',self.study))
        self.reg={'snapshots':{'cnn':{'seed1':'a'*64}}}
        self.attempt=self.root/'runs/pod-example';self.attempt.mkdir(parents=True)
        self.now=time.time();self.log=self.study/'cnn-seed1-controller-001.log'
        self.process=dict(stage='cnn_seed1',snapshot='a'*64,registration_sha256=queue.REGISTRATION_SHA,
                          started=self.now,log=str(self.log))
        self.launch=dict(snapshot_id='a'*64,start_unix_time=self.now)
        write(self.attempt/'launch.json',self.launch)
        self.record=dict(kind='pod_attempt',attempt=str(self.attempt),attempt_id=self.attempt.name,snapshot_id='a'*64)
        write(self.study/'cnn-seed1-process-001.json',self.process);write(self.log,self.record)

    def test_expected_attempt_found_and_foreign_source_rejected(self):
        self.assertEqual(queue.attempt_for('cnn_seed1',self.reg),self.attempt)
        write(self.study/'cnn-seed1-process-001.json',{**self.process,'snapshot':'b'*64})
        with self.assertRaises(ValueError):queue.attempt_for('cnn_seed1',self.reg)

    def test_foreign_directory_and_stale_attempt_rejected(self):
        write(self.log,{**self.record,'attempt':str(self.root/'other/pod-example')})
        with self.assertRaises(ValueError):queue.attempt_for('cnn_seed1',self.reg)
        write(self.log,self.record)
        write(self.attempt/'launch.json',{**self.launch,'start_unix_time':self.now-500})
        with self.assertRaises(ValueError):queue.attempt_for('cnn_seed1',self.reg)

    def test_delayed_publication_is_bounded(self):
        self.log.write_text('')
        self.assertIsNone(queue.attempt_for('cnn_seed1',self.reg))
        write(self.study/'cnn-seed1-process-001.json',{**self.process,'started':self.now-500})
        with self.assertRaises(ValueError):queue.attempt_for('cnn_seed1',self.reg)

    def test_wrong_checkpoint_or_audit_cannot_advance_stage(self):
        q=object.__new__(queue.Queue);path=self.study/'audit.json'
        for change in ({'status':'failed'},{'attempt':'another'},{'training_snapshot':'b'*64}):
            write(path,{**dict(status='passed',attempt=self.attempt.name,training_snapshot='a'*64),**change})
            with self.assertRaises(ValueError):q.validate_receipt(path,self.attempt)

    def test_stop_and_source_failure_precede_subprocess(self):
        q=object.__new__(queue.Queue);q.plan={};q.directory=self.study;q.deadline=time.time()+100;q.calls=0
        with patch.object(queue,'inspect',return_value=None),patch.object(queue.subprocess,'run') as run:
            (self.study/'stop').touch()
            with self.assertRaises(RuntimeError):q.invoke(['would-launch'],'example')
            run.assert_not_called()
        (self.study/'stop').unlink()
        with patch.object(queue,'inspect',side_effect=ValueError('source changed')),patch.object(queue.subprocess,'run') as run:
            with self.assertRaises(ValueError):q.invoke(['would-launch'],'example')
            run.assert_not_called()


if __name__=='__main__':unittest.main()
