import importlib.util
from pathlib import Path
import signal
import tempfile
import unittest

path=Path(__file__).parent/'flygo/data/handover.py'
spec=importlib.util.spec_from_file_location('tested_handover',path)
handover=importlib.util.module_from_spec(spec);spec.loader.exec_module(handover)


class HandoverTests(unittest.TestCase):
    def test_zombies_and_reused_pids_do_not_hold_an_allocation(self):
        record=dict(pid=32,identity='100')
        self.assertFalse(handover.active(record,lambda _:dict(identity='101',state='R')))
        self.assertFalse(handover.active(record,lambda _:dict(identity='100',state='Z')))
        self.assertTrue(handover.active(record,lambda _:dict(identity='100',state='D')))
        self.assertTrue(handover.active(record,lambda _:dict(identity='100',state='R')))

    def test_live_engine_blocks_even_after_python_worker_exits(self):
        h=dict(workers={'0':dict(processes=[dict(pid=32,identity='100'),dict(pid=33,identity='101')])})
        read=lambda pid:None if pid==32 else dict(identity='101',state='R')
        self.assertFalse(handover.lane_clear(h,0,read))

    def test_supervisor_release_waits_for_all_recorded_processes(self):
        h=dict(supervisor=dict(pid=20,identity='80'),workers={'0':dict(processes=[dict(pid=32,identity='100')])})
        live={20:dict(identity='80',state='T'),32:dict(identity='100',state='R')};signals=[]
        with tempfile.TemporaryDirectory() as tmp:
            receipt=Path(tmp)/'released.json'
            self.assertFalse(handover.release_previous(h,receipt,live.get,lambda *x:signals.append(x)))
            self.assertFalse(receipt.exists());self.assertEqual(signals,[])
            live[32]['state']='Z'
            self.assertTrue(handover.release_previous(h,receipt,live.get,lambda *x:signals.append(x)))
            self.assertEqual(signals,[(20,signal.SIGCONT)])
            self.assertTrue(handover.release_previous(h,receipt,live.get,lambda *x:signals.append(x)))
            self.assertEqual(len(signals),1)


if __name__=='__main__':unittest.main()
