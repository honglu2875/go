"""Concurrency regressions using controlled futures, with no engine dependency."""
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import threading
import unittest

path=Path(__file__).parent/'flygo/data/stream.py'
spec=importlib.util.spec_from_file_location('tested_stream',path)
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)


class StreamTests(unittest.TestCase):
    def start(self,submit,publish,can_admit,stopping,limit):
        outcomes=[]
        def execute():
            try:outcomes.append(stream.run(submit,publish,can_admit,stopping,concurrency=2,limit=limit,poll_seconds=.01))
            except BaseException as error:outcomes.append(error)
        worker=threading.Thread(target=execute,daemon=True);worker.start()
        return worker,outcomes

    def test_refills_before_the_slow_game_finishes(self):
        futures=[Future() for _ in range(3)];admitted=[];published=[];refilled=threading.Event()
        futures[1].set_result('short');futures[2].set_result('replacement')
        def submit():
            i=len(admitted);admitted.append(i)
            if i==2:refilled.set()
            return futures[i],dict(sequence=i)
        worker,outcomes=self.start(submit,lambda meta,value:published.append((meta['sequence'],value)),lambda:True,lambda:False,3)
        try:
            self.assertTrue(refilled.wait(2));self.assertFalse(futures[0].done())
        finally:futures[0].set_result('slow');worker.join(2)
        self.assertFalse(worker.is_alive());self.assertEqual(outcomes,[dict(submitted=3,completed=3)])
        self.assertEqual({i for i,_ in published},{0,1,2})

    def test_stop_drains_and_does_not_refill(self):
        futures=[Future(),Future()];admitted=[];stop=threading.Event();published=[];ready=threading.Event()
        def submit():
            i=len(admitted);admitted.append(i)
            if i==1:ready.set()
            return futures[i],dict(sequence=i)
        worker,outcomes=self.start(submit,lambda meta,value:published.append(value),lambda:True,stop.is_set,None)
        self.assertTrue(ready.wait(2));stop.set();futures[0].set_result(0);futures[1].set_result(1);worker.join(2)
        self.assertEqual(admitted,[0,1]);self.assertEqual(outcomes,[dict(submitted=2,completed=2)])

    def test_pressure_blocks_new_admission_and_then_recovers(self):
        admitted=[];blocked=threading.Event();allowed=threading.Event();future=Future();future.set_result(7)
        def can_admit():blocked.set();return allowed.is_set()
        def submit():admitted.append(0);return future,dict(sequence=0)
        worker,outcomes=self.start(submit,lambda *_:None,can_admit,lambda:False,1)
        self.assertTrue(blocked.wait(2));self.assertEqual(admitted,[]);allowed.set();worker.join(2)
        self.assertEqual(outcomes,[dict(submitted=1,completed=1)])

    def test_failure_propagates_without_publishing_invalid_result(self):
        bad=Future();bad.set_exception(ValueError('bad labels'));pending=Future();admitted=[];published=[]
        def submit():
            i=len(admitted);admitted.append(i);return (bad if i==0 else pending),dict(sequence=i)
        worker,outcomes=self.start(submit,lambda *_:published.append(1),lambda:True,lambda:False,2);worker.join(2)
        self.assertFalse(worker.is_alive());self.assertIsInstance(outcomes[0],ValueError);self.assertEqual(published,[])


if __name__=='__main__':unittest.main()
