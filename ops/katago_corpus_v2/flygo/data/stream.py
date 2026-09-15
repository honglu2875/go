"""Bounded continuous admission, draining every admitted job on a clean stop."""
from concurrent.futures import wait,FIRST_COMPLETED
import time


def run(submit,publish,can_admit,stopping,*,concurrency,limit=None,tick=None,poll_seconds=1.):
    if concurrency<1 or (limit is not None and limit<1) or poll_seconds<=0:
        raise ValueError('Invalid bounded stream configuration')
    pending={};submitted=completed=0
    while True:
        while len(pending)<concurrency and not stopping() and (limit is None or submitted<limit) and can_admit():
            future,metadata=submit()
            if future in pending:raise ValueError('Submission reused an active future')
            pending[future]=metadata;submitted+=1
        if tick is not None:tick(pending)
        if not pending:
            if stopping() or (limit is not None and submitted==limit):
                return dict(submitted=submitted,completed=completed)
            time.sleep(poll_seconds)
            continue
        done,_=wait(pending,timeout=poll_seconds,return_when=FIRST_COMPLETED)
        for future in sorted(done,key=lambda f:pending[f]['sequence']):
            publish(pending[future],future.result())
            del pending[future];completed+=1
