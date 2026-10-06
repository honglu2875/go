"""Wait for one owned attempt, then audit its final reports without live-error latching."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--review-sha256',required=True);args=parser.parse_args()
    if '/' in args.attempt or not args.attempt.startswith('pod-'):raise ValueError('Invalid attempt')
    study=Path(__file__).resolve().parent;root=study.parents[2]
    result_path=root/'runs'/args.attempt/'result.json';started=time.time()
    outcome=dict(kind='throughput_completion_review',attempt=args.attempt,started=started)
    try:
        while not result_path.exists():
            if time.time()-started>4800:raise TimeoutError('Attempt closure did not arrive')
            time.sleep(20)
        raw=result_path.read_bytes();closed=json.loads(raw)
        outcome['attempt_result_sha256']=hashlib.sha256(raw).hexdigest()
        if closed['status']!='passed':raise ValueError('Attempt did not pass: '+str(closed.get('error')))
        review=study/'review_benchmark.py'
        if hashlib.sha256(review.read_bytes()).hexdigest()!=args.review_sha256:raise ValueError('Review operator changed')
        subprocess.run([sys.executable,'-B',str(review),'--attempt',args.attempt,'--output',str(study/'RESULTS_001.json')],
                       cwd=root,check=True,timeout=180)
        outcome.update(status='passed',review_sha256=args.review_sha256,
                       results_sha256=hashlib.sha256((study/'RESULTS_001.json').read_bytes()).hexdigest())
    except BaseException as error:
        outcome.update(status='failed',error=repr(error));raise
    finally:
        outcome['finished']=time.time()
        with (study/'completion-001.json').open('x') as f:json.dump(outcome,f,indent=2);f.write('\n')


if __name__=='__main__':main()
