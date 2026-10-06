"""Run the pinned varied-batch reviewer after bounded process closure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    if '/' in args.attempt or not args.attempt.startswith('pod-'):
        raise ValueError('Invalid attempt')
    result = dict(kind='varied_runtime_completion', attempt=args.attempt, started=time.time())
    try:
        closed = ROOT / 'runs' / args.attempt / 'result.json'
        deadline = time.monotonic() + 3300
        while not closed.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Qualification closure did not arrive')
            time.sleep(15)
        review = STUDY / 'review_varied.py'
        if hashlib.sha256(review.read_bytes()).hexdigest() != args.review_sha256:
            raise ValueError('Review operator changed')
        subprocess.run([sys.executable, '-B', str(review), '--attempt', args.attempt], cwd=ROOT, check=True)
        report = STUDY / 'VARIED_001.json'
        result.update(status='passed', qualification_status=json.loads(report.read_text())['status'],
                      report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished'] = time.time()
        with (STUDY / 'varied-completion-001.json').open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
