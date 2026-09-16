#!/usr/bin/env python3
"""Run a frozen joint policy/value learner or its continuation qualification."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after-turn',type=int);a=p.parse_args()
    c=json.loads(a.config.read_text())
    if c.get('kind')!='fixed_joint_learning' or c.get('platform') not in ('cpu','tpu'):
        raise ValueError('Expected an explicit joint-learning platform')
    os.environ['JAX_PLATFORMS']=c['platform']
    from train_joint import entry
    entry(a)


if __name__=='__main__':main()
