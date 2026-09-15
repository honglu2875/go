#!/usr/bin/env python3
"""Run only a frozen, bounded architecture comparison or inference service."""
import argparse
from pathlib import Path
import json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after-turn',type=int)
    args=p.parse_args()
    c=json.loads(args.config.read_text())
    if c['kind']=='paired_policy_evaluation':
        from evaluate_games import entry
    else: raise ValueError('Unsupported baseline job')
    entry(args)
if __name__=='__main__': main()
