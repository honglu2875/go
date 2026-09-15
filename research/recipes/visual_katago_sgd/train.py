#!/usr/bin/env python3
import argparse
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-turn', type=int)
    from train_sgd import entry
    entry(p.parse_args())
