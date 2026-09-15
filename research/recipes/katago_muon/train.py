"""CPU numerical qualification entry point; this recipe launches no learner."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workspace-root', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.config.read_text())
    if c.get('kind') != 'katago_muon_cpu_qualification' or c.get('platform') != 'cpu':
        raise ValueError('Expected an explicit CPU optimizer qualification')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from qualify import run
    run(args, c)


if __name__ == '__main__':
    main()
