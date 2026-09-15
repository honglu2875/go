"""Qualify the trained joint model evaluation path with explicit CPU placement."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workspace-root',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config.get('kind') not in ('joint_trained_inference_qualification','joint_gtp_cpu_qualification','joint_katago_cpu_qualification') or config.get('platform')!='cpu':
        raise ValueError('Expected explicit CPU trained-inference qualification')
    os.environ['JAX_PLATFORMS']='cpu'
    if config['kind']=='joint_katago_cpu_qualification':
        from qualify_matches import run
    elif config['kind']=='joint_gtp_cpu_qualification':
        from qualify_gtp import run
    else:
        from qualify_trained import run
    run(args,config)


if __name__=='__main__':main()
