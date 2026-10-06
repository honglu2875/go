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
    if c.get('kind')=='dense_batch_qualification':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('Batch qualification cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from qualify_batches import entry
        entry(a)
        return
    if c.get('kind')=='moe_system_qualification':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('System qualification cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from qualify_system import entry
        entry(a)
        return
    if c.get('kind')=='moe_kernel_qualification':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('Kernel qualification cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from qualify_kernels import entry
        entry(a)
        return
    if c.get('kind')=='joint19_padding_varied':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('Runtime qualification cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from verify_varied import entry
        entry(a)
        return
    if c.get('kind')=='joint19_runtime_diagnostic':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('Diagnostic cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from diagnose import entry
        entry(a)
        return
    if c.get('kind')=='joint19_throughput_benchmark':
        if a.resume is not None or a.stop_after_turn is not None:raise ValueError('Benchmark cannot resume learning')
        os.environ['JAX_PLATFORMS']='tpu'
        from benchmark import entry
        entry(a)
        return
    if c.get('kind')!='fixed_joint_learning' or c.get('platform') not in ('cpu','tpu'):
        raise ValueError('Expected an explicit joint-learning platform')
    os.environ['JAX_PLATFORMS']=c['platform']
    from train_joint import entry
    entry(a)


if __name__=='__main__':main()
