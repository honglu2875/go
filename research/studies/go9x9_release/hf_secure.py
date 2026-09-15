"""Run hf with a token loaded in memory; credentials never enter argv or logs."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys


def environment():
    token = None
    for line in Path('/workspace/go/.env.hf').read_text().splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        if line.startswith('HF_TOKEN='):
            values = shlex.split(line.split('=', 1)[1], comments=True)
            if len(values) != 1:
                raise ValueError('Invalid HF_TOKEN assignment')
            token = values[0]
    if not token:
        raise ValueError('HF_TOKEN is missing')
    root = Path('/dev/shm/go-hf-release-cache')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = dict(os.environ, HF_TOKEN=token, HF_HOME=str(root),
               HF_XET_CACHE=str(root / 'xet'), HF_HUB_DISABLE_PROGRESS_BARS='1',
               HF_HUB_VERBOSITY='error', HF_HUB_DISABLE_TELEMETRY='1',
               TMPDIR=str(root))
    return env, token


if __name__ == '__main__':
    env, secret = environment()
    result = subprocess.run(['/home/go-user/.local/bin/hf', *sys.argv[1:]],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for stream, target in ((result.stdout, sys.stdout), (result.stderr, sys.stderr)):
        target.write(stream.replace(secret, '[REDACTED]'))
    raise SystemExit(result.returncode)
