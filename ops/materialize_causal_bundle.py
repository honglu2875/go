#!/usr/bin/env python3
"""Materialize a verified inference checkpoint under its original logical paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--manifest-sha256',required=True);p.add_argument('--archive-sha256',required=True)
a=p.parse_args();root=a.workspace_root.resolve();bundle=a.bundle.resolve()
def digest(path):
    with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
if digest(bundle/'manifest.json')!=a.manifest_sha256 or digest(bundle/'payload.tar')!=a.archive_sha256:
    raise ValueError('Candidate bundle identity differs')
m=json.loads((bundle/'manifest.json').read_text());d=m['descriptor']
if m['kind']!='portable_causal_artifact_paths' or len(m['files'])>32:
    raise ValueError('Unexpected candidate bundle')
target=(root/d['training_result_path']).resolve().parent
if not target.is_relative_to(root/'runs') or target.name!='artifacts':raise ValueError('Artifact target escapes runs')
contents={}
with tarfile.open(bundle/'payload.tar','r:')as archive:
    members=archive.getmembers()
    if len(members)!=len(m['files']) or {x.name for x in members}!=set(m['files']) or sum(x.size for x in members)>512*2**20:
        raise ValueError('Archive dimensions differ')
    for member in members:
        destination=(root/member.name).resolve()
        if not member.isfile() or not destination.is_relative_to(target) or str(destination.relative_to(root))!=member.name:
            raise ValueError('Only canonical regular inference files are allowed')
        data=archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest()!=m['files'][member.name]:raise ValueError('Archive member hash differs')
        contents[destination.relative_to(target)]=data
created=False
if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.causal-bundle-',dir=target.parent))
    for name,data in contents.items():
        path=temporary/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        path.chmod(0o444)
    for directory in sorted([p for p in temporary.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True)+[temporary]:
        fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    temporary.rename(target);fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);created=True
for name,expected in m['files'].items():
    if digest(root/name)!=expected:raise ValueError('Existing or published candidate bytes differ')
source=root/'.gozero/snapshots'/d['training_snapshot']
sys.path.insert(0,str(source/'packages/gozero/src'));sys.dont_write_bytecode=True
from gozero.causal_artifacts import validate
validate(root,d)
print(json.dumps({'status':'passed','created':created,'manifest_sha256':a.manifest_sha256,'model_export_sha256':d['model_export_sha256'],'files':len(m['files'])}))
