"""Freeze 19x19 producer sources and create a separate, RAM-only generation contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    source=Path(__file__).resolve().parent
    root=Path('/dev/shm/go-corpus19');root.mkdir(exist_ok=True)
    prior=Path('/dev/shm/gozero/environments/0780619799010a206823')
    native=next((prior/'site-packages/flygo').glob('_native*.so'))
    assert digest(native)==json.loads((prior/'snapshot.json').read_text())['native_sha256']
    sources={str(p.relative_to(source/'flygo')):p.read_bytes() for p in sorted((source/'flygo').rglob('*.py'))}
    version=hashlib.sha256(b'go-corpus19-v2\0'+native.read_bytes())
    for name,content in sorted(sources.items()):version.update(name.encode()+b'\0'+content)
    key=version.hexdigest()
    environment=root/'environments'/key
    if not environment.exists():
        site=environment/'site-packages';(site/'flygo').mkdir(parents=True)
        for name,content in sources.items():
            p=site/'flygo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        os.link(native,site/'flygo'/native.name)
        for name in ('numpy','numpy.libs'):
            shutil.copytree(prior/'site-packages'/name,site/name,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__'))
        (environment/'entry.py').write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/'site-packages'))\nfrom flygo.data.service import main\nmain()\n")
        (environment/'snapshot.json').write_text(json.dumps(dict(snapshot=key,native_sha256=digest(native),
            numpy='2.5.3',source_sha256={name:hashlib.sha256(content).hexdigest() for name,content in sources.items()},created=time.time()),indent=2)+'\n')
        for path in environment.rglob('*'):
            if path.is_file() and path.suffix=='.py':path.chmod(0o444)
    old=json.loads(Path('/dev/shm/gozero/runs/expert-v1/config.json').read_text())
    teacher={**old['contract']['opponents'][-1], 'role':'fixed 19x19 raw policy/value teacher', 'board_sizes':[19]}
    contract=dict(schema_version=2,board_size=19,komi=7.5,rules='positional-area-multisuicide-v1',
        teacher=teacher,opponents=old['contract']['opponents'],engine_sha256=old['contract']['engine_sha256'],
        visits=16,max_moves=1444,opening_moves=16,
        behavior='16-ply 0.75 root-edge-visits + 0.25 raw-policy exploration; root order afterward',
        opponent_assignment='one equal-runtime worker per checkpoint per host; eight strata including teacher self-play',
        labels='fixed teacher raw policy/value every ply; search labels eligible only on teacher turns',
        neural_history='ignorePreRootHistory=false, ignoreAllHistory=false; full causal move prefix',
        search_determinism='concurrent search is not bitwise deterministic; published records immutable',
        production_stream='continuous-v2: eight independently refilled game slots; monotonic persisted admission sequence',
        cpu_inference_threads=dict(ordinary_opponents=dict(teacher=7,opponent=1),teacher_selfplay=dict(teacher=4,opponent=4)))
    contract_id=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    run=root/'runs/expert19-v2';run.mkdir(parents=True,exist_ok=True)
    path=run/'contract.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:path.write_text(json.dumps(contract,indent=2)+'\n')
    for record in [teacher,*contract['opponents']]:
        suffix='txt.gz' if record.get('url','').endswith('.txt.gz') else 'bin.gz'
        src=Path('/dev/shm/gozero/artifacts')/record['sha256']/('model.'+suffix)
        assert digest(src)==record['sha256']
        dest=root/'artifacts'/record['sha256']/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(src,dest)
    binary=Path('/dev/shm/gozero/artifacts')/contract['engine_sha256']/'katago'
    assert digest(binary)==contract['engine_sha256']
    dest=root/'artifacts'/contract['engine_sha256']/'katago';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(binary,dest)
    deployment=dict(root=str(root),environment=str(environment),contract=str(path),contract_id=contract_id,
                    run_id='expert19-v2',snapshot=key,prepared=time.time())
    (source/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    print(json.dumps(deployment),flush=True)


if __name__=='__main__':main()
