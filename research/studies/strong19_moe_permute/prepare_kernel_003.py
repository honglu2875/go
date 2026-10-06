"""Freeze the amended tile grid without changing earlier registrations."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json,freeze,verify


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    with path.open('xb') as stream:stream.write(canonical_json(obj))
    path.chmod(0o444)


def main():
    previous=STUDY/'kernel-registration-002.json'
    plan=json.loads(previous.read_text())
    old=ROOT/'.gozero/snapshots'/plan['snapshot'];verify(old)
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    relative=Path('packages/gozero/src/gozero/moe.py')
    left,right=functions(old/relative),functions(ROOT/relative)
    names=('initialize','grouped_dot','_dispatch_permutation','_dispatch_permutation_forward',
           '_dispatch_permutation_backward','_gather_permutation','_gather_permutation_forward',
           '_gather_permutation_backward','feed_forward','router_metrics','active_flops')
    assert all(left[n]==right[n] for n in names)
    installed=ROOT/'.venv/lib/python3.12/site-packages/jax/experimental/pallas/ops/tpu/megablox/gmm.py'
    equivalence=STUDY/'kernel-source-equivalence-003.json'
    write(equivalence,dict(status='passed',reference_snapshot=old.name,
          unchanged_functions={n:hashlib.sha256(right[n].encode()).hexdigest() for n in names},
          source_sha256=sha(ROOT/relative),reference_source_sha256=sha(old/relative),
          installed_gmm_sha256=sha(installed),operator_sha256=sha(Path(__file__)),
          scope='Numeric routing/grouped-dot/custom-gradient functions unchanged. Shared validator has one additional model-scope field, unused by this benchmark. The benchmark adds a relative-L2 gate.'))
    config=json.loads((STUDY/'kernel-proposal-config-002.json').read_text())
    cases=[]
    for offset in range(0,len(config['cases']),5):
        group=config['cases'][offset:offset+5]
        assert len(group)==5 and group[0]['tiling']==[256,512,512] and not group[0]['permutation_vjp']
        cases.extend(group)
        cases.extend({**group[0],'tiling':tile} for tile in ([256,384,384],[512,384,384]))
    assert len(cases)==70
    config['cases']=cases
    path=STUDY/'kernel-proposal-config-003.json';write(path,config)
    snapshot=freeze(ROOT,Path('research/recipes/strong19_moe_permute'),path,ROOT/'.gozero/snapshots')
    plan.update(created=time.time(),snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),
                supersedes_preparation=previous.name,output_directory='kernel-stage-003',
                scope='70 fixed same-device control/candidate cases, adding 384-aligned tiles and a 1% per-leaf relative-L2 oracle gate. No checkpoint, dataset or model learning.')
    for name in ('PLAN-003.md','prepare_kernel_003.py'):
        p=STUDY/name;plan['operators'][str(p.relative_to(ROOT))]=sha(p)
    plan['prerequisites'][str(equivalence.relative_to(ROOT))]=sha(equivalence)
    path=STUDY/'kernel-registration-003.json';write(path,plan)
    print(json.dumps(dict(status='prepared',snapshot=snapshot.name,registration_sha256=sha(path),cases=70,accelerator_jobs_started=False)))


if __name__=='__main__':main()
