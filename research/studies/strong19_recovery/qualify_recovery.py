"""Actual disk-backup recovery after erasing disposable inputs and checkpoints."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.durable_files import atomic_json,sha256


def main():
    plan=json.loads((STUDY/'harness-plan-001.json').read_text())
    operator=ROOT/'research/studies/strong19_train/qualify_harness.py'
    assert sha256(operator)=='88389cf66f567a2610430c98f2a23218432b492dd0276b115be626459751e2a9'
    spec=importlib.util.spec_from_file_location('original_harness',operator)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    start=time.time();result=dict(status='running',models={},created=start,operator_sha256=sha256(Path(__file__)))
    try:
        for arm,snapshot in plan['snapshots'].items():
            folder=STUDY/'harness-001'/arm;snap=ROOT/'.gozero/snapshots'/snapshot
            full=harness.run_stage(sys.executable,snap,folder/'full')
            prefix=harness.run_stage(sys.executable,snap,folder/'prefix',stop=2)
            cp=Path(prefix['latest_checkpoint']['path']);backup=Path(prefix['latest_checkpoint']['disk']['target'])
            mirror=json.loads((backup/'mirror.json').read_text())
            for name,r in mirror['files'].items():
                assert sha256(backup/name)==r['sha256'] and (backup/name).stat().st_size==r['bytes']
            fixture=Path(plan['fixture']);assert fixture.parent==Path('/dev/shm/gozero-datasets')
            shutil.rmtree(fixture);shutil.rmtree(cp);cp.with_suffix('.group.json').unlink()
            assert not fixture.exists() and not cp.exists()
            shutil.copytree(plan['durable_fixture'],fixture)
            cp.mkdir()
            for name in ('manifest.json','arrays.npz','state.json','actors.json'):
                shutil.copyfile(backup/'host-0'/name,cp/name)
            shutil.copyfile(backup/'host-0/group.json',cp.with_suffix('.group.json'))
            resumed=harness.run_stage(sys.executable,snap,folder/'resumed',resume=cp)
            audit=harness.audit_model(folder,snap,full,prefix,resumed)
            result['models'][arm]=dict(audit,disposable_fixture_removed=True,primary_checkpoint_removed=True,
                restored_from_disk_bundle=True,fresh_process=True)
        assert len({r['position_exposures'] for r in result['models'].values()})==1
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result.update(seconds=time.time()-start,scope='CPU four-device full-vs-2+2 exact continuation, both model variants, actual input/checkpoint loss and disk restore. Multi-host TPU remapping is a separate qualification.')
        atomic_json(STUDY/'harness-qualification-001.json',result,replace=False)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
