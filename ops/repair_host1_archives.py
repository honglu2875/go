"""Verified lossless archival of already-audited closed legacy arrays on host1."""
import argparse,gzip,hashlib,json,os,shutil,sys,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path('/workspace/go');source=root/'.gozero/snapshots/970bedd265b21422f981097a9f9e4f881e77ddc94e38f4bd1aacfe6a0f782e41'
sys.path.insert(0,str(source/'packages/gozero/src'))
from gozero import checkpoint_archives as archives,checkpoints
from gozero.snapshots import verify,read_json,canonical_json
verify(source)
if checkpoints.sha256(a.plan)!=a.plan_sha256 or a.output.exists():raise ValueError('Plan or output changed')
plan=read_json(a.plan);report={'schema_version':1,'kind':'host1_lossless_archival_repair','started_unix':time.time(),'plan_sha256':a.plan_sha256,'source':source.name,'operator_sha256':checkpoints.sha256(Path(__file__)),'status':'running','entries':[],'free_before':shutil.disk_usage(root).free}
a.output.parent.mkdir(parents=True,exist_ok=True)
try:
 selected=[row for row in plan['entries'] if (root/row['original_path']).is_file()]
 if not selected:raise ValueError('No expected legacy arrays found')
 first=selected[0];row=archives.archive(root,first['original_path'],first['manifest_sha256'])
 # Exercise the standard restore and checkpoint reader on RAM-backed scratch.
 with tempfile.TemporaryDirectory(prefix='gozero-archive-roundtrip-',dir='/dev/shm') as temporary:
  scratch=Path(temporary)
  for name in (row['archive_path'],row['receipt_path']):
   dest=scratch/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
  restored=archives.restore(scratch,row['receipt_path'],row['receipt_sha256'],destination='restored/arrays.npz')
  original=root/first['original_path']
  for name in ('manifest.json','state.json','actors.json'):shutil.copyfile(original.with_name(name),restored.with_name(name))
  state,arrays,actors=checkpoints.read(restored.parent,expected_manifest_sha256=first['manifest_sha256'])
  if checkpoints.sha256(restored)!=first['original_sha256']:raise ValueError('RAM round trip changed original bytes')
  report['roundtrip']={'ram_backed':True,'array_count':len(arrays),'original_sha256':first['original_sha256'],'original_preserved':original.is_file()}
  del arrays,state,actors
 with a.output.with_suffix('.progress.jsonl').open('x') as log:
  for item in selected:
   if shutil.disk_usage(root).free>=6*2**30:break
   record=archives.archive(root,item['original_path'],item['manifest_sha256'],remove_original=True)
   if record['original_sha256']!=item['original_sha256'] or record['archive_sha256']!=item['archive_sha256']:raise ValueError('Previously audited archival identity differs')
   report['entries'].append(record);log.write(canonical_json(record).decode());log.flush()
   print(json.dumps({'archived':len(report['entries']),'free_bytes':shutil.disk_usage(root).free}),flush=True)
 report['free_after']=shutil.disk_usage(root).free
 if report['free_after']<3*2**30:raise ValueError('Insufficient restored staging headroom')
 report['status']='passed'
except BaseException as e:report['error']=repr(e);raise
finally:
 report['finished_unix']=time.time();a.output.write_bytes(canonical_json(report));print(json.dumps({'status':report['status'],'output':str(a.output)}),flush=True)
