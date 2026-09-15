"""Audit full-model native/GTP qualification receipts for both baseline arms."""
import argparse,json,math
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import verify,read_json,canonical_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--cnn',required=True);p.add_argument('--transformer',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
 if a.output.exists():raise FileExistsError(a.output)
 root=SOURCE.parents[2];arms={};evidence={}
 for name,attempt,registration in [('cnn',a.cnn,'cnn_service_registration.json'),('transformer',a.transformer,'cnn_transformer_service_registration.json')]:
  regpath=root/'research/studies/visual_causal'/registration;reg=read_json(regpath);evidence[str(regpath.relative_to(root))]=sha256(regpath)
  d=root/'runs'/attempt;closed=read_json(d/'result.json');evidence[str((d/'result.json').relative_to(root))]=sha256(d/'result.json')
  if closed['status']!='passed' or closed['snapshot_id']!=reg['snapshot_id']:raise ValueError('Service source or completion differs')
  c=read_json(root/'.gozero/snapshots'/reg['snapshot_id']/'resolved_config.json');reports=[]
  for host in range(4):
   path=d/f'rank-{host}/artifacts/result.json';r=read_json(path);evidence[str(path.relative_to(root))]=sha256(path)
   if r['status']!='passed' or r['candidate_sha256']!=reg['candidate_sha256'] or not r['training_complete']:raise ValueError('Wrong candidate or incomplete service')
   q=r['qualification']
   if not q['all_heads_compared'] or q['exact_leaf_predictions']<=0 or not 0<=q['maximum_absolute_error']<=c['qualification']['tolerance']:raise ValueError('Numerical qualification failed')
   if len(r['gtp_probe']['clients'])!=2 or any(x['boards_checked']!=4 for x in r['gtp_probe']['clients']):raise ValueError('Incomplete GTP probe')
   reports.append(r)
  arms[name]={'attempt':attempt,'exact_leaf_predictions':sum(r['qualification']['exact_leaf_predictions'] for r in reports),
    'maximum_absolute_error':max(r['qualification']['maximum_absolute_error'] for r in reports),
    'maximum_policy_total_variation':max(q['maximum_total_variation'] for r in reports for q in r['qualification']['legal_policy_comparisons'].values()),
    'greedy_differences':sum(q['greedy_differences'] for r in reports for q in r['qualification']['legal_policy_comparisons'].values()),
    'gtp_boards':sum(x['boards_checked'] for r in reports for x in r['gtp_probe']['clients']),
    'attempt_chip_hours':closed['reserved_chip_hours']}
 a.output.write_bytes(canonical_json({'schema_version':1,'kind':'baseline_full_model_service_audit','status':'passed','operator_snapshot':SOURCE.name,'arms':arms,'evidence':evidence,
  'scope':'Full training-path versus exact native leaf inference under explicit floating-point tolerance, plus real GTP client boards. Distinct architectures and history contracts; no bitwise search-equivalence, speedup or strength claim.'}));print(json.dumps(arms))
if __name__=='__main__':main()
