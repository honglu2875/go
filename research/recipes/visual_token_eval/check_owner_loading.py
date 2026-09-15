#!/usr/bin/env python3
"""Exercise the evaluation parameter reader on real audited checkpoints, on CPU."""
import argparse
import gc
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json,canonical_json,verify
from gozero.checkpoints import sha256
import evaluate_games

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(SOURCE)
    rows=[]
    for case in read_json(a.inputs)['cases']:
        params=evaluate_games.owner_parameters(a.workspace_root,case)
        rows.append({'label':case['label'],'parameter_count':sum(x.size for x in params.values()),
            'parameter_elements_sha256':evaluate_games.array_digest(params)})
        del params;gc.collect()
        try:evaluate_games.owner_parameters(a.workspace_root,{**case,'training_result_sha256':'0'*64})
        except ValueError:pass
        else:raise ValueError('Modified result hash was accepted')
    result={'status':'passed','operator_snapshot':SOURCE.name,'inputs_sha256':sha256(a.inputs),'cases':rows,
        'scope':'Authenticated CPU reads of real p_ arrays and deliberate changed-result-hash rejection; no inference or learning.'}
    with a.output.open('xb') as f:f.write(canonical_json(result))
    a.output.chmod(0o444);print(canonical_json({'status':'passed','sha256':sha256(a.output)}).decode().strip())

if __name__=='__main__':main()
