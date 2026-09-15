#!/usr/bin/env python3
"""Export exact parameters from a committed checkpoint for learning-curve evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact,checkpoint_parameters,validate_candidate
from gozero.snapshots import canonical_json,read_json,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lineage',required=True)
    args=p.parse_args();verify(SOURCE);root=args.artifacts_root.resolve();checkpoint=artifact(root,args.checkpoint)
    state=read_json(checkpoint/'state.json')
    reference={'path':str(checkpoint.relative_to(root)),'manifest_sha256':sha256(checkpoint/'manifest.json'),
               'group_sha256':sha256(checkpoint.parent/(checkpoint.name+'.group.json'))}
    state,parameters=checkpoint_parameters(root,reference,state['snapshot_id'],state['counters']['updates'])
    output=args.output.resolve()
    if not output.is_relative_to(root):raise ValueError('Export must remain inside the artifact root')
    output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    weights=output/'model_export.npz';np.savez(weights,**{key:parameters[key] for key in sorted(parameters)})
    candidate={'schema_version':2,'training_snapshot':state['snapshot_id'],'checkpoint':reference,
               'model_export_path':str(weights.relative_to(root)),'model_export_sha256':sha256(weights),
               'network_version':state['counters']['updates'],'training_lineage':args.lineage}
    validate_candidate(root,candidate)
    (output/'candidate.json').write_bytes(canonical_json(candidate))
    receipt={'schema_version':1,'kind':'verified_checkpoint_model_export','status':'passed','export_source_snapshot':SOURCE.name,
             'candidate_sha256':sha256(output/'candidate.json'),'checkpoint_turn':state['turn'],
             'local_real_moves':state['counters']['real_moves'],'world_size':state['world_size'],
             'native_binary_sha256':state['native_sha256'],'model_export_sha256':candidate['model_export_sha256'],
             'parameter_arrays':len(parameters),'no_training_performed':True}
    (output/'receipt.json').write_bytes(canonical_json(receipt));verify(SOURCE)
    for file in output.iterdir():file.chmod(0o444)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
