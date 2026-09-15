"""Check source control-flow and default model-norm logging semantics."""
import ast
from collections import defaultdict
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=STUDY/'source-runtime-observation-001.json'
    if output.exists():raise FileExistsError(output)
    reference=json.loads((ROOT/'research/studies/katago_muon/reference-source-001.json').read_text())
    revision=reference['revision']
    source=ROOT/'.gozero/external/katago-source'/revision/('KataGo-'+revision)
    train=source/'python/train.py'
    helper=source/'python/katago/train/trainloop_helpers.py'
    metrics=source/'python/katago/train/metrics_logging.py'
    for path in (train,helper):
        if sha(path)!=reference['files'][str(path.relative_to(source))]:raise ValueError('Pinned source changed')
    namespace=dict(np=np,logging=logging,math=math,json=json)
    extracted=[]
    for path,names in ((helper,{'set_snapshot_metrics'}),(metrics,{'accumulate_metrics','log_metrics'})):
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=names:raise ValueError('Missing source functions')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        extracted.extend(dict(path=str(path.relative_to(source)),function=n.name,line=n.lineno,end_line=n.end_lineno) for n in nodes)
    observations=[]
    for weight in (0.,1.):
        sums=defaultdict(float);weights=defaultdict(float)
        for value in (3.,9.):
            data={'norm_normal_batch':value,'norm_input_batch':value/2}
            namespace['accumulate_metrics'](sums,weights,data,256,.999,weight)
            namespace['set_snapshot_metrics'](sums,weights,data,list(data))
            stream=io.StringIO();namespace['log_metrics'](sums,weights,data,stream)
            displayed=json.loads(stream.getvalue())
            for key,expected in data.items():
                if displayed[key]!=expected or sums[key]/weights[key]!=expected:
                    raise ValueError('Snapshot replacement semantics differ')
            observations.append(dict(accumulation_weight=weight,current_norm=value,
                post_log_normal_ratio=sums['norm_normal_batch']/weights['norm_normal_batch'],
                retained_weight=weights['norm_normal_batch']))
    tree=ast.parse(train.read_text());parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(sub_epochs)']
    if len(loops)!=1:raise ValueError('Ambiguous subepoch loop')
    loop=loops[0]
    reset=[n for n in loop.body if isinstance(n,ast.Assign) and ast.unparse(n)=='lookahead_counter = 0']
    if len(reset)!=1:raise ValueError('Counter reset is not at subepoch entry')
    parent=parents[loop];index=parent.body.index(loop)
    following=parent.body[index+1:]
    flush=[n for n in following if isinstance(n,ast.If) and ast.unparse(n.test)=='lookahead_k is not None']
    if len(flush)!=1 or 'param.data.copy_(slow_param_data)' not in ast.unparse(flush[0]):
        raise ValueError('Epoch slow-parameter flush changed')
    report=dict(kind='katago_source_runtime_observation',status='passed',created=time.time(),revision=revision,
        operator_sha256=sha(Path(__file__)),source_sha256={str(p.relative_to(source)):sha(p) for p in (train,helper,metrics)},
        executed_source_functions=extracted,norm_snapshot_observations=observations,
        subepoch_loop_line=loop.lineno,subepoch_counter_reset_line=reset[0].lineno,
        slow_parameter_flush_line=flush[0].lineno,flush_outside_subepoch_loop=True,
        findings=['Default print-only model norms replace prior metric sum/weight; the ratio is the latest pre-update norm snapshot, not an EMA.',
                  'Lookahead counter resets at each subepoch start. Fast parameters are copied from slow parameters after the complete subepoch loop, at epoch end.'],
        scope='Observation of unmodified pinned source functions and AST nesting, not a schedule integration, neural training result or historical checkpoint configuration.')
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    output.chmod(0o444)
    print(json.dumps(dict(status='passed',sha256=sha(output),norm_cases=len(observations))))


if __name__=='__main__':main()
