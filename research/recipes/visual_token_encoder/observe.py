"""Read-only pod progress, with a separate append-only observation journal."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time


def read_rows(path):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    return rows


def observe(root):
    attempts=sorted((root/'runs').glob('pod-*/launch.json'))
    open_attempts=[x.parent for x in attempts if not x.with_name('result.json').exists()]
    selected=open_attempts or [attempts[-1].parent]
    output={'time':datetime.now(timezone.utc).isoformat(),'open_pod_attempts':len(open_attempts),'attempts':[]}
    for attempt in selected:
        launch=json.loads((attempt/'launch.json').read_text())
        row={'attempt':attempt.name,'snapshot':launch['snapshot_id'],'status':'open'}
        if (attempt/'result.json').exists():row['status']=json.loads((attempt/'result.json').read_text())['status']
        log=attempt/'rank-0/stdout.log';records=read_rows(log)
        for kind,key in [('visual_update','learning'),('visual_heldout','validation')]:
            matches=[x for x in records if x.get('kind')==kind]
            if matches:
                last=matches[-1];metrics=last.get('metrics',last)
                row[key]={'turn':last['turn'],**{k:metrics[k] for k in ['expert_kl','expert_top1','accepted','cumulative_learning_seconds'] if k in metrics}}
        if log.exists():row['seconds_since_stdout_change']=round(time.time()-log.stat().st_mtime,1)
        output['attempts'].append(row)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--journal',type=Path);a=p.parse_args();result=observe(a.workspace_root)
    raw=json.dumps(result,sort_keys=True,separators=(',',':'))+'\n'
    if a.journal:
        a.journal.parent.mkdir(parents=True,exist_ok=True)
        with a.journal.open('a') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(raw,end='',flush=True)


if __name__=='__main__':main()
