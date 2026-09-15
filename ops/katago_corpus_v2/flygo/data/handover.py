"""Transfer a generation allocation only after its recorded processes exit."""
import json
import os
from pathlib import Path
import signal
import time


def probe(pid):
    try:fields=Path('/proc',str(pid),'stat').read_text().rpartition(')')[2].split()
    except FileNotFoundError:return None
    return dict(identity=fields[19],state=fields[0])


def active(record,read_process=probe):
    if type(record['pid']) is not int or record['pid']<=1 or not str(record['identity']).isdigit():
        raise ValueError('Invalid process identity')
    current=read_process(record['pid'])
    return current is not None and current['identity']==record['identity'] and current['state']!='Z'


def lane_clear(h,index,read_process=probe):
    return not any(active(record,read_process) for record in h['workers'][str(index)]['processes'])


def all_clear(h,read_process=probe):
    return all(lane_clear(h,int(index),read_process) for index in h['workers'])


def release_previous(h,receipt,read_process=probe,send_signal=os.kill):
    if receipt.exists():
        if json.loads(receipt.read_text())['supervisor']!=h['supervisor']:
            raise ValueError('Previous-supervisor release identity changed')
        return True
    if not all_clear(h,read_process):return False
    if active(h['supervisor'],read_process):
        send_signal(h['supervisor']['pid'],signal.SIGCONT)
    record=dict(status='released',supervisor=h['supervisor'],released_unix=time.time(),
                all_recorded_workers_and_engines_inactive=True)
    with receipt.open('x') as f:json.dump(record,f,indent=2)
    receipt.chmod(0o444)
    return True
