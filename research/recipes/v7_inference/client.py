"""Bounded process glue for exact native V7 history/suffix replay."""
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time

import numpy as np

MAX_REQUEST=4*1024*1024
MAX_OUTPUT=64*1024*1024


class Replay:
    def __init__(self,binary,binary_sha256,*,size,komi=7.5,timeout=30.):
        binary=Path(binary)
        with binary.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=binary_sha256:raise ValueError('Feature binary changed')
        if type(size) is not int or not 1<=size<=52 or not 0<timeout<=120:
            raise ValueError('Invalid replay size/timeout')
        if not np.isfinite(komi) or abs(komi)>1000 or float(np.float32(komi))!=komi:
            raise ValueError('Komi must be finite exact float32')
        self.size,self.komi,self.timeout=size,komi,timeout;self.counter=0
        self.process=subprocess.Popen([str(binary.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,bufsize=0)

    def _read(self,n,deadline):
        chunks=[]
        while n:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError('Feature response timed out')
            value=os.read(self.process.stdout.fileno(),min(n,1<<20))
            if not value:raise EOFError('Feature worker closed')
            chunks.append(value);n-=len(value)
        return b''.join(chunks)

    def __call__(self,histories,starts=None):
        size=self.size;area=size*size
        if not isinstance(histories,(list,tuple)) or not 1<=len(histories)<=128:
            raise ValueError('Invalid history batch')
        if any(not isinstance(h,(list,tuple)) or len(h)>=2048 or
               any(type(a) is not int or not 0<=a<=area for a in h) for h in histories):
            raise ValueError('Invalid complete history')
        starts=[0]*len(histories) if starts is None else starts
        if not isinstance(starts,(list,tuple)) or len(starts)!=len(histories) or any(
                type(start) is not int or not 0<=start<=len(h) for start,h in zip(starts,histories)):
            raise ValueError('Invalid suffix offsets')
        counts=[len(h)+1-start for h,start in zip(histories,starts)];rows=sum(counts)
        spatial_bytes=rows*area*22;global_bytes=rows*19*4;audit_bytes=rows*(2*area+1)
        if spatial_bytes+global_bytes+audit_bytes>MAX_OUTPUT:raise ValueError('Output exceeds bound')
        self.counter+=1
        request=dict(version=1,id=self.counter,size=size,komi=self.komi,histories=histories,starts=starts)
        raw=json.dumps(request,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAX_REQUEST:raise ValueError('Request exceeds bound')
        deadline=time.monotonic()+self.timeout
        try:
            data=memoryview(struct.pack('<I',len(raw))+raw)
            while data:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not select.select([],[self.process.stdin],[],remaining)[1]:
                    raise TimeoutError('Feature request timed out')
                written=os.write(self.process.stdin.fileno(),data[:4096]);data=data[written:]
            header_size=struct.unpack('<I',self._read(4,deadline))[0]
            if not 0<header_size<=65536:raise ValueError('Invalid feature response header')
            response=json.loads(self._read(header_size,deadline))
            if response.get('id')!=self.counter or response.get('version')!=1:raise ValueError('Feature response identity differs')
            if response.get('status')=='error':
                if set(response)!={'version','id','status','error'}:raise ValueError('Malformed error response')
                # A framed application rejection leaves the stream usable.
                return_error=response['error']
            else:
                return_error=None
                offsets=np.concatenate(([0],np.cumsum(counts,dtype=np.int64)))
                expected=dict(version=1,id=self.counter,status='ok',size=size,rows=rows,offsets=offsets.tolist(),
                              spatial_bytes=spatial_bytes,global_bytes=global_bytes,audit_bytes=audit_bytes)
                if response!=expected:raise ValueError('Feature response shape differs')
                spatial=np.frombuffer(self._read(spatial_bytes,deadline),np.uint8).reshape(rows,size,size,22)
                glob=np.frombuffer(self._read(global_bytes,deadline),'<f4').reshape(rows,19)
                audit=np.frombuffer(self._read(audit_bytes,deadline),np.uint8).reshape(rows,2*area+1)
                if np.any(spatial>1) or not np.all(spatial[...,0]) or not np.isfinite(glob).all() or np.any(audit[:,:area]>2) or np.any(audit[:,area:]>1):
                    raise ValueError('Feature payload invalid')
                result=[dict(spatial=spatial[lo:hi],global_features=glob[lo:hi],
                             stones=audit[lo:hi,:area].reshape(-1,size,size),legal=audit[lo:hi,area:].astype(bool))
                        for lo,hi in zip(offsets[:-1],offsets[1:])]
        except BaseException:
            self.close();raise
        if return_error is not None:raise ValueError('Native history rejected: '+return_error)
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.process.stdin.close();self.process.stdout.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
