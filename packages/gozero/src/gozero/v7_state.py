"""Canonical native leaf identity, separate from the network's V7 features."""
import hashlib
import struct

import numpy as np


def validate_history(history,*,size,capacity):
    if (not isinstance(history,(tuple,list)) or len(history)>=capacity or
            any(type(a) is not int or not 0<=a<=size*size for a in history)):
        raise ValueError('Invalid or overflowing complete action history')
    return tuple(history)


def state_digest(stones,legal,*,history,size,komi):
    history=validate_history(history,size=size,capacity=2048)
    stones=np.asarray(stones);legal=np.asarray(legal)
    if (stones.shape!=(size,size) or legal.shape!=(size*size+1,) or
            not np.isin(stones,(0,1,2)).all() or not np.isin(legal,(False,True)).all() or not legal[-1]):
        raise ValueError('Invalid nonterminal native board or legal mask')
    if not np.isfinite(komi) or float(np.float32(komi))!=komi:raise ValueError('Komi must be exact finite float32')
    player=1+len(history)%2;passed=bool(history and history[-1]==size*size)
    header=b'gozero-v7-leaf-state-v1\0'+struct.pack('<HfBB',size,komi,player,passed)
    return hashlib.sha256(header+stones.astype(np.uint8).tobytes()+legal.astype(np.uint8).tobytes()).hexdigest()


def digest_from_native(features,*,history,size,komi):
    """Read Rust's six audit planes; model inputs come from the V7 worker."""
    history=validate_history(history,size=size,capacity=2048)
    x=np.asarray(features,np.float32)
    if x.size!=size*size*6:raise ValueError('Expected history=1 native audit planes')
    x=x.reshape(size,size,6)
    if not np.isfinite(x).all() or not np.isin(x[...,[0,1,2,5]],(0.,1.)).all() or np.any(x[...,0]+x[...,1]>1):
        raise ValueError('Invalid native audit planes')
    black=len(history)%2==0;passed=bool(history and history[-1]==size*size)
    expected=(float(black),np.float32((-komi if black else komi)/(size*size)),float(passed)/2.)
    for channel,value in zip((2,3,4),expected):
        if not np.all(x[...,channel]==value):raise ValueError('Native leaf metadata and full history differ')
    own,other=x[...,0],x[...,1]
    stones=own+2*other if black else 2*own+other
    legal=np.concatenate((x[...,5].reshape(-1),np.ones(1,np.float32)))
    return state_digest(stones,legal,history=history,size=size,komi=komi)


def row_digest(row,history,*,size,komi,index=-1):
    return state_digest(row['stones'][index],row['legal'][index],history=history,size=size,komi=komi)
