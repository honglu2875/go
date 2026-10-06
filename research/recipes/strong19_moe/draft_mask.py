"""Training frames [draft board, full board, action]; draft keys never enter history."""
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import _ComputableMask


def visible(q,k,valid_length):
    qframe,qkind=q//3,q%3;kframe,kkind=k//3,k%3
    past=(kframe<qframe)&(kkind!=0)
    current=(kframe==qframe)&(((qkind==0)&(kkind==0))|((qkind==1)&(kkind==1))|((qkind==2)&(kkind>=1)))
    return (past|current)&(k<valid_length)


class DraftMask(_ComputableMask):
    def __init__(self,shape,valid_length):
        if not 0<valid_length<=shape[1] or valid_length%3:raise ValueError('Invalid draft frame extent')
        self.valid_length=valid_length
        super().__init__(shape,lambda q,k:visible(q,k,valid_length))

    def __eq__(self,other):
        if not isinstance(other,type(self)):return NotImplemented
        return self.shape==other.shape and self.valid_length==other.valid_length and np.array_equal(self.q_sequence,other.q_sequence)

    def __hash__(self):return hash((type(self),self.shape,self.valid_length,self.q_sequence.tobytes()))
