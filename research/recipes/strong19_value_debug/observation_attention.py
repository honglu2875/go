"""Known board tokens may communicate; predicted/future actions remain hidden.

The stride is [board patches, policy readout, actual action]. A board is a
single observation, so its patches need no artificial raster-order causality.
Splash's lazy computable mask avoids allocating a full history-square array.
"""
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import _ComputableMask


def visible(queries, keys, stride):
    causal = keys <= queries
    if not stride:
        return causal
    same_board = ((queries // stride == keys // stride)
                  & (queries % stride < stride - 1)
                  & (keys % stride < stride - 1))
    return causal | same_board


class ObservationMask(_ComputableMask):
    """Lazy block-prefix mask, with an explicit limit for padded kernel keys."""

    def __init__(self, shape, stride, valid_length):
        if type(stride) is not int or stride < 3:
            raise ValueError('Observation stride must include a patch, readout and action')
        if not 0 < valid_length <= shape[1]:
            raise ValueError('Invalid unpadded key extent')
        self.stride = stride
        self.valid_length = valid_length
        super().__init__(shape, lambda q, k: visible(q, k, stride) & (k < valid_length))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.shape == other.shape and self.stride == other.stride
                and self.valid_length == other.valid_length
                and np.array_equal(self.q_sequence, other.q_sequence))

    def __hash__(self):
        return hash((type(self), self.shape, self.stride, self.valid_length,
                     self.q_sequence.tobytes()))
