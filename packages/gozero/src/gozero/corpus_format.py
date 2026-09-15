"""Lossless, memory-mapped KataGo feature/target corpus format."""
import numpy as np

KIND = 'katago_raw_teacher_sequences'
SPLITS = {'train': 0, 'validation': 1, 'test': 2}
GAME_DTYPE = np.dtype([
    ('game_id', 'S64'), ('opening_family', 'S64'), ('trajectory', 'S64'),
    ('split', 'u1'), ('opponent', 'u1'), ('expert_color', 'u1'), ('length', '<i4')])


def array_specs(size, positions, games):
    area = size*size
    return {
        'spatial': ((positions, (area*22+7)//8), np.dtype('u1')),
        'global_features': ((positions, 19), np.dtype('<f4')),
        'policies': ((positions, area+1), np.dtype('<f4')),
        'values': ((positions,), np.dtype('<f4')),
        'actions': ((positions,), np.dtype('<i4')),
        'legal': ((positions, (area+8)//8), np.dtype('u1')),
        'expert_offsets': ((games+1,), np.dtype('<i8')),
        'games': ((games,), GAME_DTYPE),
    }


def unpack_spatial(packed, size):
    return np.unpackbits(packed, axis=-1, count=size*size*22, bitorder='little').reshape(
        *packed.shape[:-1], size, size, 22)


def unpack_legal(packed, size):
    return np.unpackbits(packed, axis=-1, count=size*size+1, bitorder='little').astype(bool)
