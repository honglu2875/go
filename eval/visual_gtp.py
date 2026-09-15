#!/usr/bin/env python3
"""Native Go search with complete visual-history evaluation by a local TPU owner."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.native import load_library
from gozero.snapshots import read_json, verify
from gozero.visual_artifacts import load_module
from gozero.visual_history import observation_sha256
from gozero.visual_rpc import Client
from learned_gtp import Engine as BaseEngine, serve


def contract(c):
    fields = ('schema_version kind size komi scoring simulations cpuct fpu_reduction gumbel '
              'max_search_edges max_game_moves cache_positions native_receipt_sha256')
    if (not isinstance(c, dict) or set(c) != set(fields.split()) or c['schema_version'] != 1
            or c['kind'] != 'visual_mcts_inference'):
        raise ValueError('Invalid visual search contract')
    for key, low, high in [('size', 1, 25), ('simulations', 0, 1000000),
                           ('max_game_moves', 2, 1000000), ('cache_positions', 2, 1000000),
                           ('max_search_edges', c['size'] ** 2 + 1, 1000000)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError('Invalid ' + key)
    if (not isinstance(c['cpuct'], (int, float)) or not math.isfinite(c['cpuct']) or c['cpuct'] < 0
            or not math.isfinite(c['komi']) or c['scoring'] not in ('raw_area', 'pass_alive_area')
            or c['max_game_moves'] + c['simulations'] >= c['cache_positions']):
        raise ValueError('Search rules or complete-context capacity differ')
    if c['gumbel'] is not None and c['gumbel'].get('gumbel_scale') != 0:
        raise ValueError('Evaluation search must disable Gumbel sampling noise')
    return c


def identity(source, candidate, inference):
    return {'source': source.name, 'candidate_sha256': sha256(candidate),
            'inference_sha256': sha256(inference)}


class Engine(BaseEngine):
    def __init__(self, args):
        manifest = verify(SOURCE); self.np = np
        for path in [args.candidate, args.inference_config]:
            if not path.resolve().is_relative_to(SOURCE):
                raise ValueError('Candidate and search configuration must be frozen with the adapter')
        candidate = read_json(args.candidate); c = contract(read_json(args.inference_config))
        self.observations = load_module(SOURCE, manifest['recipe'], 'observations')
        if sha256(args.native_receipt) != c['native_receipt_sha256']:
            raise ValueError('Native inference receipt differs from registered search')
        receipt = read_json(args.native_receipt)
        self.native = load_library(args.native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
        if getattr(self.native, 'CAUSAL_GAME_ABI_VERSION', None) != 1:
            raise ValueError('Pending-leaf complete histories are unavailable')
        self.rpc = Client(args.socket, identity(SOURCE, args.candidate, args.inference_config))
        self.size, self.komi, self.scoring = (c[k] for k in ('size', 'komi', 'scoring'))
        self.supported_size = self.size; self.history = 1
        self.simulations, self.cpuct = c['simulations'], c['cpuct']
        self.fpu_reduction, self.gumbel, self.score_utility = c['fpu_reduction'], c['gumbel'], None
        self.max_game_moves, self.max_search_edges = c['max_game_moves'], c['max_search_edges']
        self.cache_positions, self.network = c['cache_positions'], candidate['network_version']
        self.reset()
        print(json.dumps({'kind': 'engine_ready', 'adapter': 'visual_exact_cached_rpc',
                          **identity(SOURCE, args.candidate, args.inference_config), 'slot': self.rpc.slot,
                          'native_binary_sha256': receipt['binary_sha256'], 'simulations': self.simulations}),
              file=sys.stderr, flush=True)

    def reset(self):
        self.game = self.native.Game(json.dumps({'size': self.size, 'komi': self.komi, 'scoring': self.scoring,
            'history': 1, 'simulations': self.simulations, 'cpuct': self.cpuct,
            'fpu_reduction': self.fpu_reduction, 'gumbel': self.gumbel, 'max_search_edges': self.max_search_edges}, allow_nan=False))
        self.plies = 0; self.last_stats = None

    def evaluate_leaf(self, request, features):
        history = self.game.request_history(request).tolist()
        if not self.plies <= len(history) <= self.plies + self.simulations or len(history) >= self.cache_positions:
            raise ValueError('Native pending leaf exceeds complete-history search contract')
        board = self.observations.from_native(features.reshape(self.size, self.size, 6))
        response = self.rpc.predict(history, observation_sha256(board))
        return np.asarray(response['expert_logits'], np.float32), response['value']

    def command(self, name, args):
        if name == 'version' and not args:
            return 'gozero-visual-exact-cached-rpc-v1'
        if name in ('play', 'genmove'):
            if self.plies >= self.max_game_moves:
                raise ValueError('Registered game cap reached')
            response = super().command(name, args); self.plies += 1
            return response
        return super().command(name, args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'inference-config', 'native-receipt', 'socket'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(); engine = Engine(args)
    try:
        serve(engine)
    finally:
        engine.rpc.close()


if __name__ == '__main__':
    main()
