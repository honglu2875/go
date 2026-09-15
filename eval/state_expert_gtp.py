#!/usr/bin/env python3
"""Pinned expert-only models evaluate exact native MCTS leaf inputs."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.causal_artifacts import validate
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import read_json, verify
from causal_gtp import inference_contract
from learned_gtp import Engine as BoardEngine, serve


class Engine(BoardEngine):
    def __init__(self, args):
        import jax
        import jax.numpy as jnp
        import numpy as np
        self.jax = jax; self.np = np; verify(SOURCE)
        if jax.default_backend() != 'cpu' or jax.process_count() != 1:
            raise ValueError('Expert GTP adapter requires single-process CPU inference')
        descriptor = read_json(args.candidate)
        if descriptor['kind'] != 'state_expert_policy':
            raise ValueError('This adapter requires an explicit state-expert descriptor')
        identity = validate(args.artifacts_root, descriptor)
        snapshot, config = identity['snapshot'], identity['config']
        infer = args.inference_config.resolve()
        if not infer.is_relative_to(SOURCE):
            raise ValueError('Expert inference contract must be frozen with the adapter')
        c = inference_contract(read_json(infer), config['model'])
        if c['simulations'] != args.simulations or c['cpuct'] != args.cpuct:
            raise ValueError('Command-line search budget differs from frozen expert inference')
        dataset = artifact(args.artifacts_root, str(Path(config['dataset']['path']) / 'manifest.json'))
        if sha256(dataset) != config['dataset']['manifest_sha256']:
            raise ValueError('Expert training dataset identity differs')
        data = read_json(dataset)
        if data['kind'] != 'board_causal_teacher_dataset' or any(data['rules'][k] != c[k] for k in ('size', 'komi', 'scoring')):
            raise ValueError('Inference rules differ from exact-state training data')
        self.history = c['history']; self.net = config['model']; self.size = c['size']; self.komi = c['komi']; self.scoring = c['scoring']
        self.fpu_reduction = c['fpu_reduction']; self.score_utility = None; self.gumbel = c['gumbel']
        self.supported_size = self.size; self.max_game_moves = c['max_game_moves']; self.max_search_edges = c['max_search_edges']
        self.network = descriptor['network_version']; self.simulations = c['simulations']; self.cpuct = c['cpuct']
        recipe = snapshot / identity['manifest']['recipe']
        # Each GTP process loads one frozen recipe. Reject an existing helper
        # from another snapshot instead of accidentally sharing Python's cache.
        for name in ('history_model', 'spatial_model', 'state_features'):
            prior = sys.modules.get(name)
            if prior is not None and Path(prior.__file__).resolve() != (recipe / (name + '.py')).resolve():
                raise ValueError('Conflicting frozen expert helper already loaded: ' + name)
        sys.path.insert(0, str(recipe))
        spec = importlib.util.spec_from_file_location('frozen_state_expert_' + snapshot.name, recipe / 'model.py')
        model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)
        from state_features import native_inputs
        self.native_inputs = native_inputs
        initial = model.initialize(config['seed'], self.net)
        leaves, tree = jax.tree.flatten(initial)
        schema = [{'path': jax.tree_util.keystr(path), 'shape': list(a.shape), 'dtype': str(a.dtype)}
                  for path, a in jax.tree_util.tree_flatten_with_path(initial)[0]]
        if schema != identity['model_schema']:
            raise ValueError('Expert parameter tree differs from trained checkpoint')
        self.params = tree.unflatten([jnp.asarray(identity['arrays'][f'p_{i:04d}']) for i in range(len(leaves))])
        receipt = read_json(args.native_receipt)
        if receipt['snapshot_id'] != args.inference_native_snapshot:
            raise ValueError('Expert inference native source differs')
        verify(artifact(args.artifacts_root, '.gozero/snapshots/' + receipt['snapshot_id']))
        self.native = load_library(args.native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
        if getattr(self.native, 'CAUSAL_GAME_ABI_VERSION', None) != 1:
            raise ValueError('Native pending-leaf history ABI is unavailable')
        self.tokens = np.full((1, self.net['max_tokens']), self.size ** 2 + 2, np.int32)
        self.tokens[0, 0] = self.size ** 2 + 1
        def predict(params, tokens, lengths, stones, legal, passed):
            return model.leaf_predictions(params, tokens, lengths, stones, legal, passed, self.net)
        dummy = (self.params, self.tokens, np.zeros(1, np.int32), np.zeros((1, self.size ** 2), np.uint8),
                 np.ones((1, self.size ** 2 + 1), bool), np.zeros(1, bool))
        self.forward = jax.jit(predict).lower(*dummy).compile(); jax.block_until_ready(self.forward(*dummy))
        self.reset()
        print(json.dumps({'kind': 'engine_ready', 'adapter': 'state_expert_native_leaf', 'candidate_sha256': sha256(args.candidate),
            'training_snapshot': snapshot.name, 'model_code_sha256': identity['model_code_sha256'],
            'helper_code_sha256': {name: sha256(recipe / name) for name in ('history_model.py', 'spatial_model.py', 'state_features.py')},
            'weights_sha256': descriptor['model_export_sha256'], 'native_snapshot': receipt['snapshot_id'],
            'native_sha256': receipt['binary_sha256'], 'inference_config_sha256': sha256(infer),
            'backend': jax.default_backend(), 'simulations_excluding_root': self.simulations, 'cpuct': self.cpuct,
            'maximum_game_moves': self.max_game_moves, 'context_tokens': self.net['max_tokens'],
            'expert_architecture': self.net['architecture'], 'state_feature_channels': 9,
            'fixed_observer': identity['training_result']['fixed_observer'], 'observer_used_as_search_policy': False}), file=sys.stderr, flush=True)

    def reset(self):
        self.game = self.native.Game(json.dumps({'size': self.size, 'komi': self.komi, 'history': self.history,
            'scoring': self.scoring, 'simulations': self.simulations, 'cpuct': self.cpuct,
            'max_search_edges': self.max_search_edges, 'fpu_reduction': self.fpu_reduction, 'gumbel': self.gumbel}, allow_nan=False))
        self.last_stats = None; self.plies = 0

    def evaluate_leaf(self, request, features):
        history = self.game.request_history(request); length = len(history)
        if length < self.plies or length > self.plies + self.simulations or length >= self.net['max_tokens']:
            raise ValueError('Pending leaf violates the fixed comparison context')
        stones, _, legal, passed = self.native_inputs(features, history, size=self.size, history_planes=self.history, komi=self.komi)
        legal = self.np.concatenate((legal, self.np.ones((1, 1), bool)), axis=-1)
        self.tokens.fill(self.size ** 2 + 2); self.tokens[0, 0] = self.size ** 2 + 1; self.tokens[0, 1:length+1] = history
        logits, value = self.jax.device_get(self.forward(self.params, self.tokens, self.np.asarray([length], self.np.int32), stones, legal, passed))
        return logits[0], float(value[0])

    def command(self, name, args):
        if name == 'version' and not args:
            return 'gozero-state-expert-native-leaf-v1'
        if name in ('play', 'genmove'):
            if self.plies >= self.max_game_moves:
                raise ValueError('Registered expert game cap reached')
            response = super().command(name, args); self.plies += 1; return response
        return super().command(name, args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True); p.add_argument('--native-receipt', type=Path, required=True)
    p.add_argument('--artifacts-root', type=Path, required=True); p.add_argument('--simulations', type=int, required=True)
    p.add_argument('--cpuct', type=float, required=True); p.add_argument('--inference-config', type=Path, required=True)
    p.add_argument('--inference-native-snapshot', required=True); a = p.parse_args()
    if os.environ.get('JAX_PLATFORMS', 'cpu') != 'cpu':
        p.error('This GTP adapter is CPU-only')
    os.environ['JAX_PLATFORMS'] = 'cpu'; serve(Engine(a))


if __name__ == '__main__':
    main()
