"""Typed, coarse interface to the source-qualified native rollout extension.

Importing this module does not initialize JAX. Native feature/replay storage is
transferred to NumPy; rollout/search loops and worker threads remain in Rust.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROW_META_FIELDS = ('game_id', 'network', 'action', 'simulations', 'neural_evaluations', 'terminal_evaluations')


def load_library(path: Path, expected_sha256: str):
    path = path.resolve(strict=True)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError('Native extension hash mismatch')
    name = '_gozero_native'
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path or previous._gozero_binary_sha256 != expected_sha256:
            raise RuntimeError('A different native extension is already loaded; use a fresh process')
        return previous
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError('Cannot load native extension')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if module.ABI_VERSION not in (1, 2):
        raise ValueError('Unsupported native interface version')
    module._gozero_binary_sha256 = expected_sha256
    sys.modules[name] = module
    return module


@dataclass(frozen=True)
class InferenceBatch:
    round: int
    network: int
    features: np.ndarray
    active: np.ndarray

    @property
    def active_count(self):
        return int(np.count_nonzero(self.active))


@dataclass(frozen=True)
class CompletedRows:
    features: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    root_values: np.ndarray
    metadata: np.ndarray
    ownership: np.ndarray
    games: list[dict]


class Actors:
    def __init__(self, config: dict, library: Path, library_sha256: str, *, checkpoint: str | None = None):
        # Canonicalization also rejects NaN and ensures our shape view reflects
        # the exact configuration accepted by Rust, rather than a mutable caller dict.
        text = json.dumps(config, sort_keys=True, allow_nan=False)
        self.config = json.loads(text)
        self._module = load_library(library, library_sha256)
        if self._module.ABI_VERSION != 2:
            raise ValueError('Actors with ownership targets requires native interface version 2')
        self._actors = self._module.Actors(text, checkpoint)
        self._closed = False
        self._shape = (self.config['size'], self.config['size'], 2*self.config['history']+4)
        self._actions = self.config['size']**2+1

    def _batch(self, values):
        sequence, network, features, active = values
        features = features.reshape((len(active), *self._shape))
        features.flags.writeable = False
        active.flags.writeable = False
        return InferenceBatch(sequence, network, features, active)

    def start(self, network: int):
        return self._batch(self._actors.start(network))

    def checkpoint(self) -> str:
        """Complete native state at a real-move boundary, including all RNGs.

        Full superko and observation histories are reconstructed by replay on
        restore. Checkpointing an in-flight search is rejected.
        """
        return self._actors.checkpoint()

    def evaluate(self, batch: InferenceBatch, policy_logits, values):
        policies = np.ascontiguousarray(policy_logits, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        return self._batch(self._actors.evaluate(batch.round, batch.network, policies, values))

    def prefetch(self, batch: InferenceBatch, limit: int):
        """Prepare up to limit direct root children per game, actor-major.

        The preceding ordinary batch becomes stale. Complete this request even
        when every slot is padding, then continue the returned ordinary batch.
        """
        if getattr(self._module, 'ROOT_PREFETCH_ABI_VERSION', None) != 1:
            raise ValueError('Root prefetch requires its qualified native interface version 1')
        return self._batch(self._actors.prefetch(batch.round, batch.network, limit))

    def evaluate_prefetch(self, batch: InferenceBatch, policy_logits, values):
        policies = np.ascontiguousarray(policy_logits, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        return self._batch(self._actors.evaluate_prefetch(batch.round, batch.network, policies, values))

    def prefetch_counters(self):
        """Process-lifetime (active prefetched NN rows, consumed cache hits).

        Their difference at a committed move boundary is wasted prediction work.
        These diagnostic counters restart at zero after native restoration.
        """
        return self._actors.prefetch_counters()

    def commit(self):
        features, policies, outcomes, root_values, metadata, ownership, games = self._actors.commit()
        rows = len(outcomes)
        features = features.reshape((rows, *self._shape))
        policies = policies.reshape((rows, self._actions))
        metadata = metadata.reshape((rows, len(ROW_META_FIELDS)))
        ownership = ownership.reshape((rows, self.config['size'], self.config['size']))
        for array in (features, policies, outcomes, root_values, metadata, ownership):
            array.flags.writeable = False
        return CompletedRows(features, policies, outcomes, root_values, metadata, ownership, json.loads(games))

    def close(self):
        if not self._closed:
            self._actors.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
