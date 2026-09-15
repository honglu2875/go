"""Coarse Go interface. Rules, search, worker threads and history live in Rust.

Actions are row-major from the top left; size**2 is pass. Absolute stone/color
codes are 0=empty, 1=Black, 2=White. Neural values and ownership targets use the
player-to-move perspective. Features are NHWC with 2*history+4 channels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

from . import _native

if _native.ABI_VERSION != 1:
    raise ImportError("Unsupported FlyGo native ABI; rebuild the extension")


@dataclass(frozen=True)
class GumbelConfig:
    max_considered_actions: int = 16
    value_scale: float = 0.1
    maxvisit_init: float = 50.0
    rescale_values: bool = True
    gumbel_scale: float = 0.0


@dataclass(frozen=True)
class GameConfig:
    size: int = 9
    komi: float = 7.5
    scoring: str = "pass_alive_area"
    history: int = 4
    simulations: int = 0
    cpuct: float = 1.5
    fpu_reduction: float | None = None
    gumbel: GumbelConfig | None = None
    max_search_edges: int = 100_000

    @property
    def actions(self) -> int:
        return self.size**2 + 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.size, self.size, 2 * self.history + 4


@dataclass(frozen=True)
class ActorConfig(GameConfig):
    games: int = 32
    workers: int = 4
    worker_cpus: tuple[int, ...] = ()
    max_game_moves: int = 324
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.0
    temperature_early: float = 1.0
    temperature_late: float = 0.0
    temperature_moves: int = 16
    seed: int = 1
    actor_offset: int = 0


def _json(config) -> str:
    return json.dumps(asdict(config), sort_keys=True, allow_nan=False)


def _readonly(array: np.ndarray, shape=None) -> np.ndarray:
    if shape is not None:
        array = array.reshape(shape)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class BoardState:
    size: int
    to_play: int
    terminal: bool
    white_score: float | None
    stones: np.ndarray


@dataclass(frozen=True)
class SearchRequest:
    identity: tuple[int, int, int, bool] | None
    features: np.ndarray


@dataclass(frozen=True)
class SearchResult:
    action: int
    policy: np.ndarray
    value: float
    simulations: int
    neural_evaluations: int
    terminal_evaluations: int


class Game:
    """One externally driven game; use Actors for batched high-throughput search."""

    def __init__(self, config: GameConfig = GameConfig()):
        self.config = config
        self._game = _native.Game(_json(config))

    def play(self, color: int, action: int) -> None:
        self._game.play(color, action)

    def state(self) -> BoardState:
        size, color, terminal, score, stones = self._game.state()
        return BoardState(size, color, terminal, score if terminal else None,
                          _readonly(stones, (size, size)))

    def legal(self) -> np.ndarray:
        return _readonly(np.asarray(self._game.legal(), dtype=np.int32))

    def ownership(self) -> np.ndarray:
        """Absolute color codes under the configured scoring profile."""
        return _readonly(self._game.ownership(), (self.config.size, self.config.size))

    def _request(self, result) -> SearchRequest:
        identity, features = result
        shape = self.config.shape if identity is not None else (0,)
        return SearchRequest(identity, _readonly(features, shape))

    def start(self, network: int = 0) -> SearchRequest:
        return self._request(self._game.start(network))

    def evaluate(self, request: SearchRequest, logits, value: float) -> SearchRequest:
        if request.identity is None:
            raise ValueError("No pending evaluation")
        return self._request(self._game.evaluate(
            request.identity, np.ascontiguousarray(logits, dtype=np.float32), value))

    def request_history(self, request: SearchRequest) -> np.ndarray:
        if request.identity is None:
            raise ValueError("No pending evaluation")
        return _readonly(self._game.request_history(request.identity))

    def inspect_search(self) -> dict:
        return json.loads(self._game.inspect_search())

    def finish(self) -> SearchResult:
        action, policy, value, simulations, neural, terminal = self._game.finish()
        return SearchResult(action, _readonly(policy), value, simulations, neural, terminal)


@dataclass(frozen=True)
class InferenceBatch:
    round: int
    network: int
    features: np.ndarray
    active: np.ndarray

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active))


ROW_META_FIELDS = ("game_id", "network", "action", "simulations",
                   "neural_evaluations", "terminal_evaluations")


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
    """Persistent native games with one batched evaluation request per round.

    commit() advances actual moves and returns targets only for completed games.
    Truncated games have metadata but no invented terminal targets. Checkpoint
    at a committed move boundary; worker count may change on restoration.
    """

    def __init__(self, config: ActorConfig = ActorConfig(), *, checkpoint: str | None = None):
        self.config = config
        self._actors = _native.Actors(_json(config), checkpoint)

    def _batch(self, result) -> InferenceBatch:
        sequence, network, features, active = result
        return InferenceBatch(sequence, network,
                              _readonly(features, (len(active), *self.config.shape)),
                              _readonly(active))

    def start(self, network: int = 0) -> InferenceBatch:
        return self._batch(self._actors.start(network))

    def evaluate(self, batch: InferenceBatch, policy_logits, values) -> InferenceBatch:
        return self._batch(self._actors.evaluate(
            batch.round, batch.network,
            np.ascontiguousarray(policy_logits, dtype=np.float32),
            np.ascontiguousarray(values, dtype=np.float32)))

    def commit(self) -> CompletedRows:
        features, policies, outcomes, values, metadata, ownership, games = self._actors.commit()
        rows = len(outcomes)
        size = self.config.size
        return CompletedRows(
            _readonly(features, (rows, *self.config.shape)),
            _readonly(policies, (rows, self.config.actions)),
            _readonly(outcomes), _readonly(values),
            _readonly(metadata, (rows, len(ROW_META_FIELDS))),
            _readonly(ownership, (rows, size, size)), json.loads(games))

    def checkpoint(self) -> str:
        return self._actors.checkpoint()

    def close(self) -> None:
        self._actors.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


@dataclass(frozen=True)
class ReplayObservations:
    stones: np.ndarray
    legal: np.ndarray
    outcomes: list[dict]


def _integers(values, dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("Packed histories require one-dimensional integer arrays")
    bounds = np.iinfo(dtype)
    if array.size and (int(array.min()) < bounds.min or int(array.max()) > bounds.max):
        raise ValueError("Packed history integer is outside the native dtype range")
    return np.ascontiguousarray(array, dtype=dtype)


def replay_observations(actions, offsets, *, size: int = 9, komi: float = 7.5,
                        scoring: str = "pass_alive_area", starts=None) -> ReplayObservations:
    """Replay packed full games; return pre-action rows, optionally only suffixes.

    offsets has games+1 entries, starts has one relative row per game. The Rust
    engine validates all prefix moves and retains exact positional superko.
    """
    config = json.dumps(dict(size=size, komi=komi, scoring=scoring), allow_nan=False)
    actions = _integers(actions, np.int32)
    offsets = _integers(offsets, np.int64)
    if starts is None:
        result = _native.replay_observations(config, actions, offsets)
    else:
        result = _native.replay_suffix_observations(
            config, actions, offsets, _integers(starts, np.int64))
    stones, legal, outcomes = result
    return ReplayObservations(_readonly(stones, (-1, size, size)),
                              _readonly(legal, (-1, size**2 + 1)), json.loads(outcomes))
