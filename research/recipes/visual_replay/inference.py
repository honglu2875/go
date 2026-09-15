"""Batched exact-history tree traversal with donated decoder KV buffers.

Each client owns one cache path. Sibling search leaves rewind to the exact
common action prefix; future cache entries are masked until overwritten.
Predictions for a retained prefix can be served without another TPU call.
This is exact neural evaluation reuse, not speculative MCTS acceptance.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from gozero.visual_history import Replay, observation_sha256

if __package__:
    from . import model
else:
    import model


class Runner:
    def __init__(self, params, c, native, rules, *, slots, cache_positions,
                 network_version, exit_depth=None, max_block=16, suffix_replay=False):
        if (type(slots) is not int or slots < 1 or slots % len(jax.local_devices())
                or not 1 <= cache_positions <= c['max_positions']
                or type(max_block) is not int or not 1 <= max_block <= 32
                or type(network_version) is not int or not 0 <= network_version < 2**32):
            raise ValueError('Invalid fixed inference dimensions or model version')
        self.c, self.slots, self.capacity, self.version = c, slots, cache_positions, network_version
        self.depth, self.max_block = exit_depth, max_block
        if type(suffix_replay) is not bool:
            raise ValueError('Explicit suffix replay boolean required')
        self.suffix_replay = suffix_replay
        self.replay = Replay(native, rules, cache_positions); self.size = rules['size']
        self.stride = model.layout(self.size, c)['stride']
        self.mesh = Mesh(np.asarray(jax.local_devices()), ('data',))
        self.replicated = NamedSharding(self.mesh, P())
        self.batched = NamedSharding(self.mesh, P('data'))
        self.params = jax.device_put(params, self.replicated)
        self.compiled = {}; self.compilation = {}
        self.stats = {'requests': 0, 'prefix_hits': 0, 'dispatches': 0, 'appended_positions': 0,
                      'padded_position_slots': 0, 'replay_seconds': 0., 'dispatch_seconds': 0., 'total_seconds': 0.,
                      'native_encoded_positions': 0, 'native_validated_history_moves': 0}
        root = self.replay([[]])[0]
        root_batch = np.broadcast_to(root, (slots, *root.shape)).copy()
        def initialize(p, o, a, n):
            return model.forward(p, o, a, n, c, with_cache=True, network_version=network_version,
                                 cache_positions=cache_positions, exit_depth=exit_depth)
        depth = c['layers'] if exit_depth is None else exit_depth
        self.cache_specs = {'keys': (P('data'),) * depth, 'values': (P('data'),) * depth,
                            'lengths': P('data'), 'valid': P('data'), 'network_version': P()}
        fn = jax.shard_map(initialize, mesh=self.mesh, in_specs=(P(), P('data'), P('data'), P('data')),
                           out_specs=(P('data'), self.cache_specs), check_vma=False)
        start = time.perf_counter()
        inputs = (self.params, self.put(root_batch), self.put(np.zeros((slots, 1), np.int32)),
                  self.put(np.ones(slots, np.int32)))
        init = jax.jit(fn).lower(*inputs).compile()
        predictions, self.cache = init(*inputs)
        predictions = jax.device_get(predictions)
        self.compilation['root'] = time.perf_counter() - start
        self.paths = [() for _ in range(slots)]
        self.predictions = [[{k: v[row, 0].copy() for k, v in predictions.items()}] for row in range(slots)]
        self.hashes = [[observation_sha256(root[0])] for _ in range(slots)]

    def put(self, x):
        return jax.device_put(x, self.batched)

    def warmup(self):
        """Compile and exercise every fixed block bucket with no active rows."""
        start = time.perf_counter(); horizon = 1
        while horizon <= self.max_block:
            obs = np.zeros((self.slots, horizon, self.size, self.size, 6), np.float32)
            actions = np.zeros((self.slots, horizon), np.int32)
            counts = np.zeros(self.slots, np.int32); rewind = np.full(self.slots, -1, np.int32)
            inputs = (self.params, self.cache, *map(self.put, (obs, actions, counts, rewind)))
            prediction, self.cache = self._step(horizon, inputs)(*inputs)
            jax.block_until_ready((prediction, self.cache)); horizon *= 2
        self.compilation['warmup_seconds'] = time.perf_counter() - start

    def _step(self, horizon, inputs):
        if horizon not in self.compiled:
            def append(p, cache, observations, actions, counts, rewind):
                cache = {**cache, 'lengths': jnp.where(rewind >= 0, rewind, cache['lengths'])}
                return model.score_continuation(p, cache, observations, actions, counts, self.c,
                    network_version=self.version, exit_depth=self.depth)
            fn = jax.shard_map(append, mesh=self.mesh,
                in_specs=(P(), self.cache_specs, P('data'), P('data'), P('data'), P('data')),
                out_specs=(P('data'), self.cache_specs), check_vma=False)
            started = time.perf_counter()
            lowered = jax.jit(fn, donate_argnums=(1,)).lower(*inputs)
            compiled = lowered.compile(); memory = compiled.memory_analysis()
            self.compiled[horizon] = compiled
            self.compilation[str(horizon)] = {'seconds': time.perf_counter() - started,
                'memory_bytes': {k: int(getattr(memory, k)) for k in
                    ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes']}}
        return self.compiled[horizon]

    def score(self, requests):
        """Map unique slot IDs to (complete action history, exact leaf-board hash)."""
        started = time.perf_counter(); changed = {}; results = {}
        for slot, (history, leaf_hash) in requests.items():
            if type(slot) is not int or not 0 <= slot < self.slots:
                raise ValueError('Unknown inference slot')
            tape = self.replay.validate(history)
            if not isinstance(leaf_hash, str) or len(leaf_hash) != 64:
                raise ValueError('Expected exact native leaf observation hash')
            old = self.paths[slot]; common = 0
            while common < min(len(old), len(tape)) and old[common] == tape[common]:
                common += 1
            if common == len(tape):
                if self.hashes[slot][common] != leaf_hash:
                    raise ValueError('Cached prefix and native pending board differ')
                results[slot] = self.predictions[slot][common]
            else:
                changed[slot] = [tape, common, leaf_hash]
        # Validate the entire request batch before mutating any retained path.
        start_replay = time.perf_counter()
        histories = [item[0] for item in changed.values()]
        starts = [item[1] if self.suffix_replay else 0 for item in changed.values()]
        replayed = self.replay.suffix(histories, starts) if self.suffix_replay else self.replay(histories)
        self.stats['replay_seconds'] += time.perf_counter() - start_replay
        for (slot, (tape, common, leaf_hash)), observations, begin in zip(changed.items(), replayed, starts):
            if (observation_sha256(observations[-1]) != leaf_hash
                    or observation_sha256(observations[common - begin]) != self.hashes[slot][common]):
                raise ValueError('Replayed path and exact native pending board differ')
            changed[slot].extend((observations, begin))
        self.stats['native_encoded_positions'] += sum(len(o) for o in replayed)
        self.stats['native_validated_history_moves'] += sum(map(len, histories))
        self.stats['requests'] += len(requests); self.stats['prefix_hits'] += len(results)
        while changed:
            longest = max(len(item[0]) - item[1] for item in changed.values())
            horizon = min(self.max_block, 1 << (longest - 1).bit_length())
            obs = np.zeros((self.slots, horizon, self.size, self.size, 6), np.float32)
            actions = np.zeros((self.slots, horizon), np.int32)
            counts = np.zeros(self.slots, np.int32); rewind = np.full(self.slots, -1, np.int32)
            for slot, (tape, common, _, exact, begin) in changed.items():
                n = min(horizon, len(tape) - common); counts[slot] = n
                obs[slot, :n] = exact[common + 1 - begin:common + n + 1 - begin]
                actions[slot, :n] = tape[common:common + n]
                rewind[slot] = (common + 1) * self.stride - 1
            start_dispatch = time.perf_counter()
            inputs = (self.params, self.cache, *map(self.put, (obs, actions, counts, rewind)))
            fn = self._step(horizon, inputs)
            prediction, self.cache = fn(*inputs)
            prediction, valid, lengths = jax.device_get((prediction, self.cache['valid'], self.cache['lengths']))
            self.stats['dispatch_seconds'] += time.perf_counter() - start_dispatch
            if not valid.all() or any(lengths[slot] != rewind[slot] + counts[slot] * self.stride for slot in changed):
                raise RuntimeError('Decoder cache rejected a prevalidated exact continuation')
            self.stats['dispatches'] += 1; self.stats['appended_positions'] += int(counts.sum())
            self.stats['padded_position_slots'] += self.slots * horizon
            for slot in list(changed):
                tape, common, leaf_hash, exact, begin = changed[slot]; n = int(counts[slot])
                self.predictions[slot][common + 1:] = [{k: v[slot, i].copy() for k, v in prediction.items()} for i in range(n)]
                self.hashes[slot][common + 1:] = [observation_sha256(o) for o in exact[common + 1 - begin:common + n + 1 - begin]]
                self.paths[slot] = tape[:common + n]
                if common + n == len(tape):
                    results[slot] = self.predictions[slot][-1]; del changed[slot]
                else:
                    changed[slot][1] += n
        self.stats['total_seconds'] += time.perf_counter() - started
        return results
