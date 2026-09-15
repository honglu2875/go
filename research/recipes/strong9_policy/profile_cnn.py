"""Complete CNN inference on unpacked V7 board features."""
import hashlib
import os
import time
import numpy as np
import jax


def run(params,data,net,global_batch,world,output):
    import compute_budget
    import katago
    from gozero.jaxpr_cost import analyze
    local_n = 128 // world
    from gozero.corpus_format import unpack_spatial
    first_board = unpack_spatial(data.shards[0]['spatial'][:1], data.size)
    spatial = global_batch({'s': np.repeat(first_board, local_n, axis=0)})['s']
    glob = global_batch({'g': np.repeat(data.shards[0]['global_features'][:1], local_n, axis=0)})['g']
    fn = lambda p,s,g: katago.forward(p,s,g,net)
    start = time.perf_counter()
    graph = jax.make_jaxpr(fn)(params, spatial, glob)
    arithmetic = analyze(graph)
    analytical = compute_budget.cnn(net, data.size, 128)
    if arithmetic['counts']['multiply_add_flops'] != analytical['multiply_add_flops_per_batch']:
        raise ValueError('Traced and analytic complete-model decode arithmetic disagree')
    if arithmetic['unaccounted_primitives']:
        raise ValueError('Decode arithmetic contains unaudited operations')
    lowered = jax.jit(fn).lower(params, spatial, glob)
    executable = lowered.compile()
    hlo = lowered.compiler_ir('hlo').as_hlo_text()
    with (output / 'decode.hlo').open('x') as f:
        f.write(hlo); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(executable(params, spatial, glob))
    durations = []
    for _ in range(10):
        start = time.perf_counter(); jax.block_until_ready(executable(params, spatial, glob))
        durations.append(time.perf_counter() - start)
    result = {'batch_size': 128, 'board_size': data.size,
        'analytical': analytical, 'jaxpr': arithmetic,
        'compiler_cost_estimate': executable.cost_analysis(),
        'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'compile_seconds': compile_seconds, 'host_dispatch_latency_seconds': durations,
        'scope': 'Warm complete CNN neural inference; CPU rule feature generation and transfers excluded.'}
    return result
