"""Trace full-size distributed update schemas without allocating model arrays."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import read_json, verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.preparation) != args.preparation_sha256:
        raise ValueError('Preparation changed')
    preparation = read_json(args.preparation)
    if preparation['kind'] != 'joint_source_muon_full_size_preparation' or preparation['status'] != 'prepared':
        raise ValueError('Wrong preparation')
    snapshot = ROOT / '.gozero/snapshots' / preparation['snapshot']
    manifest = verify(snapshot)
    config = read_json(snapshot / 'resolved_config.json')
    sys.path.insert(0, str(snapshot / manifest['recipe']))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    import joint
    import learner
    import optimizer
    import train_config
    train_config.validate(config)
    if len(jax.devices()) != config['expected_devices'] or any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('Use the explicit simulated CPU mesh for abstract tracing')
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    opt = {k: v for k, v in config['learner'].items() if k not in ('games_per_host', 'augmentation')}
    opt['horizon_steps'] = config['steps']
    params = jax.eval_shape(lambda: joint.initialize(config['seed'], config['model'], config['value_model']))
    state = jax.eval_shape(lambda p: optimizer.initialize(p, opt, jnp.ones((13,), jnp.float32)), params)
    nbytes = lambda tree: sum(math.prod(a.shape) * np.dtype(a.dtype).itemsize for a in jax.tree.leaves(tree))
    schema = lambda tree: jax.tree.map(lambda a: (tuple(a.shape), str(a.dtype)), tree)
    elements = sum(math.prod(a.shape) for a in jax.tree.leaves(params))
    if elements != 233220870 or any(a.dtype != jnp.float32 for a in jax.tree.leaves(params)):
        raise ValueError('Full-size joint CNN master schema differs')
    if schema(state['lookahead']['slow']) != schema(params):
        raise ValueError('Slow weights do not cover all master parameters')
    batch = config['expected_processes'] * config['learner']['games_per_host']
    size = config['model']['max_board_size']
    operation = learner.step(config['model'], opt, value_weight=config['training']['value_weight'],
        path=config['training']['path'], chunk_frames=config['training']['chunk_frames'], mesh=mesh)
    cases = []
    started = time.monotonic()
    for length in config['dataset']['buckets']:
        s = jax.ShapeDtypeStruct
        inputs = dict(spatial=s((batch, length, size, size, 22), jnp.float32),
            global_features=s((batch, length, 19), jnp.float32),
            actions=s((batch, length), jnp.int32), counts=s((batch,), jnp.int32),
            policies=s((batch, length, size*size+1), jnp.float32),
            legal=s((batch, length, size*size+1), jnp.bool_),
            values=s((batch, length), jnp.float32))
        before = time.monotonic()
        graph, out = jax.make_jaxpr(operation, return_shape=True)(params, state, inputs)
        if schema(out[0]) != schema(params) or schema(out[1]) != schema(state):
            raise ValueError('Update changes model or complete optimizer schema')
        if (any(a.shape != () for a in jax.tree.leaves(out[2]))
                or out[2]['accepted'].dtype != jnp.bool_
                or out[1]['step'].dtype != jnp.int32
                or out[1]['lookahead']['counter'].dtype != jnp.int32):
            raise ValueError('Global scalar metric/counter schema differs')
        maps = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == 'shard_map']
        if not maps:
            raise ValueError('The complete collective gradient path was not traced')
        cases.append(dict(positions=length, global_sequences=batch,
            sequences_per_device=batch//len(jax.devices()),
            global_input_bytes=nbytes(inputs), per_device_input_bytes=nbytes(inputs)//len(jax.devices()),
            trace_seconds=time.monotonic()-before, shard_map_equations=len(maps),
            scalar_metrics=len(out[2]), complete_schema_preserved=True))
        print(json.dumps(cases[-1]), flush=True)
    if sha(args.preparation) != args.preparation_sha256:
        raise ValueError('Preparation changed during tracing')
    verify(snapshot)
    record = dict(kind='joint_source_muon_full_size_abstract_schema', status='passed',
        created=time.time(), seconds=time.monotonic()-started,
        snapshot=snapshot.name, preparation_sha256=args.preparation_sha256,
        operator_sha256=sha(Path(__file__)), jax_version=jax.__version__,
        master_parameters=elements, master_parameter_bytes=nbytes(params),
        optimizer_state_bytes=nbytes(state),
        optimizer_bytes_by_component={k:nbytes(v) for k,v in state.items()},
        master_parameter_leaves=len(params), cases=cases,
        scope='Abstract shapes on a simulated data-parallel CPU mesh. Includes '
            'the actual SUM-gradient collective and complete Muon/AuxAdam, '
            'Lookahead and scalar state. No full parameter arrays allocated, '
            'TPU execution, peak-HBM, timing or distributed-recovery claim. '
            'State/input byte counts are not a peak-memory estimate.')
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    args.output.chmod(0o444)
    print(json.dumps(dict(status='passed', seconds=record['seconds'], sha256=sha(args.output))), flush=True)


if __name__ == '__main__':
    main()
