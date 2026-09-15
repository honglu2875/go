"""Hash evaluation specifications together with their referenced input files.

Future trained model descriptors may be declared deferred explicitly. All
other panel children, configurations and reference descriptors must exist at
registration. Actual external weights/binaries remain verified by the runner.
"""
from pathlib import Path

from .checkpoints import sha256
from .snapshots import read_json


def collect(source, roots, *, deferred_models=()):
    source = Path(source).resolve(strict=True)

    def relative(name):
        if not isinstance(name, (str, Path)) or Path(name).is_absolute():
            raise ValueError('Evaluation references must be relative source paths')
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('Evaluation input escapes source or is a symlink')
        return path.relative_to(source).as_posix()

    deferred = {relative(name) for name in deferred_models}
    files, used_deferred = {}, set()

    def visit(name, *, model=False):
        name = relative(name)
        if model and name in deferred:
            used_deferred.add(name)
            return
        if name in deferred:
            raise ValueError('Only produced model descriptors may be deferred')
        if name in files:
            return
        path = source / name
        files[name] = sha256(path)
        if path.suffix != '.json':
            return
        document = read_json(path)
        if not isinstance(document, dict):
            raise ValueError('Expected an evaluation configuration object')
        if 'matches' in document:
            for match in document['matches']:
                visit(match['spec'])
        # Candidate descriptors name external run artifacts and are leaves of
        # the source manifest. validate_candidate audits those artifacts later.
        if 'training_snapshot' in document:
            return
        if 'candidate' in document:
            visit(document['candidate'], model=True)
            visit('eval/katago_build.json')
            visit(document.get('katago_weights') or 'eval/katago_9x9.json')
        if 'opponent' in document:
            visit(document['opponent'], model=True)
        for key in ('katago_config', 'referee_config', 'causal_inference'):
            if key in document:
                visit(document[key])

    roots = [relative(name) for name in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Distinct root evaluation specifications required')
    for name in roots:
        visit(name)
    if deferred != used_deferred:
        raise ValueError('Unused deferred descriptor declaration')
    return {'schema_version': 1, 'kind': 'evaluation_input_closure', 'roots': roots,
            'deferred_model_descriptors': sorted(deferred), 'files': dict(sorted(files.items()))}


def verify(source, manifest):
    expected = collect(source, manifest['roots'], deferred_models=manifest['deferred_model_descriptors'])
    if expected != manifest:
        raise ValueError('Evaluation input closure differs from registration')
    return expected
