"""Own the original learning schema plus explicit native and fork identities."""
import re
from pathlib import PurePosixPath

from base_config import fields, validate as validate_base


def reference(value, label):
    if not isinstance(value, str) or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
        raise ValueError('Invalid ' + label + ' path')


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected SHA-256 identity')


def validate(c):
    fields(c, 'schema_version platform expected_processes expected_devices seed selfplay_turns checkpoint_every log_every actors model learner native initialization', 'config')
    validate_base({k: v for k, v in c.items() if k not in ('native', 'initialization')})
    fields(c['native'], 'receipt receipt_sha256', 'native')
    reference(c['native']['receipt'], 'native receipt'); digest(c['native']['receipt_sha256'])
    init = c['initialization']
    if init != {'kind': 'fresh'}:
        fields(init, 'kind descriptor sha256', 'initialization')
        if init['kind'] != 'fork':
            raise ValueError('Unsupported initialization kind')
        reference(init['descriptor'], 'fork descriptor'); digest(init['sha256'])
    return c
