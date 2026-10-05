"""Reproducible physical snapshots attached to generated artifacts."""
from ..config import DEFAULT_ROBOT
def artifact_metadata(parameters=DEFAULT_ROBOT, **settings):
    return {'schema_version': 2, 'physical_parameters': parameters.snapshot(),
            'physical_fingerprint': parameters.fingerprint(), 'settings': settings}
