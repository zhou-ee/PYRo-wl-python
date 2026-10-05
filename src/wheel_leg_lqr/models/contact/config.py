"""Contact assumptions and numerical integration defaults, not CAD hardware data."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ContactSettings:
    ground_stiffness: float = 5.0e3
    ground_damping: float = 2.0e2
    stop_stiffness: float = 1.0e4
    stop_damping: float = 3.0e2
    integration_substeps: int = 5
DEFAULT_CONTACT = ContactSettings()
