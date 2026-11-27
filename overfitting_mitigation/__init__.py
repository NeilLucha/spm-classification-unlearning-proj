"""Optional regularization and overfitting mitigation for SPM experiments."""

from .config import MitigationConfig, load_mitigation_config
from .integration import configure_model, configure_datamodule

__all__ = [
    "MitigationConfig",
    "load_mitigation_config",
    "configure_model",
    "configure_datamodule",
]
