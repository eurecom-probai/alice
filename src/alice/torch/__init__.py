# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""PyTorch backend for ALICE."""

from .mi import (
    estimate_mi_fast,
    prepare_mi_model,
    velocity_field_masked,
)
from .utils import load_model

__all__ = [
    "estimate_mi_fast",
    "load_model",
    "prepare_mi_model",
    "velocity_field_masked",
]
