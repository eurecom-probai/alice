# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""PyTorch backend for ALICE."""

from .mi import (
    estimate_mi_minde,
    estimate_mi_minde_fast,
    prepare_minde_model,
    velocity_field_masked,
)
from .utils import load_model

__all__ = [
    "estimate_mi_minde",
    "estimate_mi_minde_fast",
    "load_model",
    "prepare_minde_model",
    "velocity_field_masked",
]
