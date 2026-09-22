# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""ALICE inference API."""

from .torch import (
    estimate_mi_minde,
    estimate_mi_minde_fast,
    load_model,
    prepare_minde_model,
    velocity_field_masked,
)

__all__ = [
    "estimate_mi_minde",
    "estimate_mi_minde_fast",
    "load_model",
    "prepare_minde_model",
    "velocity_field_masked",
]
