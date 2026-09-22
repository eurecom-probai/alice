# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Mutual-information estimation for the PyTorch ALICE backend."""

from .minde import estimate_mi_minde, velocity_field_masked
from .minde_fast import estimate_mi_minde as estimate_mi_minde_fast
from .minde_fast import prepare_minde_model

__all__ = [
    "estimate_mi_minde",
    "estimate_mi_minde_fast",
    "prepare_minde_model",
    "velocity_field_masked",
]
