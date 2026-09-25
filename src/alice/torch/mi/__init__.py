# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Mutual-information estimation for the PyTorch ALICE backend."""

from .alice_mi import estimate_mi, velocity_field_masked
from .alice_mi_fast import estimate_mi_fast, prepare_mi_model

__all__ = [
    "estimate_mi",
    "estimate_mi_fast",
    "prepare_mi_model",
    "velocity_field_masked",
]
