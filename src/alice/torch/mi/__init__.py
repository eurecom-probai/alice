# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Mutual-information estimation for the PyTorch ALICE backend."""

from .alice_mi_fast import estimate_mi_fast, prepare_mi_model
from .velocity import velocity_field_masked

__all__ = [
    "estimate_mi_fast",
    "prepare_mi_model",
    "velocity_field_masked",
]
