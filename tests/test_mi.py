# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alice.torch.mi import estimate_mi_minde


class ZeroResidualModel(nn.Module):
    """Small mask-capable model whose MI estimate is exactly zero."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(use_mask_channel=True, variable_dim=True)

    def forward(self, *, qry_z_t, **kwargs):
        del kwargs
        return SimpleNamespace(logits=torch.zeros_like(qry_z_t) + self.anchor)


def test_estimate_mi_minde_returns_zero_for_zero_residual_model():
    model = ZeroResidualModel().eval()
    samples = torch.randn(40, 4)

    estimate, standard_error = estimate_mi_minde(
        model,
        samples,
        slice(0, 2),
        slice(2, 4),
        n_context=16,
        n_eval=12,
        n_t_samples=4,
        chunk=5,
        generator=torch.Generator().manual_seed(7),
        return_std=True,
    )

    assert estimate == 0.0
    assert standard_error == 0.0


def test_estimate_mi_minde_validates_the_partition():
    model = ZeroResidualModel().eval()
    samples = torch.randn(10, 4)

    with pytest.raises(ValueError, match="disjoint and cover all 4"):
        estimate_mi_minde(model, samples, slice(0, 3), slice(2, 4), n_t_samples=1)


def test_estimate_mi_minde_requires_mask_channel():
    model = ZeroResidualModel().eval()
    model.config.use_mask_channel = False

    with pytest.raises(ValueError, match="use_mask_channel=True"):
        estimate_mi_minde(model, torch.randn(10, 2), slice(0, 1), slice(1, 2), n_t_samples=1)
