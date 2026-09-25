# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

import math
import os

import pytest
import torch

from alice import estimate_mi, load_model

MODEL_ID = "eurecom-probai/alice-small-stage2"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "SKIP_REAL_MODEL_TESTS" in os.environ,
        reason="SKIP_REAL_MODEL_TESTS is set",
    ),
]


@pytest.fixture(scope="module")
def small_stage2_model():
    return load_model(MODEL_ID, device="cpu")


def _gaussian_joint(rho: float, n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=generator)
    independent_noise = torch.randn(n, 1, generator=generator)
    y = rho * x + math.sqrt(1.0 - rho**2) * independent_noise
    return torch.cat([x, y], dim=-1)


def _true_gaussian_mi(rho: float) -> float:
    return -0.5 * math.log1p(-(rho**2))


@pytest.mark.parametrize(
    "rho", (0.0, 0.5, 0.8), ids=("independent", "rho-0.5", "rho-0.8")
)
def test_small_stage2_estimates_gaussians(small_stage2_model, rho):
    n_context = 2048
    n_eval = 128
    estimate, standard_error = estimate_mi(
        small_stage2_model,
        _gaussian_joint(rho, n=n_context + n_eval, seed=100 + int(10 * rho)),
        slice(0, 1),
        slice(1, 2),
        n_context=n_context,
        n_eval=n_eval,
        n_t_samples=8,
        chunk=128,
        generator=torch.Generator().manual_seed(42),
        return_std=True,
    )

    assert math.isfinite(estimate)
    assert math.isfinite(standard_error)
    assert estimate >= 0.0
    assert standard_error >= 0.0
    tolerance = {0.0: 0.02, 0.5: 0.08, 0.8: 0.10}[rho]
    assert abs(estimate - _true_gaussian_mi(rho)) < tolerance
    if rho == 0.0:
        assert estimate < 0.02
    elif rho == 0.5:
        assert estimate > 0.1
    else:
        assert estimate > 0.5
