# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Parity and cache-lifetime checks for batched MI inference."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alice import estimate_mi_fast


class MaskedModel(nn.Module):
    """Nonzero, context-dependent field with observable support encoding."""

    def __init__(self):
        """Initialize a scalar parameter and cache counters."""
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.7))
        self.config = SimpleNamespace(use_mask_channel=True, variable_dim=True)
        self.builds = 0
        self.calls = 0

    def _support_cache(self, context):
        """Record support encoding and retain context-dependent values."""
        self.builds += 1
        return context.mean(dim=1, keepdim=True)

    def predict_query_logits_cached(
        self, ctx_clean, qry_z_t, qry_t, qry_mask, support_cache
    ):
        """Evaluate a nonlinear row-independent masked field."""
        self.calls += 1
        return self.anchor * qry_t * (qry_z_t + qry_mask + support_cache).sin()

    def forward(self, ctx_clean, **kwargs):
        """Evaluate the same field with freshly encoded context."""
        return SimpleNamespace(
            logits=self.predict_query_logits_cached(
                ctx_clean, **kwargs, support_cache=self._support_cache(ctx_clean)
            )
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_fast_matches_reference_and_rng(dtype, normalize, chunk):
    """Preserve means, errors and RNG state with padding and strided slices."""
    model = MaskedModel().eval()
    samples = torch.randn(
        23, 4, dtype=dtype, generator=torch.Generator().manual_seed(11)
    )
    samples[0:3] = samples[0].clone()
    kwargs = {
        "n_context": 9,
        "n_eval": 11,
        "n_t_samples": 3,
        "chunk": chunk,
        "normalize": normalize,
        "return_std": True,
        "pad_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    rng = torch.Generator().manual_seed(42)
    expected = estimate_mi_fast(
        model,
        samples,
        slice(0, 4, 2),
        slice(1, 4, 2),
        use_cache=False,
        fuse_queries=False,
        generator=rng,
        **kwargs,
    )
    state = rng.get_state()
    for options in ({}, {"fuse_queries": False}, {"use_cache": False}):
        rng.manual_seed(42)
        builds = model.builds
        actual = estimate_mi_fast(
            model,
            samples,
            slice(0, 4, 2),
            slice(1, 4, 2),
            generator=rng,
            **kwargs,
            **options,
        )
        assert actual == pytest.approx(expected, abs=1e-4)
        assert torch.equal(state, rng.get_state())
        if options.get("use_cache", True):
            assert model.builds == builds + 1


def test_fallback_without_cache_api():
    """Models exposing only forward retain equivalent estimates."""
    from tests.test_mi import ZeroResidualModel

    result = estimate_mi_fast(
        ZeroResidualModel().eval(),
        torch.randn(10, 2),
        slice(0, 1),
        slice(1, 2),
        n_t_samples=1,
    )
    assert result == 0.0


def test_training_mode_rejected():
    """Dropout-dependent training forwards cannot safely reuse support."""
    with pytest.raises(ValueError, match="eval-only"):
        estimate_mi_fast(MaskedModel(), torch.randn(10, 2), slice(0, 1), slice(1, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 2, 17])
def test_batched_normalization_matches_columnwise(n, dtype):
    """Preserve ties, out-of-range interpolation and noncontiguous inputs."""
    from alice.torch.mi.alice_mi_fast import _normalize_columns

    reference = torch.randn(8, n, dtype=dtype).T
    reference[:, 0] = 1.0
    values = torch.randn(8, 23, dtype=dtype).T * 10
    values[:n] = reference
    if n == 1:
        expected = torch.zeros_like(values)
    else:
        columns = []
        for ref, val in zip(reference.T, values.T):
            ordered = ref.sort().values
            index = torch.searchsorted(ordered, val.contiguous()).clamp(1, n - 1)
            fraction = (
                (val - ordered[index - 1])
                / (ordered[index] - ordered[index - 1]).clamp_min(1e-12)
            ).clamp(0, 1)
            probability = (index + fraction) / (n + 1)
            columns.append(torch.distributions.Normal(0, 1).icdf(probability))
        expected = torch.stack(columns, dim=1)
    torch.testing.assert_close(
        _normalize_columns(reference, values), expected, rtol=0, atol=1e-4
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("coordinates", [1, 3])
def test_projected_attention_preserves_projection_rounding(dtype, coordinates):
    """Match the native module, including low-precision bias rounding."""
    from alice.torch.mi.alice_mi_fast import _attention, _project_keys_values

    attention = nn.MultiheadAttention(16, 4, batch_first=True, dtype=dtype).eval()
    query = torch.randn(1, coordinates, 7, 16, dtype=dtype)
    support = torch.randn(1, coordinates, 11, 16, dtype=dtype)
    with torch.inference_mode():
        expected, _ = attention(
            query.reshape(coordinates, 7, 16),
            support.reshape(coordinates, 11, 16),
            support.reshape(coordinates, 11, 16),
            need_weights=False,
        )
        actual = _attention(attention, query, *_project_keys_values(attention, support))
    torch.testing.assert_close(actual.reshape_as(expected), expected, rtol=0, atol=1e-4)
