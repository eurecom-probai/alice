# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Checkpoint parity for projected attention and full-graph query capture."""

import os

import pytest
import torch

from alice import (
    estimate_mi_fast,
    load_model,
    prepare_mi_model,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "SKIP_REAL_MODEL_TESTS" in os.environ, reason="SKIP_REAL_MODEL_TESTS is set"
    ),
]


@pytest.fixture(scope="module")
def model():
    """Load the checkpoint revision used for decoder parity measurements."""
    return load_model(
        "eurecom-probai/alice-small-stage2",
        revision="794828d2323873a19abfb077254b549e4629127b",
    )


@pytest.mark.parametrize("dim", [2, 5, 10])
def test_projected_logits_match_checkpoint(model, dim):
    """Compare logits directly before the MI reduction can hide differences."""
    prepared = prepare_mi_model(model, compile=False)
    with torch.inference_mode():
        for seed in [1, 2]:
            generator = torch.Generator().manual_seed(seed)
            context = torch.randn(1, 32, dim, generator=generator)
            states = torch.randn(1, 7, dim, generator=generator)
            times = torch.rand(1, 7, 1, generator=generator)
            times[:, 0] = 0
            masks = (torch.rand(1, 7, dim, generator=generator) > 0.5).float()
            expected = model(
                ctx_clean=context, qry_z_t=states, qry_t=times, qry_mask=masks
            ).logits
            actual = prepared.predict_query_logits_cached(
                context, states, times, masks, prepared._support_cache(context)
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_mi_matches_reference_with_tail_and_new_context(model, compiled):
    """Capture a full graph, preserve padded partitions and rebuild each context."""
    prepared = prepare_mi_model(model, compile=compiled, backend="eager")
    for seed in [3, 4]:
        samples = torch.randn(37, 4, generator=torch.Generator().manual_seed(seed))
        kwargs = {
            "n_context": 32,
            "n_t_samples": 3,
            "chunk": 7,
            "return_std": True,
            "pad_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }
        expected = estimate_mi_fast(
            model,
            samples,
            slice(0, 4, 2),
            slice(1, 4, 2),
            use_cache=False,
            fuse_queries=False,
            generator=torch.Generator().manual_seed(42),
            **kwargs,
        )
        actual = estimate_mi_fast(
            prepared,
            samples,
            slice(0, 4, 2),
            slice(1, 4, 2),
            generator=torch.Generator().manual_seed(42),
            **kwargs,
        )
        assert actual == pytest.approx(expected, abs=1e-4)
    assert not model.training


@pytest.mark.parametrize(
    "options",
    [
        {"use_cache": False},
        {"cache_projections": False},
        {"fuse_queries": False},
    ],
)
def test_prepared_ablation_options(model, options):
    """Allow independent fallback to the checkpoint's original execution paths."""
    prepared = prepare_mi_model(model, compile=False)
    samples = torch.randn(35, 2, generator=torch.Generator().manual_seed(4))
    kwargs = {"n_context": 32, "n_t_samples": 2, "chunk": 5, "normalize": False}
    expected = estimate_mi_fast(
        model,
        samples,
        slice(0, 1),
        slice(1, 2),
        use_cache=False,
        fuse_queries=False,
        generator=torch.Generator().manual_seed(42),
        **kwargs,
    )
    actual = estimate_mi_fast(
        prepared,
        samples,
        slice(0, 1),
        slice(1, 2),
        generator=torch.Generator().manual_seed(42),
        **kwargs,
        **options,
    )
    assert actual == pytest.approx(expected, abs=1e-4)


def test_prepared_rejects_training_after_preparation(model):
    """Prevent stale wrapper mode from allowing a training-mode wrapped model."""
    prepared = prepare_mi_model(model, compile=False)
    model.train()
    try:
        with pytest.raises(ValueError, match="eval-only"):
            estimate_mi_fast(prepared, torch.randn(5, 2), slice(0, 1), slice(1, 2))
    finally:
        model.eval()
