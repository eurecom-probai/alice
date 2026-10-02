# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Input-gradient contracts without network access or real checkpoints."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from alice import estimate_mi_fast
from alice.torch.mi.alice_mi_fast import (
    _disjoint_split,
    _normalize_columns,
    _soft_normalize_columns,
)
from tests.test_alice_mi_fast import MaskedModel


class GradCacheModel(MaskedModel):
    """Declare the test field's cache methods safe for input autograd."""

    supports_input_grad_cache = True

    def predict_query_logits_cached(
        self, ctx_clean, qry_z_t, qry_t, qry_mask, support_cache
    ):
        self.calls += 1
        return (
            self.anchor
            * (0.3 + qry_t)
            * (
                qry_z_t
                + qry_z_t.flip(-1)
                + qry_mask.mean(-1, keepdim=True)
                + support_cache
            ).sin()
        )


def estimate(model, samples, **options):
    return estimate_mi_fast(
        model,
        samples,
        slice(0, 1),
        slice(1, 2),
        generator=torch.Generator().manual_seed(42),
        n_context=5,
        n_eval=4,
        n_t_samples=2,
        chunk=3,
        normalize=options.pop("normalize", False),
        differentiable=options.pop("differentiable", True),
        **options,
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("fuse_queries", [False, True])
@pytest.mark.parametrize("checkpoint_queries", [False, True])
def test_paths_match_values_input_gradients_and_finite_differences(
    normalize,
    use_cache,
    fuse_queries,
    checkpoint_queries,
):
    model = GradCacheModel().double().eval().requires_grad_(False)
    samples = torch.randn(
        9, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
    ).requires_grad_()
    options = {
        "normalize": normalize,
        "use_cache": use_cache,
        "fuse_queries": fuse_queries,
        "checkpoint_queries": checkpoint_queries,
        "return_std": True,
    }
    result = estimate(model, samples, **options)
    expected = estimate(
        model,
        samples,
        normalize=normalize,
        use_cache=False,
        fuse_queries=False,
        return_std=True,
    )
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
    grad = torch.autograd.grad(sum(result), samples)[0]
    reference_grad = torch.autograd.grad(sum(expected), samples)[0]
    torch.testing.assert_close(grad, reference_grad, atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda z: estimate(model, z, **options), (samples,))
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    inference = estimate(
        model, samples, normalize=normalize, differentiable=False, return_std=True
    )
    assert all(isinstance(value, float) for value in inference)
    if not normalize:
        assert inference == pytest.approx(
            tuple(value.item() for value in result), abs=1e-12
        )


def test_context_and_evaluation_both_reach_design_and_add():
    model = GradCacheModel().double().eval().requires_grad_(False)
    base = torch.randn(
        9, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(4)
    )
    context, evaluation = _disjoint_split(9, 5, 4, torch.Generator().manual_seed(42))
    directions = torch.zeros(2, 9, 2, dtype=torch.float64)
    directions[0, context, 1] = base[context, 0]
    directions[1, evaluation, 1] = base[evaluation, 0]
    gradients = []
    for direction in (directions[0], directions[1], directions.sum(0)):
        design = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        mi = estimate(model, base + design * direction, checkpoint_queries=True)
        gradients.append(torch.autograd.grad(mi, design)[0])
    assert all(value.abs() > 1e-5 for value in gradients[:2])
    torch.testing.assert_close(gradients[2], gradients[0] + gradients[1])


def test_unknown_inference_cache_falls_back_to_forward():
    class LegacyModel(MaskedModel):
        @torch.no_grad()
        def _support_cache(self, context):
            raise AssertionError("inference-only support cache must not be used")

        @torch.no_grad()
        def predict_query_logits_cached(self, **kwargs):
            raise AssertionError("inference-only query cache must not be used")

        def forward(self, ctx_clean, qry_z_t, qry_t, qry_mask):
            return SimpleNamespace(
                logits=self.anchor
                * (0.3 + qry_t)
                * (
                    qry_z_t
                    + qry_z_t.flip(-1)
                    + qry_mask.mean(-1, keepdim=True)
                    + ctx_clean.mean(1, keepdim=True)
                ).sin()
            )

    model = LegacyModel().eval().requires_grad_(False)
    samples = torch.randn(
        9, 2, generator=torch.Generator().manual_seed(7), requires_grad=True
    )
    result = estimate(model, samples)
    gradient = torch.autograd.grad(result, samples)[0]
    context, evaluation = _disjoint_split(9, 5, 4, torch.Generator().manual_seed(42))
    assert gradient[context].abs().sum() > 0
    assert gradient[evaluation].abs().sum() > 0


@pytest.mark.parametrize("checkpoint_queries", [False, True])
def test_each_optimization_iteration_rebuilds_and_releases_cache(checkpoint_queries):
    class ObservedModel(GradCacheModel):
        def _support_cache(self, context):
            cache = super()._support_cache(context)
            self.cache_reference = weakref.ref(cache)
            return cache

    model = ObservedModel().eval().requires_grad_(False)
    design = torch.tensor(0.2, requires_grad=True)
    base = torch.randn(9, 2, generator=torch.Generator().manual_seed(9))
    for iteration in range(4):
        samples = base * design
        result = estimate(model, samples, checkpoint_queries=checkpoint_queries)
        gradient = torch.autograd.grad(result, design)[0]
        with torch.no_grad():
            design.add_(gradient, alpha=0.01)
        del result, samples, gradient
        gc.collect()
        assert model.builds == iteration + 1
        assert model.cache_reference() is None


@pytest.mark.parametrize("context_manager", [torch.no_grad, torch.inference_mode])
def test_disabled_autograd_fails_without_changing_flags(context_manager):
    model = GradCacheModel().eval().requires_grad_(False)
    samples = torch.randn(9, 2, requires_grad=True)
    with (
        context_manager(),
        pytest.raises(ValueError, match="requires enabled autograd"),
    ):
        estimate(model, samples)
    assert torch.is_grad_enabled()
    assert all(not p.requires_grad for p in model.parameters())


def test_inference_tensors_and_unfrozen_parameters_rejected():
    model = GradCacheModel().eval()
    with pytest.raises(ValueError, match="frozen model"):
        estimate(model, torch.randn(9, 2, requires_grad=True))
    assert model.anchor.requires_grad
    model.requires_grad_(False)
    with torch.inference_mode():
        samples = torch.randn(9, 2)
    with pytest.raises(ValueError, match="created outside inference_mode"):
        estimate(model, samples)
    with pytest.raises(ValueError, match="requires differentiable=True"):
        estimate(model, samples, differentiable=False, checkpoint_queries=True)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_frozen_low_precision_model_and_rng(dtype):
    model = GradCacheModel().to(dtype).eval().requires_grad_(False)
    samples = torch.randn(
        9,
        2,
        dtype=dtype,
        generator=torch.Generator().manual_seed(7),
        requires_grad=True,
    )
    rng = torch.Generator().manual_seed(42)
    kwargs = {
        "n_context": 5,
        "n_eval": 4,
        "n_t_samples": 2,
        "generator": rng,
        "normalize": False,
        "chunk": 3,
    }
    estimate_mi_fast(model, samples, slice(0, 1), slice(1, 2), **kwargs)
    expected_state = rng.get_state()
    rng.manual_seed(42)
    result = estimate_mi_fast(
        model,
        samples,
        slice(0, 1),
        slice(1, 2),
        differentiable=True,
        checkpoint_queries=True,
        **kwargs,
    )
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    gradient = torch.autograd.grad(result, samples)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(rng.get_state(), expected_state)


def test_degenerate_standard_error_has_finite_zero_gradient():
    model = GradCacheModel().eval().requires_grad_(False)
    model.anchor.zero_()
    samples = torch.randn(9, 2, requires_grad=True)
    result = estimate(model, samples, return_std=True)
    assert result[1].item() == 0
    gradient = torch.autograd.grad(sum(result), samples)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_empirical_context_ranks_and_singleton_have_zero_local_gradient():
    for n in (1, 5):
        context = torch.arange(n, dtype=torch.float64).reshape(n, 1).requires_grad_()
        normalized = _normalize_columns(context, context)
        gradient = torch.autograd.grad(normalized.sum(), context)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_rank_crossing_is_not_a_local_gradient_check():
    reference = torch.tensor([[-1.0], [0.0], [2.0]], dtype=torch.float64)
    value = torch.tensor([[-0.01]], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(_normalize_columns(reference, value).sum(), value)[
        0
    ]
    step = 0.1  # Crosses the reference knot at zero: not a rank-stable perturbation.
    secant = (
        _normalize_columns(reference, value + step)
        - _normalize_columns(reference, value - step)
    ) / (2 * step)
    assert not torch.allclose(derivative, secant, atol=1e-3, rtol=1e-3)


def test_inference_only_forward_fails_clearly():
    class DetachedForward(MaskedModel):
        @torch.no_grad()
        def forward(self, **kwargs):
            return super().forward(**kwargs)

    model = DetachedForward().eval().requires_grad_(False)
    with pytest.raises(RuntimeError, match="do not support input autograd"):
        estimate(model, torch.randn(9, 2, requires_grad=True))


@pytest.mark.parametrize("differentiable", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_normalization_selection_matches_external_context_only_fit(
    differentiable, normalize
):
    model = GradCacheModel().double().eval().requires_grad_(False)
    samples = torch.randn(
        9, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
    ).requires_grad_()
    context, _ = _disjoint_split(9, 5, 4, torch.Generator().manual_seed(42))
    normalizer = _soft_normalize_columns if differentiable else _normalize_columns
    transformed = normalizer(samples[context], samples) if normalize else samples
    actual = estimate(
        model, samples, normalize=normalize, differentiable=differentiable
    )
    expected = estimate(
        model, transformed, normalize=False, differentiable=differentiable
    )
    if differentiable:
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, samples)[0],
            torch.autograd.grad(expected, samples)[0],
        )
    else:
        assert actual == pytest.approx(expected, abs=1e-12)


def test_soft_normalization_reference_and_query_gradients_at_ties():
    reference = torch.tensor(
        [[-0.2], [0.0], [0.0], [0.3]], dtype=torch.float64, requires_grad=True
    )
    query = torch.tensor(
        [[-0.1], [0.0], [0.1]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(_soft_normalize_columns, (reference, query))
    assert torch.autograd.gradcheck(
        lambda x: _soft_normalize_columns(x, x), (reference,)
    )
    gradient = torch.autograd.grad(
        _soft_normalize_columns(reference, reference)[0].sum(), reference
    )[0]
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("n_context", [1, 4])
def test_soft_normalization_constant_columns_tails_and_chunks(dtype, n_context):
    reference = torch.zeros(n_context, 2, dtype=dtype, requires_grad=True)
    values = (
        torch.linspace(-1, 1, 130, dtype=dtype)[:, None]
        .expand(-1, 2)
        .clone()
        .requires_grad_()
    )
    result = _soft_normalize_columns(reference, values)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert result.device == values.device
    assert torch.isfinite(result).all()
    with torch.no_grad():
        expected = _soft_normalize_columns(reference, values)
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(result.sum(), (reference, values))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
