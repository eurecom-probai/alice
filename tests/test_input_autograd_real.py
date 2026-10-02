# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Input-gradient parity against ordinary checkpoint forwards."""

import os

import pytest
import torch

from alice import estimate_mi_fast, load_model, prepare_mi_model
from alice.torch.mi.alice_mi_fast import _disjoint_split

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "SKIP_REAL_MODEL_TESTS" in os.environ, reason="SKIP_REAL_MODEL_TESTS is set"
    ),
]


@pytest.fixture(scope="module")
def model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    loaded = load_model(
        os.environ.get("ALICE_TEST_MODEL", "eurecom-probai/alice-small-stage2"),
        revision="794828d2323873a19abfb077254b549e4629127b",
    ).requires_grad_(False)
    yield loaded
    torch.set_num_threads(previous_threads)


def evaluate(model, samples, **options):
    return estimate_mi_fast(
        model,
        samples,
        slice(0, 4, 2),
        slice(1, 4, 2),
        n_context=8,
        n_eval=3,
        n_t_samples=2,
        chunk=4,
        generator=torch.Generator().manual_seed(42),
        return_std=True,
        differentiable=options.pop("differentiable", True),
        normalize=options.pop("normalize", False),
        pad_mask=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        **options,
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"cache_projections": False},
        {"fuse_queries": False},
        {"checkpoint_queries": True},
        {"checkpoint_queries": True, "use_cache": False},
        {"checkpoint_queries": True, "cache_projections": False, "fuse_queries": False},
    ],
)
def test_real_cached_values_and_gradients_match_forward(model, normalize, options):
    samples = torch.randn(
        11, 4, generator=torch.Generator().manual_seed(12)
    ).requires_grad_()
    reference = evaluate(model, samples, normalize=normalize, use_cache=False)
    reference_grad = torch.autograd.grad(sum(reference), samples)[0]
    actual = evaluate(model, samples, normalize=normalize, **options)
    gradient = torch.autograd.grad(sum(actual), samples)[0]
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, atol=1e-4, rtol=1e-4)
    if options.get("fuse_queries", True):
        torch.testing.assert_close(gradient, reference_grad, atol=2e-4, rtol=2e-4)
    else:
        # Different GEMM row counts perturb FP32 backward more than forward.
        # The FP64 comparison below isolates this from a graph-path error.
        assert (gradient - reference_grad).norm() <= 1e-3 * reference_grad.norm()
    if not normalize:
        context, evaluation = _disjoint_split(
            11, 8, 3, torch.Generator().manual_seed(42)
        )
        assert gradient[context].norm() > 0
        assert gradient[evaluation].norm() > 0
    inference = evaluate(model, samples, normalize=normalize, differentiable=False)
    if not normalize:
        assert inference == pytest.approx(
            tuple(value.item() for value in actual), abs=1e-4
        )
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_low_precision_backward_is_finite_and_matches_cache_paths(model, dtype):
    original = {name: value.clone() for name, value in model.state_dict().items()}
    model.to(dtype)
    try:
        samples = torch.randn(
            11, 4, generator=torch.Generator().manual_seed(12)
        ).requires_grad_()
        reference = evaluate(model, samples, use_cache=False)
        reference_grad = torch.autograd.grad(sum(reference), samples)[0]
        actual = evaluate(model, samples, checkpoint_queries=True)
        gradient = torch.autograd.grad(sum(actual), samples)[0]
        assert all(value.dtype == torch.float32 for value in actual)
        assert torch.isfinite(gradient).all()
        # Shared caches sum adjoints before encoding backward; uncached graphs
        # sum after it. Bound the relative L2 discrepancy by four dtype epsilons.
        tolerance = 4 * torch.finfo(dtype).eps
        assert (gradient - reference_grad).norm() <= tolerance * reference_grad.norm()
        for value, expected in zip(actual, reference):
            torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)
    finally:
        model.float().load_state_dict(original)


def test_real_fused_and_unfused_gradients_match_in_float64(model):
    model.double()
    try:
        samples = (
            torch.randn(11, 4, generator=torch.Generator().manual_seed(12))
            .double()
            .requires_grad_()
        )
        reference = evaluate(model, samples, use_cache=False)
        reference_grad = torch.autograd.grad(sum(reference), samples)[0]
        actual = evaluate(model, samples, fuse_queries=False, checkpoint_queries=True)
        gradient = torch.autograd.grad(sum(actual), samples)[0]
        torch.testing.assert_close(gradient, reference_grad, atol=1e-10, rtol=1e-10)
    finally:
        model.float()


def test_prepared_fullgraph_query_supports_input_backward(model):
    prepared = prepare_mi_model(model, backend="eager")
    samples = torch.randn(
        11, 4, generator=torch.Generator().manual_seed(12)
    ).requires_grad_()
    reference = evaluate(model, samples)
    reference_grad = torch.autograd.grad(sum(reference), samples)[0]
    actual = evaluate(prepared, samples, checkpoint_queries=True)
    gradient = torch.autograd.grad(sum(actual), samples)[0]
    torch.testing.assert_close(gradient, reference_grad, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_available_accelerators_preserve_input_gradients(model, device, dtype):
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} is unavailable")
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 is unavailable")
    original = {name: value.clone() for name, value in model.state_dict().items()}
    model.to(device=device, dtype=dtype)
    try:
        samples = (
            torch.randn(11, 4, generator=torch.Generator().manual_seed(12))
            .to(device)
            .requires_grad_()
        )
        reference = evaluate(model, samples, use_cache=False, normalize=True)
        reference_grad = torch.autograd.grad(sum(reference), samples)[0]
        actual = evaluate(model, samples, checkpoint_queries=True, normalize=True)
        gradient = torch.autograd.grad(sum(actual), samples)[0]
        assert torch.isfinite(gradient).all()
        tolerance = max(1e-3, 4 * torch.finfo(dtype).eps)
        assert (gradient - reference_grad).norm() <= tolerance * reference_grad.norm()
        for value, expected in zip(actual, reference):
            torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)
    finally:
        model.to(device="cpu", dtype=torch.float32).load_state_dict(original)


def test_fixed_context_cache_is_autograd_compatible(model):
    from alice.torch.mi.alice_mi_fast import _CachedModel

    rng = torch.Generator().manual_seed(7)
    context = torch.randn(1, 8, 2, generator=rng)
    states = torch.randn(3, 2, generator=rng).requires_grad_()
    times = torch.rand(3, 1, generator=rng)
    masks = torch.ones_like(states)
    predictor = _CachedModel(model, context, True, True, differentiable=True)
    assert not torch.is_inference(predictor.cache[1])
    actual = predictor.velocity(states, times, masks, context, predictor.cache)
    gradient = torch.autograd.grad(actual.sum(), states)[0]
    reference = _CachedModel(model, context, False, False, differentiable=True)
    expected = reference.velocity(states, times, masks, context, None)
    expected_gradient = torch.autograd.grad(expected.sum(), states)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=2e-4, rtol=2e-4)
