# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Cached, batched ALICE mutual-information estimation in velocity form."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..utils import _model_device, _model_dtype


def _normal_quantile(probability: Tensor) -> Tensor:
    r"""_normal_quantile(probability) -> Tensor

    Apply the standard-normal inverse CDF using ``erfinv``.

    Args:
        probability (Tensor): Probabilities in ``[0, 1]``. Values are
            clamped slightly away from the endpoints for numerical stability.

    Returns:
        Tensor: Standard-normal quantiles with the same shape and dtype as
        ``probability``.
    """

    return math.sqrt(2.0) * torch.erfinv(
        (2.0 * probability - 1.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    )


def _disjoint_split(
    n: int,
    n_context: int | None,
    n_eval: int | None,
    generator: torch.Generator | None,
):
    r"""_disjoint_split(n, n_context, n_eval, generator) -> tuple[Tensor, Tensor]

    Resolve and apply a random disjoint context/evaluation split.

    If either size is ``None``, infer it from the other size and ``n``. If
    both are ``None``, use the first half for context and the remainder for
    evaluation. The returned indices are generated on the generator's device;
    the estimator moves them to the sample device before indexing.

    Args:
        n (int): Total number of available rows.
        n_context (int | None): Number of context rows, or ``None`` to infer.
        n_eval (int | None): Number of evaluation rows, or ``None`` to infer.
        generator (torch.Generator | None): Generator controlling the random
            permutation. Default: ``None``.

    Returns:
        tuple[Tensor, Tensor]: Context and evaluation index tensors.

    Raises:
        ValueError: If either split is empty or their sum exceeds ``n``.
    """

    if n_context is None and n_eval is None:
        n_context = n // 2
        n_eval = n - n_context
    elif n_eval is None:
        n_eval = n - n_context
    elif n_context is None:
        n_context = n - n_eval
    if n_context < 1 or n_eval < 1:
        raise ValueError(
            f"need n_context >= 1 and n_eval >= 1; got {n_context}, {n_eval}"
        )
    if n_context + n_eval > n:
        raise ValueError(f"n_context + n_eval = {n_context + n_eval} exceeds n = {n}")

    generator_device = (
        torch.device(generator.device) if generator is not None else torch.device("cpu")
    )
    permutation = torch.randperm(n, generator=generator, device=generator_device)
    return permutation[:n_context], permutation[n_context : n_context + n_eval]


def _random_tensor(
    kind: str,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    r"""_random_tensor(kind, shape, *, generator, device, dtype) -> Tensor

    Draw a reproducible random tensor, using the generator's device when a
    generator is supplied and moving the result to ``device`` afterward.

    Args:
        kind (str): ``"uniform"`` for ``torch.rand``; any other value selects
            ``torch.randn``.
        shape (tuple[int, ...]): Shape of the tensor to draw.
        generator (torch.Generator | None): Optional random generator.
        device (torch.device): Destination device.
        dtype (torch.dtype): Output dtype.

    Returns:
        Tensor: Random values on ``device`` with the requested ``shape`` and
        ``dtype``.
    """

    generator_device = (
        torch.device(generator.device) if generator is not None else device
    )
    if kind == "uniform":
        values = torch.rand(
            shape, generator=generator, device=generator_device, dtype=dtype
        )
    else:
        values = torch.randn(
            shape, generator=generator, device=generator_device, dtype=dtype
        )
    return values.to(device=device)


def _normalize_columns(reference: Tensor, values: Tensor) -> Tensor:
    """Apply empirical Gaussian copulas to all columns with batched searches."""
    n = reference.shape[0]
    if n == 1:
        return values * 0.0
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    sorted_ref = reference.to(dtype).T.contiguous().sort(dim=-1).values
    x = values.to(dtype).T.contiguous()
    indices = torch.searchsorted(sorted_ref, x).clamp(1, n - 1)
    lower = sorted_ref.gather(1, indices - 1)
    upper = sorted_ref.gather(1, indices)
    fraction = ((x - lower) / (upper - lower).clamp_min(1e-12)).clamp(0.0, 1.0)
    positions = (torch.arange(n, device=x.device, dtype=dtype) + 1.0) / (n + 1.0)
    probability = positions[indices - 1] + fraction * (
        positions[indices] - positions[indices - 1]
    )
    return _normal_quantile(probability).T.to(values.dtype).contiguous()


def _soft_normalize_columns(reference: Tensor, values: Tensor) -> Tensor:
    r"""Apply a context-fitted Gaussian-kernel CDF and normal quantile.

    For each column, use bandwidth ``sqrt(0.2**2 * variance + 0.001**2)``
    with population variance fitted only on ``reference``. Average
    ``Phi((value - reference) / bandwidth)`` over context rows and shrink
    probabilities to ``[1e-4, 1 - 1e-4]`` before the inverse normal CDF.
    Gradients include query values, context values and the fitted bandwidth.

    Inputs have shapes ``(n_context, d)`` and ``(n_values, d)``. Computation
    preserves FP64 and otherwise uses FP32; output retains that working dtype
    and the input device. Singleton contexts use the same smooth formula.
    Deterministic chunks of 128 query rows are checkpointed when inputs require
    gradients, avoiding retention of all pairwise kernel activations.
    """
    dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    reference, values = reference.to(dtype), values.to(dtype)
    bandwidth = (0.2**2 * reference.var(dim=0, correction=0) + 0.001**2).sqrt()

    def transform(query: Tensor, context: Tensor, width: Tensor) -> Tensor:
        """Gaussianize one query chunk with explicit fitted-CDF dependencies."""
        distance = (query[:, None, :] - context[None, :, :]) / width
        cdf = (0.5 * (1 + torch.erf(distance / math.sqrt(2)))).mean(dim=1)
        probability = 1e-4 + (1 - 2e-4) * cdf
        return math.sqrt(2) * torch.erfinv(2 * probability - 1)

    use_checkpoint = torch.is_grad_enabled() and (
        reference.requires_grad or values.requires_grad
    )
    return torch.cat(
        [
            checkpoint(transform, query, reference, bandwidth, use_reentrant=False)
            if use_checkpoint
            else transform(query, reference, bandwidth)
            for query in values.split(128, dim=0)
        ],
        dim=0,
    )


def _project_keys_values(
    attention: nn.MultiheadAttention, states: Tensor
) -> tuple[Tensor, Tensor]:
    """Project context once and store contiguous per-head attention keys/values."""
    b, d, n, width = states.shape
    bias = attention.in_proj_bias
    sequence = states.reshape(b * d, n, width).transpose(0, 1)
    projected = F.linear(
        sequence,
        attention.in_proj_weight[width:],
        None if bias is None else bias[width:],
    )
    key, value = projected.chunk(2, dim=-1)
    shape = (n, b * d, attention.num_heads, width // attention.num_heads)
    return (
        key.reshape(shape).permute(1, 2, 0, 3).contiguous(),
        value.reshape(shape).permute(1, 2, 0, 3).contiguous(),
    )


def _attention(
    attention: nn.MultiheadAttention, query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """Read cached keys/values using MultiheadAttention's projection layout.

    Sequence-first projection preserves low-precision bias rounding and GEMM
    row order. Changing these layouts can alter bfloat16 MI beyond 1e-3.
    """
    b, d, n, width = query.shape
    bias = attention.in_proj_bias
    sequence = query.reshape(b * d, n, width).transpose(0, 1)
    q = F.linear(
        sequence,
        attention.in_proj_weight[:width],
        None if bias is None else bias[:width],
    )
    q = q.reshape(n, b * d, attention.num_heads, width // attention.num_heads).permute(
        1, 2, 0, 3
    )
    output = F.scaled_dot_product_attention(q, key, value, dropout_p=0.0)
    output = output.permute(2, 0, 1, 3).contiguous().reshape(n * b * d, width)
    output = attention.out_proj(output).reshape(n, b * d, width).transpose(0, 1)
    return output.reshape(b, d, n, width)


def _supports_projected_cache(model: Any) -> bool:
    """Recognize the induced-group decoder and its ordinary packed-QKV reads."""
    if type(model).__name__ != "InducedGroupICLDenoiserModel":
        return False
    decoder = getattr(model, "query_decoder", None)
    if type(decoder).__name__ != "QueryDecoder":
        return False
    for name in ("latent_attention", "support_attention"):
        attention = getattr(decoder, name, None)
        if (
            not isinstance(attention, nn.MultiheadAttention)
            or not attention._qkv_same_embed_dim
        ):
            return False
        if (
            attention.bias_k is not None
            or attention.bias_v is not None
            or attention.add_zero_attn
        ):
            return False
    return True


class _InputGradModel(nn.Module):
    """Use induced-group components without inference-only cache decorators."""

    def __init__(self, model: nn.Module):
        """Wrap the model without changing its parameters, methods or mode."""
        super().__init__()
        self.model = model
        self.config = model.config
        self.training = model.training

    def forward(self, **kwargs):
        """Retain the original forward when support caching is disabled."""
        return self.model(**kwargs)

    def _support_cache(self, context: Tensor):
        """Build a fresh graph, support representation and latents with autograd."""
        graph, support = self.model._encode_support(context, ctx_mask=None)
        latents = self.model._encode_latents(support, graph, support_padding_mask=None)
        return graph, support, latents

    def predict_query_logits_cached(
        self,
        ctx_clean: Tensor,
        qry_z_t: Tensor,
        qry_t: Tensor,
        qry_mask: Tensor,
        support_cache: Any,
    ) -> Tensor:
        """Decode queries using explicit graph-bearing support tensors."""
        graph, support, latents = support_cache
        query = self.model._encode_query(qry_z_t, qry_t, qry_mask, graph)
        query = self.model._decode_query(
            query, latents, support, graph, support_padding_mask=None
        )
        return self.model.head(self.model.final_norm(query)).squeeze(-1).transpose(1, 2)


class _ProjectedModel(_InputGradModel):
    """Eval-only induced-group adapter with per-context decoder projection caching."""

    def __init__(self, model: nn.Module):
        """Wrap the model and expose the compilable query integrand."""
        super().__init__(model)
        self.terms = self._terms

    def _support_cache(self, context: Tensor, *, differentiable: bool = False):
        """Encode support and cache normalized decoder key/value projections."""
        graph, support, latents = (
            super()._support_cache(context)
            if differentiable
            else self.model._support_cache(context)
        )
        decoder = self.model.query_decoder
        latent_k, latent_v = _project_keys_values(
            decoder.latent_attention, decoder.latent_key_value_norm(latents)
        )
        support_k, support_v = _project_keys_values(
            decoder.support_attention, decoder.support_key_value_norm(support)
        )
        return graph, latent_k, latent_v, support_k, support_v

    def _query(
        self,
        states: Tensor,
        times: Tensor,
        masks: Tensor,
        graph: Any,
        latent_k: Tensor,
        latent_v: Tensor,
        support_k: Tensor,
        support_v: Tensor,
    ) -> Tensor:
        """Run independent query encoding and decoding using cached projections."""
        query = self.model._encode_query(states, times, masks, graph)
        decoder = self.model.query_decoder
        query = query + _attention(
            decoder.latent_attention,
            decoder.latent_query_norm(query),
            latent_k,
            latent_v,
        )
        query = query + _attention(
            decoder.support_attention,
            decoder.support_query_norm(query),
            support_k,
            support_v,
        )
        query = query + decoder.ffn(decoder.ffn_norm(query))
        query = decoder.graph_mixer(query, graph)
        return self.model.head(self.model.final_norm(query)).squeeze(-1).transpose(1, 2)

    def _terms(
        self,
        clean: Tensor,
        full: Tensor,
        times: Tensor,
        masks: Tensor,
        keep: Tensor,
        x_slice: slice,
        y_slice: slice,
        graph: Any,
        latent_k: Tensor,
        latent_v: Tensor,
        support_k: Tensor,
        support_v: Tensor,
    ) -> Tensor:
        """Construct all six query groups and reduce their weighted velocity gaps."""
        n, d = full.shape
        dtype = self.model.input_proj.weight.dtype
        states = (
            torch.where(masks[:, None].bool(), full[None], clean[None])
            .reshape(3 * n, d)
            .to(dtype)
        )
        query_masks = masks[:, None].expand(3, n, d).reshape(3 * n, d).to(dtype)
        query_times = times.repeat(3, 1).to(dtype)
        logits = self._query(
            torch.cat((states, states))[None],
            torch.cat((query_times, torch.zeros_like(query_times)))[None],
            torch.cat((query_masks, query_masks))[None],
            graph,
            latent_k,
            latent_v,
            support_k,
            support_v,
        )[0]
        at_t, at_zero = logits.chunk(2)
        velocities = (states + at_t - at_zero).to(full.dtype).reshape(3, n, d)
        diff_x = (velocities[0] - velocities[1])[:, x_slice] * keep[x_slice]
        diff_y = (velocities[0] - velocities[2])[:, y_slice] * keep[y_slice]
        gap = diff_x.square().sum(-1) + diff_y.square().sum(-1)
        return (1.0 - times[:, 0]) / times[:, 0] * gap

    def predict_query_logits_cached(
        self,
        ctx_clean: Tensor,
        qry_z_t: Tensor,
        qry_t: Tensor,
        qry_mask: Tensor,
        support_cache: Any,
    ) -> Tensor:
        """Evaluate query logits from a cache belonging to this model and context."""
        return self._query(qry_z_t, qry_t, qry_mask, *support_cache)


def prepare_mi_model(
    model: nn.Module, *, compile: bool = True, backend: str = "inductor"
) -> nn.Module:
    """Prepare an eval-mode induced-group model for repeated MI estimation.

    Return an adapter sharing the model's parameters without modifying its
    methods. With ``compile=True``, lazily compile the tensor query path with
    ``torch.compile(fullgraph=True, dynamic=False)``. Keep and reuse this
    adapter to amortize compilation; new query shapes may trigger compilation.
    The compiled region includes query construction and the MI integrand;
    use it with ``use_cache=True``, ``cache_projections=True`` and
    ``fuse_queries=True``. Support construction and random draws remain eager.
    Compiler speedups depend on the device; benchmark warm calls before use.
    Context-dependent caches are rebuilt for every estimation, even when the
    compiled query is reused. Compilation failures propagate to the caller.
    """
    if model.training:
        raise ValueError("fast MI inference is eval-only; call model.eval() first")
    if not _supports_projected_cache(model):
        raise ValueError(
            "projected/compiled inference requires an induced-group ALICE checkpoint"
        )
    prepared = _ProjectedModel(model)
    if compile:
        prepared.terms = torch.compile(
            prepared._terms, backend=backend, fullgraph=True, dynamic=False
        )
    return prepared


class _CachedModel:
    """Bind one support cache to an eval-mode model without mutating it."""

    def __init__(
        self,
        model: Any,
        context: Tensor,
        use_cache: bool,
        cache_projections: bool,
        differentiable: bool = False,
    ):
        """Build a fresh cache for this estimation call when supported."""
        if isinstance(model, _ProjectedModel) and not cache_projections:
            model = model.model
        if use_cache and cache_projections and _supports_projected_cache(model):
            model = _ProjectedModel(model)
        if differentiable and use_cache:
            if _supports_projected_cache(model):
                model = _InputGradModel(model)
            elif not isinstance(model, _ProjectedModel):
                # Unknown checkpoint caches may hide no_grad or detach calls.
                # Only an explicit capability declaration opts them in.
                use_cache = getattr(model, "supports_input_grad_cache", False) is True
        self.model = model
        self.context = context
        self.differentiable = differentiable
        self.cache = None
        if (
            use_cache
            and callable(getattr(model, "_support_cache", None))
            and callable(getattr(model, "predict_query_logits_cached", None))
        ):
            if isinstance(model, _ProjectedModel):
                self.cache = model._support_cache(
                    context, differentiable=differentiable
                )
            else:
                self.cache = model._support_cache(context)

    def velocity(
        self,
        states: Tensor,
        times: Tensor,
        masks: Tensor,
        context: Tensor,
        cache: Any,
    ) -> Tensor:
        """Evaluate all masked states and their zero-time corrections together."""
        dtype = context.dtype
        states = states.to(dtype).unsqueeze(0)
        times = times.to(dtype).unsqueeze(0)
        masks = masks.to(dtype).unsqueeze(0)
        inputs = {
            "ctx_clean": context,
            "qry_z_t": torch.cat((states, states), dim=1),
            "qry_t": torch.cat((times, torch.zeros_like(times)), dim=1),
            "qry_mask": torch.cat((masks, masks), dim=1),
        }
        if cache is not None:
            logits = self.model.predict_query_logits_cached(
                **inputs, support_cache=cache
            )
        else:
            logits = getattr(self.model(**inputs), "logits", None)
            if logits is None:
                raise RuntimeError("ALICE model did not return logits")
        if (
            self.differentiable
            and (states.requires_grad or context.requires_grad)
            and not logits.requires_grad
        ):
            raise RuntimeError(
                "model query logits do not support input autograd; use a checkpoint "
                "with a gradient-capable forward/cache implementation"
            )
        at_t, at_zero = logits.chunk(2, dim=1)
        return (states + at_t - at_zero).squeeze(0)

    def terms(
        self,
        clean: Tensor,
        full: Tensor,
        times: Tensor,
        masks: Tensor,
        keep: Tensor,
        x_slice: slice,
        y_slice: slice,
        context: Tensor,
        cache: Any,
        fuse_queries: bool,
    ) -> Tensor:
        """Evaluate a deterministic chunk with explicit context/cache dependencies."""
        if (
            isinstance(self.model, _ProjectedModel)
            and cache is not None
            and fuse_queries
        ):
            return self.model.terms(
                clean, full, times, masks, keep, x_slice, y_slice, *cache
            )
        x_noised = clean.clone()
        x_noised[:, x_slice] = full[:, x_slice]
        y_noised = clean.clone()
        y_noised[:, y_slice] = full[:, y_slice]
        states = (full, x_noised, y_noised)
        field_masks = tuple(mask.expand_as(full) for mask in masks)
        if fuse_queries:
            velocities = self.velocity(
                torch.cat(states),
                times.repeat(3, 1),
                torch.cat(field_masks),
                context,
                cache,
            ).to(full.dtype)
            v_full, v_x_cond, v_y_cond = velocities.chunk(3)
        else:
            v_full, v_x_cond, v_y_cond = (
                self.velocity(state, times, mask, context, cache).to(full.dtype)
                for state, mask in zip(states, field_masks)
            )
        diff_x = (v_full - v_x_cond)[:, x_slice] * keep[x_slice]
        diff_y = (v_full - v_y_cond)[:, y_slice] * keep[y_slice]
        gap = diff_x.square().sum(dim=-1) + diff_y.square().sum(dim=-1)
        return (1.0 - times[:, 0]) / times[:, 0] * gap


def estimate_mi_fast(
    model: Any,
    joint_samples: Tensor,
    x_slice: slice,
    y_slice: slice,
    *,
    n_context: int | None = None,
    n_eval: int | None = None,
    n_t_samples: int = 64,
    t_min: float = 1e-3,
    t_max: float = 1.0 - 1e-3,
    chunk: int = 256,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    return_std: bool = False,
    normalize: bool = True,
    pad_mask: Tensor | None = None,
    use_cache: bool = True,
    fuse_queries: bool = True,
    cache_projections: bool = True,
    differentiable: bool = False,
    checkpoint_queries: bool = False,
) -> float | tuple[float, float] | Tensor | tuple[Tensor, Tensor]:
    r"""estimate_mi_fast(model, joint_samples, x_slice, y_slice, *, ...)

    Estimate ``I(X;Y)`` with cached, fused ALICE MI inference.

    Support and decoder key/value projection caching default to enabled.
    Projection caching specializes the induced-group architecture;
    ``cache_projections=False`` skips decoder projection caching.
    Support caching applies to checkpoints exposing the support-cache API;
    other models use ordinary forwards. The cache lives only for this call.
    ``fuse_queries=True`` batches the three masks into one forward, including
    their zero-time corrections (up to ``6 * chunk`` model query rows). Set it
    false to reduce peak query memory or benchmark caching alone. Random draws,
    model precision, boundary correction, and the reduction are preserved.

    ``joint_samples`` contains clean joint draws.  The two slices must be
    disjoint and cover every column.  The rows are split into a clean context
    that binds the in-context velocity field and an evaluation set used for
    the Monte Carlo integral.  When ``normalize`` is true, both sets are
    Gaussian-copula normalized using the context as the reference distribution.

    For each evaluation row and sampled time, the estimator evaluates the
    joint query ``[X_t, Y_t]``, the ``X_t`` query with clean ``Y``, and the
    ``Y_t`` query with clean ``X``.  These three queries share the same noise
    draw.  The integral uses the derived weight ``(1 - t) / t`` and is
    multiplied by ``t_max - t_min`` because times are sampled uniformly on
    that interval.

    With ``differentiable=True``, return tensors and preserve input gradients
    through context and queries. All model parameters must already be frozen;
    their flags and gradients are never modified. Enclosing ``no_grad()`` or
    ``inference_mode()`` and inference-created samples are rejected. A new
    autograd graph and cache are built for each call. Do not change inputs or
    model state between this call and backward, including checkpoint recomputation.

    Recognized induced-group checkpoints use their ordinary encoding/decoding
    components to bypass inference-only cache methods; no checkpoint update is
    needed. Other checkpoints use forward unless they explicitly declare
    ``supports_input_grad_cache=True``, guaranteeing both cache methods preserve
    context and query autograd. Their forward must itself support input autograd.

    With ``normalize=True``, differentiable calls use a smooth Gaussian-kernel
    CDF fitted only on context, including gradients through its values and
    bandwidth. The bandwidth is ``sqrt(0.2**2 * variance + 0.001**2)`` and CDF
    probabilities shrink to ``[1e-4, 1 - 1e-4]``. Ordinary inference uses hard
    empirical ranks, so normalized MI values can differ between modes. No
    transform is detached or given a straight-through derivative. Use
    ``normalize=False`` for caller-provided preprocessing in either mode.

    Differentiable reductions use FP32 for FP16/BF16 samples, otherwise the
    sample dtype. Model operations retain the model dtype; low-precision
    gradients may need caller-managed loss scaling. Random draws precede any
    activation checkpoint and follow the same order as inference calls.

    Args:
        model (torch.nn.Module): Eval-mode ALICE model trained with
            ``config.use_mask_channel=True``.
        joint_samples (Tensor): Clean joint samples with shape ``(N, d)``.
        x_slice (slice): Columns belonging to the ``X`` block.
        y_slice (slice): Columns belonging to the ``Y`` block.
        n_context (int | None, optional): Number of clean rows used as
            context. Defaults to half of ``N``.
        n_eval (int | None, optional): Number of rows used for the Monte Carlo
            integral. Defaults to all rows not used for context.
        n_t_samples (int, optional): Number of uniformly sampled times per
            evaluation row. Default: ``64``.
        t_min (float, optional): Lower integration endpoint. Must be positive.
            Default: ``1e-3``.
        t_max (float, optional): Upper integration endpoint. Must be greater
            than ``t_min`` and at most one. Default: ``1.0 - 1e-3``.
        chunk (int, optional): Monte Carlo rows per chunk, producing up to
            ``6 * chunk`` model query rows with fusion. Default: ``256``.
        generator (torch.Generator | None, optional): Generator controlling
            the split, times, and Gaussian noise. Default: ``None``.
        device (torch.device | str | None, optional): Device for model field
            evaluations. Defaults to the model parameter device.
        return_std (bool, optional): Also return the Monte Carlo standard
            error. Default: ``False``.
        normalize (bool, optional): Apply Gaussian-copula normalization to
            context and evaluation samples. Default: ``True``.
        pad_mask (Tensor | None, optional): Length-``d`` mask, nonzero on
            iid Gaussian-padded coordinates to exclude from the reduction.
            Default: ``None``.

        use_cache (bool, optional): Reuse support encoding when supported.
            Default: ``True``.
        fuse_queries (bool, optional): Batch all three masked fields together.
            Uses approximately three times the query activation memory.
            Default: ``True``.
        cache_projections (bool, optional): Cache decoder attention keys and
            values for compatible induced-group checkpoints. Requires
            ``use_cache=True``. Default: ``True``.
        differentiable (bool, optional): Return tensors with input autograd,
            requiring an eval-mode model with frozen parameters. Default: ``False``.
        checkpoint_queries (bool, optional): Recompute each query chunk's
            activations during backward using non-reentrant checkpointing.
            Requires ``differentiable=True``. Context/cache tensors and their
            encoding graph remain resident; support encoding is not checkpointed
            when cached. Default: ``False``.

    Returns:
        float | tuple[float, float] | Tensor | tuple[Tensor, Tensor]: MI in nats.
        Differentiable mode returns scalar tensors on the model evaluation device.
        With ``return_std=True``, return ``(estimate, standard_error)``; this is
        Monte Carlo standard error, not estimator accuracy. Its derivative is
        defined as zero at nonpositive estimated variance.

    Raises:
        ValueError: If the samples, model mask-channel capability, partition,
            integration interval, chunk size, split sizes, or padding mask
            have invalid shapes or values, or autograd requirements are violated.
        RuntimeError: If the model omits logits or returns detached logits for
            inputs requiring gradients.
    """

    if checkpoint_queries and not differentiable:
        raise ValueError("checkpoint_queries=True requires differentiable=True")
    if differentiable:
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise ValueError(
                "differentiable=True requires enabled autograd outside no_grad() "
                "and inference_mode()"
            )
        if torch.is_inference(joint_samples):
            raise ValueError("joint_samples must be created outside inference_mode()")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "differentiable=True requires frozen model parameters; "
                "call model.requires_grad_(False) first"
            )
    with torch.enable_grad() if differentiable else torch.inference_mode():
        result = _estimate_mi_tensor(
            model,
            joint_samples,
            x_slice,
            y_slice,
            n_context=n_context,
            n_eval=n_eval,
            n_t_samples=n_t_samples,
            t_min=t_min,
            t_max=t_max,
            chunk=chunk,
            generator=generator,
            device=device,
            return_std=return_std,
            normalize=normalize,
            pad_mask=pad_mask,
            use_cache=use_cache,
            fuse_queries=fuse_queries,
            cache_projections=cache_projections,
            differentiable=differentiable,
            checkpoint_queries=checkpoint_queries,
        )
        if differentiable:
            return result
        if isinstance(result, tuple):
            return tuple(float(value) for value in result)
        return float(result)


def _estimate_mi_tensor(
    model: Any,
    joint_samples: Tensor,
    x_slice: slice,
    y_slice: slice,
    *,
    n_context: int | None = None,
    n_eval: int | None = None,
    n_t_samples: int = 64,
    t_min: float = 1e-3,
    t_max: float = 1.0 - 1e-3,
    chunk: int = 256,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    return_std: bool = False,
    normalize: bool = True,
    pad_mask: Tensor | None = None,
    use_cache: bool = True,
    fuse_queries: bool = True,
    cache_projections: bool = True,
    differentiable: bool = False,
    checkpoint_queries: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute MI tensors in the caller-selected gradient mode."""

    if getattr(model, "training", False) or (
        isinstance(model, _ProjectedModel) and model.model.training
    ):
        raise ValueError("fast MI inference is eval-only; call model.eval() first")
    if joint_samples.ndim != 2:
        raise ValueError(
            f"joint_samples must have shape (N, d); got {tuple(joint_samples.shape)}"
        )
    if not torch.is_floating_point(joint_samples):
        raise ValueError(
            f"joint_samples must use a floating dtype; got {joint_samples.dtype}"
        )
    if n_t_samples < 1:
        raise ValueError(f"n_t_samples must be >= 1; got {n_t_samples}")
    if not 0.0 < t_min < t_max <= 1.0:
        raise ValueError(
            f"require 0 < t_min < t_max <= 1; got t_min={t_min}, t_max={t_max}"
        )
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1; got {chunk}")

    config = getattr(model, "config", None)
    if config is None or not getattr(config, "use_mask_channel", False):
        raise ValueError(
            "estimate_mi_fast requires a model trained with a mask channel "
            "(config.use_mask_channel=True)"
        )

    dim = joint_samples.shape[-1]
    if (
        not getattr(config, "variable_dim", False)
        and getattr(config, "d_y", None) != dim
    ):
        raise ValueError(
            f"model.config.d_y={getattr(config, 'd_y', None)} must match joint width d={dim}"
        )

    covered = list(range(dim)[x_slice]) + list(range(dim)[y_slice])
    if sorted(covered) != list(range(dim)) or len(covered) != dim:
        raise ValueError(
            f"x_slice and y_slice must be disjoint and cover all {dim} columns; "
            f"got x={range(dim)[x_slice]}, y={range(dim)[y_slice]}"
        )

    if pad_mask is not None and tuple(pad_mask.shape) != (dim,):
        raise ValueError(
            f"pad_mask must have shape ({dim},); got {tuple(pad_mask.shape)}"
        )

    context_indices, eval_indices = _disjoint_split(
        joint_samples.shape[0], n_context, n_eval, generator
    )
    context_indices = context_indices.to(joint_samples.device)
    eval_indices = eval_indices.to(joint_samples.device)
    context = joint_samples[context_indices]
    eval_z0 = joint_samples[eval_indices]
    if normalize:
        normalizer = _soft_normalize_columns if differentiable else _normalize_columns
        normalized = normalizer(context, torch.cat((context, eval_z0)))
        context, eval_z0 = normalized.split((context.shape[0], eval_z0.shape[0]))

    run_device = _model_device(model, device)
    predictor = _CachedModel(
        model,
        context.unsqueeze(0).to(
            device=run_device, dtype=_model_dtype(model, context.dtype)
        ),
        use_cache,
        cache_projections,
        differentiable,
    )
    dtype = eval_z0.dtype
    eval_device = eval_z0.device
    t = t_min + (t_max - t_min) * _random_tensor(
        "uniform",
        (eval_z0.shape[0], n_t_samples, 1),
        generator=generator,
        device=eval_device,
        dtype=dtype,
    )
    epsilon = _random_tensor(
        "normal",
        (eval_z0.shape[0], n_t_samples, dim),
        generator=generator,
        device=eval_device,
        dtype=dtype,
    )
    z0 = eval_z0.unsqueeze(1).to(run_device)
    t = t.to(run_device)
    epsilon = epsilon.to(run_device)
    z_full = (1.0 - t) * z0 + t * epsilon

    mask_x = torch.zeros(dim, device=run_device, dtype=dtype)
    mask_x[x_slice] = 1.0
    mask_y = torch.zeros(dim, device=run_device, dtype=dtype)
    mask_y[y_slice] = 1.0
    mask_all = torch.ones(dim, device=run_device, dtype=dtype)
    keep = torch.ones(dim, device=run_device, dtype=dtype)
    if pad_mask is not None:
        keep = 1.0 - pad_mask.to(device=run_device, dtype=dtype)

    masks = torch.stack((mask_all, mask_x, mask_y))
    total_rows = eval_z0.shape[0] * n_t_samples
    accumulation_dtype = (
        torch.float32
        if differentiable and dtype in (torch.float16, torch.bfloat16)
        else dtype
    )
    sum_term = torch.zeros((), device=run_device, dtype=accumulation_dtype)
    sum_squared_term = torch.zeros_like(sum_term)
    flat_zfull = z_full.reshape(-1, dim)
    flat_t = t.reshape(-1, 1)

    for start in range(0, total_rows, chunk):
        stop = min(start + chunk, total_rows)
        rows = torch.arange(start, stop, device=run_device) // n_t_samples
        clean = z0[:, 0][rows]
        full = flat_zfull[start:stop]
        times = flat_t[start:stop]

        arguments = (
            clean.to(accumulation_dtype),
            full.to(accumulation_dtype),
            times.to(accumulation_dtype),
            masks,
            keep,
            x_slice,
            y_slice,
            predictor.context,
            predictor.cache,
            fuse_queries,
        )
        term = (
            checkpoint(predictor.terms, *arguments, use_reentrant=False)
            if checkpoint_queries
            else predictor.terms(*arguments)
        )
        sum_term = sum_term + term.sum()
        if return_std:
            sum_squared_term = sum_squared_term + term.square().sum()

    interval = t_max - t_min
    estimate = sum_term / total_rows * interval
    if not return_std:
        return estimate
    variance = (sum_squared_term - sum_term.square() / total_rows) / max(
        total_rows - 1, 1
    )
    if differentiable:
        # Define a finite zero derivative for degenerate/clamped MC variance.
        positive = variance > 0
        root = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        root = torch.where(positive, root, torch.zeros_like(root))
    else:
        root = variance.clamp_min(0.0).sqrt()
    standard_error = root / math.sqrt(total_rows) * interval
    return estimate, standard_error


__all__ = ["estimate_mi_fast", "prepare_mi_model"]
