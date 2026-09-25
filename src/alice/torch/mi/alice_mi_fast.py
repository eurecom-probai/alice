# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Cached, batched ALICE MI inference with the same Monte Carlo estimator."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils import _model_device, _model_dtype
from .alice_mi import _disjoint_split, _normal_quantile, _random_tensor


def _normalize_columns(reference: Tensor, values: Tensor) -> Tensor:
    """Apply empirical Gaussian copulas to all columns with batched searches."""
    n = reference.shape[0]
    if n == 1:
        return torch.zeros_like(values)
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


class _ProjectedModel(nn.Module):
    """Eval-only induced-group adapter with per-context decoder projection caching."""

    def __init__(self, model: nn.Module):
        """Wrap the model without changing its parameters, methods or mode."""
        super().__init__()
        self.model = model
        self.config = model.config
        self.training = model.training
        self.terms = self._terms

    def forward(self, **kwargs):
        """Retain the original forward when support caching is disabled."""
        return self.model(**kwargs)

    def _support_cache(self, context: Tensor):
        """Encode support and cache normalized decoder key/value projections."""
        graph, support, latents = self.model._support_cache(context)
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
        self, model: Any, context: Tensor, use_cache: bool, cache_projections: bool
    ):
        """Build a fresh cache for this estimation call when supported."""
        if isinstance(model, _ProjectedModel) and not cache_projections:
            model = model.model
        if use_cache and cache_projections and _supports_projected_cache(model):
            model = _ProjectedModel(model)
        self.model = model
        self.context = context
        self.cache = None
        if (
            use_cache
            and callable(getattr(model, "_support_cache", None))
            and callable(getattr(model, "predict_query_logits_cached", None))
        ):
            self.cache = model._support_cache(context)

    def velocity(self, states: Tensor, times: Tensor, masks: Tensor) -> Tensor:
        """Evaluate all masked states and their zero-time corrections together."""
        dtype = self.context.dtype
        states = states.to(dtype).unsqueeze(0)
        times = times.to(dtype).unsqueeze(0)
        masks = masks.to(dtype).unsqueeze(0)
        inputs = {
            "ctx_clean": self.context,
            "qry_z_t": torch.cat((states, states), dim=1),
            "qry_t": torch.cat((times, torch.zeros_like(times)), dim=1),
            "qry_mask": torch.cat((masks, masks), dim=1),
        }
        if self.cache is not None:
            logits = self.model.predict_query_logits_cached(
                **inputs, support_cache=self.cache
            )
        else:
            logits = getattr(self.model(**inputs), "logits", None)
            if logits is None:
                raise RuntimeError("ALICE model did not return logits")
        at_t, at_zero = logits.chunk(2, dim=1)
        return (states + at_t - at_zero).squeeze(0)


@torch.inference_mode()
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
) -> float | tuple[float, float]:
    r"""estimate_mi_fast(model, joint_samples, x_slice, y_slice, *, ...) -> float | tuple[float, float]

    Estimate ``I(X;Y)`` with cached, fused ALICE MI inference.

    Support and decoder key/value projection caching default to enabled.
    Projection caching specializes the induced-group architecture;
    ``cache_projections=False`` uses the checkpoint's unmodified cached decoder.
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

    Returns:
        float | tuple[float, float]: MI estimate in nats. When
        ``return_std=True``, return ``(estimate, standard_error)``.

    Raises:
        ValueError: If the samples, model mask-channel capability, partition,
            integration interval, chunk size, split sizes, or padding mask
            have invalid shapes or values.
    """

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
        normalized = _normalize_columns(context, torch.cat((context, eval_z0)))
        context, eval_z0 = normalized.split((context.shape[0], eval_z0.shape[0]))

    run_device = _model_device(model, device)
    predictor = _CachedModel(
        model,
        context.unsqueeze(0).to(
            device=run_device, dtype=_model_dtype(model, context.dtype)
        ),
        use_cache,
        cache_projections,
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
    keep_x = keep_y = None
    if pad_mask is not None:
        keep = 1.0 - pad_mask.to(device=run_device, dtype=dtype)
        keep_x, keep_y = keep[x_slice], keep[y_slice]

    masks = torch.stack((mask_all, mask_x, mask_y))
    total_rows = eval_z0.shape[0] * n_t_samples
    sum_term = torch.zeros((), device=run_device, dtype=dtype)
    sum_squared_term = torch.zeros((), device=run_device, dtype=dtype)
    flat_zfull = z_full.reshape(-1, dim)
    flat_t = t.reshape(-1, 1)

    for start in range(0, total_rows, chunk):
        stop = min(start + chunk, total_rows)
        rows = torch.arange(start, stop, device=run_device) // n_t_samples
        clean = z0[:, 0][rows]
        full = flat_zfull[start:stop]
        times = flat_t[start:stop]

        if (
            isinstance(predictor.model, _ProjectedModel)
            and predictor.cache is not None
            and fuse_queries
        ):
            term = predictor.model.terms(
                clean, full, times, masks, keep, x_slice, y_slice, *predictor.cache
            )
            sum_term = sum_term + term.sum()
            if return_std:
                sum_squared_term = sum_squared_term + term.square().sum()
            continue

        x_noised = clean.clone()
        x_noised[:, x_slice] = full[:, x_slice]
        y_noised = clean.clone()
        y_noised[:, y_slice] = full[:, y_slice]

        states = (full, x_noised, y_noised)
        field_masks = tuple(mask.expand_as(full) for mask in (mask_all, mask_x, mask_y))
        if fuse_queries:
            velocities = predictor.velocity(
                torch.cat(states), times.repeat(3, 1), torch.cat(field_masks)
            ).to(dtype)
            v_full, v_x_cond, v_y_cond = velocities.chunk(3)
        else:
            v_full, v_x_cond, v_y_cond = (
                predictor.velocity(state, times, mask).to(dtype)
                for state, mask in zip(states, field_masks)
            )
        diff_x = (v_full - v_x_cond)[:, x_slice]
        diff_y = (v_full - v_y_cond)[:, y_slice]
        if keep_x is not None:
            diff_x = diff_x * keep_x
            diff_y = diff_y * keep_y
        gap = diff_x.square().sum(dim=-1) + diff_y.square().sum(dim=-1)
        term = ((1.0 - times.squeeze(-1)) / times.squeeze(-1)) * gap
        sum_term = sum_term + term.sum()
        if return_std:
            sum_squared_term = sum_squared_term + term.square().sum()

    interval = t_max - t_min
    estimate = float(sum_term / total_rows * interval)
    if not return_std:
        return estimate
    variance = (sum_squared_term - sum_term.square() / total_rows) / max(
        total_rows - 1, 1
    )
    standard_error = float(
        variance.clamp_min(0.0).sqrt() / math.sqrt(total_rows) * interval
    )
    return estimate, standard_error


__all__ = ["estimate_mi_fast", "prepare_mi_model"]
