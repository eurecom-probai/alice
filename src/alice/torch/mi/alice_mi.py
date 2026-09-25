# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

r"""ALICE mutual-information estimator in rectified-flow velocity form.

ALICE estimates ``I(X;Y)`` from the gap between the joint score and the
concatenation of conditional scores, with one block held clean in each
conditional query.  For the rectified-flow interpolant
``z_t = (1 - t) z_0 + t epsilon``, the score--velocity relation is
``s = ((1 - t) v - z_t) / t``.  Because each score comparison uses the same
noised coordinates, the ``-z_t / t`` terms cancel blockwise and the estimator
uses the derived velocity weight ``(1 - t) / t``:

.. math::

    I(X;Y) = \int_0^1 \frac{1-t}{t} \,
        \mathbb{E}\left[\lVert v_{\mathrm{joint}} -
        [v_{X\mid Y\ \mathrm{clean}}, v_{Y\mid X\ \mathrm{clean}}]\rVert^2\right] \, dt.

The model must support partial noising through a per-coordinate mask channel.
The three velocity evaluations share each Monte Carlo row's noise draw, so
the noised block is identical in the joint and matching conditional query.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ..utils import _model_device, _model_dtype


def _predict_velocity(
    model: Any,
    ctx_clean: Tensor,
    qry_z_t: Tensor,
    qry_t: Tensor,
    qry_mask: Tensor,
) -> Tensor:
    r"""_predict_velocity(model, ctx_clean, qry_z_t, qry_t, qry_mask) -> Tensor

    Evaluate the boundary-corrected velocity
    ``v(z_t, t) = z_t + m(z_t, t) - m(z_t, 0)``.

    The model receives concatenated query copies at ``t`` and ``0`` in one
    forward pass.  This keeps the residual input and mask identical for the
    boundary correction and preserves the exact ``v(z, 0) = z`` boundary.

    Args:
        model (torch.nn.Module): ALICE model returning query ``logits``.
        ctx_clean (Tensor): Clean context with shape ``(B, N_ctx, d)``.
        qry_z_t (Tensor): Query state with shape ``(B, N_qry, d)``.
        qry_t (Tensor): Query times with shape ``(B, N_qry, 1)``.
        qry_mask (Tensor): Per-coordinate mask with shape matching
            ``qry_z_t``; one marks noised coordinates and zero marks clean
            coordinates.

    Returns:
        Tensor: Boundary-corrected velocity with shape ``(B, N_qry, d)``.

    Raises:
        RuntimeError: If the model output does not contain ``logits``.
    """

    combined_qry_z = torch.cat([qry_z_t, qry_z_t], dim=1)
    combined_qry_t = torch.cat([qry_t, torch.zeros_like(qry_t)], dim=1)
    combined_mask = torch.cat([qry_mask, qry_mask], dim=1)
    output = model(
        ctx_clean=ctx_clean,
        qry_z_t=combined_qry_z,
        qry_t=combined_qry_t,
        qry_mask=combined_mask,
    )
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("ALICE model did not return logits")
    logits_t, logits_0 = logits.chunk(2, dim=1)
    return qry_z_t + logits_t - logits_0


def velocity_field_masked(
    model: Any,
    context: Tensor,
    *,
    device: torch.device | str | None = None,
    chunk: int = 256,
):
    r"""velocity_field_masked(model, context, *, device=None, chunk=256) -> Callable

    Bind ``model`` and a clean context to a masked velocity callable.

    ``mask`` uses one for coordinates noised at ``t`` and zero for coordinates
    held clean.  The wrapper evaluates flattened query rows in chunks and
    returns results on the query tensor's device and with its dtype.  Chunking
    changes throughput only: query tokens attend to the fixed context and not
    to one another.

    Args:
        model (torch.nn.Module): ALICE model with a mask channel.
        context (Tensor): Clean context samples with shape ``(N_ctx, d)``.
        device (torch.device | str | None, optional): Device for model
            forwards. Defaults to the model parameter device, or CPU for a
            parameterless model. Default: ``None``.
        chunk (int, optional): Maximum number of flattened query rows per
            model call. Default: ``256``.

    Returns:
        Callable: Function ``field(x_t, t, mask)``. ``x_t`` may have any
        shape ``(..., d)``; ``t`` must provide one ``(1,)`` value per query
        row, and ``mask`` must broadcast to ``x_t``. The returned tensor has
        the same shape, device, and dtype as ``x_t``.

    Raises:
        ValueError: If ``context`` is not two-dimensional or ``chunk`` is
            smaller than one. The returned callable raises ``ValueError`` for
            a query width mismatch or a time tensor with the wrong row count.
    """

    if context.ndim != 2:
        raise ValueError(f"context must have shape (N, d); got {tuple(context.shape)}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1; got {chunk}")

    run_device = _model_device(model, device)
    model_dtype = _model_dtype(model, context.dtype)
    ctx = context.unsqueeze(0).to(device=run_device, dtype=model_dtype)
    dim = context.shape[-1]

    @torch.inference_mode()
    def field(x_t: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        shape = x_t.shape
        if x_t.ndim < 2 or x_t.shape[-1] != dim:
            raise ValueError(f"query must have shape (..., {dim}); got {tuple(shape)}")
        flat = x_t.reshape(-1, dim)
        flat_t = t.reshape(-1, 1)
        flat_mask = mask.expand(shape).reshape(-1, dim)
        if flat_t.shape[0] != flat.shape[0]:
            raise ValueError(
                f"t must have one value per query row; got {tuple(t.shape)} for {tuple(shape)}"
            )

        outputs = []
        for start in range(0, flat.shape[0], chunk):
            stop = min(start + chunk, flat.shape[0])
            qz = flat[start:stop].unsqueeze(0).to(device=run_device, dtype=model_dtype)
            qt = (
                flat_t[start:stop].unsqueeze(0).to(device=run_device, dtype=model_dtype)
            )
            qm = (
                flat_mask[start:stop]
                .unsqueeze(0)
                .to(device=run_device, dtype=model_dtype)
            )
            velocity = _predict_velocity(model, ctx, qz, qt, qm)
            outputs.append(velocity.squeeze(0).to(device=flat.device, dtype=flat.dtype))
        return torch.cat(outputs, dim=0).reshape(shape)

    return field


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


def _gaussian_copula_normalize(
    reference: Tensor, values: Tensor | None = None
) -> Tensor:
    r"""_gaussian_copula_normalize(reference, values=None) -> Tensor

    Map values to standard-normal copula scores using the empirical CDF fitted
    independently to each reference column.  This preserves the rank
    structure while putting every coordinate on a common marginal scale.

    Args:
        reference (Tensor): Fitting samples with shape ``(N, d)``. At least
            one row is required.
        values (Tensor | None, optional): Samples with shape ``(M, d)`` to
            transform. When ``None``, transform ``reference`` itself. Default:
            ``None``.

    Returns:
        Tensor: Normal scores with the shape and dtype of ``values``. A
        one-row reference maps every value to zero.

    Raises:
        ValueError: If the input ranks or widths are incompatible, or if the
            reference is empty.
    """

    if values is None:
        values = reference
    if reference.ndim != 2:
        raise ValueError(
            f"reference must have shape (N, d); got {tuple(reference.shape)}"
        )
    if values.ndim != 2 or values.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"values must have shape (M, {reference.shape[-1]}); got {tuple(values.shape)}"
        )
    n, dim = reference.shape
    if n < 1:
        raise ValueError("reference must contain at least one row")
    if n == 1:
        return torch.zeros_like(values)

    # erfinv is most portable in float32/float64; restore the caller's dtype.
    work_dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
    ref = reference.to(work_dtype)
    x = values.to(work_dtype)
    positions = (torch.arange(n, device=ref.device, dtype=work_dtype) + 1.0) / (n + 1.0)
    result = torch.empty_like(x)
    for column in range(dim):
        sorted_ref = torch.sort(ref[:, column]).values
        x_column = x[:, column].contiguous()
        indices = torch.searchsorted(sorted_ref, x_column).clamp(1, n - 1)
        lower = sorted_ref[indices - 1]
        upper = sorted_ref[indices]
        fraction = ((x_column - lower) / (upper - lower).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        result[:, column] = _normal_quantile(
            positions[indices - 1]
            + fraction * (positions[indices] - positions[indices - 1])
        )
    return result.to(values.dtype)


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


def estimate_mi(
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
) -> float | tuple[float, float]:
    r"""estimate_mi(model, joint_samples, x_slice, y_slice, *, ...) -> float | tuple[float, float]

    Estimate ``I(X;Y)`` with ALICE rectified-flow velocity differences.

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
        chunk (int, optional): Query rows processed per model call. Default:
            ``256``.
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

    Returns:
        float | tuple[float, float]: MI estimate in nats. When
        ``return_std=True``, return ``(estimate, standard_error)``.

    Raises:
        ValueError: If the samples, model mask-channel capability, partition,
            integration interval, chunk size, split sizes, or padding mask
            have invalid shapes or values.
    """

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
            "estimate_mi requires a model trained with a mask channel "
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
        eval_z0 = _gaussian_copula_normalize(context, eval_z0)
        context = _gaussian_copula_normalize(context)

    run_device = _model_device(model, device)
    field = velocity_field_masked(model, context, device=run_device, chunk=chunk)
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
    keep_x = keep_y = None
    if pad_mask is not None:
        keep = 1.0 - pad_mask.to(device=run_device, dtype=dtype)
        keep_x, keep_y = keep[x_slice], keep[y_slice]

    total_rows = eval_z0.shape[0] * n_t_samples
    sum_term = torch.zeros((), device=run_device, dtype=dtype)
    sum_squared_term = torch.zeros((), device=run_device, dtype=dtype)
    flat_z0 = z0.expand(-1, n_t_samples, -1).reshape(-1, dim)
    flat_zfull = z_full.reshape(-1, dim)
    flat_t = t.reshape(-1, 1)

    for start in range(0, total_rows, chunk):
        stop = min(start + chunk, total_rows)
        clean = flat_z0[start:stop]
        full = flat_zfull[start:stop]
        times = flat_t[start:stop]

        x_noised = clean.clone()
        x_noised[:, x_slice] = full[:, x_slice]
        y_noised = clean.clone()
        y_noised[:, y_slice] = full[:, y_slice]

        v_full = field(full, times, mask_all)
        v_x_cond = field(x_noised, times, mask_x)
        v_y_cond = field(y_noised, times, mask_y)
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


__all__ = ["estimate_mi"]
