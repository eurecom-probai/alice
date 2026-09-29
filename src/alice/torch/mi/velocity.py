# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Masked rectified-flow velocity evaluation for ALICE models."""

from __future__ import annotations

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


__all__ = ["velocity_field_masked"]
