# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

r"""Shared model and Hub utilities for the ALICE PyTorch backend.

The helpers in this module are independent of the mutual-information
reduction itself.  They load Hugging Face checkpoints and resolve the device
and dtype used by model-facing inference helpers.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    device: str = "cpu",
    dtype: Any | None = None,
):
    r"""load_model(model_id, *, revision=None, device="cpu", dtype=None) -> PreTrainedModel

    Load an ALICE checkpoint from a Hugging Face Hub id or a local
    ``save_pretrained`` directory and put it into evaluation mode.

    The checkpoint's custom model implementation is loaded through the
    Transformers AutoClass mechanism with ``trust_remote_code=True``.

    Args:
        model_id (str): Hugging Face model id or local checkpoint directory.
        revision (str | None, optional): Hub branch, tag, or commit to load.
            Default: ``None``.
        device (str, optional): Device receiving the loaded model. Default:
            ``"cpu"``.
        dtype (Any | None, optional): Optional dtype forwarded to
            ``AutoModel.from_pretrained``. Default: ``None``.

    Returns:
        PreTrainedModel: The loaded ALICE model on ``device`` in evaluation
        mode.
    """

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    if dtype is not None:
        kwargs["dtype"] = dtype
    return AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()


def _model_device(model: Any, device: torch.device | str | None) -> torch.device:
    r"""_model_device(model, device) -> torch.device

    Resolve the device on which model forwards should run.

    An explicit ``device`` takes precedence.  Otherwise, use the device of the
    model's first parameter; parameterless test doubles fall back to CPU.

    Args:
        model (torch.nn.Module): Model whose first parameter defines the
            default runtime device.
        device (torch.device | str | None): Explicit device override. When
            ``None``, infer the device from ``model``. Default: ``None``.

    Returns:
        torch.device: The explicit device or the model's parameter device.
    """

    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_dtype(model: Any, fallback: torch.dtype) -> torch.dtype:
    r"""_model_dtype(model, fallback) -> torch.dtype

    Resolve the dtype used for model-facing tensors.

    Args:
        model (torch.nn.Module): Model whose first parameter defines the
            default inference dtype.
        fallback (torch.dtype): Dtype used when ``model`` has no parameters.

    Returns:
        torch.dtype: The first parameter's dtype, or ``fallback`` for a
        parameterless model.
    """

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return fallback


__all__ = ["load_model"]
