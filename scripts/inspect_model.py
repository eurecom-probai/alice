# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

"""Load an ALICE model with Transformers AutoClass and run one small forward pass."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig

from alice.torch.utils.hub import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Hugging Face model id or local checkpoint directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    model = load_model(args.model_id, revision=args.revision, device=args.device)
    parameters = list(model.parameters())
    n_parameters = sum(parameter.numel() for parameter in parameters)
    n_trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)

    print(f"model_id: {args.model_id}")
    print(f"model_class: {type(model).__name__}")
    print(f"config_class: {type(config).__name__}")
    print(f"parameters: {n_parameters:,} ({n_parameters / 1e6:.2f}M)")
    print(f"trainable_parameters: {n_trainable:,}")
    print(f"dtype: {parameters[0].dtype if parameters else 'none'}")
    print(f"device: {next(model.parameters()).device}")
    print("config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    d_y = int(getattr(config, "d_y", 1))
    kwargs = {
        "ctx_clean": torch.randn(1, 2, d_y, device=args.device),
        "qry_z_t": torch.randn(1, 3, d_y, device=args.device),
        "qry_t": torch.rand(1, 3, 1, device=args.device),
    }
    if getattr(config, "use_mask_channel", False):
        kwargs["qry_mask"] = torch.ones_like(kwargs["qry_z_t"])
    with torch.inference_mode():
        output = model(**kwargs)
    print(f"forward_logits_shape: {tuple(output.logits.shape)}")


if __name__ == "__main__":
    main()
