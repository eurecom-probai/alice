# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#


import pytest

import alice
import alice.jax
import alice.torch
import alice.torch.mi
from alice.torch import utils
from alice.torch.mi.alice_mi import estimate_mi, velocity_field_masked
from alice.torch.mi.alice_mi_fast import estimate_mi_fast, prepare_mi_model


def test_root_api_defaults_to_torch_backend():
    assert alice.load_model is alice.torch.load_model
    assert not hasattr(alice.jax, "load_model")


@pytest.mark.parametrize(
    "name, implementation",
    [
        ("estimate_mi", estimate_mi),
        ("estimate_mi_fast", estimate_mi_fast),
        ("prepare_mi_model", prepare_mi_model),
        ("velocity_field_masked", velocity_field_masked),
    ],
)
def test_mi_api_exports_match_implementations(name, implementation):
    for namespace in (alice, alice.torch, alice.torch.mi):
        assert getattr(namespace, name) is implementation
        assert name in namespace.__all__


def test_load_model_uses_autoclass_and_configures_runtime(monkeypatch):
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    def from_pretrained(model_id, **kwargs):
        calls["model_id"] = model_id
        calls["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(utils.AutoModel, "from_pretrained", from_pretrained)
    loaded = utils.load_model(
        "org/checkpoint", revision="v1", device="cuda", dtype="float16"
    )

    assert isinstance(loaded, FakeModel)
    assert calls == {
        "model_id": "org/checkpoint",
        "kwargs": {"trust_remote_code": True, "revision": "v1", "dtype": "float16"},
        "device": "cuda",
        "eval": True,
    }
