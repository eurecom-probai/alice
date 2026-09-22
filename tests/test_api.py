# Required Notice: Copyright 2026 EURECOM (https://www.eurecom.fr/)
#

from types import SimpleNamespace

import alice
import alice.jax
import alice.torch
import alice.torch.utils as utils


def test_root_api_defaults_to_torch_backend():
    assert alice.load_model is alice.torch.load_model
    assert not hasattr(alice.jax, "load_model")


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
    loaded = utils.load_model("org/checkpoint", revision="v1", device="cuda", dtype="float16")

    assert isinstance(loaded, FakeModel)
    assert calls == {
        "model_id": "org/checkpoint",
        "kwargs": {"trust_remote_code": True, "revision": "v1", "dtype": "float16"},
        "device": "cuda",
        "eval": True,
    }
