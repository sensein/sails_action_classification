"""
Tests for src/sailsprep/action_model_testing/Videomaev2/utils/checkpoint.py

`checkpoint.py` does `from modeling_finetune import vit_base_patch16_224`
inside a function body, so we inject a lightweight stub `modeling_finetune`
module into sys.modules before calling into it, avoiding the real 86M-param
ViT-B and the checkpoint download.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "Videomaev2" / "utils" / "checkpoint.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("videomae2_checkpoint", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod():
    return _load()


class _TinyViT(nn.Module):
    def __init__(self, num_classes=710, **kw):
        super().__init__()
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.head(self.fc_norm(torch.zeros(x.size(0), 768)))


@pytest.fixture(autouse=True)
def stub_modeling_finetune(monkeypatch):
    fake = types.ModuleType("modeling_finetune")
    fake.vit_base_patch16_224 = lambda num_classes=710, **kw: _TinyViT(num_classes)
    monkeypatch.setitem(sys.modules, "modeling_finetune", fake)
    return fake


@pytest.fixture()
def dummy_ckpt(tmp_path):
    p = tmp_path / "vit_b_k710.pth"
    torch.save(_TinyViT().state_dict(), str(p))
    return str(p)


class TestLoadPretrainedVitbK710:
    def test_loads_existing_checkpoint_without_downloading(self, mod, dummy_ckpt):
        with patch("torch.hub.download_url_to_file") as dl:
            model, missing, unexpected = mod.load_pretrained_vitb_k710(dummy_ckpt)
        dl.assert_not_called()
        assert isinstance(model, nn.Module)
        assert missing == [] and unexpected == []

    def test_downloads_when_missing(self, mod, tmp_path):
        ckpt_path = str(tmp_path / "missing" / "vit_b_k710.pth")

        def _fake_download(url, dst):
            torch.save(_TinyViT().state_dict(), dst)

        with patch("torch.hub.download_url_to_file", side_effect=_fake_download) as dl:
            mod.load_pretrained_vitb_k710(ckpt_path)
        dl.assert_called_once()


class TestBuildVideomae2Vitb:
    def test_replaces_classifier_head(self, mod, dummy_ckpt):
        model = mod.build_videomae2_vitb(num_classes=5, ckpt_path=dummy_ckpt, freeze_all_but_last_block=False)
        assert model.head.out_features == 5

    def test_freezes_all_but_last_block_and_head(self, mod, dummy_ckpt):
        model = mod.build_videomae2_vitb(num_classes=5, ckpt_path=dummy_ckpt, freeze_all_but_last_block=True)
        assert model.head.weight.requires_grad
        assert model.fc_norm.weight.requires_grad

    def test_no_freeze_all_trainable(self, mod, dummy_ckpt):
        model = mod.build_videomae2_vitb(num_classes=5, ckpt_path=dummy_ckpt, freeze_all_but_last_block=False)
        assert all(p.requires_grad for p in model.parameters())

    def test_import_error_raises_helpful_message(self, mod, dummy_ckpt, monkeypatch):
        monkeypatch.delitem(sys.modules, "modeling_finetune", raising=False)
        with pytest.raises(ImportError, match="modeling_finetune"):
            mod.build_videomae2_vitb(num_classes=5, ckpt_path=dummy_ckpt)
