"""
Tests for src/sailsprep/action_model_testing/slow_fast/experiments/ablation/train.py

`train.py` parses `--version` and runs its version-specific config block at
import time (outside any function/`__main__` guard), and calls
`os.makedirs(MODEL_SAVE_DIR, ...)` against a hardcoded absolute path. We
patch `sys.argv` and `os.makedirs` around the import so loading the module
is safe and deterministic in tests.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2  # noqa: F401
import pandas as pd
import pytest
import pytorch_lightning  # noqa: F401
import pytorchvideo.data.encoded_video  # noqa: F401
import torch
import torch.nn as nn

# Warm sys.modules for these heavy/fragile packages before repeatedly
# exec_module-ing train.py below — reimporting them fresh mid-test-run can
# trip rare lazy-submodule bugs in this environment's cv2/torch builds.

_MODULE_PATH = (
    Path(__file__).parents[5]
    / "sailsprep" / "action_model_testing" / "slow_fast" / "experiments" / "ablation" / "train.py"
)


def _stub_pytorchvideo_transforms():
    """
    `pytorchvideo.transforms` is unimportable in this environment: it pulls in
    `torchvision.transforms.functional_tensor`, which newer torchvision
    releases removed. Stub the names train.py actually uses so the import
    succeeds; the stubs are simple passthrough transforms adequate for
    exercising train.py's own logic (which never calls them directly at
    import time — only inside DataLoader pipelines this test doesn't build).
    """
    mod = types.ModuleType("pytorchvideo.transforms")

    class _Identity:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, x):
            return x

    for name in ("ApplyTransformToKey", "RandomShortSideScale", "ShortSideScale", "UniformTemporalSubsample"):
        setattr(mod, name, _Identity)
    return mod


def _load(version: str = "v2", module_alias: str = "ablation_train"):
    spec = importlib.util.spec_from_file_location(module_alias, _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    with (
        patch.object(sys, "argv", ["train.py", "--version", version]),
        patch("os.makedirs"),
        patch.dict(sys.modules, {"pytorchvideo.transforms": _stub_pytorchvideo_transforms()}),
    ):
        spec.loader.exec_module(mod)
    return mod


class TestVersionConfigs:
    def test_v1_baseline_defaults(self):
        mod = _load("v1", "ablation_train_v1")
        assert mod.FREEZE_BACKBONE is True
        assert mod.USE_CLASS_WEIGHTS is False
        assert mod.USE_OVERSAMPLING is False
        assert mod.LEARNING_RATE == pytest.approx(1e-4)

    def test_v2_class_weights(self):
        mod = _load("v2", "ablation_train_v2")
        assert mod.USE_CLASS_WEIGHTS is True
        assert mod.FREEZE_BACKBONE is True

    def test_v3_unfreeze_lower_lr(self):
        mod = _load("v3", "ablation_train_v3")
        assert mod.FREEZE_BACKBONE is False
        assert mod.LEARNING_RATE == pytest.approx(1e-5)

    def test_v4_oversampling(self):
        mod = _load("v4", "ablation_train_v4")
        assert mod.USE_OVERSAMPLING is True

    def test_v8_larger_batch_size(self):
        mod = _load("v8", "ablation_train_v8")
        assert mod.BATCH_SIZE == 8

    def test_v9_smaller_crop(self):
        mod = _load("v9", "ablation_train_v9")
        assert mod.CROP_SIZE == 224
        assert mod.SIDE_SIZE == 224

    def test_v10_slow_r50(self):
        mod = _load("v10", "ablation_train_v10")
        assert mod.MODEL_NAME == "slow_r50"

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown version"):
            _load("v99", "ablation_train_bad")


class TestClipDataModuleSetup:
    def _make_module(self):
        return _load("v2", "ablation_train_setup")

    def _fake_splits(self):
        train = pd.DataFrame({
            "video_path": ["a.mp4", "b.mp4"],
            "class_name": ["walk", "run"],
            "label_encoded": [0, 4],
        })
        val = pd.DataFrame({"video_path": ["c.mp4"], "class_name": ["walk"], "label_encoded": [0]})
        test = pd.DataFrame({"video_path": ["d.mp4"], "class_name": ["walk"], "label_encoded": [0]})
        return train, val, test

    def test_setup_computes_class_weights_when_enabled(self, tmp_path):
        mod = self._make_module()
        mod.MODEL_SAVE_DIR = str(tmp_path)
        dm = mod.ClipDataModule()
        with patch.object(mod, "load_splits_from_csv", return_value=self._fake_splits()):
            dm.setup()
        assert dm.class_weights is not None
        # value_counts() only covers classes actually present in the fixture (walk, run)
        assert dm.class_weights.shape[0] == 2

    def test_setup_no_class_weights_when_disabled(self, tmp_path):
        mod = _load("v1", "ablation_train_no_weights")
        mod.MODEL_SAVE_DIR = str(tmp_path)
        dm = mod.ClipDataModule()
        with patch.object(mod, "load_splits_from_csv", return_value=self._fake_splits()):
            dm.setup()
        assert dm.class_weights is None


class TestSlowFastFineTuneHead:
    def _make_fake_slowfast(self, num_classes_original=400):
        class _FakeHead(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.proj = nn.Linear(in_f, out_f)

            def forward(self, x):
                return self.proj(x.mean([2, 3, 4]))

        class _FakeSlowFast(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Identity() for _ in range(6)] + [_FakeHead(256, num_classes)]
                )

            def forward(self, x):
                return self.blocks[-1].proj(torch.zeros(x[0].shape[0], 256))

        return _FakeSlowFast(num_classes_original)

    def test_head_replaced_for_num_classes(self, tmp_path):
        mod = _load("v2", "ablation_train_head")
        with patch.object(mod.torch.hub, "load", return_value=self._make_fake_slowfast()):
            m = mod.SlowFastFineTune(num_classes=5, freeze_backbone=False)
        assert m.model.blocks[-1].proj.out_features == 5

    def test_freeze_backbone_only_last_block_trainable(self, tmp_path):
        mod = _load("v2", "ablation_train_freeze")
        with patch.object(mod.torch.hub, "load", return_value=self._make_fake_slowfast()):
            m = mod.SlowFastFineTune(num_classes=5, freeze_backbone=True)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert all("blocks.6" in n for n in trainable)

    def test_class_weights_registered_as_buffer(self, tmp_path):
        mod = _load("v2", "ablation_train_weights")
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with patch.object(mod.torch.hub, "load", return_value=self._make_fake_slowfast()):
            m = mod.SlowFastFineTune(num_classes=5, freeze_backbone=False, class_weights=weights)
        assert torch.allclose(m.class_weights, weights)
