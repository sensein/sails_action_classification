"""
Unit tests for src/sailsprep/action_model_testing/pyskl/fusion/extract_logits.py

All tests use mocks — no GPU / real checkpoints / real datasets needed.
"""
import importlib
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_stub_modules():
    stubs = {
        "mmcv": MagicMock(),
        "mmcv.runner": MagicMock(),
        "mmcv.parallel": MagicMock(),
        "pyskl": MagicMock(),
        "pyskl.datasets": MagicMock(),
        "pyskl.models": MagicMock(),
        "torch": MagicMock(),
    }
    cfg = MagicMock()
    stubs["mmcv"].Config.fromfile.return_value = cfg
    return stubs


@pytest.fixture(autouse=True)
def stub_heavy_imports(monkeypatch):
    stubs = _make_stub_modules()
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield stubs


MODULE_ROOT = Path(__file__).parents[4] / "sailsprep" / "action_model_testing" / "pyskl" / "fusion"


def _load(filename: Path):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestExtractLogits:
    @pytest.fixture()
    def module(self):
        return _load(MODULE_ROOT / "extract_logits.py")

    # -- extract_logits_from_model ------------------------------------------

    def test_returns_logits_and_labels(self, module, stub_heavy_imports):
        stubs = stub_heavy_imports
        n_samples, n_classes = 8, 4

        fake_batch = {"label": MagicMock(item=MagicMock(return_value=2))}
        stubs["mmcv"].Config.fromfile.return_value.data.test = MagicMock()
        stubs["pyskl.datasets"].build_dataset.return_value = MagicMock()
        stubs["pyskl.datasets"].build_dataloader.return_value = [fake_batch] * n_samples

        fake_model = MagicMock()
        fake_model.return_value = np.zeros((1, n_classes))
        stubs["pyskl.models"].build_model.return_value = fake_model
        stubs["mmcv.runner"].load_checkpoint = MagicMock()
        stubs["mmcv.parallel"].MMDataParallel.return_value = fake_model

        logits, labels = module.extract_logits_from_model("fake.py", "fake.pth", split="test")

        assert logits.shape == (n_samples, n_classes)
        assert labels.shape == (n_samples,)
        assert all(l == 2 for l in labels)

    def test_val_split_uses_val_dataset(self, module, stub_heavy_imports):
        stubs = stub_heavy_imports
        cfg = stubs["mmcv"].Config.fromfile.return_value

        fake_batch = {"label": MagicMock(item=MagicMock(return_value=0))}
        fake_model = MagicMock(return_value=np.zeros((1, 4)))
        stubs["pyskl.datasets"].build_dataloader.return_value = [fake_batch]
        stubs["pyskl.models"].build_model.return_value = fake_model
        stubs["mmcv.parallel"].MMDataParallel.return_value = fake_model

        module.extract_logits_from_model("fake.py", "fake.pth", split="val")
        stubs["pyskl.datasets"].build_dataset.assert_called_once_with(cfg.data.val)

    def test_invalid_split_raises(self, module, stub_heavy_imports):
        with pytest.raises(ValueError, match="Unknown split"):
            module.extract_logits_from_model("fake.py", "fake.pth", split="train")

    # -- main / logit saving -----------------------------------------------

    def test_main_saves_pkl(self, module, stub_heavy_imports, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "work_dirs").mkdir()

        fake_ckpt = str(tmp_path / "best.pth")
        Path(fake_ckpt).touch()

        def side_effect(*a, **kw):
            labels = np.array([0, 1])
            return np.zeros((2, 4)), labels

        with patch("glob.glob", return_value=[fake_ckpt]), \
            patch.object(module, "extract_logits_from_model", side_effect=side_effect), \
            patch("sys.argv", ["extract_logits.py", "--dataset", "rmm"]):
            module.main()

        out = tmp_path / "work_dirs" / "fusion_rmm_test_logits.pkl"
        assert out.exists()
        data = pickle.loads(out.read_bytes())
        assert "logits" in data and "labels" in data

    def test_main_skips_missing_checkpoints(self, module, stub_heavy_imports, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "work_dirs").mkdir()

        with patch("glob.glob", return_value=[]), \
             patch("sys.argv", ["extract_logits.py", "--dataset", "rmm"]):
            module.main()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_label_mismatch_raises(self, module, stub_heavy_imports, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "work_dirs").mkdir()

        fake_ckpt = str(tmp_path / "best.pth")
        Path(fake_ckpt).touch()

        call_count = {"n": 0}
        def side_effect(*a, **kw):
            call_count["n"] += 1
            labels = np.array([0, 1]) if call_count["n"] == 1 else np.array([1, 0])
            return np.zeros((2, 4)), labels

        with patch("glob.glob", return_value=[fake_ckpt]), \
            patch.object(module, "extract_logits_from_model", side_effect=side_effect), \
            patch("sys.argv", ["extract_logits.py", "--dataset", "rmm"]), \
            pytest.raises(AssertionError, match="Label mismatch"):
            module.main()
