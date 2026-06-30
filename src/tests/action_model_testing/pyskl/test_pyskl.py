# save as: src/tests/pyskl_test.py
"""
Unit tests for:
  src/sailsprep/action_model_testing/pyskl/extract_logits.py
  src/sailsprep/action_model_testing/pyskl/generate_posec3d_configs.py
  src/sailsprep/action_model_testing/pyskl/generate_stgcnpp_configs.py

All tests use mocks — no GPU / real checkpoints / real datasets needed.
Run: pytest src/tests/pyskl_test.py -v
"""
import importlib
import os
import pickle
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: stub out heavy third-party imports so tests run without mmcv/pyskl
# ---------------------------------------------------------------------------

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
    # Config.fromfile must return something with .data, .model attributes
    cfg = MagicMock()
    stubs["mmcv"].Config.fromfile.return_value = cfg
    return stubs


@pytest.fixture(autouse=True)
def stub_heavy_imports(monkeypatch):
    """Inject stubs for mmcv / pyskl / torch before each test."""
    stubs = _make_stub_modules()
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield stubs


# ---------------------------------------------------------------------------
# Import modules under test (after stubs are in place)
# ---------------------------------------------------------------------------

MODULE_ROOT = Path(__file__).parents[3] / "sailsprep" / "action_model_testing" / "pyskl"


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# generate_stgcnpp_configs tests
# ---------------------------------------------------------------------------

class TestGenerateStgcnppConfigs:
    @pytest.fixture()
    def script(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return _load(MODULE_ROOT / "generate_stgcnpp_configs.py")

    def test_creates_all_feat_files(self, script, tmp_path):
        """Should produce j/b/jm/bm configs for each dataset."""
        for ds in ("rmm", "loco"):
            for feat in ("j", "b", "jm", "bm"):
                p = tmp_path / f"configs/custom/stgcnpp_{ds}/{feat}.py"
                assert p.exists(), f"Missing {p}"

    def test_rmm_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/stgcnpp_rmm/j.py").read_text()
        assert "num_classes=4" in content

    def test_loco_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/stgcnpp_loco/j.py").read_text()
        assert "num_classes=5" in content

    def test_feat_name_in_config(self, script, tmp_path):
        content = (tmp_path / "configs/custom/stgcnpp_rmm/bm.py").read_text()
        assert "'bm'" in content

    def test_work_dir_in_config(self, script, tmp_path):
        content = (tmp_path / "configs/custom/stgcnpp_rmm/j.py").read_text()
        assert "stgcnpp" in content
        assert "rmm" in content

    def test_train_pipeline_present(self, script, tmp_path):
        content = (tmp_path / "configs/custom/stgcnpp_loco/b.py").read_text()
        assert "train_pipeline" in content
        assert "val_pipeline" in content
        assert "test_pipeline" in content


# ---------------------------------------------------------------------------
# generate_posec3d_configs tests
# ---------------------------------------------------------------------------

class TestGeneratePosec3dConfigs:
    @pytest.fixture()
    def script(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return _load(MODULE_ROOT / "generate_posec3d_configs.py")

    def test_creates_joint_files(self, script, tmp_path):
        for ds in ("rmm", "loco"):
            p = tmp_path / f"configs/custom/posec3d_{ds}/joint.py"
            assert p.exists(), f"Missing {p}"

    def test_rmm_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/posec3d_rmm/joint.py").read_text()
        assert "num_classes=4" in content

    def test_loco_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/posec3d_loco/joint.py").read_text()
        assert "num_classes=5" in content

    def test_backbone_type(self, script, tmp_path):
        content = (tmp_path / "configs/custom/posec3d_rmm/joint.py").read_text()
        assert "ResNet3dSlowOnly" in content

    def test_work_dir_uses_dataset_name(self, script, tmp_path):
        content = (tmp_path / "configs/custom/posec3d_loco/joint.py").read_text()
        assert "posec3d_loco" in content

    def test_heatmap_format(self, script, tmp_path):
        content = (tmp_path / "configs/custom/posec3d_rmm/joint.py").read_text()
        assert "NCTHW_Heatmap" in content


# ---------------------------------------------------------------------------
# extract_logits tests
# ---------------------------------------------------------------------------

class TestExtractLogits:
    @pytest.fixture()
    def module(self):
        return _load(MODULE_ROOT / "extract_logits.py")

    # -- extract_logits_from_model ------------------------------------------

    def test_returns_logits_and_labels(self, module, stub_heavy_imports):
        stubs = stub_heavy_imports
        n_samples, n_classes = 8, 4

        # Fake dataloader yields dicts with label tensors
        fake_batch = {"label": MagicMock(item=MagicMock(return_value=2))}
        stubs["mmcv"].Config.fromfile.return_value.data.test = MagicMock()
        stubs["pyskl.datasets"].build_dataset.return_value = MagicMock()
        stubs["pyskl.datasets"].build_dataloader.return_value = [fake_batch] * n_samples

        fake_model = MagicMock()
        fake_model.return_value = np.zeros((1, n_classes))  # model forward
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

        # Provide one fake batch so np.vstack doesn't receive an empty list
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

        # Patch glob to return a fake checkpoint
        fake_ckpt = str(tmp_path / "best.pth")
        Path(fake_ckpt).touch()

        def side_effect(*a, **kw):
            # Consistent labels across calls so main() doesn't hit the
            # "Label mismatch" assertion — this test is for the success path.
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