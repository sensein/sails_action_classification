"""
Unit tests for src/sailsprep/action_model_testing/pyskl/scripts/generate_ctrgcn_configs.py

All tests use mocks — no GPU / real checkpoints / real datasets needed.
"""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

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


MODULE_ROOT = Path(__file__).parents[4] / "sailsprep" / "action_model_testing" / "pyskl" / "scripts"


def _load(filename: Path):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGenerateCtrgcnConfigs:
    @pytest.fixture()
    def script(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return _load(MODULE_ROOT / "generate_ctrgcn_configs.py")

    def test_creates_all_feat_files(self, script, tmp_path):
        for ds in ("rmm", "loco"):
            for feat in ("j", "b", "jm", "bm"):
                p = tmp_path / f"configs/custom/ctrgcn_{ds}/{feat}.py"
                assert p.exists(), f"Missing {p}"

    def test_rmm_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_rmm/j.py").read_text()
        assert "num_classes=4" in content

    def test_loco_num_classes(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_loco/j.py").read_text()
        assert "num_classes=5" in content

    def test_backbone_type(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_rmm/j.py").read_text()
        assert "CTRGCN" in content

    def test_feat_name_in_config(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_rmm/bm.py").read_text()
        assert "'bm'" in content

    def test_work_dir_uses_dataset_name(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_loco/j.py").read_text()
        assert "ctrgcn_loco" in content

    def test_train_pipeline_present(self, script, tmp_path):
        content = (tmp_path / "configs/custom/ctrgcn_loco/b.py").read_text()
        assert "train_pipeline" in content
        assert "val_pipeline" in content
        assert "test_pipeline" in content
