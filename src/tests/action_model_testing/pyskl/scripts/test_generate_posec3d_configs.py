"""
Unit tests for src/sailsprep/action_model_testing/pyskl/scripts/generate_posec3d_configs.py

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
