"""
src/tests/tracking_pose_model_testing/test_rtmlib.py

Smoke test for the RTMLib wholebody video-annotation script.
  Script under test : src/sailsprep/tracking_pose_model_testing/rtmlib.py
  This test file    : src/tests/tracking_pose_model_testing/test_rtmlib.py

This script defines no functions -- it is pure top-level orchestration
(model construction + a hardcoded "/video" folder scan). There is no
business logic to unit test directly, so this file only verifies the module
executes top-to-bottom without raising once its one unusual dependency
(`rtmlib`, not installed here) is stubbed, `os.listdir` is patched to return
no files (so the video loop body never runs), and a couple of expected
top-level constants are present.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_rtmlib.py -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest


@contextlib.contextmanager
def _scoped_modules(stub_map: dict):
    saved = {k: sys.modules.get(k) for k in stub_map}
    sys.modules.update(stub_map)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _make_stub_modules() -> dict:
    rtmlib_mod = types.ModuleType("rtmlib")
    rtmlib_mod.Wholebody = mock.MagicMock(return_value=mock.MagicMock())
    rtmlib_mod.draw_skeleton = mock.MagicMock()
    return {"rtmlib": rtmlib_mod}


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "rtmlib.py"


@pytest.mark.unit
class TestRtmlibSmoke:

    def test_module_loads_without_raising(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("rtmlib_pipeline", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            _scoped_modules(_make_stub_modules()),
            mock.patch("os.listdir", return_value=[]),
            mock.patch("os.makedirs"),
            mock.patch("builtins.print"),
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert mod.device == "cuda"
        assert mod.backend == "onnxruntime"
        assert mod.openpose_skeleton is False
        assert mod.input_folder == "/video"
        assert mod.output_folder == "/video_output"
