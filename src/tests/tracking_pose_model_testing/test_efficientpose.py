"""
src/tests/tracking_pose_model_testing/test_efficientpose.py

Unit tests for the EfficientPose tracking pipeline utilities.
  Script under test : src/sailsprep/tracking_pose_model_testing/efficientpose.py
  This test file    : src/tests/tracking_pose_model_testing/test_efficientpose.py

The script imports `pymediainfo` (not installed) and a bare `from utils import
helpers` (not a real package in this environment), and its top-level code
calls `os.listdir('input')` and `os.makedirs(...)`. These are stubbed so the
module can be exec'd safely and its real functions exercised directly.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_efficientpose.py -v
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: mock.patch.dict(sys.modules, {...}) snapshots/restores the ENTIRE
# sys.modules dict, wiping out any module imported for the first time during
# exec_module. `_scoped_modules` only saves/restores the specific keys we
# stub, so newly-imported real modules (numpy, torch, etc.) are unaffected.
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Stub dependencies unavailable in this environment
# ─────────────────────────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _make_stub_modules() -> dict:
    pymediainfo_mod = _stub("pymediainfo", MediaInfo=mock.MagicMock())

    helpers_mod = _stub(
        "utils.helpers",
        keras_BilinearWeights=mock.MagicMock(),
        Swish=mock.MagicMock(return_value=mock.MagicMock()),
        eswish=mock.MagicMock(),
        swish1=mock.MagicMock(),
        preprocess=mock.MagicMock(),
        extract_coordinates=mock.MagicMock(),
        display_camera=mock.MagicMock(),
        display_body_parts=mock.MagicMock(),
        display_segments=mock.MagicMock(),
    )
    utils_mod = _stub("utils")
    utils_mod.helpers = helpers_mod

    return {
        "pymediainfo": pymediainfo_mod,
        "utils": utils_mod,
        "utils.helpers": helpers_mod,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load the pipeline script
# ─────────────────────────────────────────────────────────────────────────────

def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "efficientpose.py"

_module_cache: types.ModuleType | None = None


def _load_pipeline() -> types.ModuleType:
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not PIPELINE_SCRIPT.exists():
        pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("efficientpose", PIPELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with (
        _scoped_modules(_make_stub_modules()),
        mock.patch("os.makedirs"),
        mock.patch("os.listdir", return_value=[]),
        mock.patch("builtins.print"),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    _module_cache = mod
    return mod


@pytest.fixture(scope="session")
def pipeline() -> types.ModuleType:
    return _load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: perform_tracking — validation paths (no model loading needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPerformTrackingValidation:

    def test_invalid_framework_returns_false(self, pipeline):
        result = pipeline.perform_tracking(
            video=False, file_path="x.jpg", model_name="II_Lite",
            framework_name="not_a_framework", visualize=False, store=False,
        )
        assert result is False

    def test_invalid_model_variant_returns_false(self, pipeline):
        result = pipeline.perform_tracking(
            video=False, file_path="x.jpg", model_name="not_a_model",
            framework_name="tflite", visualize=False, store=False,
        )
        assert result is False

    def test_valid_framework_case_insensitive(self, pipeline):
        # Framework/model checks lower() their inputs before validating.
        result = pipeline.perform_tracking(
            video=False, file_path="x.jpg", model_name="BOGUS",
            framework_name="TFLite", visualize=False, store=False,
        )
        assert result is False  # model_name invalid -> short-circuits before model load


# ─────────────────────────────────────────────────────────────────────────────
# Tests: save — CSV writing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSave:

    def test_video_csv_has_frame_column_and_rows(self, pipeline, tmp_path):
        file_path = tmp_path / "myvideo.mp4"
        coordinates = [
            [("nose", 1.0, 2.0), ("left_eye", 3.0, 4.0)],
            [("nose", 5.0, 6.0), ("left_eye", 7.0, 8.0)],
        ]
        pipeline.save(video=True, file_path=str(file_path), coordinates=coordinates)

        csv_path = tmp_path / "myvideo_coordinates.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            reader = list(csv.DictReader(f))
        assert reader[0]["frame"] == "1"
        assert reader[0]["nose_x"] == "1.0"
        assert reader[1]["frame"] == "2"

    def test_image_csv_has_no_frame_column(self, pipeline, tmp_path):
        file_path = tmp_path / "myimage.png"
        coordinates = [[("nose", 1.0, 2.0)]]
        pipeline.save(video=False, file_path=str(file_path), coordinates=coordinates)

        csv_path = tmp_path / "myimage_coordinates.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            header = f.readline().strip()
        assert "frame" not in header
        assert "nose_x" in header


# ─────────────────────────────────────────────────────────────────────────────
# Tests: infer — framework branches with mocked model objects
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestInfer:

    def test_keras_lite_branch_calls_predict(self, pipeline):
        model = mock.MagicMock()
        model.predict.return_value = np.zeros((1, 10, 10, 3))
        batch = np.zeros((1, 224, 224, 3))
        out = pipeline.infer(batch, model, lite=True, framework="keras")
        model.predict.assert_called_once_with(batch)
        assert out.shape == (1, 10, 10, 3)

    def test_keras_non_lite_branch_takes_last_output(self, pipeline):
        model = mock.MagicMock()
        model.predict.return_value = [np.zeros((1, 5)), np.ones((1, 6))]
        batch = np.zeros((1, 224, 224, 3))
        out = pipeline.infer(batch, model, lite=False, framework="k")
        np.testing.assert_array_equal(out, np.ones((1, 6)))

    def test_tensorflow_lite_branch(self, pipeline):
        model = mock.MagicMock()
        model.get_input_details.return_value = [{"index": 0}]
        model.get_output_details.return_value = [{"index": 1}]
        model.get_tensor.return_value = np.zeros((1, 17, 3))
        batch = np.zeros((1, 224, 224, 3))
        out = pipeline.infer(batch, model, lite=True, framework="tflite")
        model.set_tensor.assert_called_once()
        model.invoke.assert_called_once()
        assert out.shape == (1, 17, 3)

    def test_tensorflow_branch(self, pipeline):
        model = mock.MagicMock()
        model.graph.get_tensor_by_name.return_value = "tensor"
        model.run.return_value = np.zeros((1, 17, 3))
        batch = np.zeros((1, 224, 224, 3))
        out = pipeline.infer(batch, model, lite=False, framework="tf")
        model.run.assert_called_once_with("tensor", {"input_res1:0": batch})
        assert out.shape == (1, 17, 3)

    def test_pytorch_branch(self, pipeline):
        # efficientpose.infer() does `from torch import autograd, from_numpy`
        # *inside* the pytorch branch, i.e. at call time -- so it re-reads
        # whatever is in sys.modules["torch"] when infer() runs, not what was
        # there when this test module was imported. Other test files in this
        # directory (e.g. test_hrnet.py) permanently replace sys.modules
        # ["torch"] with a bare/empty stub without restoring it, so relying
        # on the real torch package being present here is order-dependent.
        # A small self-contained fake torch (scoped via _scoped_modules) is
        # used instead so this test is correct regardless of what other test
        # files have done to sys.modules["torch"].
        class _FakeTensor:
            def __init__(self, arr):
                self._arr = arr

            def float(self):
                return self

            def detach(self):
                return self

            def numpy(self):
                return self._arr

        def _from_numpy(arr):
            return _FakeTensor(arr)

        def _variable(tensor, requires_grad=False):
            return tensor

        fake_torch = _stub(
            "torch",
            from_numpy=_from_numpy,
            autograd=_stub("torch.autograd", Variable=_variable),
        )

        class FakeModel:
            def __call__(self, x):
                return _FakeTensor(np.zeros((1, 3, 8, 8), dtype=np.float32))

        batch = np.zeros((1, 8, 8, 3), dtype=np.float32)
        with _scoped_modules({"torch": fake_torch}):
            out = pipeline.infer(batch, FakeModel(), lite=False, framework="pytorch")
        # rollaxis(1, 4) moves channel axis to the end
        assert out.shape == (1, 8, 8, 3)
