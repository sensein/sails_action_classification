"""
src/tests/tracking_pose_model_testing/test_sam2_yolov8.py

Unit tests for the SAM2 + YOLOv8 video-segmentation pipeline utilities.
  Script under test : src/sailsprep/tracking_pose_model_testing/sam2_yolov8.py
  This test file    : src/tests/tracking_pose_model_testing/test_sam2_yolov8.py

The script builds a real SAM2 video predictor and a YOLO model, then
immediately calls `process_all_videos("/video", "/output_videos", predictor)`
at import time. `sam2.build_sam.build_sam2_video_predictor` and
`ultralytics.YOLO` are stubbed (scoped via `mock.patch.dict`) so this heavy
top-level code executes harmlessly, and `os.listdir` is patched to return an
empty list so the hardcoded "/video" folder scan finds nothing.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_sam2_yolov8.py -v
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import PIL  # noqa: F401
import pytest

matplotlib.use("Agg")


def _heal_real_module(name: str, required_attr: str) -> None:
    """
    Some other test files in this directory replace sys.modules[name] with a
    bare/incomplete stub without restoring it (e.g. test_hrnet.py permanently
    stubs cv2 with only a handful of attributes). If that happened before
    this file's tests run, drop the stub and re-import the real package so
    functions like cv2.VideoWriter_fourcc are available here.
    """
    mod = sys.modules.get(name)
    if mod is None or not hasattr(mod, required_attr):
        sys.modules.pop(name, None)
        globals()[name] = importlib.import_module(name)


_heal_real_module("cv2", "VideoWriter_fourcc")

# ─────────────────────────────────────────────────────────────────────────────
# Load the pipeline script with heavy top-level calls stubbed
#
# NOTE: mock.patch.dict(sys.modules, {...}) snapshots and restores the ENTIRE
# sys.modules dict, which wipes out any module imported for the FIRST time
# during exec_module (e.g. matplotlib/torch/PIL, if not already imported
# elsewhere). That leaves a stale reference inside the exec'd module's
# namespace pointing at a now-orphaned module object, while later `import`
# statements create a *second*, distinct module object -- breaking
# isinstance() checks (e.g. matplotlib Patch subclasses) and, for C
# extensions like numpy, raising "cannot load module more than once per
# process". `_scoped_modules` only saves/restores the specific keys we stub.
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


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "sam2_yolov8.py"

_module_cache: types.ModuleType | None = None


def _make_stub_modules() -> dict:
    sam2_build_sam = types.ModuleType("sam2.build_sam")
    sam2_build_sam.build_sam2_video_predictor = mock.MagicMock(return_value=mock.MagicMock())

    ultralytics_mod = types.ModuleType("ultralytics")
    ultralytics_mod.YOLO = mock.MagicMock(return_value=mock.MagicMock())

    # sam2_yolov8.py only needs cuda/mps availability checks and
    # torch.device(...) at import time (the cuda-only branch below that is
    # never reached since is_available() is False). Some other test files in
    # this directory (e.g. test_hrnet.py) permanently replace
    # sys.modules["torch"] with a bare stub without restoring it, so a
    # minimal scoped torch stub (restored afterwards via `_scoped_modules`)
    # keeps this test correct regardless of what other test files have done
    # to sys.modules["torch"].
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = mock.MagicMock(is_available=mock.MagicMock(return_value=False))
    torch_mod.backends = mock.MagicMock(mps=mock.MagicMock(is_available=mock.MagicMock(return_value=False)))
    torch_mod.device = mock.MagicMock(side_effect=lambda x: types.SimpleNamespace(type=x))

    return {
        "torch": torch_mod,
        "sam2.build_sam": sam2_build_sam,
        "ultralytics": ultralytics_mod,
    }


def _load_pipeline() -> types.ModuleType:
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not PIPELINE_SCRIPT.exists():
        pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("sam2_yolov8", PIPELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    # Re-heal immediately before exec: this fixture (and thus exec_module)
    # runs lazily at TEST-EXECUTION time, which is after collection has
    # finished for every test file in the session. If a file collected
    # after this one also permanently pollutes sys.modules["cv2"], the
    # collection-time heal above is no longer enough -- the pipeline module
    # would bind its own `cv2` name to the polluted stub at exec_module time.
    _heal_real_module("cv2", "VideoWriter_fourcc")

    with (
        _scoped_modules(_make_stub_modules()),
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_video(path: Path, n_frames: int = 4, size=(32, 32)) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: extract_frames_to_jpegs
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractFramesToJpegs:

    def test_frame_count_and_files(self, pipeline, tmp_path):
        video_path = make_dummy_video(tmp_path / "video.mp4", n_frames=5)
        out_dir = tmp_path / "frames"
        count = pipeline.extract_frames_to_jpegs(str(video_path), str(out_dir))
        assert count == 5
        jpgs = sorted(out_dir.glob("*.jpg"))
        assert len(jpgs) == 5
        assert jpgs[0].name == "00000.jpg"

    def test_creates_output_dir(self, pipeline, tmp_path):
        video_path = make_dummy_video(tmp_path / "video2.mp4", n_frames=1)
        out_dir = tmp_path / "does" / "not" / "exist"
        pipeline.extract_frames_to_jpegs(str(video_path), str(out_dir))
        assert out_dir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_frame_paths
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetFramePaths:

    def test_sorted_jpg_only(self, pipeline, tmp_path):
        (tmp_path / "00002.jpg").write_bytes(b"x")
        (tmp_path / "00000.jpg").write_bytes(b"x")
        (tmp_path / "00001.jpg").write_bytes(b"x")
        (tmp_path / "ignore.txt").write_bytes(b"x")

        paths = pipeline.get_frame_paths(str(tmp_path))
        names = [Path(p).name for p in paths]
        assert names == ["00000.jpg", "00001.jpg", "00002.jpg"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: run_yolo_on_frame
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRunYoloOnFrame:

    def test_extracts_boxes(self, pipeline):
        det1 = mock.MagicMock()
        det1.xyxy = [mock.MagicMock()]
        det1.xyxy[0].cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0, 4.0])

        det2 = mock.MagicMock()
        det2.xyxy = [mock.MagicMock()]
        det2.xyxy[0].cpu.return_value.numpy.return_value = np.array([5.0, 6.0, 7.0, 8.0])

        result = mock.MagicMock()
        result.boxes = [det1, det2]
        yolo_model = mock.MagicMock(return_value=[result])

        boxes = pipeline.run_yolo_on_frame(np.zeros((10, 10, 3)), yolo_model)

        assert len(boxes) == 2
        np.testing.assert_allclose(boxes[0], [1.0, 2.0, 3.0, 4.0])
        assert boxes[0].dtype == np.float32

    def test_no_detections(self, pipeline):
        result = mock.MagicMock()
        result.boxes = []
        yolo_model = mock.MagicMock(return_value=[result])
        boxes = pipeline.run_yolo_on_frame(np.zeros((10, 10, 3)), yolo_model)
        assert boxes == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests: show_mask / show_points / show_box (matplotlib drawing helpers)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestVisualizationHelpers:

    def test_show_mask_runs(self, pipeline):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        pipeline.show_mask(mask, ax, obj_id=1)
        plt.close(fig)

    def test_show_mask_random_color(self, pipeline):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        mask = np.ones((5, 5))
        pipeline.show_mask(mask, ax, random_color=True)
        plt.close(fig)

    def test_show_points_runs(self, pipeline):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        coords = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([1, 0, 1])
        pipeline.show_points(coords, labels, ax)
        plt.close(fig)

    def test_show_box_runs(self, pipeline):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        pipeline.show_box([1, 2, 10, 20], ax)
        plt.close(fig)
