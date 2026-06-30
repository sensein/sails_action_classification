"""
src/tests/test_vit_pose.py

Unit tests for the ViTPose pose-estimation pipeline utilities.
  Script under test : src/sailsprep/tracking_pose_model_testing/vit_pose.py
  This test file    : src/tests/test_vit_pose.py

All heavy ML dependencies (torch, transformers, cv2, PIL) are stubbed so the
suite runs on any machine without a GPU or model checkpoints.

Usage:
    poetry run pytest src/tests/test_vit_pose.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stub every heavy dependency BEFORE the script is executed
# ─────────────────────────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# cv2
_stub(
    "cv2",
    COLOR_BGR2RGB=4,
    CAP_PROP_FRAME_WIDTH=3,
    CAP_PROP_FRAME_HEIGHT=4,
    CAP_PROP_FPS=5,
    CAP_PROP_FRAME_COUNT=7,
    VideoCapture=mock.MagicMock(),
    cvtColor=mock.MagicMock(return_value=np.zeros((4, 4, 3), np.uint8)),
)

# PIL
_pil = _stub("PIL")
_pil_image = _stub("PIL.Image")
_pil_image.fromarray = mock.MagicMock(return_value=mock.MagicMock())
_pil.Image = _pil_image

# torch — needs cuda, no_grad, zeros, long
_torch_ctx = mock.MagicMock()
_torch_ctx.__enter__ = lambda s: None
_torch_ctx.__exit__  = mock.MagicMock(return_value=False)

_torch = _stub(
    "torch",
    cuda=types.SimpleNamespace(is_available=lambda: False),
    no_grad=mock.MagicMock(return_value=_torch_ctx),
    zeros=mock.MagicMock(return_value=mock.MagicMock()),
    long=0,
)

# transformers — AutoProcessor + VitPoseForPoseEstimation
_fake_processor = mock.MagicMock()
_fake_processor.from_pretrained = mock.MagicMock(return_value=_fake_processor)
_fake_processor.post_process_pose_estimation.return_value = [[]]

_fake_model = mock.MagicMock()
_fake_model.from_pretrained = mock.MagicMock(return_value=_fake_model)
_fake_model.eval.return_value = None
_fake_model.config.id2label = {0: "nose", 1: "left_eye", 2: "right_eye"}

_stub(
    "transformers",
    AutoProcessor=_fake_processor,
    VitPoseForPoseEstimation=_fake_model,
)

# tqdm
_tqdm_ctx = mock.MagicMock()
_tqdm_ctx.__enter__ = lambda s: s
_tqdm_ctx.__exit__  = mock.MagicMock(return_value=False)
_tqdm_ctx.update    = mock.MagicMock()
_stub("tqdm", tqdm=mock.MagicMock(return_value=_tqdm_ctx))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Import the pipeline script as a module, suppressing all side-effects
# ─────────────────────────────────────────────────────────────────────────────

def _find_src_root(start: Path) -> Path:
    """Walk up from this file until we find the `src` directory."""
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")

_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = (
    _SRC_ROOT
    / "sailsprep"
    / "tracking_pose_model_testing"
    / "vit_pose.py"
)

_module_cache: types.ModuleType | None = None


def _load_pipeline() -> types.ModuleType:
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not PIPELINE_SCRIPT.exists():
        pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("vit_pose", PIPELINE_SCRIPT)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with (
        mock.patch("os.makedirs"),
        mock.patch("sys.argv", ["vit_pose.py"]),
        mock.patch("builtins.print"),
        mock.patch(
            "pandas.read_csv",
            return_value=pd.DataFrame(columns=["video_path", "h5_file_path"]),
        ),
        mock.patch("os.path.exists", return_value=False),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    _module_cache = mod
    return mod


@pytest.fixture(scope="session")
def pipeline() -> types.ModuleType:
    return _load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────────────────────

_STABLE_BBOX = (100, 200, 200, 400)


def _stable_map(n: int = 30, bbox: tuple = _STABLE_BBOX) -> dict:
    return {i: bbox for i in range(n)}


def _store(n: int = 20, noise_frame: int | None = None) -> dict:
    kp_names = ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"]
    s = {
        i: {kp: (100.0 + j, 200.0 + j, 0.9) for j, kp in enumerate(kp_names)}
        for i in range(n)
    }
    if noise_frame is not None:
        s[noise_frame] = {
            kp: (9000.0 + j, 9000.0 + j, 0.9) for j, kp in enumerate(kp_names)
        }
    return s


def _bmap(n: int = 20) -> dict:
    return {i: (50, 50, 250, 450) for i in range(n)}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _kp_features
# ─────────────────────────────────────────────────────────────────────────────

class TestKpFeatures:

    def test_empty_returns_none(self, pipeline):
        assert pipeline._kp_features({}) is None

    def test_centroid_two_points(self, pipeline):
        feat = pipeline._kp_features({"a": (0.0, 0.0, 1.0), "b": (4.0, 2.0, 1.0)})
        np.testing.assert_allclose(feat["centroid"], [2.0, 1.0])

    def test_single_point_spread_is_nan(self, pipeline):
        feat = pipeline._kp_features({"x": (5.0, 3.0, 0.9)})
        assert np.isnan(feat["spread_ar"])

    def test_spread_ar_horizontal(self, pipeline):
        feat = pipeline._kp_features({"a": (0.0, 5.0, 1.0), "b": (10.0, 5.0, 1.0)})
        assert feat["spread_ar"] == pytest.approx(10.0)

    def test_pts_shape(self, pipeline):
        kmap = {str(i): (float(i), float(i * 2), 0.8) for i in range(7)}
        assert pipeline._kp_features(kmap)["pts"].shape == (7, 2)

    def test_centroid_three_symmetric_points(self, pipeline):
        kmap = {"a": (0.0, 0.0, 1.0), "b": (6.0, 0.0, 1.0), "c": (3.0, 6.0, 1.0)}
        feat = pipeline._kp_features(kmap)
        np.testing.assert_allclose(feat["centroid"], [3.0, 2.0], atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: clean_bbox_map
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanBboxMap:

    def test_empty_input(self, pipeline):
        out, *_ = pipeline.clean_bbox_map({})
        assert out == {}

    def test_stable_sequence_unchanged(self, pipeline):
        _, ne, nc, nar, nf, _ = pipeline.clean_bbox_map(_stable_map())
        assert ne == 0 and nc == 0 and nar == 0 and nf == 0

    def test_outlier_x2_corrected(self, pipeline):
        bmap = _stable_map(30)
        bmap[15] = (100, 200, 700, 400)
        cleaned, *_ = pipeline.clean_bbox_map(bmap, n_passes=1)
        assert cleaned[15][2] < 700, "Outlier x2 should be pulled toward median"

    def test_all_output_bboxes_valid(self, pipeline):
        bmap = _stable_map(30)
        bmap[10] = (10, 10, 9, 400)        # deliberately broken x2 < x1
        cleaned, *_ = pipeline.clean_bbox_map(bmap)
        for f, (x1, y1, x2, y2) in cleaned.items():
            assert x2 > x1 and y2 > y1, f"Degenerate bbox at frame {f}"

    def test_single_frame_passthrough(self, pipeline):
        bmap = {0: (10, 20, 110, 220)}
        cleaned, *_ = pipeline.clean_bbox_map(bmap)
        assert cleaned[0] == (10, 20, 110, 220)

    def test_zero_passes_nothing_changed(self, pipeline):
        bmap = _stable_map(30)
        bmap[15] = (100, 200, 999, 400)
        _, ne, nc, nar, nf, _ = pipeline.clean_bbox_map(bmap, n_passes=0)
        assert nf == 0

    def test_output_keys_match_input(self, pipeline):
        bmap = _stable_map(20)
        cleaned, *_ = pipeline.clean_bbox_map(bmap)
        assert set(cleaned.keys()) == set(bmap.keys())

    def test_per_pass_list_length(self, pipeline):
        bmap = _stable_map(30)
        *_, per_pass = pipeline.clean_bbox_map(bmap, n_passes=3)
        assert len(per_pass) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: post_filter_keypoints
# ─────────────────────────────────────────────────────────────────────────────

class TestPostFilterKeypoints:

    def test_empty_input(self, pipeline):
        cleaned, flagged, n = pipeline.post_filter_keypoints({}, {})
        assert cleaned == {} and flagged == set() and n == 0

    def test_all_frames_present_in_output(self, pipeline):
        s = _store(20)
        cleaned, *_ = pipeline.post_filter_keypoints(s, _bmap(20))
        assert set(cleaned.keys()) == set(s.keys())

    def test_stable_store_nothing_flagged(self, pipeline):
        _, flagged, n = pipeline.post_filter_keypoints(_store(20), _bmap(20))
        assert n == 0

    def test_flagged_frames_produce_empty_dicts(self, pipeline):
        s = _store(30, noise_frame=15)
        cleaned, flagged, _ = pipeline.post_filter_keypoints(s, _bmap(30))
        for f in flagged:
            assert cleaned[f] == {}, f"Flagged frame {f} should be empty in output"

    def test_zero_passes_nothing_flagged(self, pipeline):
        s = _store(30, noise_frame=15)
        _, flagged, n = pipeline.post_filter_keypoints(s, _bmap(30), n_passes=0)
        assert n == 0 and len(flagged) == 0

    def test_return_types(self, pipeline):
        cleaned, flagged, n = pipeline.post_filter_keypoints(_store(10), _bmap(10))
        assert isinstance(cleaned, dict)
        assert isinstance(flagged, set)
        assert isinstance(n, int)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_bbox_map  (h5py mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBboxMap:

    def _make_vb1(self, rows: list[tuple]) -> np.ndarray:
        dt = np.dtype([
            ("c0", np.int64), ("c1", np.int64), ("c2", np.int64),
            ("c3", np.int64), ("c4", np.int64), ("c5", np.int64),
        ])
        return np.array(rows, dtype=dt)

    def _patch_h5(self, vb1: np.ndarray):
        import h5py
        table_data = mock.MagicMock()
        table_data.__getitem__ = mock.MagicMock(return_value=vb1)
        dataset = mock.MagicMock()
        dataset.__getitem__ = mock.MagicMock(return_value=table_data)
        file_mock = mock.MagicMock()
        file_mock.__enter__.return_value = file_mock
        file_mock.__exit__ = mock.MagicMock(return_value=False)
        file_mock.__getitem__ = mock.MagicMock(return_value=dataset)
        return mock.patch.object(h5py, "File", return_value=file_mock)

    def test_bbox_values_parsed_correctly(self, pipeline):
        vb1 = self._make_vb1([
            (0, 0, 10, 20, 110, 220),
            (3, 0, 15, 25, 115, 225),
        ])
        with self._patch_h5(vb1):
            result = pipeline.load_bbox_map("dummy.h5")
        assert result[0] == (10, 20, 110, 220)
        assert result[3] == (15, 25, 115, 225)

    def test_empty_table_returns_empty_dict(self, pipeline):
        dt = np.dtype([
            ("c0", np.int64), ("c1", np.int64), ("c2", np.int64),
            ("c3", np.int64), ("c4", np.int64), ("c5", np.int64),
        ])
        with self._patch_h5(np.array([], dtype=dt)):
            result = pipeline.load_bbox_map("dummy.h5")
        assert result == {}

    def test_frame_index_used_as_key(self, pipeline):
        vb1 = self._make_vb1([(7, 0, 50, 60, 150, 160)])
        with self._patch_h5(vb1):
            result = pipeline.load_bbox_map("dummy.h5")
        assert 7 in result and 0 not in result