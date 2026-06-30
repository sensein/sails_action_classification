"""
src/tests/test_hrnet.py

Unit tests for the HRNet wholebody pose-estimation pipeline utilities.
  Script under test : src/sailsprep/tracking_pose_model_testing/hrnet.py
  This test file    : src/tests/test_hrnet.py

All heavy ML dependencies (mmpose, mmdet, torch, cv2) are stubbed so the
suite runs on any machine without a GPU or model checkpoints.

Usage:
    poetry run pytest src/tests/test_hrnet.py -v
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
#     (must happen at module level so they are in sys.modules on first import)
# ─────────────────────────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> types.ModuleType:
    """Register a minimal fake module in sys.modules (no-op if already real)."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod          # intentionally overwrites so ML stubs win
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

# mmcv / mmengine
_stub("mmcv", imread=mock.MagicMock())
_stub("mmengine")
_stub("mmengine.registry", init_default_scope=mock.MagicMock())

# Build a realistic fake pose_estimator so attribute assignments in the script
# (pose_estimator.cfg.visualizer.radius = 3) don't raise.
_fake_vis = mock.MagicMock(radius=3, line_width=2)
_fake_cfg = mock.MagicMock(visualizer=_fake_vis)
_fake_cfg.get.return_value = None
_fake_pose_est = mock.MagicMock(cfg=_fake_cfg)

_stub("mmpose")
_stub(
    "mmpose.apis",
    inference_topdown=mock.MagicMock(return_value=[]),
    init_model=mock.MagicMock(return_value=_fake_pose_est),
)
_stub("mmpose.evaluation")
_stub("mmpose.evaluation.functional", nms=mock.MagicMock())
_stub("mmpose.structures", merge_data_samples=mock.MagicMock())

# mmdet
_stub("mmdet")
_stub(
    "mmdet.apis",
    inference_detector=mock.MagicMock(),
    init_detector=mock.MagicMock(return_value=mock.MagicMock()),
)

# torch
_stub("torch")

# tqdm — stub to suppress progress bars in tests
_tqdm_ctx = mock.MagicMock()
_tqdm_ctx.__enter__ = lambda s: s
_tqdm_ctx.__exit__ = mock.MagicMock(return_value=False)
_tqdm_ctx.update = mock.MagicMock()
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
    / "hrnet.py"
)

_module_cache: types.ModuleType | None = None


def _load_pipeline() -> types.ModuleType:
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not PIPELINE_SCRIPT.exists():
        pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("hrnet", PIPELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with (
        mock.patch("os.chdir"),
        mock.patch("os.makedirs"),
        mock.patch("sys.argv", ["hrnet.py"]),
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
    """Session-scoped fixture: import the pipeline script exactly once."""
    return _load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories shared by multiple test classes
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(kps: np.ndarray, scores: np.ndarray) -> mock.MagicMock:
    """Minimal mock mirroring mmpose PoseDataSample used by the validator."""
    r = mock.MagicMock()
    r.pred_instances.keypoints = kps[np.newaxis]        # (1, 133, 2)
    r.pred_instances.keypoint_scores = scores[np.newaxis]  # (1, 133)
    return r


def _blank(n_kp: int = 133, score: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Return zeroed keypoints + flat score array."""
    return np.zeros((n_kp, 2), dtype=float), np.full(n_kp, score, dtype=float)


def _standing() -> tuple[np.ndarray, np.ndarray]:
    """
    Anatomically plausible standing-person keypoints for a 200×500 bbox.
    Enough landmarks are placed so every anatomical check in the validator
    has valid reference points.
    """
    kps, scores = _blank()
    kps[0]  = [100,  40]    # nose
    kps[5]  = [ 80, 130]    # L shoulder
    kps[6]  = [120, 130]    # R shoulder
    kps[9]  = [ 70, 280]    # L wrist
    kps[10] = [130, 280]    # R wrist
    kps[11] = [ 85, 260]    # L hip
    kps[12] = [115, 260]    # R hip
    kps[13] = [ 80, 350]    # L knee
    kps[14] = [120, 350]    # R knee
    kps[15] = [ 75, 460]    # L ankle
    kps[16] = [125, 460]    # R ankle
    return kps, scores


_BBOX = np.array([[0, 0, 200, 500]], dtype=float)
_STABLE_BBOX = (100, 200, 200, 400)


def _stable_map(n: int = 30, bbox: tuple = _STABLE_BBOX) -> dict:
    return {i: bbox for i in range(n)}


def _store(n: int = 20, noise_frame: int | None = None) -> dict:
    s = {
        i: {f"kp_{k:03d}": (100.0 + k, 200.0 + k, 0.9) for k in range(17)}
        for i in range(n)
    }
    if noise_frame is not None:
        s[noise_frame] = {
            f"kp_{k:03d}": (9000.0 + k, 9000.0 + k, 0.9) for k in range(17)
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

    def test_spread_ar_horizontal_line(self, pipeline):
        # kw = 10, kh = 0  →  spread_ar = 10 / max(0, 1.0) = 10
        feat = pipeline._kp_features({"a": (0.0, 5.0, 1.0), "b": (10.0, 5.0, 1.0)})
        assert feat["spread_ar"] == pytest.approx(10.0)

    def test_pts_shape(self, pipeline):
        kmap = {str(i): (float(i), float(i * 2), 0.8) for i in range(7)}
        assert pipeline._kp_features(kmap)["pts"].shape == (7, 2)

    def test_centroid_three_symmetric_points(self, pipeline):
        # centroid of (0,0), (6,0), (3,6) = (3, 2)
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
        bmap[15] = (100, 200, 700, 400)   # x2 far too large
        cleaned, *_ = pipeline.clean_bbox_map(bmap, n_passes=1)
        assert cleaned[15][2] < 700, "Outlier x2 should be pulled toward median"

    def test_all_output_bboxes_valid(self, pipeline):
        bmap = _stable_map(30)
        bmap[10] = (10, 10, 9, 400)       # deliberately broken x2 < x1
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
        # Early exit on nf==0 is allowed, so length is ≤ n_passes
        assert len(per_pass) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_pose_predictions_133
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatePosePredictions133:

    def test_empty_inputs_return_empty(self, pipeline):
        assert pipeline.validate_pose_predictions_133([], np.empty((0, 4))) == []

    def test_133_kp_structure_preserved(self, pipeline):
        kps, scores = _standing()
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert len(out) == 1
        assert out[0].pred_instances.keypoint_scores.shape == (1, 133)

    def test_ankle_above_shoulder_zeroed(self, pipeline):
        kps, scores = _standing()
        kps[15] = [80, 40]    # L ankle placed near nose — anatomically impossible
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert out[0].pred_instances.keypoint_scores[0, 15] == 0.0, (
            "L ankle above shoulder should be zeroed"
        )

    def test_ankle_above_knee_zeroed(self, pipeline):
        kps, scores = _standing()
        kps[15] = [75, 280]   # L ankle above L knee (y=350)
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert out[0].pred_instances.keypoint_scores[0, 15] == 0.0, (
            "Ankle above its knee should be zeroed"
        )

    def test_face_kp_below_hips_zeroed(self, pipeline):
        kps, scores = _standing()
        kps[30] = [100, 490]  # face kp far below hip level (hip y≈260)
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert out[0].pred_instances.keypoint_scores[0, 30] == 0.0

    def test_hand_kp_far_from_wrist_zeroed(self, pipeline):
        kps, scores = _standing()
        kps[95] = [950, 950]  # L hand kp far from L wrist at (70, 280)
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert out[0].pred_instances.keypoint_scores[0, 95] == 0.0

    def test_low_wrist_score_zeroes_all_left_hand_kps(self, pipeline):
        kps, scores = _standing()
        scores[9] = 0.1       # L wrist below 0.3 threshold
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        hand = out[0].pred_instances.keypoint_scores[0, 91:112]
        assert (hand == 0.0).all(), (
            "All L hand kps should be zeroed when wrist confidence is too low"
        )

    def test_low_right_wrist_zeroes_right_hand(self, pipeline):
        kps, scores = _standing()
        scores[10] = 0.1      # R wrist low
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        rhand = out[0].pred_instances.keypoint_scores[0, 112:133]
        assert (rhand == 0.0).all()

    def test_valid_nose_score_retained(self, pipeline):
        """Nose is above any leg/face/hand check — its score must survive."""
        kps, scores = _standing()
        out = pipeline.validate_pose_predictions_133([_make_result(kps, scores)], _BBOX)
        assert out[0].pred_instances.keypoint_scores[0, 0] > 0

    def test_multiple_persons_each_validated(self, pipeline):
        kps1, scores1 = _standing()
        kps2, scores2 = _standing()
        results = [_make_result(kps1, scores1), _make_result(kps2, scores2)]
        bboxes  = np.tile(_BBOX, (2, 1))
        out = pipeline.validate_pose_predictions_133(results, bboxes)
        assert len(out) == 2


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
# Tests: load_bbox_map  (h5py mocked — no real file needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBboxMap:
    """
    The script reads a pandas HDFStore layout:
        f["bboxes/table"][()]  →  structured array with field "values_block_1"
        values_block_1 rows   →  (frame_idx, pad, x1, y1, x2, y2)
    """

    def _make_vb1(self, rows: list[tuple]) -> np.ndarray:
        dt = np.dtype(
            [("c0", np.int64), ("c1", np.int64), ("c2", np.int64),
             ("c3", np.int64), ("c4", np.int64), ("c5", np.int64)]
        )
        return np.array(rows, dtype=dt)

    def _patch_h5(self, vb1: np.ndarray):
        """Build a mock h5py.File context manager for the given vb1 array."""
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
        import h5py  # noqa: F401  (import triggers stub if not real)

        vb1 = self._make_vb1([
            (0, 0, 10, 20, 110, 220),
            (3, 0, 15, 25, 115, 225),
        ])
        with self._patch_h5(vb1):
            result = pipeline.load_bbox_map("dummy.h5")

        assert result[0] == (10, 20, 110, 220)
        assert result[3] == (15, 25, 115, 225)

    def test_empty_table_returns_empty_dict(self, pipeline):
        import h5py  # noqa: F401

        dt = np.dtype([("c0", np.int64), ("c1", np.int64), ("c2", np.int64),
                       ("c3", np.int64), ("c4", np.int64), ("c5", np.int64)])
        empty = np.array([], dtype=dt)

        with self._patch_h5(empty):
            result = pipeline.load_bbox_map("dummy.h5")

        assert result == {}

    def test_frame_index_used_as_key(self, pipeline):
        import h5py  # noqa: F401

        vb1 = self._make_vb1([(7, 0, 50, 60, 150, 160)])
        with self._patch_h5(vb1):
            result = pipeline.load_bbox_map("dummy.h5")

        assert 7 in result
        assert 0 not in result