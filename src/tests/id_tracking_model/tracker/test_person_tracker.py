"""Tests for sailsprep.id_tracking_model.tracker.person_tracker."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sailsprep.id_tracking_model.tracker.person_tracker import (
    CameraMotionCompensator,
    PersonTracker,
    TrackerConfig,
    calculate_center_distance_similarity,
    calculate_combined_similarity,
    calculate_iou,
    calculate_scene_crowding,
    get_adaptive_thresholds,
    is_spatially_plausible,
    predict_motion_with_camera_compensation,
    update_kalman_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random BGR frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_kalman(bbox: list[float] | None = None):
    """Return a real KalmanFilter initialised to bbox (skips if filterpy absent)."""
    pytest.importorskip("filterpy", reason="filterpy not installed")
    from sailsprep.id_tracking_model.tracker.person_tracker import create_kalman_filter

    return create_kalman_filter(bbox or [10.0, 20.0, 110.0, 120.0])


# ---------------------------------------------------------------------------
# TrackerConfig
# ---------------------------------------------------------------------------


class TestTrackerConfig:
    def test_defaults(self):
        cfg = TrackerConfig()
        assert cfg.base_iou_threshold == pytest.approx(0.20)
        assert cfg.base_motion_confidence == pytest.approx(0.25)
        assert cfg.base_center_weight == pytest.approx(0.75)
        assert cfg.max_lost_frames == 150
        assert cfg.confidence_decay_rate == pytest.approx(0.06)
        assert cfg.max_jump_factor == pytest.approx(2.5)

    def test_frozen(self):
        cfg = TrackerConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.max_lost_frames = 999  # type: ignore[misc]

    def test_custom_values(self):
        cfg = TrackerConfig(base_iou_threshold=0.5, max_lost_frames=50)
        assert cfg.base_iou_threshold == pytest.approx(0.5)
        assert cfg.max_lost_frames == 50


# ---------------------------------------------------------------------------
# calculate_iou
# ---------------------------------------------------------------------------


class TestCalculateIou:
    def test_identical_boxes(self):
        box = [0.0, 0.0, 10.0, 10.0]
        assert calculate_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert calculate_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # boxes share a 5×5 region; total union = 75
        iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0

    def test_touching_edge_returns_zero(self):
        # boxes touch at an edge — no area intersection
        assert calculate_iou([0, 0, 5, 5], [5, 0, 10, 5]) == pytest.approx(0.0)

    def test_one_inside_other(self):
        iou = calculate_iou([0, 0, 20, 20], [5, 5, 15, 15])
        # inner area = 100, outer = 400, union = 400, iou = 0.25
        assert iou == pytest.approx(0.25)

    def test_numpy_array_input(self):
        b1 = np.array([0.0, 0.0, 10.0, 10.0])
        b2 = np.array([0.0, 0.0, 10.0, 10.0])
        assert calculate_iou(b1, b2) == pytest.approx(1.0)

    def test_returns_float(self):
        result = calculate_iou([0, 0, 5, 5], [0, 0, 5, 5])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# calculate_center_distance_similarity
# ---------------------------------------------------------------------------


class TestCalculateCenterDistanceSimilarity:
    def test_identical_boxes_return_one(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert calculate_center_distance_similarity(box, box) == pytest.approx(1.0)

    def test_far_apart_boxes_return_zero_or_less(self):
        sim = calculate_center_distance_similarity([0, 0, 10, 10], [1000, 1000, 1010, 1010])
        assert sim == pytest.approx(0.0)

    def test_returns_float(self):
        result = calculate_center_distance_similarity([0, 0, 10, 10], [5, 5, 15, 15])
        assert isinstance(result, float)

    def test_range_zero_to_one(self):
        sim = calculate_center_distance_similarity([0, 0, 50, 50], [20, 20, 70, 70])
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# calculate_combined_similarity
# ---------------------------------------------------------------------------


class TestCalculateCombinedSimilarity:
    def test_identical_boxes(self):
        box = [0.0, 0.0, 100.0, 100.0]
        sim = calculate_combined_similarity(box, box, center_weight=0.5)
        assert sim == pytest.approx(1.0)

    def test_weight_zero_equals_iou(self):
        b1, b2 = [0, 0, 10, 10], [5, 5, 15, 15]
        assert calculate_combined_similarity(b1, b2, 0.0) == pytest.approx(
            calculate_iou(b1, b2)
        )

    def test_weight_one_equals_center_sim(self):
        b1, b2 = [0, 0, 10, 10], [5, 5, 15, 15]
        assert calculate_combined_similarity(b1, b2, 1.0) == pytest.approx(
            calculate_center_distance_similarity(b1, b2)
        )

    def test_returns_float(self):
        result = calculate_combined_similarity([0, 0, 5, 5], [0, 0, 5, 5], 0.5)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# calculate_scene_crowding
# ---------------------------------------------------------------------------


class TestCalculateSceneCrowding:
    def test_empty_returns_zero(self):
        assert calculate_scene_crowding([]) == pytest.approx(0.0)

    def test_single_box_returns_zero(self):
        assert calculate_scene_crowding([[0, 0, 10, 10]]) == pytest.approx(0.0)

    def test_very_close_boxes_return_one(self):
        # two boxes almost on top of each other
        result = calculate_scene_crowding([[0, 0, 10, 10], [1, 1, 11, 11]])
        assert result == pytest.approx(1.0)

    def test_far_boxes_return_zero(self):
        result = calculate_scene_crowding([[0, 0, 10, 10], [1000, 1000, 1010, 1010]])
        assert result == pytest.approx(0.0)

    def test_returns_float(self):
        result = calculate_scene_crowding([[0, 0, 10, 10], [5, 5, 15, 15]])
        assert isinstance(result, float)

    def test_crowding_values_in_range(self):
        result = calculate_scene_crowding([[0, 0, 50, 50], [60, 60, 110, 110]])
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# get_adaptive_thresholds
# ---------------------------------------------------------------------------


class TestGetAdaptiveThresholds:
    def test_no_crowding(self):
        cfg = TrackerConfig()
        iou_t, cw, mc = get_adaptive_thresholds(cfg, 0.0)
        assert iou_t == pytest.approx(cfg.base_iou_threshold)
        assert cw == pytest.approx(cfg.base_center_weight)
        assert mc == pytest.approx(cfg.base_motion_confidence)

    def test_full_crowding(self):
        cfg = TrackerConfig()
        iou_t, cw, mc = get_adaptive_thresholds(cfg, 1.0)
        assert iou_t == pytest.approx(cfg.base_iou_threshold + 0.20)
        assert cw == pytest.approx(cfg.base_center_weight - 0.35)
        assert mc == pytest.approx(cfg.base_motion_confidence + 0.25)

    def test_returns_three_floats(self):
        result = get_adaptive_thresholds(TrackerConfig(), 0.5)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# is_spatially_plausible
# ---------------------------------------------------------------------------


class TestIsSpatiallyPlausible:
    def test_same_location_is_plausible(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert is_spatially_plausible(box, box, max_jump_factor=2.5) is True

    def test_large_jump_not_plausible(self):
        det = [0.0, 0.0, 10.0, 10.0]
        pred = [1000.0, 1000.0, 1010.0, 1010.0]
        assert is_spatially_plausible(det, pred, max_jump_factor=2.5) is False

    def test_small_jump_plausible(self):
        det = [0.0, 0.0, 100.0, 100.0]
        pred = [5.0, 5.0, 105.0, 105.0]
        assert is_spatially_plausible(det, pred, max_jump_factor=2.5) is True

    def test_returns_bool(self):
        box = [0.0, 0.0, 10.0, 10.0]
        result = is_spatially_plausible(box, box, 2.5)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# CameraMotionCompensator
# ---------------------------------------------------------------------------


class TestCameraMotionCompensator:
    def test_init_state(self):
        cmc = CameraMotionCompensator()
        assert cmc.prev_frame_gray is None
        assert cmc.prev_points is None
        assert len(cmc.motion_history) == 0
        assert cmc.orb is None

    def test_first_frame_returns_zero(self):
        cmc = CameraMotionCompensator()
        frame = _make_frame()
        dx, dy = cmc.estimate_camera_motion(frame)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_first_frame_sets_prev(self):
        cmc = CameraMotionCompensator()
        frame = _make_frame()
        cmc.estimate_camera_motion(frame)
        assert cmc.prev_frame_gray is not None

    def test_second_frame_returns_tuple_of_floats(self):
        cmc = CameraMotionCompensator()
        frame1 = _make_frame()
        frame2 = _make_frame()
        cmc.estimate_camera_motion(frame1)
        dx, dy = cmc.estimate_camera_motion(frame2)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_identical_frames_small_motion(self):
        cmc = CameraMotionCompensator()
        frame = _make_frame()
        cmc.estimate_camera_motion(frame)
        dx, dy = cmc.estimate_camera_motion(frame.copy())
        # identical frame → near-zero motion
        assert abs(dx) < 5.0
        assert abs(dy) < 5.0

    def test_motion_history_grows(self):
        cmc = CameraMotionCompensator()
        frame = _make_frame()
        cmc.estimate_camera_motion(frame)
        cmc.estimate_camera_motion(frame.copy())
        assert len(cmc.motion_history) >= 1

    def test_motion_history_maxlen(self):
        cmc = CameraMotionCompensator()
        frame = _make_frame()
        for _ in range(10):
            cmc.estimate_camera_motion(frame.copy())
        assert len(cmc.motion_history) <= 5


# ---------------------------------------------------------------------------
# create_kalman_filter / predict / update
# ---------------------------------------------------------------------------


filterpy_skip = pytest.mark.skipif(
    __import__("importlib").util.find_spec("filterpy") is None,
    reason="filterpy not installed",
)


@filterpy_skip
class TestKalmanUtilities:
    def test_create_returns_object(self):
        kf = _make_kalman([0, 0, 100, 100])
        assert kf is not None

    def test_initial_state_matches_bbox(self):
        bbox = [10.0, 20.0, 110.0, 120.0]
        kf = _make_kalman(bbox)
        cx_expected = (10 + 110) / 2
        cy_expected = (20 + 120) / 2
        w_expected = 100.0
        h_expected = 100.0
        assert kf.x[0] == pytest.approx(cx_expected)
        assert kf.x[1] == pytest.approx(cy_expected)
        assert kf.x[2] == pytest.approx(w_expected)
        assert kf.x[3] == pytest.approx(h_expected)
        # velocities initialised to zero
        assert kf.x[4] == pytest.approx(0.0)
        assert kf.x[5] == pytest.approx(0.0)

    def test_predict_returns_array_and_confidence(self):
        kf = _make_kalman()
        pred, conf = predict_motion_with_camera_compensation(kf, 0, (0.0, 0.0))
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (4,)
        assert isinstance(conf, float)

    def test_confidence_at_zero_missed(self):
        kf = _make_kalman()
        _, conf = predict_motion_with_camera_compensation(kf, 0, (0.0, 0.0))
        assert conf == pytest.approx(1.0)

    def test_confidence_decays_with_missed_updates(self):
        kf1 = _make_kalman()
        kf2 = _make_kalman()
        _, conf0 = predict_motion_with_camera_compensation(kf1, 0, (0.0, 0.0))
        _, conf5 = predict_motion_with_camera_compensation(kf2, 5, (0.0, 0.0))
        assert conf5 < conf0

    def test_confidence_floor_is_0_1(self):
        kf = _make_kalman()
        _, conf = predict_motion_with_camera_compensation(kf, 9999, (0.0, 0.0))
        assert conf == pytest.approx(0.1)

    def test_camera_motion_shifts_prediction(self):
        kf1 = _make_kalman([50, 50, 150, 150])
        kf2 = _make_kalman([50, 50, 150, 150])
        pred_no_motion, _ = predict_motion_with_camera_compensation(kf1, 0, (0.0, 0.0))
        pred_with_motion, _ = predict_motion_with_camera_compensation(kf2, 0, (10.0, 5.0))
        # x coords shifted right by ~10, y by ~5
        assert pred_with_motion[0] == pytest.approx(pred_no_motion[0] + 10.0, abs=1e-3)
        assert pred_with_motion[1] == pytest.approx(pred_no_motion[1] + 5.0, abs=1e-3)

    def test_update_kalman_filter_does_not_raise(self):
        kf = _make_kalman()
        update_kalman_filter(kf, [20.0, 30.0, 120.0, 130.0])

    def test_update_then_predict_reasonable(self):
        kf = _make_kalman([0, 0, 100, 100])
        update_kalman_filter(kf, [10, 10, 110, 110])
        pred, conf = predict_motion_with_camera_compensation(kf, 0, (0.0, 0.0))
        # prediction should be a valid bbox (x2 > x1, y2 > y1)
        assert pred[2] > pred[0]
        assert pred[3] > pred[1]

    def test_custom_cfg_affects_decay(self):
        cfg_fast = TrackerConfig(confidence_decay_rate=0.5)
        kf1 = _make_kalman()
        kf2 = _make_kalman()
        _, conf_default = predict_motion_with_camera_compensation(kf1, 3, (0.0, 0.0))
        _, conf_fast = predict_motion_with_camera_compensation(kf2, 3, (0.0, 0.0), cfg=cfg_fast)
        assert conf_fast < conf_default


# ---------------------------------------------------------------------------
# PersonTracker
# ---------------------------------------------------------------------------


class TestPersonTracker:
    def test_default_device(self):
        pt = PersonTracker()
        assert pt.device == "cuda:0"

    def test_custom_device(self):
        pt = PersonTracker(device="cpu")
        assert pt.device == "cpu"

    def test_initial_state(self):
        pt = PersonTracker()
        assert pt.frame_count == 0
        assert pt.next_track_id == 1
        assert pt.active_tracks == {}
        assert pt._detector is None
        assert pt._pose is None

    def test_cfg_default(self):
        pt = PersonTracker()
        assert isinstance(pt.cfg, TrackerConfig)

    def test_cfg_custom(self):
        cfg = TrackerConfig(max_lost_frames=42)
        pt = PersonTracker(tracker_config=cfg)
        assert pt.cfg.max_lost_frames == 42

    def test_camera_compensator_initialised(self):
        pt = PersonTracker()
        assert isinstance(pt._camera, CameraMotionCompensator)

    def test_ensure_models_raises_without_configs(self):
        pt = PersonTracker()
        with pytest.raises((ImportError, ValueError)):
            pt._ensure_models()

    def test_process_video_missing_file_raises(self):
        pt = PersonTracker()
        with pytest.raises(FileNotFoundError):
            pt.process_video("/nonexistent/path/video.mp4", "/tmp/out.mp4")

    def test_process_video_writes_output(self, tmp_path: Path):
        """process_video should create an output file from a synthetic video."""
        import cv2

        in_path = tmp_path / "in.mp4"
        out_path = tmp_path / "out.mp4"

        # Write a tiny 5-frame video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_path), fourcc, 10.0, (64, 64))
        rng = np.random.default_rng(1)
        for _ in range(5):
            frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        pt = PersonTracker()
        pt.process_video(str(in_path), str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_process_video_increments_frame_count(self, tmp_path: Path):
        import cv2

        in_path = tmp_path / "in.mp4"
        out_path = tmp_path / "out.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_path), fourcc, 10.0, (64, 64))
        rng = np.random.default_rng(2)
        n_frames = 7
        for _ in range(n_frames):
            writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        writer.release()

        pt = PersonTracker()
        pt.process_video(str(in_path), str(out_path))

        assert pt.frame_count == n_frames

    def test_process_folder_processes_all_videos(self, tmp_path: Path):
        import cv2

        in_dir = tmp_path / "videos"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        rng = np.random.default_rng(3)
        for name in ("a.mp4", "b.mp4"):
            writer = cv2.VideoWriter(str(in_dir / name), fourcc, 10.0, (32, 32))
            for _ in range(3):
                writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
            writer.release()

        pt = PersonTracker()
        pt.process_folder(str(in_dir), str(out_dir))

        assert (out_dir / "a.mp4").exists()
        assert (out_dir / "b.mp4").exists()

    def test_process_folder_creates_out_dir(self, tmp_path: Path):
        import cv2

        in_dir = tmp_path / "vids"
        out_dir = tmp_path / "nested" / "out"
        in_dir.mkdir()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_dir / "v.mp4"), fourcc, 10.0, (32, 32))
        rng = np.random.default_rng(4)
        for _ in range(2):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        writer.release()

        pt = PersonTracker()
        pt.process_folder(str(in_dir), str(out_dir))

        assert out_dir.exists()

    def test_active_tracks_cleared_between_runs(self, tmp_path: Path):
        import cv2

        in_path = tmp_path / "in.mp4"
        out_path = tmp_path / "out.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_path), fourcc, 10.0, (32, 32))
        rng = np.random.default_rng(5)
        for _ in range(2):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        writer.release()

        pt = PersonTracker()
        pt.active_tracks[99] = {}  # inject stale state
        pt.process_video(str(in_path), str(out_path))
        assert 99 not in pt.active_tracks


# ---------------------------------------------------------------------------
# Convenience API wrappers
# ---------------------------------------------------------------------------


class TestConvenienceApi:
    def test_process_video_wrapper(self, tmp_path: Path):
        import cv2
        from sailsprep.id_tracking_model.tracker.person_tracker import (
            process_video as convenience_process_video,
        )

        in_path = tmp_path / "in.mp4"
        out_path = tmp_path / "out.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_path), fourcc, 10.0, (32, 32))
        rng = np.random.default_rng(6)
        for _ in range(3):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        writer.release()

        convenience_process_video(str(in_path), str(out_path))
        assert out_path.exists()

    def test_process_folder_wrapper(self, tmp_path: Path):
        import cv2
        from sailsprep.id_tracking_model.tracker.person_tracker import (
            process_folder as convenience_process_folder,
        )

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(in_dir / "v.mp4"), fourcc, 10.0, (32, 32))
        rng = np.random.default_rng(7)
        for _ in range(2):
            writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        writer.release()

        convenience_process_folder(str(in_dir), str(out_dir))
        assert (out_dir / "v.mp4").exists()