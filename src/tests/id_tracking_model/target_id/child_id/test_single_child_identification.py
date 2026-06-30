"""
Tests for single_child_identification.py

Run with:
    poetry run pytest src/tests/test_single_child_identification.py -v
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sailsprep.id_tracking_model.target_id.child_id.single_child_identification import (
    AnnotationInfo,
    ChildIdentificationConfig,
    ChildResult,
    EdgeScore,
    Evidence,
    NodeScore,
    SingleChildIdentifier,
    SkeletonRatios,
    Track,
    Tracklet,
    _evenly_spaced_indices,
    _sigmoid,
    _smart_frame_selection,
    aggregate_skeleton_ratios_over_track,
    compute_keypoint_rigidity_score,
    compute_scale_invariant_ratios,
    crop_bbox_from_frame,
    deepface_predict_age,
    identify_single_child,
    is_body_in_bbox,
    map_age_to_child_prob,
    median_age_from_face_crops,
    ratio_to_child_score,
    siglip_predict_child_prob,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_kp(
    nose_conf: float = 0.9,
    shoulder_conf: float = 0.9,
    hip_conf: float = 0.9,
    knee_conf: float = 0.9,
    ankle_conf: float = 0.9,
    elbow_conf: float = 0.9,
    wrist_conf: float = 0.9,
) -> np.ndarray:
    """Return a synthetic 17-keypoint COCO array shaped (17, 3) with [x, y, conf].

    Coordinates are chosen so that a child-like skeleton is produced by default:
      - large head relative to narrow shoulders
      - short legs relative to torso
    """
    kp = np.zeros((17, 3), dtype=float)

    # nose (0)
    kp[0] = [50.0, 10.0, nose_conf]
    # eyes (1, 2) – not used by ratios but need to exist
    kp[1] = [48.0, 8.0, 0.5]
    kp[2] = [52.0, 8.0, 0.5]
    # ears (3, 4)
    kp[3] = [46.0, 10.0, 0.5]
    kp[4] = [54.0, 10.0, 0.5]
    # shoulders (5 L, 6 R) – narrow: 40 wide, centred at x=50, y=30
    kp[5] = [40.0, 30.0, shoulder_conf]   # L shoulder
    kp[6] = [60.0, 30.0, shoulder_conf]   # R shoulder
    # elbows (7 L, 8 R)
    kp[7] = [38.0, 50.0, elbow_conf]
    kp[8] = [62.0, 50.0, elbow_conf]
    # wrists (9 L, 10 R)
    kp[9]  = [36.0, 65.0, wrist_conf]
    kp[10] = [64.0, 65.0, wrist_conf]
    # hips (11 L, 12 R) – similar width to shoulders → child shoulder/hip ≈ 1
    kp[11] = [42.0, 60.0, hip_conf]       # L hip
    kp[12] = [58.0, 60.0, hip_conf]       # R hip
    # knees (13 L, 14 R) – short legs
    kp[13] = [42.0, 75.0, knee_conf]
    kp[14] = [58.0, 75.0, knee_conf]
    # ankles (15 L, 16 R)
    kp[15] = [42.0, 85.0, ankle_conf]
    kp[16] = [58.0, 85.0, ankle_conf]

    return kp


def _make_adult_kp() -> np.ndarray:
    """Adult skeleton: wider shoulders vs hips, longer legs, smaller head."""
    kp = np.zeros((17, 3), dtype=float)
    kp[0]  = [50.0,  5.0, 0.9]   # nose – far above shoulders → small head ratio
    kp[1]  = [48.0,  3.0, 0.5]
    kp[2]  = [52.0,  3.0, 0.5]
    kp[3]  = [46.0,  5.0, 0.5]
    kp[4]  = [54.0,  5.0, 0.5]
    kp[5]  = [30.0, 30.0, 0.9]   # L shoulder – wide
    kp[6]  = [70.0, 30.0, 0.9]   # R shoulder
    kp[7]  = [27.0, 55.0, 0.9]
    kp[8]  = [73.0, 55.0, 0.9]
    kp[9]  = [25.0, 78.0, 0.9]
    kp[10] = [75.0, 78.0, 0.9]
    kp[11] = [38.0, 60.0, 0.9]   # L hip – narrower than shoulders
    kp[12] = [62.0, 60.0, 0.9]   # R hip
    kp[13] = [38.0, 90.0, 0.9]   # long thighs
    kp[14] = [62.0, 90.0, 0.9]
    kp[15] = [38.0, 120.0, 0.9]  # long shanks
    kp[16] = [62.0, 120.0, 0.9]
    return kp


def _make_track(
    track_id: int = 1,
    start: int = 0,
    end: int = 60,
    fps: float = 30.0,
    keypoints: Optional[List[Any]] = None,
) -> Track:
    return Track(
        id=track_id,
        start_frame=start,
        end_frame=end,
        fps=fps,
        keypoints=keypoints,
    )


def _make_tracklet(
    parent_id: int = 1,
    start: int = 0,
    end: int = 60,
    fps: float = 30.0,
    keypoints: Optional[List[Any]] = None,
) -> Tracklet:
    return Tracklet(
        parent_id=parent_id,
        start_frame=start,
        end_frame=end,
        fps=fps,
        keypoints=keypoints,
    )


def _cfg(**kwargs: Any) -> ChildIdentificationConfig:
    """Return a config with all heavy external calls disabled."""
    defaults = dict(
        age_estimation_method="siglip",
        enable_skeleton_ratios=False,
        enable_rigidity_detection=False,
        min_track_frames=1,
    )
    defaults.update(kwargs)
    return ChildIdentificationConfig(**defaults)


def _ann() -> AnnotationInfo:
    return AnnotationInfo()


# ---------------------------------------------------------------------------
# Dataclass / structure tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_track_duration_frames(self) -> None:
        t = _make_track(start=0, end=29)
        assert t.duration_frames() == 30

    def test_track_duration_seconds(self) -> None:
        t = _make_track(start=0, end=29, fps=30.0)
        assert t.duration_seconds() == pytest.approx(1.0)

    def test_track_duration_zero_fps(self) -> None:
        t = _make_track(start=0, end=29, fps=0.0)
        # Should not raise; fps defaults to 1.0 inside duration_seconds
        assert t.duration_seconds() == pytest.approx(30.0)

    def test_tracklet_id_equals_parent_id(self) -> None:
        tl = _make_tracklet(parent_id=7)
        assert tl.id == 7

    def test_tracklet_duration(self) -> None:
        tl = _make_tracklet(start=10, end=39, fps=30.0)
        assert tl.duration_frames() == 30
        assert tl.duration_seconds() == pytest.approx(1.0)

    def test_evidence_defaults(self) -> None:
        ev = Evidence()
        assert ev.p_age is None
        assert ev.p_skeleton is None
        assert ev.p_rigidity is None
        assert ev.flags == []

    def test_config_defaults(self) -> None:
        cfg = ChildIdentificationConfig()
        assert cfg.sampling_percentage == 0.25
        assert cfg.age_estimation_method == "siglip"
        assert cfg.enable_skeleton_ratios is True
        assert cfg.enable_rigidity_detection is True


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_input(self) -> None:
        assert _sigmoid(0.0, 1.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert _sigmoid(100.0, 1.0) > 0.99

    def test_large_negative(self) -> None:
        assert _sigmoid(-100.0, 1.0) < 0.01

    def test_tau_zero_clamped(self) -> None:
        # tau is clamped to 1e-6; should not raise
        result = _sigmoid(1.0, 0.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# map_age_to_child_prob
# ---------------------------------------------------------------------------


class TestMapAgeToChildProb:
    def test_young_child(self) -> None:
        p = map_age_to_child_prob(3.0, child_years_threshold=10.0, tau=2.5)
        assert p > 0.9

    def test_adult(self) -> None:
        p = map_age_to_child_prob(35.0, child_years_threshold=10.0, tau=2.5)
        assert p < 0.1

    def test_boundary(self) -> None:
        p = map_age_to_child_prob(10.0, child_years_threshold=10.0, tau=2.5)
        assert p == pytest.approx(0.5)

    def test_monotone_decreasing(self) -> None:
        ages = [2, 5, 8, 10, 15, 25, 40]
        probs = [map_age_to_child_prob(a, 10.0, 2.5) for a in ages]
        assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))


# ---------------------------------------------------------------------------
# _evenly_spaced_indices
# ---------------------------------------------------------------------------


class TestEvenlySpacedIndices:
    def test_empty(self) -> None:
        assert _evenly_spaced_indices(0, 5) == []

    def test_k_one(self) -> None:
        result = _evenly_spaced_indices(10, 1)
        assert len(result) == 1
        assert result[0] in range(10)

    def test_full_coverage(self) -> None:
        result = _evenly_spaced_indices(5, 5)
        assert sorted(result) == [0, 1, 2, 3, 4]

    def test_endpoints_included(self) -> None:
        result = _evenly_spaced_indices(100, 10)
        assert 0 in result
        assert 99 in result

    def test_k_larger_than_n(self) -> None:
        result = _evenly_spaced_indices(3, 10)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# _smart_frame_selection
# ---------------------------------------------------------------------------


class TestSmartFrameSelection:
    def _make_kp_list(self, n: int, conf: float = 0.8) -> List[Any]:
        """n frames, each with a single high-confidence keypoint at index 0."""
        kp = [[0.0, 0.0, conf]] * 17
        return [kp] * n

    def test_returns_k_indices(self) -> None:
        kp_list = self._make_kp_list(20)
        result = _smart_frame_selection(kp_list, k=5)
        assert len(result) == 5

    def test_empty_input(self) -> None:
        assert _smart_frame_selection([], k=5) == []

    def test_indices_in_range(self) -> None:
        kp_list = self._make_kp_list(30)
        result = _smart_frame_selection(kp_list, k=8)
        assert all(0 <= i < 30 for i in result)

    def test_sorted_output(self) -> None:
        kp_list = self._make_kp_list(20)
        result = _smart_frame_selection(kp_list, k=6)
        assert result == sorted(result)

    def test_falls_back_when_low_confidence(self) -> None:
        """When no frames meet min_pose_confidence, all frames are candidates."""
        kp_list = self._make_kp_list(10, conf=0.1)
        result = _smart_frame_selection(kp_list, k=3, min_pose_confidence=0.9)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# compute_scale_invariant_ratios
# ---------------------------------------------------------------------------


class TestComputeScaleInvariantRatios:
    def test_child_skeleton(self) -> None:
        kp = _make_kp()
        ratios = compute_scale_invariant_ratios(kp, min_confidence=0.3)
        # All four ratios should be populated
        assert ratios.head_shoulder is not None
        assert ratios.leg_torso is not None
        assert ratios.shoulder_hip is not None
        assert ratios.arm_torso is not None

    def test_none_keypoints(self) -> None:
        ratios = compute_scale_invariant_ratios(None)
        assert ratios == SkeletonRatios()

    def test_too_few_keypoints(self) -> None:
        short_kp = np.zeros((10, 3))
        ratios = compute_scale_invariant_ratios(short_kp)
        assert ratios == SkeletonRatios()

    def test_low_confidence_returns_none_fields(self) -> None:
        kp = _make_kp(shoulder_conf=0.1, hip_conf=0.1)
        ratios = compute_scale_invariant_ratios(kp, min_confidence=0.5)
        # Without shoulders/hips most ratios cannot be computed
        assert ratios.leg_torso is None
        assert ratios.shoulder_hip is None

    def test_child_has_large_head_shoulder_ratio(self) -> None:
        child_kp = _make_kp()
        adult_kp = _make_adult_kp()
        child_r = compute_scale_invariant_ratios(child_kp)
        adult_r = compute_scale_invariant_ratios(adult_kp)
        assert child_r.head_shoulder is not None
        assert adult_r.head_shoulder is not None
        assert child_r.head_shoulder > adult_r.head_shoulder

    def test_child_has_shorter_legs(self) -> None:
        child_kp = _make_kp()
        adult_kp = _make_adult_kp()
        child_r = compute_scale_invariant_ratios(child_kp)
        adult_r = compute_scale_invariant_ratios(adult_kp)
        assert child_r.leg_torso is not None
        assert adult_r.leg_torso is not None
        assert child_r.leg_torso < adult_r.leg_torso


# ---------------------------------------------------------------------------
# ratio_to_child_score
# ---------------------------------------------------------------------------


class TestRatioToChildScore:
    def test_no_ratios_returns_none(self) -> None:
        assert ratio_to_child_score(SkeletonRatios()) is None

    def test_large_head_shoulder_high_score(self) -> None:
        r = SkeletonRatios(head_shoulder=1.2)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score >= 0.8

    def test_small_head_shoulder_low_score(self) -> None:
        r = SkeletonRatios(head_shoulder=0.4)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score <= 0.15

    def test_short_legs_high_score(self) -> None:
        r = SkeletonRatios(leg_torso=1.2)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score >= 0.8

    def test_long_legs_low_score(self) -> None:
        r = SkeletonRatios(leg_torso=2.8)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score <= 0.15

    def test_all_child_ratios(self) -> None:
        r = SkeletonRatios(head_shoulder=1.1, leg_torso=1.3, shoulder_hip=1.0, arm_torso=1.1)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score > 0.7

    def test_all_adult_ratios(self) -> None:
        r = SkeletonRatios(head_shoulder=0.4, leg_torso=2.6, shoulder_hip=1.4, arm_torso=2.4)
        score = ratio_to_child_score(r)
        assert score is not None
        assert score < 0.3

    def test_score_in_unit_interval(self) -> None:
        for hs in [0.3, 0.7, 0.9, 1.1, 1.5]:
            r = SkeletonRatios(head_shoulder=hs)
            s = ratio_to_child_score(r)
            assert s is not None
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# aggregate_skeleton_ratios_over_track
# ---------------------------------------------------------------------------


class TestAggregateSkeletonRatiosOverTrack:
    def test_empty_list(self) -> None:
        assert aggregate_skeleton_ratios_over_track([]) is None

    def test_none_entries_skipped(self) -> None:
        result = aggregate_skeleton_ratios_over_track([None, None])
        # No valid keypoints → None
        assert result is None

    def test_child_keypoints(self) -> None:
        kp = _make_kp()
        result = aggregate_skeleton_ratios_over_track([kp] * 5)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_child_scores_higher_than_adult(self) -> None:
        child_kp = _make_kp()
        adult_kp = _make_adult_kp()
        child_score = aggregate_skeleton_ratios_over_track([child_kp] * 10)
        adult_score = aggregate_skeleton_ratios_over_track([adult_kp] * 10)
        assert child_score is not None
        assert adult_score is not None
        assert child_score > adult_score

    def test_median_is_robust_to_outliers(self) -> None:
        child_kp = _make_kp()
        adult_kp = _make_adult_kp()
        # 9 child frames, 1 adult outlier
        frames = [child_kp] * 9 + [adult_kp]
        score = aggregate_skeleton_ratios_over_track(frames)
        assert score is not None
        # Median should be dominated by child frames
        assert score > 0.5


# ---------------------------------------------------------------------------
# compute_keypoint_rigidity_score
# ---------------------------------------------------------------------------


def _rigid_kp_list(n: int) -> List[np.ndarray]:
    """n identical frames → zero variation → rigid."""
    kp = _make_kp()
    return [kp.copy() for _ in range(n)]


def _moving_kp_list(n: int) -> List[np.ndarray]:
    """n frames with varying elbow/wrist positions → natural motion."""
    frames = []
    for i in range(n):
        kp = _make_kp()
        # Vary elbows and wrists sinusoidally
        offset = float(i) * 3.0
        kp[7][0] += offset       # L elbow x
        kp[8][0] -= offset       # R elbow x
        kp[9][0] += offset * 0.8
        kp[10][0] -= offset * 0.8
        frames.append(kp)
    return frames


class TestComputeKeypointRigidityScore:
    def test_too_few_frames_returns_none(self) -> None:
        kp_list = _rigid_kp_list(5)
        assert compute_keypoint_rigidity_score(kp_list, min_frames=10) is None

    def test_empty_returns_none(self) -> None:
        assert compute_keypoint_rigidity_score([], min_frames=5) is None

    def test_rigid_frames_score_zero(self) -> None:
        kp_list = _rigid_kp_list(20)
        score = compute_keypoint_rigidity_score(kp_list, min_frames=10)
        assert score is not None
        assert score == pytest.approx(0.0)

    def test_moving_frames_score_higher(self) -> None:
        moving = _moving_kp_list(20)
        rigid = _rigid_kp_list(20)
        s_moving = compute_keypoint_rigidity_score(moving, min_frames=10)
        s_rigid = compute_keypoint_rigidity_score(rigid, min_frames=10)
        assert s_moving is not None
        assert s_rigid is not None
        assert s_moving > s_rigid

    def test_score_in_unit_interval(self) -> None:
        kp_list = _moving_kp_list(20)
        score = compute_keypoint_rigidity_score(kp_list, min_frames=10)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_none_frames_skipped(self) -> None:
        kp_list: List[Any] = [None] * 20
        score = compute_keypoint_rigidity_score(kp_list, min_frames=10)
        assert score is None


# ---------------------------------------------------------------------------
# is_body_in_bbox
# ---------------------------------------------------------------------------


class TestIsBodyInBbox:
    def _roi(self, h: int = 100, w: int = 80) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _high_conf_kps(self, n: int = 10) -> List[List[float]]:
        return [[float(i), float(i), 0.9] for i in range(17)]

    def test_sufficient_keypoints_detected(self) -> None:
        result = is_body_in_bbox(self._roi(), self._high_conf_kps(), min_visible_keypoints=3)
        assert result["detected"] is True

    def test_insufficient_keypoints_rejected(self) -> None:
        low_kps = [[0.0, 0.0, 0.1]] * 17
        result = is_body_in_bbox(self._roi(), low_kps, min_visible_keypoints=3)
        assert result["detected"] is False
        assert "Too few visible keypoints" in result["reason"]

    def test_roi_size_filter_too_small(self) -> None:
        small_roi = np.zeros((10, 10, 3), dtype=np.uint8)
        result = is_body_in_bbox(
            small_roi,
            self._high_conf_kps(),
            enable_roi_size_filter=True,
            min_roi_height=50,
            min_roi_width=30,
        )
        assert result["detected"] is False
        assert "too small" in result["reason"]

    def test_roi_size_filter_passes(self) -> None:
        result = is_body_in_bbox(
            self._roi(100, 80),
            self._high_conf_kps(),
            enable_roi_size_filter=True,
            min_roi_height=50,
            min_roi_width=30,
        )
        assert result["detected"] is True

    def test_roi_size_filter_disabled(self) -> None:
        small_roi = np.zeros((5, 5, 3), dtype=np.uint8)
        result = is_body_in_bbox(
            small_roi,
            self._high_conf_kps(),
            enable_roi_size_filter=False,
            min_visible_keypoints=3,
        )
        # Only keypoint check applies when size filter is off
        assert result["detected"] is True

    def test_custom_body_keypoint_indices(self) -> None:
        kps = [[0.0, 0.0, 0.1]] * 17
        kps[5] = [0.0, 0.0, 0.95]   # Only shoulder is confident
        result = is_body_in_bbox(
            self._roi(),
            kps,
            min_visible_keypoints=1,
            body_keypoint_indices=[5],
        )
        assert result["detected"] is True


# ---------------------------------------------------------------------------
# crop_bbox_from_frame
# ---------------------------------------------------------------------------


class TestCropBboxFromFrame:
    def test_basic_crop(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:50, 20:60] = 128
        crop = crop_bbox_from_frame(frame, (20, 10, 60, 50))
        assert crop is not None
        assert crop.shape == (40, 40, 3)

    def test_none_frame(self) -> None:
        assert crop_bbox_from_frame(None, (0, 0, 10, 10)) is None

    def test_out_of_bounds_clamped(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        crop = crop_bbox_from_frame(frame, (-10, -10, 200, 200))
        assert crop is not None
        assert crop.shape == (50, 50, 3)

    def test_degenerate_bbox_returns_none(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # x1 == x2 after clamping
        crop = crop_bbox_from_frame(frame, (25, 25, 25, 50))
        assert crop is None


# ---------------------------------------------------------------------------
# median_age_from_face_crops  (DeepFace mocked)
# ---------------------------------------------------------------------------


class TestMedianAgeFromFaceCrops:
    def test_no_face_crops(self) -> None:
        age, flags = median_age_from_face_crops([])
        assert age is None
        assert "no_face_crop" in flags

    @patch(
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.deepface_predict_age"
    )
    def test_returns_median(self, mock_predict: MagicMock) -> None:
        mock_predict.return_value = (8.0, [])
        crops = [object()] * 4
        age, flags = median_age_from_face_crops(crops, sampling_percentage=1.0)
        assert age == pytest.approx(8.0)

    @patch(
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.deepface_predict_age"
    )
    def test_correct_median_odd(self, mock_predict: MagicMock) -> None:
        ages = [5.0, 7.0, 9.0]
        mock_predict.side_effect = [(a, []) for a in ages]
        crops = [object()] * 3
        age, _ = median_age_from_face_crops(crops, sampling_percentage=1.0)
        assert age == pytest.approx(7.0)

    @patch(
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.deepface_predict_age"
    )
    def test_all_failures_returns_none(self, mock_predict: MagicMock) -> None:
        mock_predict.return_value = (None, ["deepface_error"])
        crops = [object()] * 3
        age, flags = median_age_from_face_crops(crops, sampling_percentage=1.0)
        assert age is None
        assert any("deepface" in f for f in flags)

    @patch(
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.deepface_predict_age"
    )
    def test_sampling_limits_calls(self, mock_predict: MagicMock) -> None:
        mock_predict.return_value = (6.0, [])
        crops = [object()] * 100
        median_age_from_face_crops(crops, sampling_percentage=0.1, sampling_max=5)
        assert mock_predict.call_count <= 5


# ---------------------------------------------------------------------------
# deepface_predict_age  (DeepFace mocked)
# ---------------------------------------------------------------------------


class TestDeepfacePredictAge:
    def test_deepface_none(self) -> None:
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.DeepFace",
            None,
        ):
            age, flags = deepface_predict_age("img", 0.8)
        assert age is None
        assert "deepface_not_available" in flags

    def test_low_face_confidence_rejected(self) -> None:
        mock_df = MagicMock()
        mock_df.analyze.return_value = [{"face_confidence": 0.3, "age": 10}]
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.DeepFace",
            mock_df,
        ):
            age, flags = deepface_predict_age("img", face_conf_threshold=0.8)
        assert age is None
        assert any("low_face_confidence" in f for f in flags)

    def test_valid_detection(self) -> None:
        mock_df = MagicMock()
        mock_df.analyze.return_value = [{"face_confidence": 0.95, "age": 7}]
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.DeepFace",
            mock_df,
        ):
            age, flags = deepface_predict_age("img", face_conf_threshold=0.8)
        assert age == pytest.approx(7.0)
        assert flags == []

    def test_exception_returns_error_flag(self) -> None:
        mock_df = MagicMock()
        mock_df.analyze.side_effect = RuntimeError("boom")
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.DeepFace",
            mock_df,
        ):
            age, flags = deepface_predict_age("img", face_conf_threshold=0.8)
        assert age is None
        assert "deepface_error" in flags

    def test_picks_highest_confidence_face(self) -> None:
        mock_df = MagicMock()
        mock_df.analyze.return_value = [
            {"face_confidence": 0.6, "age": 5},
            {"face_confidence": 0.95, "age": 30},
        ]
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.DeepFace",
            mock_df,
        ):
            age, _ = deepface_predict_age("img", face_conf_threshold=0.5)
        assert age == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# siglip_predict_child_prob
# ---------------------------------------------------------------------------


class TestSiglipPredictChildProb:
    def test_siglip_not_available(self) -> None:
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.SIGLIP_AVAILABLE",
            False,
        ):
            prob, flags = siglip_predict_child_prob(None)
        assert prob is None
        assert "siglip_not_available" in flags

    def test_model_not_loaded(self) -> None:
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.SIGLIP_AVAILABLE",
            True,
        ):
            prob, flags = siglip_predict_child_prob(object(), model=None, processor=None)
        assert prob is None
        assert "siglip_model_not_loaded" in flags

    def test_invalid_image_format(self) -> None:
        with patch(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_identification.SIGLIP_AVAILABLE",
            True,
        ):
            grayscale = np.zeros((50, 50), dtype=np.uint8)  # 2-D, no channel dim
            prob, flags = siglip_predict_child_prob(
                grayscale, model=MagicMock(), processor=MagicMock()
            )
        assert prob is None
        assert "siglip_invalid_image_format" in flags


# ---------------------------------------------------------------------------
# SingleChildIdentifier — node scoring
# ---------------------------------------------------------------------------


class TestSingleChildIdentifierNodeScoring:
    """Tests that cover _score_node and evidence computation without I/O."""

    def _identifier(self, **cfg_kwargs: Any) -> SingleChildIdentifier:
        cfg = _cfg(**cfg_kwargs)
        return SingleChildIdentifier(cfg, _ann())

    def test_no_evidence_score_zero(self) -> None:
        ident = self._identifier()
        tl = _make_tracklet()
        node = ident._score_node(tl)
        assert node.score == pytest.approx(0.0)
        assert node.weight == pytest.approx(0.0)

    def test_skeleton_only_score(self) -> None:
        child_kp = _make_kp()
        ident = self._identifier(
            enable_skeleton_ratios=True,
            enable_rigidity_detection=False,
            age_estimation_method="siglip",
        )
        tl = _make_tracklet(keypoints=[child_kp] * 5)
        node = ident._score_node(tl)
        # Skeleton evidence for a child should push score > 0
        assert node.score > 0.0

    def test_rigidity_rejection(self) -> None:
        """A rigid (picture-like) track should be rejected (score=0)."""
        rigid_kps = _rigid_kp_list(30)
        cfg = _cfg(
            enable_skeleton_ratios=False,
            enable_rigidity_detection=True,
            rigidity_min_frames=10,
            min_rigidity_score=0.2,
        )
        ident = SingleChildIdentifier(cfg, _ann())
        tl = _make_tracklet(keypoints=rigid_kps)
        node = ident._score_node(tl)
        assert node.score == pytest.approx(0.0)
        assert any("rejected_static_picture" in f for f in node.evidence.flags)

    def test_moving_track_not_rejected(self) -> None:
        moving_kps = _moving_kp_list(30)
        cfg = _cfg(
            enable_skeleton_ratios=False,
            enable_rigidity_detection=True,
            rigidity_min_frames=10,
            min_rigidity_score=0.1,
        )
        ident = SingleChildIdentifier(cfg, _ann())
        tl = _make_tracklet(keypoints=moving_kps)
        node = ident._score_node(tl)
        # Should NOT be rejected as a static picture
        assert not any("rejected_static_picture" in f for f in node.evidence.flags)

    def test_weight_proportional_to_duration(self) -> None:
        child_kp = _make_kp()
        ident = self._identifier(
            enable_skeleton_ratios=True,
            enable_rigidity_detection=False,
        )
        short = _make_tracklet(start=0, end=30, fps=30.0, keypoints=[child_kp] * 5)
        long_ = _make_tracklet(start=0, end=150, fps=30.0, keypoints=[child_kp] * 5)
        n_short = ident._score_node(short)
        n_long = ident._score_node(long_)
        # Same score, but longer duration → higher weight
        assert n_long.weight > n_short.weight

    def test_flags_recorded_when_no_keypoints(self) -> None:
        cfg = _cfg(enable_skeleton_ratios=True, enable_rigidity_detection=True)
        ident = SingleChildIdentifier(cfg, _ann())
        tl = _make_tracklet(keypoints=None)
        node = ident._score_node(tl)
        flags = node.evidence.flags
        assert any("no_keypoints" in f for f in flags)

    def test_unknown_age_method_flag(self) -> None:
        cfg = _cfg(age_estimation_method="unknown_method")
        ident = SingleChildIdentifier(cfg, _ann())
        tl = _make_tracklet()
        node = ident._score_node(tl)
        assert any("unknown_age_method" in f for f in node.evidence.flags)


# ---------------------------------------------------------------------------
# SingleChildIdentifier — edge building
# ---------------------------------------------------------------------------


class TestBuildEdges:
    def _ident(self) -> SingleChildIdentifier:
        return SingleChildIdentifier(_cfg(continuity_gap_seconds=1.0), _ann())

    def _node(
        self,
        start: int,
        end: int,
        parent_id: int = 1,
        fps: float = 30.0,
        p_age: Optional[float] = None,
    ) -> NodeScore:
        tl = _make_tracklet(parent_id=parent_id, start=start, end=end, fps=fps)
        ev = Evidence(p_age=p_age)
        score = 0.5
        return NodeScore(tracklet=tl, score=score, weight=score * tl.duration_seconds(), evidence=ev)

    def test_no_nodes_no_edges(self) -> None:
        ident = self._ident()
        assert ident._build_edges([]) == []

    def test_overlapping_nodes_no_edge(self) -> None:
        ident = self._ident()
        nodes = [self._node(0, 60), self._node(30, 90)]
        edges = ident._build_edges(nodes)
        assert edges == []

    def test_adjacent_within_gap_creates_edge(self) -> None:
        ident = self._ident()
        # 5 frames gap at 30 fps ≈ 0.17 s < 1.0 s threshold
        nodes = [self._node(0, 59), self._node(64, 120)]
        edges = ident._build_edges(nodes)
        assert len(edges) == 1
        assert edges[0].src_index == 0
        assert edges[0].dst_index == 1

    def test_gap_too_large_no_edge(self) -> None:
        ident = self._ident()
        # 120 frames gap at 30 fps = 4 s > 1.0 s threshold
        nodes = [self._node(0, 59), self._node(179, 240)]
        edges = ident._build_edges(nodes)
        assert edges == []

    def test_same_id_bonus_applied(self) -> None:
        ident = self._ident()
        # Same parent_id → bonus should appear in reasons
        nodes = [self._node(0, 59, parent_id=1), self._node(64, 120, parent_id=1)]
        edges = ident._build_edges(nodes)
        assert len(edges) == 1
        assert "same_id_bonus" in edges[0].reasons
        assert edges[0].reasons["same_id_bonus"] > 0

    def test_different_id_no_bonus(self) -> None:
        ident = self._ident()
        nodes = [self._node(0, 59, parent_id=1), self._node(64, 120, parent_id=2)]
        edges = ident._build_edges(nodes)
        assert len(edges) == 1
        assert "same_id_bonus" not in edges[0].reasons

    def test_age_inconsistency_penalty(self) -> None:
        cfg = _cfg(
            age_inconsistency_threshold=0.3,
            age_inconsistency_penalty_weight=2.0,
            continuity_gap_seconds=1.0,
        )
        ident = SingleChildIdentifier(cfg, _ann())
        # One node strongly child (0.9), other strongly adult (0.1) → large penalty
        nodes = [
            self._node(0, 59, p_age=0.9),
            self._node(64, 120, p_age=0.1),
        ]
        edges = ident._build_edges(nodes)
        assert len(edges) == 1
        assert "age_inconsistency_penalty" in edges[0].reasons
        assert edges[0].reasons["age_inconsistency_penalty"] > 0


# ---------------------------------------------------------------------------
# SingleChildIdentifier — path selection
# ---------------------------------------------------------------------------


class TestSelectBestPath:
    def _ident(self) -> SingleChildIdentifier:
        return SingleChildIdentifier(_cfg(), _ann())

    def _node(self, start: int, end: int, score: float = 0.5, parent_id: int = 1) -> NodeScore:
        tl = _make_tracklet(parent_id=parent_id, start=start, end=end, fps=30.0)
        ev = Evidence()
        weight = score * tl.duration_seconds()
        return NodeScore(tracklet=tl, score=score, weight=weight, evidence=ev)

    def test_empty_nodes(self) -> None:
        ident = self._ident()
        assert ident._select_best_path([], []) == []

    def test_single_node(self) -> None:
        ident = self._ident()
        nodes = [self._node(0, 60)]
        path = ident._select_best_path(nodes, [])
        assert path == [0]

    def test_selects_highest_weight_single_node(self) -> None:
        ident = self._ident()
        # Node 1 is short + low score; node 2 is long + high score
        nodes = [
            self._node(0, 10, score=0.1),
            self._node(200, 400, score=0.9),
        ]
        path = ident._select_best_path(nodes, [])
        # No edges (too far apart), so each is standalone; highest weight wins
        assert 1 in path

    def test_chain_selected_when_edges_boost_score(self) -> None:
        ident = self._ident()
        # Two adjacent nodes with moderate scores
        n0 = self._node(0, 59, score=0.5)
        n1 = self._node(64, 120, score=0.5)
        nodes = [n0, n1]
        edge = EdgeScore(src_index=0, dst_index=1, score=1.0)
        path = ident._select_best_path(nodes, [edge])
        assert path == [0, 1]

    def test_path_reconstruction_correct_order(self) -> None:
        ident = self._ident()
        n0 = self._node(0, 29, score=0.4)
        n1 = self._node(35, 64, score=0.4)
        n2 = self._node(70, 99, score=0.4)
        nodes = [n0, n1, n2]
        edges = [
            EdgeScore(src_index=0, dst_index=1, score=0.5),
            EdgeScore(src_index=1, dst_index=2, score=0.5),
        ]
        path = ident._select_best_path(nodes, edges)
        assert path == [0, 1, 2]


# ---------------------------------------------------------------------------
# SingleChildIdentifier — age inconsistency penalty
# ---------------------------------------------------------------------------


class TestAgeInconsistencyPenalty:
    def _ident(self) -> SingleChildIdentifier:
        return SingleChildIdentifier(
            _cfg(age_inconsistency_threshold=0.3, age_inconsistency_penalty_weight=2.0),
            _ann(),
        )

    def _node_with_age(self, p_age: Optional[float]) -> NodeScore:
        tl = _make_tracklet()
        ev = Evidence(p_age=p_age)
        return NodeScore(tracklet=tl, score=0.5, weight=1.0, evidence=ev)

    def test_both_none_no_penalty(self) -> None:
        ident = self._ident()
        penalty = ident._compute_age_inconsistency_penalty(
            self._node_with_age(None), self._node_with_age(None)
        )
        assert penalty == pytest.approx(0.0)

    def test_similar_ages_no_penalty(self) -> None:
        ident = self._ident()
        penalty = ident._compute_age_inconsistency_penalty(
            self._node_with_age(0.8), self._node_with_age(0.75)
        )
        assert penalty == pytest.approx(0.0)

    def test_large_difference_penalised(self) -> None:
        ident = self._ident()
        penalty = ident._compute_age_inconsistency_penalty(
            self._node_with_age(0.9), self._node_with_age(0.1)
        )
        assert penalty > 0.0

    def test_penalty_capped_at_one(self) -> None:
        ident = self._ident()
        penalty = ident._compute_age_inconsistency_penalty(
            self._node_with_age(1.0), self._node_with_age(0.0)
        )
        assert penalty <= 1.0

    def test_one_side_none_partial_penalty(self) -> None:
        ident = self._ident()
        # Confident child estimate vs unknown
        penalty = ident._compute_age_inconsistency_penalty(
            self._node_with_age(0.9), self._node_with_age(None)
        )
        # Should penalise because 0.9 is far from 0.5
        assert penalty > 0.0


# ---------------------------------------------------------------------------
# SingleChildIdentifier — _split_into_tracklets
# ---------------------------------------------------------------------------


class TestSplitIntoTracklets:
    def test_short_tracks_filtered(self) -> None:
        cfg = _cfg(min_track_frames=30)
        ident = SingleChildIdentifier(cfg, _ann())
        short = _make_track(start=0, end=5)   # 6 frames < 30
        tracklets = ident._split_into_tracklets([short])
        assert tracklets == []

    def test_long_enough_track_kept(self) -> None:
        cfg = _cfg(min_track_frames=30)
        ident = SingleChildIdentifier(cfg, _ann())
        long_ = _make_track(start=0, end=59)  # 60 frames ≥ 30
        tracklets = ident._split_into_tracklets([long_])
        assert len(tracklets) == 1
        assert tracklets[0].parent_id == long_.id

    def test_meta_copied(self) -> None:
        cfg = _cfg(min_track_frames=1)
        ident = SingleChildIdentifier(cfg, _ann())
        track = _make_track(start=0, end=10)
        track.meta["custom"] = "value"
        tracklets = ident._split_into_tracklets([track])
        assert tracklets[0].meta["custom"] == "value"

    def test_multiple_tracks(self) -> None:
        cfg = _cfg(min_track_frames=1)
        ident = SingleChildIdentifier(cfg, _ann())
        tracks = [_make_track(i, start=i * 100, end=i * 100 + 50) for i in range(3)]
        tracklets = ident._split_into_tracklets(tracks)
        assert len(tracklets) == 3


# ---------------------------------------------------------------------------
# _estimate_confidence
# ---------------------------------------------------------------------------


class TestEstimateConfidence:
    def _ident(self) -> SingleChildIdentifier:
        return SingleChildIdentifier(_cfg(), _ann())

    def _node(self, score: float) -> NodeScore:
        tl = _make_tracklet()
        return NodeScore(tracklet=tl, score=score, weight=score, evidence=Evidence())

    def test_empty_returns_zero(self) -> None:
        ident = self._ident()
        assert ident._estimate_confidence([]) == pytest.approx(0.0)

    def test_average_score(self) -> None:
        ident = self._ident()
        nodes = [self._node(0.4), self._node(0.6)]
        assert ident._estimate_confidence(nodes) == pytest.approx(0.5)

    def test_clamped_to_unit_interval(self) -> None:
        ident = self._ident()
        nodes = [self._node(1.5)]  # score > 1 clamped to 1
        conf = ident._estimate_confidence(nodes)
        assert conf == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# identify_single_child — end-to-end (no external I/O)
# ---------------------------------------------------------------------------


class TestIdentifySingleChildEndToEnd:
    """Smoke tests for the public API without real video/model dependencies."""

    def test_empty_tracks_returns_result(self) -> None:
        result = identify_single_child([], _ann(), _cfg())
        assert isinstance(result, ChildResult)
        assert result.segments == []

    def test_single_track_selected(self) -> None:
        child_kp = _make_kp()
        track = _make_track(start=0, end=59, fps=30.0, keypoints=[child_kp] * 5)
        cfg = _cfg(
            min_track_frames=1,
            enable_skeleton_ratios=True,
            enable_rigidity_detection=False,
        )
        result = identify_single_child([track], _ann(), cfg)
        assert isinstance(result, ChildResult)
        assert len(result.segments) == 1
        assert result.segments[0].parent_id == track.id

    def test_short_track_filtered_returns_empty(self) -> None:
        track = _make_track(start=0, end=5, fps=30.0)
        cfg = _cfg(min_track_frames=30)
        result = identify_single_child([track], _ann(), cfg)
        assert result.segments == []

    def test_two_tracks_best_selected(self) -> None:
        child_kp = _make_kp()
        adult_kp = _make_adult_kp()
        child_track = _make_track(
            track_id=1, start=0, end=599, fps=30.0, keypoints=[child_kp] * 10
        )
        adult_track = _make_track(
            track_id=2, start=0, end=59, fps=30.0, keypoints=[adult_kp] * 10
        )
        cfg = _cfg(
            min_track_frames=1,
            enable_skeleton_ratios=True,
            enable_rigidity_detection=False,
        )
        result = identify_single_child([child_track, adult_track], _ann(), cfg)
        # The child track has higher score AND much longer duration → must win
        assert 1 in result.child_track_id_sequence

    def test_confidence_in_unit_interval(self) -> None:
        child_kp = _make_kp()
        track = _make_track(start=0, end=59, fps=30.0, keypoints=[child_kp] * 5)
        cfg = _cfg(min_track_frames=1, enable_skeleton_ratios=True, enable_rigidity_detection=False)
        result = identify_single_child([track], _ann(), cfg)
        assert 0.0 <= result.confidence <= 1.0

    def test_diagnostics_populated(self) -> None:
        track = _make_track(start=0, end=59, fps=30.0)
        cfg = _cfg(min_track_frames=1)
        result = identify_single_child([track], _ann(), cfg)
        assert "nodes" in result.diagnostics
        assert "edges" in result.diagnostics
        assert "path_indices" in result.diagnostics

    def test_chained_tracks_merged(self) -> None:
        """Two adjacent same-ID segments within gap should be chained."""
        child_kp = _make_kp()
        t1 = _make_track(track_id=1, start=0, end=59, fps=30.0, keypoints=[child_kp] * 5)
        t2 = _make_track(track_id=2, start=70, end=129, fps=30.0, keypoints=[child_kp] * 5)
        cfg = _cfg(
            min_track_frames=1,
            enable_skeleton_ratios=True,
            enable_rigidity_detection=False,
            continuity_gap_seconds=2.0,
        )
        result = identify_single_child([t1, t2], _ann(), cfg)
        # Both segments should be in the selected path
        assert len(result.segments) == 2