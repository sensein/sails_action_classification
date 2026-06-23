"""Tests for tracker_clip.py — pure logic units, heavy deps mocked."""
from __future__ import annotations

import sys
import types
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub out every heavy import BEFORE tracker_clip is imported
# ---------------------------------------------------------------------------

def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# facenet_pytorch
fpt = _make_module("facenet_pytorch")
fpt.MTCNN = MagicMock()  # type: ignore[attr-defined]

# torch
torch_mod = _make_module("torch")
torch_mod.device = MagicMock(return_value="cpu")  # type: ignore[attr-defined]
torch_mod.no_grad = MagicMock()  # type: ignore[attr-defined]
torch_mod.cuda = MagicMock()  # type: ignore[attr-defined]
torch_mod.cuda.is_available = MagicMock(return_value=False)  # type: ignore[attr-defined]
torch_mod.cuda.empty_cache = MagicMock()  # type: ignore[attr-defined]
amp_mod = _make_module("torch.amp")
amp_mod.autocast = MagicMock()  # type: ignore[attr-defined]
torch_mod.amp = amp_mod  # type: ignore[attr-defined]

# torchvision
tv = _make_module("torchvision")
tv_t = _make_module("torchvision.transforms")
tv_t.Normalize = MagicMock()  # type: ignore[attr-defined]
tv_t.ToTensor = MagicMock()  # type: ignore[attr-defined]
tv.transforms = tv_t  # type: ignore[attr-defined]

# cv2
cv2_mod = _make_module("cv2")
cv2_mod.VideoCapture = MagicMock()  # type: ignore[attr-defined]
cv2_mod.cvtColor = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))  # type: ignore[attr-defined]
cv2_mod.resize = MagicMock(return_value=np.zeros((256, 128, 3), dtype=np.uint8))  # type: ignore[attr-defined]
cv2_mod.Laplacian = MagicMock(return_value=np.zeros((50, 50)))  # type: ignore[attr-defined]
cv2_mod.Canny = MagicMock(return_value=np.zeros((50, 50), dtype=np.uint8))  # type: ignore[attr-defined]
cv2_mod.rectangle = MagicMock()  # type: ignore[attr-defined]
cv2_mod.putText = MagicMock()  # type: ignore[attr-defined]
cv2_mod.circle = MagicMock()  # type: ignore[attr-defined]
cv2_mod.line = MagicMock()  # type: ignore[attr-defined]
cv2_mod.COLOR_BGR2GRAY = 6  # type: ignore[attr-defined]
cv2_mod.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
cv2_mod.CV_64F = 6  # type: ignore[attr-defined]
cv2_mod.INTER_LINEAR = 1  # type: ignore[attr-defined]
cv2_mod.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
cv2_mod.CAP_PROP_FRAME_WIDTH = 3  # type: ignore[attr-defined]
cv2_mod.CAP_PROP_FRAME_HEIGHT = 4  # type: ignore[attr-defined]
cv2_mod.CAP_PROP_FRAME_COUNT = 7  # type: ignore[attr-defined]
cv2_mod.FONT_HERSHEY_SIMPLEX = 0  # type: ignore[attr-defined]

# sailsprep sub-packages
sp = _make_module("sailsprep")
sp_fp = _make_module("sailsprep.feature_processing")
sp_fp_t = _make_module("sailsprep.feature_processing.tracker")
sp_fp_pt = _make_module("sailsprep.feature_processing.tracker.person_tracker")

# minimal TrackerConfig dataclass
from dataclasses import dataclass, field as dc_field

@dataclass
class _TrackerConfig:
    base_iou_threshold: float = 0.3
    base_motion_confidence: float = 0.3
    base_center_weight: float = 0.5
    max_lost_frames: int = 30
    confidence_decay_rate: float = 0.05
    max_jump_factor: float = 2.0

sp_fp_pt.TrackerConfig = _TrackerConfig  # type: ignore[attr-defined]
sp_fp_pt.CameraMotionCompensator = MagicMock()  # type: ignore[attr-defined]
sp_fp_pt.calculate_combined_similarity = MagicMock(return_value=0.9)  # type: ignore[attr-defined]
sp_fp_pt.calculate_scene_crowding = MagicMock(return_value=0.0)  # type: ignore[attr-defined]
sp_fp_pt.create_kalman_filter = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
sp_fp_pt.get_adaptive_thresholds = MagicMock(return_value=(0.3, 0.5, 0.3))  # type: ignore[attr-defined]
sp_fp_pt.is_spatially_plausible = MagicMock(return_value=True)  # type: ignore[attr-defined]
sp_fp_pt.predict_motion_with_camera_compensation = MagicMock(  # type: ignore[attr-defined]
    return_value=(np.array([10, 10, 60, 80]), 0.9)
)
sp_fp_pt.update_kalman_filter = MagicMock()  # type: ignore[attr-defined]

sp_fp_u = _make_module("sailsprep.feature_processing.utils")
sp_fp_cm = _make_module("sailsprep.feature_processing.utils.cache_manager")
sp_fp_cm.CacheManager = MagicMock()  # type: ignore[attr-defined]
sp_fp_te = _make_module("sailsprep.feature_processing.utils.tracking_exporter_new")
sp_fp_te.TrackingDataCollector = MagicMock()  # type: ignore[attr-defined]

sp_id = _make_module("sailsprep.id_tracking_model")
sp_id_u = _make_module("sailsprep.id_tracking_model.utils")
sp_id_uu = _make_module("sailsprep.id_tracking_model.utils.utils")
sp_id_uu.oks_nms = MagicMock(return_value=[0, 1])  # type: ignore[attr-defined]

sp_id_t = _make_module("sailsprep.id_tracking_model.tracker")
sp_id_tc = _make_module("sailsprep.id_tracking_model.tracker.clip")

# scipy / sklearn / tqdm
scipy_mod = _make_module("scipy")
scipy_opt = _make_module("scipy.optimize")
scipy_opt.linear_sum_assignment = MagicMock(  # type: ignore[attr-defined]
    return_value=(np.array([0]), np.array([0]))
)
scipy_mod.optimize = scipy_opt  # type: ignore[attr-defined]

skl = _make_module("sklearn")
skl_mp = _make_module("sklearn.metrics")
skl_mpw = _make_module("sklearn.metrics.pairwise")
skl_mpw.cosine_similarity = MagicMock(return_value=np.array([[0.9]]))  # type: ignore[attr-defined]
skl_mp.pairwise = skl_mpw  # type: ignore[attr-defined]
skl.metrics = skl_mp  # type: ignore[attr-defined]

tqdm_mod = _make_module("tqdm")
tqdm_mod.tqdm = MagicMock()  # type: ignore[attr-defined]

# Patch the CLIP-ReID path check so import doesn't raise
import os
os.makedirs("sailsprep/id_tracking_model/tracker/clip/CLIP-ReID", exist_ok=True)

# Now import the module under test
import importlib
import importlib.util
import pathlib

def _import_tracker_clip() -> types.ModuleType:
    pkg_name = "sailsprep.id_tracking_model.tracker.clip.tracker_clip"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    try:
        return importlib.import_module(pkg_name)
    except (ImportError, ModuleNotFoundError):
        pass
    candidates = [
        pathlib.Path(__file__).parent / "tracker_clip.py",
        pathlib.Path(__file__).parent.parent / "tracker_clip.py",
        pathlib.Path("src/sailsprep/id_tracking_model/tracker/clip/tracker_clip.py"),
        pathlib.Path("sailsprep/id_tracking_model/tracker/clip/tracker_clip.py"),
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("tracker_clip", cand)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["tracker_clip"] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
    raise ImportError("Cannot find tracker_clip.py")

tracker_clip = _import_tracker_clip()

# Re-export the classes we need
PoseResult = tracker_clip.PoseResult
create_pose_results_from_cache = tracker_clip.create_pose_results_from_cache
apply_oks_nms = tracker_clip.apply_oks_nms
filter_poses_by_keypoints = tracker_clip.filter_poses_by_keypoints
RegionExtractor = tracker_clip.RegionExtractor
DetectionValidator = tracker_clip.DetectionValidator
TrackingModule = tracker_clip.TrackingModule
VisualizationModule = tracker_clip.VisualizationModule
ProcessingConfig = tracker_clip.ProcessingConfig
VisualizationConfig = tracker_clip.VisualizationConfig
PipelineConfig = tracker_clip.PipelineConfig
create_batch_config = tracker_clip.create_batch_config
CLIPReIDConfig = tracker_clip.CLIPReIDConfig
FeatureConfig = tracker_clip.FeatureConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keypoints(score: float = 0.9) -> np.ndarray:
    """17 keypoints (x, y, score) in a plausible standing-person layout."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # head
    kpts[0] = [100, 50, score]   # nose
    kpts[1] = [95,  45, score]   # left_eye
    kpts[2] = [105, 45, score]   # right_eye
    kpts[3] = [90,  50, score]   # left_ear
    kpts[4] = [110, 50, score]   # right_ear
    # shoulders
    kpts[5] = [80,  100, score]  # left_shoulder
    kpts[6] = [120, 100, score]  # right_shoulder
    # elbows
    kpts[7] = [75,  140, score]  # left_elbow
    kpts[8] = [125, 140, score]  # right_elbow
    # wrists
    kpts[9]  = [70,  180, score] # left_wrist
    kpts[10] = [130, 180, score] # right_wrist
    # hips
    kpts[11] = [85,  200, score] # left_hip
    kpts[12] = [115, 200, score] # right_hip
    # knees
    kpts[13] = [85,  260, score] # left_knee
    kpts[14] = [115, 260, score] # right_knee
    # ankles
    kpts[15] = [85,  320, score] # left_ankle
    kpts[16] = [115, 320, score] # right_ankle
    return kpts


def _make_bbox4(x1=70, y1=40, x2=140, y2=330) -> np.ndarray:
    """4-element bbox [x1,y1,x2,y2] — used in PoseResult and cache (original code expects exactly 4)."""
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _make_bbox(x1=70, y1=40, x2=140, y2=330, score=0.95) -> np.ndarray:
    """5-element bbox [x1,y1,x2,y2,score] — used in detection dicts."""
    return np.array([x1, y1, x2, y2, score], dtype=np.float32)


def _make_pose(score: float = 0.9) -> PoseResult:
    return PoseResult(keypoints=_make_keypoints(score), bbox=_make_bbox4())


def _make_detection(
    bbox: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    face: np.ndarray | None = None,
    frame_idx: int = 0,
) -> dict[str, Any]:
    if bbox is None:
        bbox = _make_bbox()[:4]
    feat = np.random.rand(512).astype(np.float32)
    feat /= np.linalg.norm(feat)
    return {
        'bbox': bbox,
        'keypoints': _make_keypoints(),
        'confidence': 0.95,
        'pose_type': 'standing',
        'face_feature': face,
        'upper_feature': upper if upper is not None else feat.copy(),
        'lower_feature': lower if lower is not None else feat.copy(),
        'sufficient_keypoints': True,
        'frame_idx': frame_idx,
    }


# ---------------------------------------------------------------------------
# PoseResult
# ---------------------------------------------------------------------------

class TestPoseResult:
    def test_to_dict_keys(self) -> None:
        pose = _make_pose()
        d = pose.to_dict()
        assert set(d.keys()) == {"keypoints", "bbox", "metadata"}

    def test_metadata_default_empty(self) -> None:
        pose = _make_pose()
        assert pose.metadata == {}

    def test_stores_keypoints_and_bbox(self) -> None:
        kpts = _make_keypoints()
        bbox = _make_bbox4()
        pose = PoseResult(keypoints=kpts, bbox=bbox)
        np.testing.assert_array_equal(pose.keypoints, kpts)
        np.testing.assert_array_equal(pose.bbox, bbox)


# ---------------------------------------------------------------------------
# create_pose_results_from_cache
# ---------------------------------------------------------------------------

class TestCreatePoseResultsFromCache:
    def test_empty(self) -> None:
        assert create_pose_results_from_cache([]) == []

    def test_converts_dicts(self) -> None:
        kpts = _make_keypoints()
        bbox = _make_bbox4()
        results = create_pose_results_from_cache([{"keypoints": kpts, "bbox": bbox}])
        assert len(results) == 1
        assert isinstance(results[0], PoseResult)
        np.testing.assert_array_equal(results[0].keypoints, kpts)

    def test_multiple(self) -> None:
        items = [{"keypoints": _make_keypoints(), "bbox": _make_bbox4()} for _ in range(3)]
        results = create_pose_results_from_cache(items)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# apply_oks_nms
# ---------------------------------------------------------------------------

class TestApplyOksNms:
    def test_empty_input(self) -> None:
        assert apply_oks_nms([], 0.9, 0.3) == []

    def test_returns_subset(self) -> None:
        poses = [_make_pose() for _ in range(3)]
        # oks_nms mock returns [0, 1]
        result = apply_oks_nms(poses, 0.9, 0.3)
        assert len(result) == 2
        assert result[0] is poses[0]
        assert result[1] is poses[1]

    def test_single_pose_kept(self) -> None:
        sp_id_uu.oks_nms.return_value = [0]
        poses = [_make_pose()]
        result = apply_oks_nms(poses, 0.9, 0.3)
        assert len(result) == 1
        sp_id_uu.oks_nms.return_value = [0, 1]  # reset


# ---------------------------------------------------------------------------
# filter_poses_by_keypoints
# ---------------------------------------------------------------------------

class TestFilterPosesByKeypoints:
    def test_high_score_sufficient(self) -> None:
        pose = _make_pose(score=0.9)
        result = filter_poses_by_keypoints([pose], kpt_threshold=0.3)
        assert result[0].metadata['sufficient_keypoints'] is True

    def test_zero_score_insufficient(self) -> None:
        pose = _make_pose(score=0.0)
        result = filter_poses_by_keypoints([pose], kpt_threshold=0.3)
        assert result[0].metadata['sufficient_keypoints'] is False

    def test_returns_same_list(self) -> None:
        poses = [_make_pose(), _make_pose()]
        result = filter_poses_by_keypoints(poses, kpt_threshold=0.3)
        assert result is poses


# ---------------------------------------------------------------------------
# RegionExtractor
# ---------------------------------------------------------------------------

class TestRegionExtractor:
    def setup_method(self) -> None:
        self.frame = np.zeros((400, 200, 3), dtype=np.uint8)
        self.frame[40:330, 70:140] = 128  # fill person region
        self.kpts = _make_keypoints(score=0.9)
        self.bbox = _make_bbox4()

    def test_extract_face_region_returns_tuple(self) -> None:
        result = RegionExtractor.extract_face_region(self.frame, self.kpts, self.bbox)
        assert result is not None
        roi, coords = result
        assert roi.ndim == 3
        assert len(coords) == 4

    def test_extract_upper_body_returns_tuple(self) -> None:
        result = RegionExtractor.extract_upper_body_region(
            self.frame, self.kpts, self.bbox, "standing"
        )
        assert result is not None
        roi, coords = result
        assert roi.ndim == 3

    def test_extract_lower_body_returns_tuple(self) -> None:
        result = RegionExtractor.extract_lower_body_region(
            self.frame, self.kpts, self.bbox, "standing"
        )
        assert result is not None
        roi, coords = result
        assert roi.ndim == 3

    def test_lower_body_lying_returns_none(self) -> None:
        result = RegionExtractor.extract_lower_body_region(
            self.frame, self.kpts, self.bbox, "lying"
        )
        assert result is None

    def test_face_low_score_fallback(self) -> None:
        kpts_low = _make_keypoints(score=0.1)
        result = RegionExtractor.extract_face_region(self.frame, kpts_low, self.bbox)
        # fallback uses bbox upper region — may return None if too small, that's ok
        # just must not raise
        assert result is None or isinstance(result, tuple)


# ---------------------------------------------------------------------------
# DetectionValidator
# ---------------------------------------------------------------------------

class TestDetectionValidator:
    def setup_method(self) -> None:
        self.validator = DetectionValidator()
        self.frame = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
        self.bbox = np.array([70, 40, 140, 330], dtype=np.float32)

    def test_returns_true_before_min_frames(self) -> None:
        result = self.validator.is_real_person(1, self.bbox, self.frame, min_frames=10)
        assert result is True

    def test_update_motion_stores_history(self) -> None:
        self.validator.update_motion(1, 5.0)
        self.validator.update_motion(1, 3.0)
        assert len(self.validator.motion_history[1]) == 2

    def test_reset_track_clears_history(self) -> None:
        self.validator.update_motion(1, 5.0)
        self.validator.reset_track(1)
        assert 1 not in self.validator.motion_history

    def test_reset_nonexistent_track_ok(self) -> None:
        self.validator.reset_track(999)  # should not raise

    def test_tiny_roi_returns_false(self) -> None:
        tiny_bbox = np.array([0, 0, 5, 5], dtype=np.float32)
        result = self.validator.is_real_person(2, tiny_bbox, self.frame, min_frames=1)
        assert result is False


# ---------------------------------------------------------------------------
# TrackingModule — geometry helpers
# ---------------------------------------------------------------------------

class TestTrackingModuleGeometry:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())

    def test_compute_bbox_geometry_area(self) -> None:
        bbox = np.array([0, 0, 50, 100], dtype=np.float32)
        area, aspect = TrackingModule._compute_bbox_geometry(bbox)
        assert area == pytest.approx(50 * 100)
        assert aspect == pytest.approx(50 / 100)

    def test_compute_bbox_geometry_square(self) -> None:
        bbox = np.array([10, 10, 60, 60], dtype=np.float32)
        area, aspect = TrackingModule._compute_bbox_geometry(bbox)
        assert area == pytest.approx(50 * 50)
        assert aspect == pytest.approx(1.0)

    def test_is_geometry_consistent_within_bounds(self) -> None:
        assert self.tm._is_geometry_consistent((1000.0, 0.5), (1000.0, 0.5)) is True

    def test_is_geometry_consistent_area_too_big(self) -> None:
        assert self.tm._is_geometry_consistent((5000.0, 0.5), (1000.0, 0.5)) is False

    def test_is_geometry_consistent_area_too_small(self) -> None:
        assert self.tm._is_geometry_consistent((100.0, 0.5), (1000.0, 0.5)) is False

    def test_ensure_track_geometry_creates(self) -> None:
        track: dict[str, Any] = {}
        self.tm._ensure_track_geometry(track, np.array([0, 0, 50, 100]))
        assert 'geometry' in track
        assert 'area_ema' in track['geometry']
        assert 'aspect_ema' in track['geometry']

    def test_ensure_track_geometry_idempotent(self) -> None:
        track: dict[str, Any] = {}
        self.tm._ensure_track_geometry(track, np.array([0, 0, 50, 100]))
        original_area = track['geometry']['area_ema']
        self.tm._ensure_track_geometry(track, np.array([0, 0, 10, 20]))
        assert track['geometry']['area_ema'] == original_area  # not overwritten

    def test_update_track_geometry_clean(self) -> None:
        track: dict[str, Any] = {}
        det = _make_detection()
        self.tm._update_track_geometry(track, det, clean=True)
        assert track['geometry']['status'] == 'stable'
        assert track['geometry']['suspect_frames'] == 0

    def test_update_track_geometry_dirty(self) -> None:
        track: dict[str, Any] = {}
        det = _make_detection()
        self.tm._update_track_geometry(track, det, clean=False)
        assert track['geometry']['status'] == 'suspect'
        assert track['geometry']['suspect_frames'] == 1


# ---------------------------------------------------------------------------
# TrackingModule — similarity
# ---------------------------------------------------------------------------

class TestTrackingModuleSimilarity:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())

    def _feat(self) -> np.ndarray:
        v = np.random.rand(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_identical_features_high_similarity(self) -> None:
        feat = self._feat()
        det = _make_detection(upper=feat.copy(), lower=feat.copy())
        profile: dict[str, Any] = {
            'upper_feature': feat.copy(),
            'lower_feature': feat.copy(),
            'face_feature': None,
        }
        sim, match_type = self.tm._compute_person_similarity(det, profile)
        assert sim > 0.5
        assert "upper" in match_type

    def test_no_features_returns_zero(self) -> None:
        det = _make_detection(upper=None, lower=None, face=None)
        profile: dict[str, Any] = {
            'upper_feature': None,
            'lower_feature': None,
            'face_feature': None,
        }
        sim, match_type = self.tm._compute_person_similarity(det, profile)
        assert sim == 0.0
        assert match_type == "none"

    def test_compute_feature_similarity_returns_float(self) -> None:
        # cosine_similarity is mocked to return 0.9 — verify result is float >= 0
        feat = self._feat()
        sim = self.tm._compute_feature_similarity(feat, feat)
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_compute_feature_similarity_clamps_negative(self) -> None:
        # Mock returns negative value — _compute_feature_similarity clamps to 0
        skl_mpw.cosine_similarity.return_value = np.array([[-0.5]])
        feat = self._feat()
        sim = self.tm._compute_feature_similarity(feat, feat)
        assert sim == pytest.approx(0.0)
        skl_mpw.cosine_similarity.return_value = np.array([[0.9]])  # restore


# ---------------------------------------------------------------------------
# TrackingModule — candidate lifecycle
# ---------------------------------------------------------------------------

class TestCandidateLifecycle:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())
        self.tm.camera_compensator = MagicMock()
        self.tm.camera_compensator.estimate_camera_motion.return_value = (0.0, 0.0)

    def test_create_candidate_returns_negative_id(self) -> None:
        det = _make_detection()
        cid = self.tm._create_candidate_track(det)
        assert cid is not None
        assert cid < 0

    def test_create_candidate_stored(self) -> None:
        det = _make_detection()
        cid = self.tm._create_candidate_track(det)
        assert cid in self.tm.candidate_tracks

    def test_update_candidate_increments_confirmation(self) -> None:
        det = _make_detection()
        cid = self.tm._create_candidate_track(det)
        assert cid is not None
        self.tm._update_candidate_track(_make_detection(), cid)
        assert self.tm.candidate_tracks[cid]['confirmation_count'] == 2

    def test_promote_candidate_after_3_confirmations(self) -> None:
        det = _make_detection()
        cid = self.tm._create_candidate_track(det)
        assert cid is not None
        track = self.tm.candidate_tracks[cid]
        track['confirmation_count'] = 3
        self.tm._promote_confirmed_candidates()
        assert cid not in self.tm.candidate_tracks
        # permanent ID should be in active_tracks
        assert len(self.tm.active_tracks) == 1

    def test_cleanup_removes_old_failed_candidates(self) -> None:
        det = _make_detection()
        cid = self.tm._create_candidate_track(det)
        assert cid is not None
        # Simulate old age
        self.tm.candidate_tracks[cid]['lost_frames'] = 15
        self.tm._cleanup_failed_candidates()
        assert cid not in self.tm.candidate_tracks

    def test_max_tracks_prevents_new_candidate(self) -> None:
        cfg = ProcessingConfig()
        cfg.max_tracks = 1
        tm = TrackingModule(cfg)
        # Fill up person_profiles to hit max
        tm.person_profiles[1] = {}
        result = tm._create_candidate_track(_make_detection())
        assert result is None


# ---------------------------------------------------------------------------
# TrackingModule — temporal overlap
# ---------------------------------------------------------------------------

class TestTemporalOverlap:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())

    def _track_with_frames(self, frame_indices: list[int]) -> dict[str, Any]:
        dets = [_make_detection(frame_idx=i) for i in frame_indices]
        return {
            'created_frame': frame_indices[0],
            'detections': deque(dets, maxlen=100),
        }

    def test_overlapping_tracks(self) -> None:
        t1 = self._track_with_frames([1, 2, 3])
        t2 = self._track_with_frames([3, 4, 5])
        assert self.tm._check_temporal_overlap(t1, t2) is True

    def test_non_overlapping_tracks(self) -> None:
        t1 = self._track_with_frames([1, 2, 3])
        t2 = self._track_with_frames([4, 5, 6])
        assert self.tm._check_temporal_overlap(t1, t2) is False


# ---------------------------------------------------------------------------
# TrackingModule — person profile update
# ---------------------------------------------------------------------------

class TestPersonProfileUpdate:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())
        self.tm.frame_count = 5

    def test_new_feature_added_to_profile(self) -> None:
        feat = np.random.rand(512).astype(np.float32)
        feat /= np.linalg.norm(feat)
        profile: dict[str, Any] = {
            'person_id': 1,
            'upper_feature': None,
            'lower_feature': None,
            'face_feature': None,
        }
        det = _make_detection(upper=feat.copy())
        self.tm._update_person_profile(profile, det)
        assert profile['upper_feature'] is not None
        np.testing.assert_array_almost_equal(profile['upper_feature'], feat)

    def test_existing_feature_ema_updated(self) -> None:
        old_feat = np.ones(512, dtype=np.float32)
        old_feat /= np.linalg.norm(old_feat)
        new_feat = np.zeros(512, dtype=np.float32)
        new_feat[0] = 1.0

        profile: dict[str, Any] = {
            'person_id': 1,
            'upper_feature': old_feat.copy(),
            'lower_feature': None,
            'face_feature': None,
        }
        det = _make_detection(upper=new_feat.copy())
        self.tm._update_person_profile(profile, det)
        # Should not be identical to old or new — it's a blend
        assert not np.allclose(profile['upper_feature'], old_feat)
        assert not np.allclose(profile['upper_feature'], new_feat)


# ---------------------------------------------------------------------------
# TrackingModule — lost tracks handling
# ---------------------------------------------------------------------------

class TestLostTracksHandling:
    def setup_method(self) -> None:
        self.tm = TrackingModule(ProcessingConfig())
        self.tm.frame_count = 10

    def _make_active_track(self, track_id: int, last_seen: int, lost_frames: int, n_dets: int = 15) -> None:
        dets = deque([_make_detection() for _ in range(n_dets)], maxlen=100)
        track: dict[str, Any] = {
            'track_id': track_id,
            'kalman': MagicMock(),
            'detections': dets,
            'last_seen': last_seen,
            'created_frame': 0,
            'lost_frames': lost_frames,
            'missed_updates': 0,
            'match_type': 'motion',
        }
        self.tm._ensure_track_geometry(track, _make_bbox4())
        self.tm.active_tracks[track_id] = track

    def test_track_moved_to_lost_after_max_lost_frames(self) -> None:
        cfg = ProcessingConfig()
        self.tm.tracker_config = cfg.tracker_config
        self.tm.tracker_config.max_lost_frames = 5
        self._make_active_track(1, last_seen=4, lost_frames=6, n_dets=15)
        self.tm._handle_lost_tracks_with_candidates(final_matches={})
        assert 1 not in self.tm.active_tracks
        assert 1 in self.tm.lost_tracks

    def test_short_track_not_moved_to_lost(self) -> None:
        self.tm.tracker_config.max_lost_frames = 5
        self._make_active_track(2, last_seen=4, lost_frames=6, n_dets=5)
        self.tm._handle_lost_tracks_with_candidates(final_matches={})
        assert 2 not in self.tm.active_tracks
        assert 2 not in self.tm.lost_tracks  # too short — discarded


# ---------------------------------------------------------------------------
# DataConfig / PipelineConfig
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_pipeline_config_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.frame_limit == 0
        assert cfg.processing.kpt_threshold == pytest.approx(0.3)
        assert cfg.features.feature_update_interval == 10

    def test_create_batch_config(self) -> None:
        cfg = create_batch_config("/tmp/out")
        assert cfg.export.output_path == "/tmp/out"
        assert cfg.features.no_deepface is True
        assert cfg.processing.combined_reid_threshold == pytest.approx(0.8)

    def test_processing_config_max_tracks_default(self) -> None:
        cfg = ProcessingConfig()
        assert cfg.max_tracks == 0

    def test_clipreid_config_defaults(self) -> None:
        cfg = CLIPReIDConfig()
        assert cfg.num_classes == 1041
        assert cfg.camera_num == 15


# ---------------------------------------------------------------------------
# VisualizationModule
# ---------------------------------------------------------------------------

class TestVisualizationModule:
    def setup_method(self) -> None:
        cfg = VisualizationConfig()
        self.vm = VisualizationModule(cfg, kpt_threshold=0.3)
        self.frame = np.zeros((400, 200, 3), dtype=np.uint8)

    def test_disabled_returns_frame_unchanged(self) -> None:
        cfg = VisualizationConfig(enable_visualization=False)
        vm = VisualizationModule(cfg)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        result = vm.draw_tracking_results(frame, [], {}, {})
        np.testing.assert_array_equal(result, frame)

    def test_returns_ndarray(self) -> None:
        pose = _make_pose()
        result = self.vm.draw_tracking_results(self.frame, [pose], {}, {})
        assert isinstance(result, np.ndarray)

    def test_out_of_bounds_det_idx_skipped(self) -> None:
        pose = _make_pose()
        # det_idx=5 is out of bounds for 1-pose list
        assignments = {5: 1}
        active = {1: {'match_type': 'motion', 'app_match_type': None}}
        result = self.vm.draw_tracking_results(self.frame, [pose], assignments, active)
        assert result is not None

    def test_candidate_track_negative_id(self) -> None:
        pose = _make_pose()
        assignments = {0: -1}
        result = self.vm.draw_tracking_results(self.frame, [pose], assignments, {})
        assert isinstance(result, np.ndarray)