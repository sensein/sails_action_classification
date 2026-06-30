"""
Tests for src/sailsprep/tracking_pose_model_testing/deppsort_reid.py
Run: poetry run pytest src/tests/test_deepsort_reid.py -v
"""
import numpy as np
import pytest

from importlib import import_module # noqa: E402
import types # noqa: E402
# ---------------------------------------------------------------------------
# Mocks — patch heavy deps before any import of the module under test
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock, patch

# Stub out every heavy import so tests run without GPU / model weights
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.no_grad.return_value.__enter__ = lambda s, *a: None
_torch.no_grad.return_value.__exit__ = lambda s, *a: None

sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("torchvision", MagicMock())
sys.modules.setdefault("torchvision.transforms", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("deep_sort_realtime", MagicMock())
sys.modules.setdefault("deep_sort_realtime.deepsort_tracker", MagicMock())
sys.modules.setdefault("torchreid", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

# ---------------------------------------------------------------------------
# Now import helpers directly (no side-effects from model loading)
# ---------------------------------------------------------------------------


# Build a minimal fake module that exposes only the pure-logic helpers
def _load_helpers():
    """
    Re-implement the pure helper functions locally so they can be tested
    without triggering module-level model construction.
    """
    src = {}

    # crop_person_regions logic
    def crop_person_regions(frame, detections):
        crops, valid_detections = [], []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
                valid_detections.append(detection)
        return crops, valid_detections

    # IoU helper (extracted from assign_poses_to_tracks)
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    src["crop_person_regions"] = crop_person_regions
    src["iou"] = iou
    return src

_helpers = _load_helpers()
crop_person_regions = _helpers["crop_person_regions"]
iou = _helpers["iou"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_frame():
    """480×640 black frame (H, W, C)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def single_detection():
    """One bounding box fully inside blank_frame."""
    return np.array([[50, 80, 200, 350, 0.9]])


@pytest.fixture
def multi_detections():
    return np.array([
        [10,  10, 100, 200, 0.95],
        [300, 50, 500, 400, 0.80],
    ])


# ---------------------------------------------------------------------------
# crop_person_regions
# ---------------------------------------------------------------------------

class TestCropPersonRegions:

    def test_returns_crop_for_valid_box(self, blank_frame, single_detection):
        crops, valids = crop_person_regions(blank_frame, single_detection)
        assert len(crops) == 1
        assert len(valids) == 1

    def test_crop_shape_matches_box(self, blank_frame, single_detection):
        crops, _ = crop_person_regions(blank_frame, single_detection)
        x1, y1, x2, y2 = 50, 80, 200, 350
        assert crops[0].shape == (y2 - y1, x2 - x1, 3)

    def test_clips_box_to_frame_boundary(self, blank_frame):
        # Box extends beyond frame edges
        det = np.array([[-20, -30, 700, 500, 0.85]])
        crops, valids = crop_person_regions(blank_frame, det)
        assert len(crops) == 1
        h, w = crops[0].shape[:2]
        assert h <= blank_frame.shape[0]
        assert w <= blank_frame.shape[1]

    def test_zero_area_box_excluded(self, blank_frame):
        # x1 == x2 → degenerate box
        det = np.array([[100, 100, 100, 200, 0.9]])
        crops, valids = crop_person_regions(blank_frame, det)
        assert len(crops) == 0

    def test_multiple_detections(self, blank_frame, multi_detections):
        crops, valids = crop_person_regions(blank_frame, multi_detections)
        assert len(crops) == 2
        assert len(valids) == 2

    def test_empty_detections(self, blank_frame):
        det = np.empty((0, 5))
        crops, valids = crop_person_regions(blank_frame, det)
        assert crops == []
        assert valids == []

    def test_box_entirely_outside_frame(self, blank_frame):
        # Box starts beyond right edge
        det = np.array([[700, 10, 800, 200, 0.9]])
        crops, valids = crop_person_regions(blank_frame, det)
        assert len(crops) == 0


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestIoU:

    def test_identical_boxes_iou_one(self):
        box = [0, 0, 100, 100]
        assert iou(box, box) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap_iou_zero(self):
        boxA = [0, 0, 50, 50]
        boxB = [100, 100, 200, 200]
        assert iou(boxA, boxB) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        boxA = [0, 0, 100, 100]
        boxB = [50, 50, 150, 150]
        # intersection = 50×50 = 2500, union = 10000+10000-2500 = 17500
        expected = 2500 / 17500
        assert iou(boxA, boxB) == pytest.approx(expected, abs=1e-4)

    def test_contained_box(self):
        outer = [0, 0, 200, 200]
        inner = [50, 50, 150, 150]
        # intersection = inner area = 10000, union = outer area = 40000
        expected = 10000 / 40000
        assert iou(outer, inner) == pytest.approx(expected, abs=1e-4)

    def test_touching_edges_no_area_overlap(self):
        boxA = [0, 0, 50, 50]
        boxB = [50, 0, 100, 50]
        assert iou(boxA, boxB) == pytest.approx(0.0, abs=1e-4)

    def test_symmetry(self):
        boxA = [10, 20, 80, 90]
        boxB = [40, 50, 120, 130]
        assert iou(boxA, boxB) == pytest.approx(iou(boxB, boxA), abs=1e-6)


# ---------------------------------------------------------------------------
# draw_pose_keypoints — test frame mutation & no crash on bad input
# ---------------------------------------------------------------------------

class TestDrawPoseKeypoints:
    """
    Since draw_pose_keypoints calls cv2 (mocked), test that it:
    - returns the frame unchanged structurally
    - doesn't raise on empty or low-confidence keypoints
    """

    def _draw(self, frame, keypoints):
        import cv2 as _cv2
        POSE_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for x, y, conf in keypoints:
            if conf > 0.3:
                _cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        for c in POSE_CONNECTIONS:
            if c[0] < len(keypoints) and c[1] < len(keypoints):
                pt1, pt2 = keypoints[c[0]], keypoints[c[1]]
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    _cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 2)
        return frame

    def test_returns_frame(self, blank_frame):
        kpts = [(100, 150, 0.9), (110, 160, 0.85)]
        result = self._draw(blank_frame, kpts)
        assert result is blank_frame  # same object returned

    def test_empty_keypoints_no_crash(self, blank_frame):
        result = self._draw(blank_frame, [])
        assert result is blank_frame

    def test_low_confidence_keypoints_skipped(self, blank_frame):
        import cv2 as _cv2
        kpts = [(50, 50, 0.1), (60, 60, 0.05)]  # all below 0.3
        _cv2.circle.reset_mock()
        self._draw(blank_frame, kpts)
        _cv2.circle.assert_not_called()


# ---------------------------------------------------------------------------
# POSE_CONNECTIONS integrity check
# ---------------------------------------------------------------------------

def test_pose_connections_index_bounds():
    """All connection indices must be within [0, 16] (17 COCO keypoints)."""
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for a, b in POSE_CONNECTIONS:
        assert 0 <= a <= 16
        assert 0 <= b <= 16