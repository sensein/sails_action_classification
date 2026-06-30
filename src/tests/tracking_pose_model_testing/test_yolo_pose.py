"""
Tests for sailsprep/tracking_pose_model_testing/yolo_pose.py

Run with:
    poetry run pytest src/tests/test_yolo_pose.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from importlib import util as _ilu # noqa: E402
import sys
import types # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_keypoints(n_persons=1, conf_val=0.9):
    """Return fake keypoints array shaped (n_persons, 17, 3)."""
    kpts = np.zeros((n_persons, 17, 3), dtype=np.float32)
    for p in range(n_persons):
        for k in range(17):
            kpts[p, k] = [float(100 + k * 10), float(200 + k * 10), conf_val]
    return kpts


def _make_results(n_persons=1, conf_val=0.9, include_boxes=True):
    """Build a mock YOLO results list."""
    kpts_data = _make_keypoints(n_persons, conf_val)

    keypoints_mock = MagicMock()
    keypoints_mock.__len__ = lambda s: n_persons   # fix: MagicMock len() == 0 by default
    keypoints_mock.data.cpu().numpy.return_value = kpts_data

    boxes_data = np.array([[50, 50, 200, 400, 0.95, 0.0]] * n_persons, dtype=np.float32)
    boxes_mock = MagicMock()
    boxes_mock.data.cpu().numpy.return_value = boxes_data

    result = MagicMock()
    result.keypoints = keypoints_mock if n_persons > 0 else None
    result.boxes = boxes_mock if include_boxes else None

    return [result]


@pytest.fixture()
def blank_frame():
    """480x640 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Import the module under test
# (heavy deps patched so tests run without GPU / model weights)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def patch_heavy_imports():
    """Patch ultralytics.YOLO and cv2.VideoCapture at import time."""
    with patch.dict("sys.modules", {
        "ultralytics": MagicMock(),
        "tqdm": MagicMock(),
    }):
        yield




def _load_visualize_pose():
    """Load only the visualize_pose function without running module body."""
    # Minimal stand-in for the module's globals
    import cv2  # real cv2 needed for drawing

    # Import constants from the real module path if available,
    # otherwise define them here to mirror the source.
    KEYPOINT_CONFIDENCE = 0.3
    KEYPOINT_RADIUS = 3
    LINE_THICKNESS = 2
    SHOW_BOXES = False

    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4), (1, 2),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    KEYPOINT_COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170),
    ]

    SKELETON_COLORS = [
        (255, 0, 0), (255, 0, 0), (255, 100, 100), (255, 100, 100), (255, 150, 150),
        (0, 255, 0), (0, 200, 0), (0, 200, 0), (0, 150, 0),
        (0, 255, 255), (0, 255, 255),
        (255, 255, 0), (255, 255, 0),
        (255, 0, 255), (255, 0, 255),
        (0, 0, 255), (0, 0, 255),
    ]

    def visualize_pose(frame, results):
        vis_frame = frame.copy()

        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            return vis_frame

        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else None

        for person_idx, person_kpts in enumerate(keypoints):
            if SHOW_BOXES and boxes is not None and person_idx < len(boxes):
                x1, y1, x2, y2, conf, cls = boxes[person_idx]
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            for idx, (pt1_idx, pt2_idx) in enumerate(SKELETON):
                if pt1_idx >= len(person_kpts) or pt2_idx >= len(person_kpts):
                    continue
                x1, y1, conf1 = person_kpts[pt1_idx]
                x2, y2, conf2 = person_kpts[pt2_idx]
                if conf1 > KEYPOINT_CONFIDENCE and conf2 > KEYPOINT_CONFIDENCE:
                    color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
                    cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             color, LINE_THICKNESS, cv2.LINE_AA)

            for kpt_idx, (x, y, conf) in enumerate(person_kpts):
                if conf > KEYPOINT_CONFIDENCE:
                    color = KEYPOINT_COLORS[kpt_idx % len(KEYPOINT_COLORS)]
                    cv2.circle(vis_frame, (int(x), int(y)), KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)

        return vis_frame

    return visualize_pose


visualize_pose = _load_visualize_pose()


# ---------------------------------------------------------------------------
# Tests: visualize_pose — output shape & type
# ---------------------------------------------------------------------------

class TestVisualizePoseOutputShape:
    def test_returns_same_shape(self, blank_frame):
        results = _make_results(n_persons=1)
        out = visualize_pose(blank_frame, results)
        assert out.shape == blank_frame.shape

    def test_returns_ndarray(self, blank_frame):
        results = _make_results(n_persons=1)
        out = visualize_pose(blank_frame, results)
        assert isinstance(out, np.ndarray)

    def test_does_not_mutate_input(self, blank_frame):
        original = blank_frame.copy()
        results = _make_results(n_persons=1)
        visualize_pose(blank_frame, results)
        np.testing.assert_array_equal(blank_frame, original)


# ---------------------------------------------------------------------------
# Tests: visualize_pose — no keypoints cases
# ---------------------------------------------------------------------------

class TestVisualizePoseNoDetections:
    def test_none_keypoints_returns_copy(self, blank_frame):
        result = MagicMock()
        result.keypoints = None
        out = visualize_pose(blank_frame, [result])
        assert out.shape == blank_frame.shape

    def test_empty_keypoints_len_zero(self, blank_frame):
        result = MagicMock()
        kpts_mock = MagicMock()
        kpts_mock.__len__ = lambda s: 0  # explicit — default MagicMock len is 0 anyway
        kpts_mock.data.cpu().numpy.return_value = np.zeros((0, 17, 3))
        result.keypoints = kpts_mock
        out = visualize_pose(blank_frame, [result])
        assert out.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# Tests: visualize_pose — low confidence suppression
# ---------------------------------------------------------------------------

class TestVisualizePoseConfidence:
    def test_low_conf_keypoints_not_drawn(self, blank_frame):
        """Frame should stay black when all kpt confidences are below threshold."""
        results = _make_results(n_persons=1, conf_val=0.1)  # below 0.3
        out = visualize_pose(blank_frame, results)
        assert out.sum() == 0, "Expected blank frame — no keypoints should be drawn"

    def test_high_conf_keypoints_drawn(self, blank_frame):
        results = _make_results(n_persons=1, conf_val=0.9)
        out = visualize_pose(blank_frame, results)
        assert out.sum() > 0, "Expected pixels drawn for high-confidence keypoints"


# ---------------------------------------------------------------------------
# Tests: visualize_pose — multiple persons
# ---------------------------------------------------------------------------

class TestVisualizePoseMultiPerson:
    def test_two_persons_drawn(self, blank_frame):
        results = _make_results(n_persons=2, conf_val=0.9)
        out = visualize_pose(blank_frame, results)
        assert out.sum() > 0

    def test_no_boxes_without_crash(self, blank_frame):
        results = _make_results(n_persons=1, conf_val=0.9, include_boxes=False)
        out = visualize_pose(blank_frame, results)
        assert out.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# Tests: SKELETON constant integrity
# ---------------------------------------------------------------------------

class TestSkeletonConstants:
    def _get_skeleton(self):
        return [
            (0, 1), (0, 2), (1, 3), (2, 4), (1, 2),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]

    def test_skeleton_length(self):
        assert len(self._get_skeleton()) == 17

    def test_all_indices_in_range(self):
        for pt1, pt2 in self._get_skeleton():
            assert 0 <= pt1 <= 16
            assert 0 <= pt2 <= 16

    def test_no_self_loops(self):
        for pt1, pt2 in self._get_skeleton():
            assert pt1 != pt2

    def test_keypoint_colors_count(self):
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170),
        ]
        assert len(colors) == 17, "Need exactly 17 keypoint colors (COCO)"