"""
Unit tests for sailsprep/tracking_pose_model_testing/bytetrack.py
Run: poetry run pytest src/tests/test_bytetrack.py -m unit
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------
from sailsprep.tracking_pose_model_testing.bytetrack import draw_pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a black BGR frame."""
    import numpy as np
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_keypoints(n: int = 17, conf: float = 0.9) -> list[list[float]]:
    """Return `n` keypoints with given confidence, scattered across frame."""
    rng = np.random.default_rng(42)
    pts = rng.integers(50, 400, size=(n, 2)).tolist()
    return [[float(x), float(y), conf] for x, y in pts]


def _low_conf_keypoints(n: int = 17) -> list[list[float]]:
    return [[100.0, 100.0, 0.1]] * n


# ---------------------------------------------------------------------------
# draw_pose — basic smoke tests (unit, no GPU, no YOLO)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDrawPose:

    def test_returns_none(self):
        """draw_pose is a side-effect fn; must return None."""
        img = _blank_frame()
        result = draw_pose(img, _make_keypoints(), track_id=1)
        assert result is None

    def test_modifies_frame_high_conf(self):
        """High-confidence keypoints → at least one pixel changed."""
        img = _blank_frame()
        original = img.copy()
        draw_pose(img, _make_keypoints(conf=0.9), track_id=None)
        assert not np.array_equal(img, original), "Frame should be modified when conf > 0.5"

    def test_no_modification_low_conf(self):
        """Low-confidence keypoints → frame unchanged."""
        img = _blank_frame()
        original = img.copy()
        draw_pose(img, _low_conf_keypoints(), track_id=None)
        assert np.array_equal(img, original), "Frame must not change when all conf <= 0.5"

    def test_with_track_id(self):
        """Track ID path must not raise; frame gets modified."""
        img = _blank_frame()
        original = img.copy()
        draw_pose(img, _make_keypoints(conf=0.9), track_id=42)
        assert not np.array_equal(img, original)

    def test_empty_keypoints(self):
        """Empty keypoints list → no crash, frame unchanged."""
        img = _blank_frame()
        original = img.copy()
        draw_pose(img, [], track_id=None)
        assert np.array_equal(img, original)

    def test_single_keypoint(self):
        """Single keypoint above threshold → at least one green pixel drawn."""
        img = _blank_frame()
        draw_pose(img, [[200.0, 200.0, 0.9]], track_id=None)
        # Green channel should have been touched at (200, 200)
        assert img[200, 200, 1] > 0

    def test_connections_drawn(self):
        """Two connected keypoints, both high conf → frame changes more than single point."""
        img_two = _blank_frame()
        img_one = _blank_frame()

        kpts_two = _low_conf_keypoints()   # start with all low
        # keypoints 0 and 1 are connected; set both high
        kpts_two[0] = [100.0, 100.0, 0.9]
        kpts_two[1] = [200.0, 200.0, 0.9]

        kpts_one = _low_conf_keypoints()
        kpts_one[0] = [100.0, 100.0, 0.9]  # only kpt 0 high

        draw_pose(img_two, kpts_two, track_id=None)
        draw_pose(img_one, kpts_one, track_id=None)

        changed_two = np.sum(img_two != _blank_frame())
        changed_one = np.sum(img_one != _blank_frame())
        assert changed_two > changed_one, "Line between two high-conf pts should add more pixels"

    @pytest.mark.parametrize("track_id", [0, 1, 99, 1000])
    def test_various_track_ids(self, track_id):
        """Various integer track IDs must not raise."""
        img = _blank_frame()
        draw_pose(img, _make_keypoints(conf=0.9), track_id=track_id)

    def test_keypoints_outside_frame(self):
        """Keypoints outside frame bounds must not crash (cv2 clips)."""
        img = _blank_frame(480, 640)
        kpts = [[-100.0, -100.0, 0.9], [9999.0, 9999.0, 0.9]] + _low_conf_keypoints(15)
        draw_pose(img, kpts, track_id=None)  # should not raise