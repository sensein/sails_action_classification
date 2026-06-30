"""
Tests for sailsprep/tracking_pose_model_testing/mediapipe_holistic.py
Run: poetry run pytest src/tests/test_mediapipe_holistic.py -v
"""

import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Block module-level side effects BEFORE any import of mediapipe_holistic.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_module_level_init():
    """
    Inject fakes for mediapipe and cv2 so module-level code doesn't
    touch real models or display dependencies.
    """
    # --- fake mediapipe ---
    def _make_mp():
        mp = types.ModuleType("mediapipe")
        solutions = types.ModuleType("mediapipe.solutions")
        drawing = MagicMock()
        drawing_styles = MagicMock()
        holistic = MagicMock()
        holistic.FACEMESH_TESSELATION = MagicMock()
        holistic.FACEMESH_CONTOURS = MagicMock()
        holistic.POSE_CONNECTIONS = MagicMock()
        holistic.HAND_CONNECTIONS = MagicMock()
        solutions.drawing_utils = drawing
        solutions.drawing_styles = drawing_styles
        solutions.holistic = holistic
        mp.solutions = solutions
        return mp

    fake_mp = _make_mp()
    sys.modules.setdefault("mediapipe", fake_mp)
    sys.modules.setdefault("mediapipe.solutions", fake_mp.solutions)
    sys.modules.setdefault("mediapipe.solutions.drawing_utils", fake_mp.solutions.drawing_utils)
    sys.modules.setdefault("mediapipe.solutions.drawing_styles", fake_mp.solutions.drawing_styles)
    sys.modules.setdefault("mediapipe.solutions.holistic", fake_mp.solutions.holistic)

    # --- fake cv2 ---
    if "cv2" not in sys.modules:
        fake_cv2 = MagicMock()
        fake_cv2.COLOR_BGR2RGB = 4
        sys.modules["cv2"] = fake_cv2

    # Clear any prior import so mocks take effect
    if "sailsprep.tracking_pose_model_testing.mediapipe_holistic" in sys.modules:
        del sys.modules["sailsprep.tracking_pose_model_testing.mediapipe_holistic"]

    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _import():
    import importlib
    return importlib.import_module(
        "sailsprep.tracking_pose_model_testing.mediapipe_holistic"
    )


# ---------------------------------------------------------------------------
# visualize_holistic
# ---------------------------------------------------------------------------

class TestVisualizeHolistic:
    def test_returns_same_shape_as_input(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = None
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        out = mod.visualize_holistic(frame, results)
        assert out.shape == frame.shape

    def test_no_landmarks_no_draw_call(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = None
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.visualize_holistic(frame, results)
        drawing_mock.draw_landmarks.assert_not_called()

    def test_face_landmarks_draws_twice(self, _patch_module_level_init):
        """face draws tessellation + contours = 2 calls."""
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = MagicMock()
        results.pose_landmarks = None
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.visualize_holistic(frame, results)
        assert drawing_mock.draw_landmarks.call_count == 2

    def test_pose_landmarks_draws_once(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = MagicMock()
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.visualize_holistic(frame, results)
        assert drawing_mock.draw_landmarks.call_count == 1

    def test_both_hands_draws_twice(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = None
        results.left_hand_landmarks = MagicMock()
        results.right_hand_landmarks = MagicMock()
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.visualize_holistic(frame, results)
        assert drawing_mock.draw_landmarks.call_count == 2

    def test_all_landmarks_draws_five_times(self, _patch_module_level_init):
        """face×2 + pose×1 + left_hand×1 + right_hand×1 = 5."""
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = MagicMock()
        results.pose_landmarks = MagicMock()
        results.left_hand_landmarks = MagicMock()
        results.right_hand_landmarks = MagicMock()
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.visualize_holistic(frame, results)
        assert drawing_mock.draw_landmarks.call_count == 5

    def test_output_is_copy_not_same_object(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = None
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        out = mod.visualize_holistic(frame, results)
        assert out is not frame

    def test_show_boxes_false_no_rectangle(self, _patch_module_level_init):
        mod = _import()
        mod.SHOW_BOXES = False
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = MagicMock()
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None
        cv2_mock = sys.modules["cv2"]
        cv2_mock.rectangle.reset_mock()
        mod.visualize_holistic(frame, results)
        cv2_mock.rectangle.assert_not_called()

    def test_show_boxes_true_with_visible_landmarks_draws_rectangle(self, _patch_module_level_init):
        mod = _import()
        mod.SHOW_BOXES = True
        frame = make_frame(480, 640)

        # Build fake pose_landmarks with 2 visible landmarks
        lm1 = MagicMock()
        lm1.x, lm1.y, lm1.visibility = 0.3, 0.4, 0.9
        lm2 = MagicMock()
        lm2.x, lm2.y, lm2.visibility = 0.6, 0.7, 0.9

        results = MagicMock()
        results.face_landmarks = None
        results.pose_landmarks = MagicMock()
        results.pose_landmarks.landmark = [lm1, lm2]
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None

        cv2_mock = sys.modules["cv2"]
        cv2_mock.rectangle.reset_mock()
        mod.visualize_holistic(frame, results)
        cv2_mock.rectangle.assert_called_once()

        # restore
        mod.SHOW_BOXES = False


# ---------------------------------------------------------------------------
# consume_stderr
# ---------------------------------------------------------------------------

class TestConsumeStderr:
    def test_consumes_without_error(self, _patch_module_level_init):
        import subprocess
        mod = _import()
        proc = MagicMock(spec=subprocess.Popen)
        proc.stderr = iter([b"line1\n", b"line2\n"])
        mod.consume_stderr(proc)  # should not raise

    def test_none_stderr_no_crash(self, _patch_module_level_init):
        import subprocess
        mod = _import()
        proc = MagicMock(spec=subprocess.Popen)
        proc.stderr = None
        mod.consume_stderr(proc)  # should not raise