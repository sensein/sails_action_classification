"""
Tests for sailsprep/tracking_pose_model_testing/face_mediapipe.py
Run: poetry run pytest src/tests/test_face_mediapipe.py -v
"""

import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Block module-level side effects BEFORE any import of face_mediapipe.
# The source file runs YOLO() + .to('cuda:0') at import time, which
# downloads weights and crashes on CPU-only machines.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_module_level_init():
    """
    Inject fakes for ultralytics.YOLO and mediapipe so the module-level
    code in face_mediapipe.py doesn't download weights or touch CUDA.
    """
    # --- fake YOLO ---
    fake_yolo_instance = MagicMock()
    fake_yolo_cls = MagicMock(return_value=fake_yolo_instance)

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = fake_yolo_cls
    sys.modules.setdefault("ultralytics", fake_ultralytics)

    # --- fake mediapipe (deep tree) ---
    def _make_mp():
        mp = types.ModuleType("mediapipe")
        solutions = types.ModuleType("mediapipe.solutions")
        drawing = MagicMock()
        holistic = MagicMock()
        holistic.FACEMESH_TESSELATION = MagicMock()
        holistic.FACEMESH_CONTOURS = MagicMock()
        solutions.drawing_utils = drawing
        solutions.holistic = holistic
        mp.solutions = solutions
        return mp

    fake_mp = _make_mp()
    sys.modules.setdefault("mediapipe", fake_mp)
    sys.modules.setdefault("mediapipe.solutions", fake_mp.solutions)
    sys.modules.setdefault("mediapipe.solutions.drawing_utils", fake_mp.solutions.drawing_utils)
    sys.modules.setdefault("mediapipe.solutions.holistic", fake_mp.solutions.holistic)

    # --- fake cv2 (avoid display deps) ---
    if "cv2" not in sys.modules:
        fake_cv2 = MagicMock()
        fake_cv2.COLOR_BGR2RGB = 4
        sys.modules["cv2"] = fake_cv2

    # Now safe to import — module-level YOLO() hits the mock
    import importlib
    if "sailsprep.tracking_pose_model_testing.face_mediapipe" in sys.modules:
        del sys.modules["sailsprep.tracking_pose_model_testing.face_mediapipe"]

    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_bbox(x1=100, y1=100, x2=300, y2=400, conf=0.9, cls=0):
    return np.array([x1, y1, x2, y2, conf, cls], dtype=np.float32)


def _import():
    """Import after mocks are in place."""
    import importlib
    mod = importlib.import_module(
        "sailsprep.tracking_pose_model_testing.face_mediapipe"
    )
    return mod


# ---------------------------------------------------------------------------
# crop_person
# ---------------------------------------------------------------------------

class TestCropPerson:
    def test_basic_crop_returns_correct_shape(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        bbox = make_bbox(100, 100, 300, 400)
        crop, (x1, y1, x2, y2) = mod.crop_person(frame, bbox, padding=0)
        assert crop.shape[0] == y2 - y1
        assert crop.shape[1] == x2 - x1

    def test_padding_applied(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        bbox = make_bbox(100, 100, 300, 400)
        _, (x1_np, y1_np, x2_np, y2_np) = mod.crop_person(frame, bbox, padding=0)
        _, (x1_p, y1_p, x2_p, y2_p) = mod.crop_person(frame, bbox, padding=20)
        assert x1_p <= x1_np
        assert y1_p <= y1_np
        assert x2_p >= x2_np
        assert y2_p >= y2_np

    def test_clamps_to_frame_boundaries(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame(480, 640)
        bbox = make_bbox(5, 5, 635, 475)
        _, (x1, y1, x2, y2) = mod.crop_person(frame, bbox, padding=50)
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 640 and y2 <= 480

    def test_zero_size_bbox_returns_empty(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        bbox = make_bbox(200, 200, 200, 200)
        crop, _ = mod.crop_person(frame, bbox, padding=0)
        assert crop.size == 0


# ---------------------------------------------------------------------------
# draw_face_mesh_only
# ---------------------------------------------------------------------------

class TestDrawFaceMeshOnly:
    def test_no_face_landmarks_no_draw_call(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = None
        # drawing mock is already in sys.modules
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.draw_face_mesh_only(frame, results, (50, 50, 300, 400))
        drawing_mock.draw_landmarks.assert_not_called()

    def test_with_face_landmarks_draws_twice(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        results = MagicMock()
        results.face_landmarks = MagicMock()
        drawing_mock = sys.modules["mediapipe"].solutions.drawing_utils
        drawing_mock.draw_landmarks.reset_mock()
        mod.draw_face_mesh_only(frame, results, (50, 50, 300, 400))
        assert drawing_mock.draw_landmarks.call_count == 2  # TESSELATION + CONTOURS


# ---------------------------------------------------------------------------
# process_frame_multi_person
# ---------------------------------------------------------------------------

class TestProcessFrameMultiPerson:
    def _yolo_result(self, boxes_data):
        result = MagicMock()
        if boxes_data is not None:
            result.boxes = MagicMock()
            result.boxes.data.cpu.return_value.numpy.return_value = boxes_data
        else:
            result.boxes = None
        return [result]

    def test_no_detections_returns_frame(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        yolo = MagicMock()
        yolo.predict.return_value = self._yolo_result(None)
        out = mod.process_frame_multi_person(frame, yolo, MagicMock())
        assert out.shape == frame.shape

    def test_single_person_calls_holistic_once(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        yolo = MagicMock()
        yolo.predict.return_value = self._yolo_result(np.array([make_bbox()]))
        holistic = MagicMock()
        holistic.process.return_value = MagicMock(face_landmarks=None)
        mod.process_frame_multi_person(frame, yolo, holistic)
        holistic.process.assert_called_once()

    def test_multi_person_calls_holistic_per_person(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        yolo = MagicMock()
        boxes = np.array([make_bbox(50, 50, 200, 400), make_bbox(300, 50, 500, 400)])
        yolo.predict.return_value = self._yolo_result(boxes)
        holistic = MagicMock()
        holistic.process.return_value = MagicMock(face_landmarks=None)
        mod.process_frame_multi_person(frame, yolo, holistic)
        assert holistic.process.call_count == 2

    def test_zero_area_bbox_skipped(self, _patch_module_level_init):
        mod = _import()
        frame = make_frame()
        yolo = MagicMock()
        yolo.predict.return_value = self._yolo_result(np.array([make_bbox(200, 200, 200, 200)]))
        holistic = MagicMock()
        mod.process_frame_multi_person(frame, yolo, holistic)
        holistic.process.assert_not_called()

    def test_yolo_filters_person_class_only(self, _patch_module_level_init):
        mod = _import()
        yolo = MagicMock()
        yolo.predict.return_value = self._yolo_result(None)
        mod.process_frame_multi_person(make_frame(), yolo, MagicMock())
        assert yolo.predict.call_args[1]["classes"] == [0]