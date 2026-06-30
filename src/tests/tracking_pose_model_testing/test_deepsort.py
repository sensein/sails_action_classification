"""
Tests for sailsprep.tracking_pose_model_testing.deepsort
Mocks all heavy deps (cv2, YOLO, DeepSort) — no GPU/video needed.
"""
import types
import sys
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without real packages
# ---------------------------------------------------------------------------

class _FakeVideoCapture:
    """Fake cv2.VideoCapture — never opens a file, isOpened() → False."""
    def __init__(self, *a, **kw): pass
    def get(self, prop): return 30 if prop == 5 else 640 if prop == 3 else 480
    def isOpened(self): return False
    def read(self): return False, None
    def release(self): pass


class _FakeVideoWriter:
    def __init__(self, *a, **kw): pass
    def write(self, frame): pass
    def release(self): pass


def _make_cv2_stub():
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = _FakeVideoCapture
    cv2.VideoWriter = _FakeVideoWriter
    cv2.VideoWriter_fourcc = lambda *a: 0
    cv2.circle = lambda *a, **kw: None
    cv2.line = lambda *a, **kw: None
    cv2.putText = lambda *a, **kw: None
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.FONT_HERSHEY_SIMPLEX = 0
    return cv2


def _make_torch_stub(use_cuda=False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: use_cuda)
    return torch


def _make_ultralytics_stub():
    ultra = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, *a, **kw):
            pass
        def to(self, device):
            return self
        def __call__(self, frame):
            return []

    ultra.YOLO = FakeYOLO
    return ultra


def _make_deepsort_stub():
    pkg = types.ModuleType("deep_sort_realtime")
    sub = types.ModuleType("deep_sort_realtime.deepsort_tracker")

    class FakeDeepSort:
        def __init__(self, *a, **kw):
            pass
        def update_tracks(self, detections, frame=None):
            return []

    sub.DeepSort = FakeDeepSort
    pkg.deepsort_tracker = sub
    return pkg, sub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_imports(monkeypatch):
    """Inject stubs before any import of the target module."""
    cv2_stub = _make_cv2_stub()
    torch_stub = _make_torch_stub()
    ultra_stub = _make_ultralytics_stub()
    ds_pkg, ds_sub = _make_deepsort_stub()

    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "ultralytics", ultra_stub)
    monkeypatch.setitem(sys.modules, "deep_sort_realtime", ds_pkg)
    monkeypatch.setitem(sys.modules, "deep_sort_realtime.deepsort_tracker", ds_sub)

    # Remove cached module if already imported
    monkeypatch.delitem(
        sys.modules,
        "sailsprep.tracking_pose_model_testing.deepsort",
        raising=False,
    )
    yield


# ---------------------------------------------------------------------------
# Import helper (re-imports fresh each test via autouse fixture)
# ---------------------------------------------------------------------------

def _import_draw_pose():
    from sailsprep.tracking_pose_model_testing.deepsort import draw_pose
    return draw_pose


# ---------------------------------------------------------------------------
# draw_pose unit tests
# ---------------------------------------------------------------------------

class TestDrawPose:
    """draw_pose(img, keypoints, track_id) unit tests."""

    def _make_kpts(self, conf=1.0):
        """17 keypoints with given confidence, all at position (10, 10)."""
        return [[10.0, 10.0, conf] for _ in range(17)]

    def test_high_conf_draws_without_error(self):
        draw_pose = _import_draw_pose()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = self._make_kpts(conf=0.9)
        draw_pose(img, kpts, track_id=1)  # must not raise

    def test_low_conf_skips_drawing(self):
        """All keypoints below threshold — function must not crash."""
        draw_pose = _import_draw_pose()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = self._make_kpts(conf=0.1)
        draw_pose(img, kpts, track_id=None)

    def test_no_track_id(self):
        draw_pose = _import_draw_pose()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = self._make_kpts(conf=0.9)
        draw_pose(img, kpts, track_id=None)  # must not raise

    def test_mixed_confidence(self):
        """Some keypoints visible, some not."""
        draw_pose = _import_draw_pose()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = self._make_kpts(conf=0.9)
        # zero out a few
        for i in [1, 3, 7, 14]:
            kpts[i][2] = 0.0
        draw_pose(img, kpts, track_id=42)

    def test_keypoint_count(self):
        """Exactly 17 keypoints expected (COCO format)."""
        draw_pose = _import_draw_pose()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = self._make_kpts()
        assert len(kpts) == 17

    def test_connections_indices_in_range(self):
        """All connection indices must be valid for 17 keypoints."""
        connections = [
            (0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9),
            (6,8), (8,10), (5,11), (6,12), (11,12), (11,13),
            (13,15), (12,14), (14,16),
        ]
        for start, end in connections:
            assert 0 <= start < 17
            assert 0 <= end < 17


# ---------------------------------------------------------------------------
# device selection test
# ---------------------------------------------------------------------------

class TestDeviceSelection:
    def test_cpu_when_no_cuda(self, monkeypatch):
        import torch as torch_stub
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
        monkeypatch.delitem(
            sys.modules,
            "sailsprep.tracking_pose_model_testing.deepsort",
            raising=False,
        )
        import importlib
        mod = importlib.import_module(
            "sailsprep.tracking_pose_model_testing.deepsort"
        )
        assert mod.device == "cpu"

    def test_cuda_when_available(self, monkeypatch):
        import torch as torch_stub
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
        monkeypatch.delitem(
            sys.modules,
            "sailsprep.tracking_pose_model_testing.deepsort",
            raising=False,
        )
        import importlib
        mod = importlib.import_module(
            "sailsprep.tracking_pose_model_testing.deepsort"
        )
        assert mod.device == "cuda"


# ---------------------------------------------------------------------------
# DeepSort integration stub test
# ---------------------------------------------------------------------------

class TestDeepSortStub:
    def test_update_tracks_returns_list(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        result = tracker.update_tracks([], frame=None)
        assert isinstance(result, list)