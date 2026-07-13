"""
src/tests/tracking_pose_model_testing/test_rtmpose.py

Unit tests for the RTMPose wholebody video-annotation pipeline.
  Script under test : src/sailsprep/tracking_pose_model_testing/rtmpose.py
  This test file    : src/tests/tracking_pose_model_testing/test_rtmpose.py

`mmcv`, `mmdet`, and `mmpose` are not installed in this environment (though
`mmengine`, `cv2`, `numpy`, and `tqdm` are real), so they are stubbed. The
script builds a detector/pose-estimator/visualizer at import time and then
scans a hardcoded "/videos" folder; `os.listdir` is patched to return an
empty list so that loop is a no-op. The only real business logic in the
file -- `visualize_img`, which wires detector/pose_estimator/visualizer
calls together -- is exercised directly with mocked collaborators.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_rtmpose.py -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest


@contextlib.contextmanager
def _scoped_modules(stub_map: dict):
    """Save/restore only the specific sys.modules keys we stub (see
    test_sam2_yolov8.py for why mock.patch.dict(sys.modules, ...) is unsafe
    here: it would wipe out modules imported for the first time during exec).
    """
    saved = {k: sys.modules.get(k) for k in stub_map}
    sys.modules.update(stub_map)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _make_stub_modules() -> dict:
    mmcv_mod = _stub("mmcv", imread=mock.MagicMock(return_value=np.zeros((4, 4, 3))))

    mmdet_mod = _stub("mmdet")
    mmdet_apis = _stub(
        "mmdet.apis",
        inference_detector=mock.MagicMock(),
        init_detector=mock.MagicMock(return_value=mock.MagicMock()),
    )

    _fake_vis_cfg = mock.MagicMock(radius=3, line_width=2)
    _fake_cfg = mock.MagicMock(visualizer=_fake_vis_cfg)
    _fake_pose_est = mock.MagicMock(cfg=_fake_cfg)

    mmpose_mod = _stub("mmpose")
    mmpose_apis = _stub(
        "mmpose.apis",
        inference_topdown=mock.MagicMock(return_value=[]),
        init_model=mock.MagicMock(return_value=_fake_pose_est),
    )
    mmpose_eval = _stub("mmpose.evaluation")
    mmpose_eval_functional = _stub("mmpose.evaluation.functional", nms=mock.MagicMock())
    mmpose_registry = _stub(
        "mmpose.registry",
        VISUALIZERS=mock.MagicMock(build=mock.MagicMock(return_value=mock.MagicMock())),
    )
    mmpose_structures = _stub("mmpose.structures", merge_data_samples=mock.MagicMock())

    return {
        "mmcv": mmcv_mod,
        "mmdet": mmdet_mod,
        "mmdet.apis": mmdet_apis,
        "mmpose": mmpose_mod,
        "mmpose.apis": mmpose_apis,
        "mmpose.evaluation": mmpose_eval,
        "mmpose.evaluation.functional": mmpose_eval_functional,
        "mmpose.registry": mmpose_registry,
        "mmpose.structures": mmpose_structures,
    }


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "rtmpose.py"

_module_cache: types.ModuleType | None = None


def _load_pipeline() -> types.ModuleType:
    global _module_cache
    if _module_cache is not None:
        return _module_cache

    if not PIPELINE_SCRIPT.exists():
        pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("rtmpose", PIPELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with (
        _scoped_modules(_make_stub_modules()),
        mock.patch("os.listdir", return_value=[]),
        mock.patch("os.makedirs"),
        mock.patch("builtins.print"),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    _module_cache = mod
    return mod


@pytest.fixture(scope="session")
def pipeline() -> types.ModuleType:
    return _load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: visualize_img wiring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestVisualizeImg:

    def _make_detector(self):
        detector = mock.MagicMock()
        detector.cfg.get.return_value = "mmdet"
        return detector

    def _make_pred_instance(self):
        pred_instance = mock.MagicMock()
        pred_instance.bboxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        pred_instance.scores = np.array([0.9])
        pred_instance.labels = np.array([0])
        return pred_instance

    def test_add_datasample_called_with_expected_kwargs(self, pipeline):
        detector = self._make_detector()
        pose_estimator = mock.MagicMock()
        visualizer = mock.MagicMock()

        pred_instance = self._make_pred_instance()
        detect_result = mock.MagicMock()
        detect_result.pred_instances.cpu.return_value.numpy.return_value = pred_instance

        with (
            mock.patch.object(pipeline, "inference_detector", return_value=detect_result),
            mock.patch.object(pipeline, "nms", return_value=np.array([0])),
            mock.patch.object(pipeline, "inference_topdown", return_value=[]),
            mock.patch.object(pipeline, "merge_data_samples", return_value=mock.MagicMock()),
            mock.patch.object(pipeline.mmcv, "imread", return_value=np.zeros((4, 4, 3))),
            mock.patch.object(pipeline, "init_default_scope"),
        ):
            pipeline.visualize_img(
                "fake.jpg", detector, pose_estimator, visualizer,
                show_interval=0, out_file=None,
            )

        assert visualizer.add_datasample.call_count == 1
        args, kwargs = visualizer.add_datasample.call_args
        assert args[0] == "result"
        assert kwargs["draw_gt"] is False
        assert kwargs["draw_heatmap"] is False
        assert kwargs["draw_bbox"] is False
        assert kwargs["show"] is False
        assert kwargs["kpt_thr"] == 0.3
        assert kwargs["wait_time"] == 0
        assert kwargs["out_file"] is None

    def test_bboxes_filtered_by_label_and_score(self, pipeline):
        detector = self._make_detector()
        pose_estimator = mock.MagicMock()
        visualizer = mock.MagicMock()

        pred_instance = mock.MagicMock()
        pred_instance.bboxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]])
        pred_instance.scores = np.array([0.9, 0.1])  # second below 0.3 threshold
        pred_instance.labels = np.array([0, 0])

        detect_result = mock.MagicMock()
        detect_result.pred_instances.cpu.return_value.numpy.return_value = pred_instance

        captured_bboxes = {}

        def fake_inference_topdown(pose_est, img_path, bboxes):
            captured_bboxes["bboxes"] = bboxes
            return []

        with (
            mock.patch.object(pipeline, "inference_detector", return_value=detect_result),
            mock.patch.object(pipeline, "nms", side_effect=lambda b, t: np.arange(len(b))),
            mock.patch.object(pipeline, "inference_topdown", side_effect=fake_inference_topdown),
            mock.patch.object(pipeline, "merge_data_samples", return_value=mock.MagicMock()),
            mock.patch.object(pipeline.mmcv, "imread", return_value=np.zeros((4, 4, 3))),
            mock.patch.object(pipeline, "init_default_scope"),
        ):
            pipeline.visualize_img(
                "fake.jpg", detector, pose_estimator, visualizer,
                show_interval=5, out_file="out.jpg",
            )

        # Only the high-confidence, person-label (0) box should survive.
        assert captured_bboxes["bboxes"].shape[0] == 1

    def test_default_scope_not_initialized_when_scope_none(self, pipeline):
        detector = self._make_detector()
        detector.cfg.get.return_value = None
        pose_estimator = mock.MagicMock()
        visualizer = mock.MagicMock()

        pred_instance = self._make_pred_instance()
        detect_result = mock.MagicMock()
        detect_result.pred_instances.cpu.return_value.numpy.return_value = pred_instance

        with (
            mock.patch.object(pipeline, "inference_detector", return_value=detect_result),
            mock.patch.object(pipeline, "nms", return_value=np.array([0])),
            mock.patch.object(pipeline, "inference_topdown", return_value=[]),
            mock.patch.object(pipeline, "merge_data_samples", return_value=mock.MagicMock()),
            mock.patch.object(pipeline.mmcv, "imread", return_value=np.zeros((4, 4, 3))),
            mock.patch.object(pipeline, "init_default_scope") as mock_init_scope,
        ):
            pipeline.visualize_img(
                "fake.jpg", detector, pose_estimator, visualizer,
                show_interval=0, out_file=None,
            )

        mock_init_scope.assert_not_called()
