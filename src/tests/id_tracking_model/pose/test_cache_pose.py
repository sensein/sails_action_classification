"""
Tests for src/sailsprep/id_tracking_model/pose/cache_pose.py

Run with:
    poetry run pytest src/tests/test_cache_pose.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

MODULE_CANDIDATES = (
    "sailsprep.id_tracking_model.pose.cache_pose",
    "src.sailsprep.id_tracking_model.pose.cache_pose",
)

# Third-party CV packages that cache_pose.py imports at module load time but
# that aren't needed for these unit tests (they require GPU / large native
# wheels that may not be installed, e.g. mmcv). We stub them out in
# sys.modules with MagicMocks so `import cache_pose` succeeds regardless of
# whether the real packages are installed.
_STUB_MODULES = (
    "mmcv",
    "mmdet",
    "mmdet.apis",
    "mmengine",
    "mmengine.registry",
    "mmpose",
    "mmpose.apis",
    "mmpose.evaluation",
    "mmpose.evaluation.functional",
    "mmpose.registry",
)


_STUB_VERSIONS = {
    "mmcv": "2.0.1",
    "mmdet": "3.2.0",
    "mmengine": "0.9.0",
    "mmpose": "1.3.0",
}


def _install_stub_modules():
    """Force-install MagicMock stand-ins for all of _STUB_MODULES.

    We always stub these rather than trying a real import first: even when
    `mmdet` itself is installed, it does an internal version check against
    `mmcv` at import time (`digit_version(mmcv.__version__)`), and once any
    one of these packages is mocked the whole chain needs to be mocked
    consistently or it cascades into AttributeErrors. These tests only need
    cache_pose's top-level `from mmdet.apis import ...` style imports to
    resolve to *something* callable/mockable; they never need the real
    detection/pose model code to run.
    """
    import sys

    installed = []
    for name in _STUB_MODULES:
        stub = MagicMock(name=name)
        if name in _STUB_VERSIONS:
            stub.__version__ = _STUB_VERSIONS[name]
        sys.modules[name] = stub
        installed.append(name)

    return installed


def _import_cache_pose():
    """Import cache_pose, trying both with and without the 'src' prefix,
    stubbing out heavy/optional CV deps (mmcv/mmdet/mmpose) if missing."""
    import importlib

    _install_stub_modules()

    last_err: Exception | None = None
    for name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as err:
            last_err = err
    raise ImportError(
        f"Could not import cache_pose under any of {MODULE_CANDIDATES}"
    ) from last_err


# --------------------------------------------------------------------------
# Fixtures: import the module under test with heavy model init calls mocked
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cp():
    """Import cache_pose with model-loading calls patched out."""
    module = _import_cache_pose()
    with patch.object(module, "init_detector", return_value=MagicMock()), \
         patch.object(module, "init_pose_estimator", return_value=MagicMock()), \
         patch.object(module, "VISUALIZERS") as mock_visualizers:
        mock_visualizers.build.return_value = MagicMock()
        yield module


@pytest.fixture
def model_config(cp):
    return cp.ModelConfig()


@pytest.fixture
def processing_config(cp):
    return cp.ProcessingConfig(
        detection_confidence_threshold=0.5,
        nms_threshold=0.7,
        nms_type="strict",
        bbox_min_height=50,
        bbox_min_width=50,
        oks_nms_threshold=0.6,
        kpt_threshold=0.3,
    )


@pytest.fixture
def visualization_config(cp):
    return cp.VisualizationConfig(
        enable_visualization=True,
        enable_pose_drawing=True,
        enable_bbox_drawing=True,
        radius=3,
        line_width=1,
    )


@pytest.fixture
def detection_module(cp, model_config, processing_config):
    with patch.object(cp, "init_detector", return_value=MagicMock()):
        return cp.DetectionModule(model_config, processing_config)


@pytest.fixture
def pose_module(cp, model_config, visualization_config, processing_config):
    with patch.object(cp, "init_pose_estimator", return_value=MagicMock()), \
         patch.object(cp, "VISUALIZERS") as mock_visualizers:
        mock_visualizers.build.return_value = MagicMock()
        return cp.PoseEstimationModule(model_config, visualization_config, processing_config)


# --------------------------------------------------------------------------
# DetResult / PoseResult wrappers
# --------------------------------------------------------------------------
class TestDetResult:
    def test_concatenates_bboxes_and_scores(self, cp):
        bboxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=float)
        scores = np.array([0.9, 0.5])

        result = cp.DetResult(bboxes, scores)

        assert result.bboxes.shape == (2, 5)
        np.testing.assert_array_almost_equal(result.bboxes[:, :4], bboxes)
        np.testing.assert_array_almost_equal(result.bboxes[:, 4], scores)

    def test_to_dict_contains_bboxes_key(self, cp):
        result = cp.DetResult(np.zeros((1, 4)), np.array([0.8]))
        d = result.to_dict()
        assert "bboxes" in d
        np.testing.assert_array_equal(d["bboxes"], result.bboxes)

    def test_empty_input(self, cp):
        result = cp.DetResult(np.empty((0, 4)), np.empty(0))
        assert result.bboxes.shape == (0, 5)


class TestPoseResult:
    def test_stores_keypoints_and_bbox(self, cp):
        keypoints = np.random.rand(17, 3)
        bbox = np.array([0, 0, 100, 200])

        result = cp.PoseResult(keypoints, bbox)

        np.testing.assert_array_equal(result.keypoints, keypoints)
        np.testing.assert_array_equal(result.bbox, bbox)
        assert result.metadata == {}

    def test_to_dict_round_trip(self, cp):
        keypoints = np.random.rand(17, 3)
        bbox = np.array([0, 0, 100, 200])
        result = cp.PoseResult(keypoints, bbox)
        result.metadata["sufficient_keypoints"] = True

        d = result.to_dict()

        np.testing.assert_array_equal(d["keypoints"], keypoints)
        np.testing.assert_array_equal(d["bbox"], bbox)
        assert d["metadata"] == {"sufficient_keypoints": True}


# --------------------------------------------------------------------------
# DetectionModule filtering logic
# --------------------------------------------------------------------------
class TestDetectionModuleProcessing:
    def test_filters_low_confidence_boxes(self, cp, detection_module):
        bboxes = np.array([
            [0, 0, 100, 100],
            [0, 0, 100, 100],
        ], dtype=float)
        scores = np.array([0.9, 0.2])  # second below threshold 0.5
        det_result = cp.DetResult(bboxes, scores)

        with patch.object(cp, "nms", return_value=np.array([0])):
            processed = detection_module._process_detection_result(det_result)

        assert len(processed.bboxes) == 1
        assert processed.bboxes[0, 4] == pytest.approx(0.9)

    def test_filters_boxes_smaller_than_minimum_dimensions(self, cp, detection_module):
        # one large valid box, one too-small box
        bboxes = np.array([
            [0, 0, 100, 100],   # 100x100 -> valid
            [0, 0, 10, 10],     # 10x10 -> too small
        ], dtype=float)
        scores = np.array([0.9, 0.95])
        det_result = cp.DetResult(bboxes, scores)

        with patch.object(cp, "nms", return_value=np.array([0])):
            processed = detection_module._process_detection_result(det_result)

        assert len(processed.bboxes) == 1
        np.testing.assert_array_almost_equal(processed.bboxes[0, :4], [0, 0, 100, 100])

    def test_returns_empty_when_no_boxes_pass_filters(self, cp, detection_module):
        bboxes = np.array([[0, 0, 5, 5]], dtype=float)
        scores = np.array([0.1])
        det_result = cp.DetResult(bboxes, scores)

        processed = detection_module._process_detection_result(det_result)

        assert len(processed.bboxes) == 0

    def test_applies_strict_nms_when_configured(self, cp, model_config, processing_config):
        processing_config.nms_type = "strict"
        with patch.object(cp, "init_detector", return_value=MagicMock()):
            module = cp.DetectionModule(model_config, processing_config)

        bboxes = np.array([[0, 0, 100, 100], [1, 1, 101, 101]], dtype=float)
        scores = np.array([0.9, 0.8])
        det_result = cp.DetResult(bboxes, scores)

        with patch.object(cp, "nms", return_value=np.array([0])) as mock_nms:
            processed = module._process_detection_result(det_result)

        mock_nms.assert_called_once()
        assert len(processed.bboxes) == 1

    def test_applies_soft_nms_when_configured(self, cp, model_config, processing_config):
        processing_config.nms_type = "soft"
        with patch.object(cp, "init_detector", return_value=MagicMock()):
            module = cp.DetectionModule(model_config, processing_config)

        bboxes = np.array([[0, 0, 100, 100], [1, 1, 101, 101]], dtype=float)
        scores = np.array([0.9, 0.8])
        det_result = cp.DetResult(bboxes, scores)

        with patch.object(cp, "soft_nms", return_value=np.array([0])) as mock_soft_nms:
            processed = module._process_detection_result(det_result)

        mock_soft_nms.assert_called_once()
        assert len(processed.bboxes) == 1

    def test_process_cached_detection_reuses_filtering(self, cp, detection_module):
        cached = {
            "bboxes": np.array([
                [0, 0, 100, 100, 0.9],
                [0, 0, 5, 5, 0.95],
            ])
        }

        with patch.object(cp, "nms", return_value=np.array([0])):
            result = detection_module.process_cached_detection(cached)

        assert len(result.bboxes) == 1


# --------------------------------------------------------------------------
# PoseEstimationModule
# --------------------------------------------------------------------------
class TestPoseEstimationModule:
    def test_estimate_single_returns_empty_for_no_detections(self, cp, pose_module):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det_result = cp.DetResult(np.empty((0, 4)), np.empty(0))

        result = pose_module.estimate_single(frame, det_result)

        assert result == []

    def test_create_pose_results_from_cache(self, cp):
        cached = [
            {"keypoints": np.random.rand(17, 3), "bbox": np.array([0, 0, 50, 50])},
            {"keypoints": np.random.rand(17, 3), "bbox": np.array([10, 10, 60, 60])},
        ]

        results = cp.PoseEstimationModule.create_pose_results_from_cache(cached)

        assert len(results) == 2
        for res, src in zip(results, cached, strict=True):
            np.testing.assert_array_equal(res.keypoints, src["keypoints"])
            np.testing.assert_array_equal(res.bbox, src["bbox"])

    def test_apply_oks_nms_empty_input_returns_empty(self, pose_module):
        assert pose_module.apply_oks_nms([]) == []

    def test_apply_oks_nms_filters_by_returned_indices(self, cp, pose_module):
        pose1 = cp.PoseResult(np.full((17, 3), 0.9), np.array([0, 0, 50, 50]))
        pose2 = cp.PoseResult(np.full((17, 3), 0.9), np.array([1, 1, 51, 51]))

        with patch.object(cp, "oks_nms", return_value=[1]) as mock_oks:
            kept = pose_module.apply_oks_nms([pose1, pose2])

        mock_oks.assert_called_once()
        assert kept == [pose2]

    def test_filter_poses_marks_sufficient_keypoints_true(self, cp, pose_module):
        keypoints = np.zeros((17, 3))
        keypoints[:, 2] = 0.9  # all keypoints highly confident
        pose = cp.PoseResult(keypoints, np.array([0, 0, 10, 10]))

        result = pose_module.filter_poses_by_keypoints([pose])

        assert result[0].metadata["sufficient_keypoints"] is True

    def test_filter_poses_marks_sufficient_keypoints_false(self, cp, pose_module):
        keypoints = np.zeros((17, 3))  # all scores 0 -> below threshold
        pose = cp.PoseResult(keypoints, np.array([0, 0, 10, 10]))

        result = pose_module.filter_poses_by_keypoints([pose])

        assert result[0].metadata["sufficient_keypoints"] is False


# --------------------------------------------------------------------------
# VisualizationModule
# --------------------------------------------------------------------------
class TestVisualizationModule:
    def test_draw_results_returns_same_shape_frame(self, cp, visualization_config):
        vis_module = cp.VisualizationModule(visualization_config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det_result = cp.DetResult(np.array([[10, 10, 50, 50]], dtype=float), np.array([0.9]))
        keypoints = np.zeros((17, 3))
        keypoints[:, 2] = 0.9
        pose_results = [cp.PoseResult(keypoints, np.array([10, 10, 50, 50]))]

        vis_frame = vis_module.draw_results(frame, pose_results, det_result)

        assert vis_frame.shape == frame.shape
        # original frame should remain untouched (draw_results copies it)
        assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))

    def test_draw_results_skips_bbox_drawing_when_disabled(self, cp, visualization_config):
        visualization_config.enable_bbox_drawing = False
        vis_module = cp.VisualizationModule(visualization_config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        det_result = cp.DetResult(np.array([[5, 5, 20, 20]], dtype=float), np.array([0.9]))

        vis_frame = vis_module.draw_results(frame, [], det_result)

        # No drawing requested -> frame stays all zeros
        assert np.array_equal(vis_frame, frame)

    def test_draw_results_handles_none_det_result(self, cp, visualization_config):
        vis_module = cp.VisualizationModule(visualization_config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        # Should not raise even though det_result is None
        vis_frame = vis_module.draw_results(frame, [], None)

        assert vis_frame.shape == frame.shape


# --------------------------------------------------------------------------
# Config dataclasses / defaults
# --------------------------------------------------------------------------
class TestConfigs:
    def test_pipeline_config_defaults(self, cp):
        config = cp.PipelineConfig()

        assert config.frame_limit == 0
        assert config.detection_only is False
        assert isinstance(config.models, cp.ModelConfig)
        assert isinstance(config.processing, cp.ProcessingConfig)
        assert isinstance(config.visualization, cp.VisualizationConfig)
        assert isinstance(config.cache, cp.CacheConfig)

    def test_create_default_config_overrides(self, cp):
        config = cp.create_default_config()

        assert config.processing.detection_confidence_threshold == 0.4
        assert config.processing.oks_nms_threshold == 0.55
        assert config.visualization.enable_visualization is False
        assert config.cache.enable_cache is True
        assert config.detection_only is False


# --------------------------------------------------------------------------
# DetectionPosePipeline.process_frame (detection-only mode)
# --------------------------------------------------------------------------
class TestPipelineProcessFrame:
    @pytest.fixture
    def pipeline(self, cp, model_config, processing_config, visualization_config):
        config = cp.PipelineConfig(
            models=model_config,
            processing=processing_config,
            visualization=visualization_config,
            detection_only=True,
        )
        with patch.object(cp, "init_detector", return_value=MagicMock()), \
             patch.object(cp, "init_pose_estimator", return_value=MagicMock()), \
             patch.object(cp, "VISUALIZERS") as mock_visualizers:
            mock_visualizers.build.return_value = MagicMock()
            return cp.DetectionPosePipeline(config)

    def test_detection_only_mode_returns_frame_with_boxes_drawn(self, cp, pipeline):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det_result = cp.DetResult(np.array([[10, 10, 60, 60]], dtype=float), np.array([0.9]))

        with patch.object(
            pipeline.detection_module, "detect_single", return_value=det_result
        ):
            vis_frame = pipeline.process_frame(frame)

        assert vis_frame.shape == frame.shape
        # Something should have been drawn (frame is no longer all zeros)
        assert not np.array_equal(vis_frame, frame)