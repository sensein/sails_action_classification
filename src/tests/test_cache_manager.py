"""
Tests for CacheManager
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sailsprep.id_tracking_model.utils.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache_manager(tmp_path):
    """Return a CacheManager whose cache_base_path lives in pytest's tmp_path."""
    return CacheManager(
        output_video_path="/data/videos/my_video.mp4",
        detection_config="/configs/rtmdet_config.py",
        pose_config="/configs/rtmpose_config.py",
        detection_confidence_threshold=0.5,
        nms_type="nms",
        nms_threshold=0.3,
        bbox_min_height=50,
        bbox_min_width=30,
        cache_base_path=str(tmp_path),
    )


@pytest.fixture
def sample_detections():
    """Two frames, each with an (N, 5) bboxes array."""
    return {
        0: {"bboxes": np.array([[10.0, 20.0, 50.0, 80.0, 0.9]], dtype=np.float32)},
        1: {"bboxes": np.array([[5.0, 5.0, 40.0, 60.0, 0.8],
                                [100.0, 100.0, 150.0, 200.0, 0.7]], dtype=np.float32)},
    }


@pytest.fixture
def sample_poses():
    """Two frames, each with a list of pose dicts (keypoints + bbox)."""
    return {
        0: [
            {
                "keypoints": np.zeros((17, 3), dtype=np.float32),
                "bbox": np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32),
            }
        ],
        1: [
            {
                "keypoints": np.ones((17, 3), dtype=np.float32),
                "bbox": np.array([5.0, 5.0, 40.0, 60.0], dtype=np.float32),
            },
            {
                "keypoints": np.full((17, 3), 2.0, dtype=np.float32),
                "bbox": np.array([100.0, 100.0, 150.0, 200.0], dtype=np.float32),
            },
        ],
    }


# ---------------------------------------------------------------------------
# __init__ / path generation
# ---------------------------------------------------------------------------

class TestInit:
    def test_video_basename_extracted(self, tmp_cache_manager):
        assert tmp_cache_manager.video_basename == "my_video"

    def test_detection_config_name_extracted(self, tmp_cache_manager):
        assert tmp_cache_manager.detection_config_name == "rtmdet_config"

    def test_pose_config_name_extracted(self, tmp_cache_manager):
        assert tmp_cache_manager.pose_config_name == "rtmpose_config"

    def test_detection_cache_path_is_h5(self, tmp_cache_manager):
        assert tmp_cache_manager.detection_cache_path.suffix == ".h5"

    def test_pose_cache_path_is_h5(self, tmp_cache_manager):
        assert tmp_cache_manager.pose_cache_path.suffix == ".h5"

    def test_detection_cache_path_contains_video_name(self, tmp_cache_manager):
        assert "my_video" in str(tmp_cache_manager.detection_cache_path)

    def test_pose_cache_path_contains_video_name(self, tmp_cache_manager):
        assert "my_video" in str(tmp_cache_manager.pose_cache_path)

    def test_detection_cache_path_contains_config_name(self, tmp_cache_manager):
        assert "rtmdet_config" in str(tmp_cache_manager.detection_cache_path)

    def test_pose_cache_path_contains_both_configs(self, tmp_cache_manager):
        path_str = str(tmp_cache_manager.pose_cache_path)
        assert "rtmdet_config" in path_str
        assert "rtmpose_config" in path_str

    def test_detection_and_pose_paths_are_different(self, tmp_cache_manager):
        assert tmp_cache_manager.detection_cache_path != tmp_cache_manager.pose_cache_path


# ---------------------------------------------------------------------------
# check_detection_cache / check_pose_cache
# ---------------------------------------------------------------------------

class TestCacheChecks:
    def test_detection_cache_missing_returns_false(self, tmp_cache_manager):
        assert tmp_cache_manager.check_detection_cache() is False

    def test_pose_cache_missing_returns_false(self, tmp_cache_manager):
        assert tmp_cache_manager.check_pose_cache() is False

    def test_detection_cache_present_returns_true(self, tmp_cache_manager):
        p = tmp_cache_manager.detection_cache_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        assert tmp_cache_manager.check_detection_cache() is True

    def test_pose_cache_present_returns_true(self, tmp_cache_manager):
        p = tmp_cache_manager.pose_cache_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        assert tmp_cache_manager.check_pose_cache() is True


# ---------------------------------------------------------------------------
# load_all_detections — no cache
# ---------------------------------------------------------------------------

class TestLoadDetectionsNoCache:
    def test_returns_none_when_no_cache(self, tmp_cache_manager):
        assert tmp_cache_manager.load_all_detections() is None


# ---------------------------------------------------------------------------
# save_all_detections + load_all_detections round-trip
# ---------------------------------------------------------------------------

class TestDetectionRoundTrip:
    def test_save_creates_file(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        assert tmp_cache_manager.detection_cache_path.exists()

    def test_load_returns_dict(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert isinstance(result, dict)

    def test_load_has_correct_frame_count(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        assert len(result) == len(sample_detections)

    def test_load_correct_frame_keys(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        assert set(result.keys()) == {0, 1}

    def test_bboxes_shape_frame0(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        assert result[0]["bboxes"].shape == sample_detections[0]["bboxes"].shape

    def test_bboxes_shape_frame1(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        assert result[1]["bboxes"].shape == sample_detections[1]["bboxes"].shape

    def test_bboxes_values_frame0(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        np.testing.assert_array_almost_equal(
            result[0]["bboxes"], sample_detections[0]["bboxes"]
        )

    def test_bboxes_values_frame1(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        result = tmp_cache_manager.load_all_detections()
        assert result is not None
        np.testing.assert_array_almost_equal(
            result[1]["bboxes"], sample_detections[1]["bboxes"]
        )

    def test_check_detection_cache_true_after_save(self, tmp_cache_manager, sample_detections):
        tmp_cache_manager.save_all_detections(sample_detections)
        assert tmp_cache_manager.check_detection_cache() is True


# ---------------------------------------------------------------------------
# load_all_poses — no cache
# ---------------------------------------------------------------------------

class TestLoadPosesNoCache:
    def test_returns_none_when_no_cache(self, tmp_cache_manager):
        assert tmp_cache_manager.load_all_poses() is None


# ---------------------------------------------------------------------------
# save_all_poses + load_all_poses round-trip
# ---------------------------------------------------------------------------

class TestPoseRoundTrip:
    def test_save_creates_file(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        assert tmp_cache_manager.pose_cache_path.exists()

    def test_load_returns_dict(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert isinstance(result, dict)

    def test_load_has_correct_frame_count(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        assert len(result) == len(sample_poses)

    def test_load_correct_frame_keys(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        assert set(result.keys()) == {0, 1}

    def test_pose_count_per_frame(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        assert len(result[0]) == 1
        assert len(result[1]) == 2

    def test_keypoints_shape_frame0(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        assert result[0][0]["keypoints"].shape == (17, 3)

    def test_bbox_shape_frame0(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        assert result[0][0]["bbox"].shape == (4,)

    def test_keypoints_values_frame0(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        np.testing.assert_array_almost_equal(
            result[0][0]["keypoints"], sample_poses[0][0]["keypoints"]
        )

    def test_bbox_values_frame1_pose1(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        result = tmp_cache_manager.load_all_poses()
        assert result is not None
        np.testing.assert_array_almost_equal(
            result[1][1]["bbox"], sample_poses[1][1]["bbox"]
        )

    def test_check_pose_cache_true_after_save(self, tmp_cache_manager, sample_poses):
        tmp_cache_manager.save_all_poses(sample_poses)
        assert tmp_cache_manager.check_pose_cache() is True


# ---------------------------------------------------------------------------
# get_cache_params
# ---------------------------------------------------------------------------

class TestGetCacheParams:
    def test_returns_dict(self, tmp_cache_manager):
        assert isinstance(tmp_cache_manager.get_cache_params(), dict)

    def test_video_basename_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["video_basename"] == "my_video"

    def test_detection_config_name_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["detection_config_name"] == "rtmdet_config"

    def test_pose_config_name_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["pose_config_name"] == "rtmpose_config"

    def test_detection_confidence_threshold_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["detection_confidence_threshold"] == 0.5

    def test_nms_type_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["nms_type"] == "nms"

    def test_nms_threshold_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["nms_threshold"] == 0.3

    def test_bbox_min_height_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["bbox_min_height"] == 50

    def test_bbox_min_width_key(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert params["bbox_min_width"] == 30

    def test_detection_cache_path_key_is_string(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert isinstance(params["detection_cache_path"], str)

    def test_pose_cache_path_key_is_string(self, tmp_cache_manager):
        params = tmp_cache_manager.get_cache_params()
        assert isinstance(params["pose_cache_path"], str)


# ---------------------------------------------------------------------------
# from_metadata — mocked CacheMetadataManager
# ---------------------------------------------------------------------------

class TestFromMetadata:
    def _make_metadata(self):
        """Build a mock metadata object with the expected attributes."""
        m = MagicMock()
        m.detection_config = "/configs/rtmdet_config.py"
        m.pose_config = "/configs/rtmpose_config.py"
        m.detection_confidence_threshold = 0.5
        m.nms_type = "nms"
        m.nms_threshold = 0.3
        m.bbox_min_height = 50
        m.bbox_min_width = 30
        return m

    def test_returns_none_when_metadata_missing(self, tmp_path):
        with patch(
            "sailsprep.id_tracking_model.utils.cache_manager.CacheMetadataManager"
        ) as MockMM:
            instance = MockMM.return_value
            instance.load_metadata.return_value = None
            instance.list_cached_videos.return_value = []

            result = CacheManager.from_metadata(
                video_filename="missing_video.mp4",
                cache_base_path=str(tmp_path),
            )
        assert result is None

    def test_returns_cache_manager_when_metadata_found(self, tmp_path):
        with patch(
            "sailsprep.id_tracking_model.utils.cache_manager.CacheMetadataManager"
        ) as MockMM:
            instance = MockMM.return_value
            instance.load_metadata.return_value = self._make_metadata()

            result = CacheManager.from_metadata(
                video_filename="my_video.mp4",
                cache_base_path=str(tmp_path),
            )
        assert isinstance(result, CacheManager)

    def test_from_metadata_video_basename(self, tmp_path):
        with patch(
            "sailsprep.id_tracking_model.utils.cache_manager.CacheMetadataManager"
        ) as MockMM:
            instance = MockMM.return_value
            instance.load_metadata.return_value = self._make_metadata()

            result = CacheManager.from_metadata(
                video_filename="my_video.mp4",
                cache_base_path=str(tmp_path),
            )
        assert result is not None
        assert result.video_basename == "my_video"

    def test_from_metadata_detection_config_name(self, tmp_path):
        with patch(
            "sailsprep.id_tracking_model.utils.cache_manager.CacheMetadataManager"
        ) as MockMM:
            instance = MockMM.return_value
            instance.load_metadata.return_value = self._make_metadata()

            result = CacheManager.from_metadata(
                video_filename="my_video.mp4",
                cache_base_path=str(tmp_path),
            )
        assert result is not None
        assert result.detection_config_name == "rtmdet_config"

    def test_from_metadata_passes_correct_format(self, tmp_path):
        with patch(
            "sailsprep.id_tracking_model.utils.cache_manager.CacheMetadataManager"
        ) as MockMM:
            instance = MockMM.return_value
            instance.load_metadata.return_value = self._make_metadata()

            CacheManager.from_metadata(
                video_filename="my_video.mp4",
                cache_base_path=str(tmp_path),
                metadata_format="json",
            )
            instance.load_metadata.assert_called_once_with("my_video", format="json")