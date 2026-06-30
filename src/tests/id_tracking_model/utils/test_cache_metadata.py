"""
Tests for CacheMetadata, CacheMetadataManager, and create_cache_metadata_from_config.
"""

import json
from pathlib import Path

import pytest
import yaml

from sailsprep.id_tracking_model.utils.cache_metadata import (
    CacheMetadata,
    CacheMetadataManager,
    create_cache_metadata_from_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_metadata() -> CacheMetadata:
    """A fully-populated CacheMetadata instance for reuse across tests."""
    return CacheMetadata(
        video_basename="test_video",
        detection_config="rtmdet_m_640-8xb32_coco-person.py",
        detection_checkpoint="rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        detection_confidence_threshold=0.5,
        nms_type="strict",
        nms_threshold=0.7,
        bbox_min_height=50,
        bbox_min_width=50,
        pose_config="td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
        pose_checkpoint="hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
        oks_nms_threshold=0.6,
        kpt_threshold=0.3,
        detection_cache_relative_path="detections/test_video/det.h5",
        pose_cache_relative_path="pose/test_video/pose.h5",
    )


@pytest.fixture()
def manager(tmp_path: Path) -> CacheMetadataManager:
    """CacheMetadataManager rooted in a pytest-managed temp directory."""
    return CacheMetadataManager(cache_base_path=str(tmp_path))


# ---------------------------------------------------------------------------
# CacheMetadata – dataclass behaviour
# ---------------------------------------------------------------------------

class TestCacheMetadata:

    def test_to_dict_contains_all_fields(self, sample_metadata: CacheMetadata) -> None:
        d = sample_metadata.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "video_basename",
            "detection_config",
            "detection_checkpoint",
            "detection_confidence_threshold",
            "nms_type",
            "nms_threshold",
            "bbox_min_height",
            "bbox_min_width",
            "pose_config",
            "pose_checkpoint",
            "oks_nms_threshold",
            "kpt_threshold",
            "detection_cache_relative_path",
            "pose_cache_relative_path",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self, sample_metadata: CacheMetadata) -> None:
        d = sample_metadata.to_dict()
        assert d["video_basename"] == "test_video"
        assert d["detection_confidence_threshold"] == 0.5
        assert d["bbox_min_height"] == 50

    def test_from_dict_roundtrip(self, sample_metadata: CacheMetadata) -> None:
        d = sample_metadata.to_dict()
        restored = CacheMetadata.from_dict(d)
        assert restored == sample_metadata

    def test_from_dict_field_values(self, sample_metadata: CacheMetadata) -> None:
        restored = CacheMetadata.from_dict(sample_metadata.to_dict())
        assert restored.video_basename == sample_metadata.video_basename
        assert restored.nms_threshold == sample_metadata.nms_threshold
        assert restored.pose_cache_relative_path == sample_metadata.pose_cache_relative_path


# ---------------------------------------------------------------------------
# CacheMetadataManager – save / load YAML
# ---------------------------------------------------------------------------

class TestCacheMetadataManagerYaml:

    def test_save_creates_yaml_file(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata, tmp_path: Path
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        expected = tmp_path / "metadata" / "test_video_cache_metadata.yaml"
        assert expected.exists()

    def test_saved_yaml_is_valid(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata, tmp_path: Path
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        path = tmp_path / "metadata" / "test_video_cache_metadata.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["video_basename"] == "test_video"

    def test_load_returns_correct_object(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        loaded = manager.load_metadata("test_video", format="yaml")
        assert loaded is not None
        assert loaded == sample_metadata

    def test_load_missing_returns_none(self, manager: CacheMetadataManager) -> None:
        result = manager.load_metadata("nonexistent_video", format="yaml")
        assert result is None

    def test_check_metadata_exists_true(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        assert manager.check_metadata_exists("test_video", format="yaml") is True

    def test_check_metadata_exists_false(self, manager: CacheMetadataManager) -> None:
        assert manager.check_metadata_exists("ghost_video", format="yaml") is False


# ---------------------------------------------------------------------------
# CacheMetadataManager – save / load JSON
# ---------------------------------------------------------------------------

class TestCacheMetadataManagerJson:

    def test_save_creates_json_file(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata, tmp_path: Path
    ) -> None:
        manager.save_metadata(sample_metadata, format="json")
        expected = tmp_path / "metadata" / "test_video_cache_metadata.json"
        assert expected.exists()

    def test_saved_json_is_valid(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata, tmp_path: Path
    ) -> None:
        manager.save_metadata(sample_metadata, format="json")
        path = tmp_path / "metadata" / "test_video_cache_metadata.json"
        with open(path) as f:
            data = json.load(f)
        assert data["video_basename"] == "test_video"

    def test_load_returns_correct_object(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="json")
        loaded = manager.load_metadata("test_video", format="json")
        assert loaded is not None
        assert loaded == sample_metadata

    def test_load_missing_returns_none(self, manager: CacheMetadataManager) -> None:
        result = manager.load_metadata("nonexistent_video", format="json")
        assert result is None

    def test_check_metadata_exists_true(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="json")
        assert manager.check_metadata_exists("test_video", format="json") is True

    def test_check_metadata_exists_false(self, manager: CacheMetadataManager) -> None:
        assert manager.check_metadata_exists("ghost_video", format="json") is False


# ---------------------------------------------------------------------------
# CacheMetadataManager – list_cached_videos
# ---------------------------------------------------------------------------

class TestListCachedVideos:

    def test_empty_when_no_metadata(self, manager: CacheMetadataManager) -> None:
        assert manager.list_cached_videos() == []

    def test_lists_yaml_video(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        assert "test_video" in manager.list_cached_videos()

    def test_lists_json_video(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="json")
        assert "test_video" in manager.list_cached_videos()

    def test_lists_multiple_videos(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")

        second = CacheMetadata(
            **{**sample_metadata.to_dict(), "video_basename": "another_video"}
        )
        manager.save_metadata(second, format="yaml")

        videos = manager.list_cached_videos()
        assert "test_video" in videos
        assert "another_video" in videos

    def test_deduplicates_yaml_and_json_for_same_video(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        manager.save_metadata(sample_metadata, format="yaml")
        manager.save_metadata(sample_metadata, format="json")
        videos = manager.list_cached_videos()
        assert videos.count("test_video") == 1

    def test_result_is_sorted(
        self, manager: CacheMetadataManager, sample_metadata: CacheMetadata
    ) -> None:
        for name in ["zebra_vid", "alpha_vid", "mango_vid"]:
            m = CacheMetadata(**{**sample_metadata.to_dict(), "video_basename": name})
            manager.save_metadata(m, format="yaml")
        videos = manager.list_cached_videos()
        assert videos == sorted(videos)


# ---------------------------------------------------------------------------
# create_cache_metadata_from_config
# ---------------------------------------------------------------------------

class TestCreateCacheMetadataFromConfig:

    def test_returns_cache_metadata_instance(self, tmp_path: Path) -> None:
        det_path = tmp_path / "detections" / "vid" / "det.h5"
        pose_path = tmp_path / "pose" / "vid" / "pose.h5"

        result = create_cache_metadata_from_config(
            video_basename="vid",
            detection_config="det_cfg.py",
            detection_checkpoint="det_ckpt.pth",
            pose_config="pose_cfg.py",
            pose_checkpoint="pose_ckpt.pth",
            processing_config={},
            detection_cache_path=det_path,
            pose_cache_path=pose_path,
            cache_base_path=tmp_path,
        )
        assert isinstance(result, CacheMetadata)

    def test_uses_default_processing_config_values(self, tmp_path: Path) -> None:
        det_path = tmp_path / "det.h5"
        pose_path = tmp_path / "pose.h5"

        result = create_cache_metadata_from_config(
            video_basename="vid",
            detection_config="det_cfg.py",
            detection_checkpoint="det_ckpt.pth",
            pose_config="pose_cfg.py",
            pose_checkpoint="pose_ckpt.pth",
            processing_config={},
            detection_cache_path=det_path,
            pose_cache_path=pose_path,
            cache_base_path=tmp_path,
        )
        assert result.detection_confidence_threshold == 0.5
        assert result.nms_type == "strict"
        assert result.nms_threshold == 0.7
        assert result.bbox_min_height == 50
        assert result.bbox_min_width == 50
        assert result.oks_nms_threshold == 0.6
        assert result.kpt_threshold == 0.3

    def test_honours_custom_processing_config(self, tmp_path: Path) -> None:
        det_path = tmp_path / "det.h5"
        pose_path = tmp_path / "pose.h5"

        cfg = {
            "detection_confidence_threshold": 0.8,
            "nms_type": "soft",
            "nms_threshold": 0.4,
            "bbox_min_height": 100,
            "bbox_min_width": 80,
            "oks_nms_threshold": 0.9,
            "kpt_threshold": 0.1,
        }
        result = create_cache_metadata_from_config(
            video_basename="vid",
            detection_config="det_cfg.py",
            detection_checkpoint="det_ckpt.pth",
            pose_config="pose_cfg.py",
            pose_checkpoint="pose_ckpt.pth",
            processing_config=cfg,
            detection_cache_path=det_path,
            pose_cache_path=pose_path,
            cache_base_path=tmp_path,
        )
        assert result.detection_confidence_threshold == 0.8
        assert result.nms_type == "soft"
        assert result.bbox_min_height == 100

    def test_relative_paths_computed_correctly(self, tmp_path: Path) -> None:
        det_path = tmp_path / "detections" / "vid" / "det.h5"
        pose_path = tmp_path / "pose" / "vid" / "pose.h5"

        result = create_cache_metadata_from_config(
            video_basename="vid",
            detection_config="det_cfg.py",
            detection_checkpoint="det_ckpt.pth",
            pose_config="pose_cfg.py",
            pose_checkpoint="pose_ckpt.pth",
            processing_config={},
            detection_cache_path=det_path,
            pose_cache_path=pose_path,
            cache_base_path=tmp_path,
        )
        assert result.detection_cache_relative_path == str(Path("detections/vid/det.h5"))
        assert result.pose_cache_relative_path == str(Path("pose/vid/pose.h5"))

    def test_video_basename_and_configs_set(self, tmp_path: Path) -> None:
        det_path = tmp_path / "det.h5"
        pose_path = tmp_path / "pose.h5"

        result = create_cache_metadata_from_config(
            video_basename="my_video",
            detection_config="det_cfg.py",
            detection_checkpoint="det_ckpt.pth",
            pose_config="pose_cfg.py",
            pose_checkpoint="pose_ckpt.pth",
            processing_config={},
            detection_cache_path=det_path,
            pose_cache_path=pose_path,
            cache_base_path=tmp_path,
        )
        assert result.video_basename == "my_video"
        assert result.detection_config == "det_cfg.py"
        assert result.pose_config == "pose_cfg.py"