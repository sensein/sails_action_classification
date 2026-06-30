"""
Cache Metadata Manager

Manages metadata files that store cache configuration parameters.
This allows tracker_clip.py to discover the correct cache files created by cache_pose.py
without needing to know the detection/pose model configurations.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CacheMetadata:
    """Metadata for cache files created by detection/pose pipeline"""

    # Video information
    video_basename: str

    # Detection configuration
    detection_config: str
    detection_checkpoint: str
    detection_confidence_threshold: float
    nms_type: str
    nms_threshold: float
    bbox_min_height: int
    bbox_min_width: int

    # Pose configuration
    pose_config: str
    pose_checkpoint: str
    oks_nms_threshold: float
    kpt_threshold: float

    # Cache paths (relative to cache_base_path)
    detection_cache_relative_path: str
    pose_cache_relative_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CacheMetadata':
        """Create from dictionary"""
        return cls(**data)


class CacheMetadataManager:
    """Manages reading and writing cache metadata files"""

    def __init__(self, cache_base_path: str = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking"):
        self.cache_base_path = Path(cache_base_path)
        self.metadata_dir = self.cache_base_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self, video_basename: str, format: str = "yaml") -> Path:
        """Get path to metadata file for a video"""
        return self.metadata_dir / f"{video_basename}_cache_metadata.{format}"

    def save_metadata(self, metadata: CacheMetadata, format: str = "yaml") -> None:
        """
        Save cache metadata to file

        Args:
            metadata: CacheMetadata object
            format: "yaml" or "json"
        """
        metadata_path = self._get_metadata_path(metadata.video_basename, format)
        metadata_dict = metadata.to_dict()

        try:
            with open(metadata_path, 'w') as f:
                if format == "yaml":
                    yaml.dump(metadata_dict, f, default_flow_style=False, sort_keys=False)
                else:  # json
                    json.dump(metadata_dict, f, indent=2)

            print(f"Saved cache metadata to: {metadata_path}")

        except Exception as e:
            print(f"Error saving cache metadata: {e}")
            raise

    def load_metadata(self, video_basename: str, format: str = "yaml") -> CacheMetadata | None:
        """
        Load cache metadata from file

        Args:
            video_basename: Video basename (without extension)
            format: "yaml" or "json"

        Returns:
            CacheMetadata object or None if not found
        """
        metadata_path = self._get_metadata_path(video_basename, format)

        if not metadata_path.exists():
            print(f"Cache metadata not found: {metadata_path}")
            return None

        try:
            with open(metadata_path) as f:
                metadata_dict = yaml.safe_load(f) if format == "yaml" else json.load(f)

            return CacheMetadata.from_dict(metadata_dict)

        except Exception as e:
            print(f"Error loading cache metadata: {e}")
            return None

    def check_metadata_exists(self, video_basename: str, format: str = "yaml") -> bool:
        """Check if metadata file exists"""
        return self._get_metadata_path(video_basename, format).exists()

    def list_cached_videos(self) -> list[str]:
        """List all videos with cache metadata"""
        yaml_files = list(self.metadata_dir.glob("*_cache_metadata.yaml"))
        json_files = list(self.metadata_dir.glob("*_cache_metadata.json"))

        videos = set()
        for f in yaml_files + json_files:
            # Extract video basename from metadata filename
            basename = f.stem.replace("_cache_metadata", "")
            videos.add(basename)

        return sorted(list(videos))


def create_cache_metadata_from_config(
    video_basename: str,
    detection_config: str,
    detection_checkpoint: str,
    pose_config: str,
    pose_checkpoint: str,
    processing_config: dict[str, Any],
    detection_cache_path: Path,
    pose_cache_path: Path,
    cache_base_path: Path
) -> CacheMetadata:
    """
    Helper function to create CacheMetadata from pipeline config

    Args:
        video_basename: Video filename without extension
        detection_config: Path to detection model config
        detection_checkpoint: Path to detection checkpoint
        pose_config: Path to pose model config
        pose_checkpoint: Path to pose checkpoint
        processing_config: Dictionary with processing parameters
        detection_cache_path: Absolute path to detection cache
        pose_cache_path: Absolute path to pose cache
        cache_base_path: Base cache directory

    Returns:
        CacheMetadata object
    """
    # Convert absolute paths to relative paths
    det_cache_relative = detection_cache_path.relative_to(cache_base_path)
    pose_cache_relative = pose_cache_path.relative_to(cache_base_path)

    return CacheMetadata(
        video_basename=video_basename,
        detection_config=detection_config,
        detection_checkpoint=detection_checkpoint,
        detection_confidence_threshold=processing_config.get('detection_confidence_threshold', 0.5),
        nms_type=processing_config.get('nms_type', 'strict'),
        nms_threshold=processing_config.get('nms_threshold', 0.7),
        bbox_min_height=processing_config.get('bbox_min_height', 50),
        bbox_min_width=processing_config.get('bbox_min_width', 50),
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        oks_nms_threshold=processing_config.get('oks_nms_threshold', 0.6),
        kpt_threshold=processing_config.get('kpt_threshold', 0.3),
        detection_cache_relative_path=str(det_cache_relative),
        pose_cache_relative_path=str(pose_cache_relative)
    )


# Example usage
if __name__ == "__main__":
    # Example: Save metadata
    manager = CacheMetadataManager()

    metadata = CacheMetadata(
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
        detection_cache_relative_path="detections/test_video/rtmdet_m_640-8xb32_coco-person_0.5_0.7_strict_50_50.h5",
        pose_cache_relative_path="pose/test_video/rtmdet_m_640-8xb32_coco-person_0.5_0.7_strict_50_50_td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.h5"
    )

    # Save as YAML (more readable)
    manager.save_metadata(metadata, format="yaml")

    # Load it back
    loaded = manager.load_metadata("test_video", format="yaml")
    if loaded is not None:
        print(f"\nLoaded metadata for: {loaded.video_basename}")
        print(f"Detection cache: {loaded.detection_cache_relative_path}")
        print(f"Pose cache: {loaded.pose_cache_relative_path}")

    # List all cached videos
    print(f"\nCached videos: {manager.list_cached_videos()}")