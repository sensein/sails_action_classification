"""
Cache Manager for Detection and Pose Results

Provides bulk loading/saving of detection and pose inference results using HDF5
for efficient reuse across multiple tracking runs with different parameters.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from sailsprep.id_tracking_model.utils.cache_metadata import CacheMetadataManager

class CacheManager:
    """Manages caching of detection and pose estimation results"""

    def __init__(
        self,
        output_video_path: str,
        detection_config: str,
        pose_config: str,
        detection_confidence_threshold: float,
        nms_type: str,
        nms_threshold: float,
        bbox_min_height: int,
        bbox_min_width: int,
        cache_base_path: str = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking"
    ):
        """
        Initialize cache manager

        Args:
            output_video_path: Path to output video (used to name cache folder)
            detection_config: Full path to detection config
            pose_config: Full path to pose config
            detection_confidence_threshold: Detection confidence threshold
            nms_threshold: NMS threshold for detections
            bbox_min_height: Minimum bbox height
            bbox_min_width: Minimum bbox width
            cache_base_path: Base directory for cache storage
        """
        self.cache_base_path = Path(cache_base_path)

        # Extract video basename (without extension) for cache folder name
        self.video_basename = Path(output_video_path).stem

        # Extract config names from paths
        self.detection_config_name = Path(detection_config).stem
        self.pose_config_name = Path(pose_config).stem

        # Store parameters for pose cache naming
        self.det_conf = detection_confidence_threshold
        self.nms_thresh = nms_threshold
        self.nms_type = nms_type
        self.bbox_h = bbox_min_height
        self.bbox_w = bbox_min_width

        # Generate cache file paths
        self.detection_cache_path = self._get_detection_cache_path()
        self.pose_cache_path = self._get_pose_cache_path()

    def _get_detection_cache_path(self) -> Path:
        """Generate detection cache file path"""
        cache_dir = self.cache_base_path / "detections" / self.video_basename
        return cache_dir / f"{self.detection_config_name}_{self.det_conf}_{self.nms_thresh}_{self.nms_type}_{self.bbox_h}_{self.bbox_w}.h5"

    def _get_pose_cache_path(self) -> Path:
        """Generate pose cache file path based on parameters"""
        cache_dir = self.cache_base_path / "pose" / self.video_basename
        filename = f"{self.detection_config_name}_{self.det_conf}_{self.nms_thresh}_{self.nms_type}_{self.bbox_h}_{self.bbox_w}_{self.pose_config_name}.h5"
        return cache_dir / filename

    def check_detection_cache(self) -> bool:
        """Check if detection cache exists"""
        return self.detection_cache_path.exists()

    def check_pose_cache(self) -> bool:
        """Check if pose cache exists"""
        return self.pose_cache_path.exists()

    def load_all_detections(self) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Load all detection results from cache

        Returns:
            Dict mapping frame_idx to detection dict with keys:
            - 'bboxes': (N, 5) array
            Returns None if cache doesn't exist
        """
        if not self.check_detection_cache():
            return None

        detections = {}

        try:
            with h5py.File(self.detection_cache_path, 'r') as f:
                # Iterate through all frame groups
                for frame_key in f.keys():
                    frame_idx = int(frame_key.split('_')[1])
                    frame_group = f[frame_key]

                    detections[frame_idx] = {
                        'bboxes': frame_group['bboxes'][:],
                    }

            return detections

        except Exception as e:
            print(f"Error loading detection cache: {e}")
            return None

    def load_all_poses(self) -> Optional[Dict[int, List[Dict[str, np.ndarray]]]]:
        """
        Load all pose results from cache

        Returns:
            Dict mapping frame_idx to list of pose dicts, each containing:
            - 'keypoints': (N_kpts, 3) array
            - 'bbox': (4,) array
            Returns None if cache doesn't exist
        """
        if not self.check_pose_cache():
            return None

        poses = {}

        try:
            with h5py.File(self.pose_cache_path, 'r') as f:
                # Iterate through all frame groups
                for frame_key in f.keys():
                    frame_idx = int(frame_key.split('_')[1])
                    frame_group = f[frame_key]

                    frame_poses = []
                    # Iterate through all poses in this frame
                    for pose_key in sorted(frame_group.keys()):
                        pose_group = frame_group[pose_key]

                        frame_poses.append({
                            'keypoints': pose_group['keypoints'][:],
                            'bbox': pose_group['bbox'][:]
                        })

                    poses[frame_idx] = frame_poses

            return poses

        except Exception as e:
            print(f"Error loading pose cache: {e}")
            return None

    def save_all_detections(self, detections: Dict[int, Dict]):
        """
        Save all detection results to cache

        Args:
            detections: Dict mapping frame_idx to a detection dictionary
                       Each dictionary should have a .bboxes attributes
        """
        # Create cache directory if needed
        self.detection_cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(self.detection_cache_path, 'w') as f:
                for frame_idx, det_result in detections.items():
                    frame_group = f.create_group(f'frame_{frame_idx}')

                    # Save with compression
                    frame_group.create_dataset(
                        'bboxes', data=det_result["bboxes"], compression='gzip', compression_opts=4
                    )

            print(f"Saved detection cache to: {self.detection_cache_path}")

        except Exception as e:
            print(f"Error saving detection cache: {e}")
            raise

    def save_all_poses(self, poses: Dict[int, List[Dict]]):
        """
        Save all pose results to cache

        Args:
            poses: Dict mapping frame_idx to list of pose_result dictionaries
                   Each pose_result should have keypoints and bbox attributes.
        """
        # Create cache directory if needed
        self.pose_cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(self.pose_cache_path, 'w') as f:
                for frame_idx, pose_list in poses.items():
                    frame_group = f.create_group(f'frame_{frame_idx}')

                    for pose_idx, pose_result in enumerate(pose_list):
                        pose_group = frame_group.create_group(f'pose_{pose_idx}')

                        # Save with compression
                        pose_group.create_dataset(
                            'keypoints', data=pose_result["keypoints"], compression='gzip', compression_opts=4
                        )

                        pose_group.create_dataset(
                            'bbox', data=pose_result["bbox"], compression='gzip', compression_opts=4
                        )

            print(f"Saved pose cache to: {self.pose_cache_path}")

        except Exception as e:
            print(f"Error saving pose cache: {e}")
            raise

    def get_cache_params(self) -> Dict:
        """
        Get cache parameters for metadata saving

        Returns:
            Dictionary with cache configuration parameters
        """
        return {
            'video_basename': self.video_basename,
            'detection_config_name': self.detection_config_name,
            'pose_config_name': self.pose_config_name,
            'detection_confidence_threshold': self.det_conf,
            'nms_type': self.nms_type,
            'nms_threshold': self.nms_thresh,
            'bbox_min_height': self.bbox_h,
            'bbox_min_width': self.bbox_w,
            'detection_cache_path': str(self.detection_cache_path),
            'pose_cache_path': str(self.pose_cache_path)
        }

    @classmethod
    def from_metadata(
        cls,
        video_filename: str,
        cache_base_path: str = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking",
        metadata_format: str = "yaml"
    ) -> Optional['CacheManager']:
        """
        Create CacheManager from cached metadata file

        Args:
            video_filename: Video filename
            cache_base_path: Base directory for cache storage
            metadata_format: "yaml" or "json"

        Returns:
            CacheManager instance or None if metadata not found

        Example:
            >>> manager = CacheManager.from_metadata("my_video")
            >>> if manager:
            >>>     pose_cache = manager.load_all_poses()
        """
        video_basename = Path(video_filename).stem

        # Load metadata
        metadata_manager = CacheMetadataManager(cache_base_path)
        metadata = metadata_manager.load_metadata(video_basename, format=metadata_format)

        if not metadata:
            print(f"No cache metadata found for video: {video_basename}")
            print(f"Available cached videos: {metadata_manager.list_cached_videos()}")
            return None

        # Create CacheManager using metadata
        # Use a dummy output_video_path since we're only loading
        dummy_output_path = f"{cache_base_path}/dummy/{video_basename}.mp4"

        return cls(
            output_video_path=dummy_output_path,
            detection_config=metadata.detection_config,
            pose_config=metadata.pose_config,
            detection_confidence_threshold=metadata.detection_confidence_threshold,
            nms_type=metadata.nms_type,
            nms_threshold=metadata.nms_threshold,
            bbox_min_height=metadata.bbox_min_height,
            bbox_min_width=metadata.bbox_min_width,
            cache_base_path=cache_base_path
        )
