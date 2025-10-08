"""Tracking utilities for sailsprep.

Exposes high-level entry points for motion-compensated, Kalman-based person
tracking suitable for use in notebooks and CLIs.
"""

from .person_tracker import (
    CameraMotionCompensator,
    PersonTracker,
    TrackerConfig,
    create_kalman_filter,
    predict_motion_with_camera_compensation,
    update_kalman_filter,
    calculate_center_distance_similarity,
    calculate_combined_similarity,
    calculate_iou,
    calculate_scene_crowding,
    get_adaptive_thresholds,
    is_spatially_plausible,
    process_folder,
    process_video,
)

__all__ = [
    "CameraMotionCompensator",
    "PersonTracker",
    "TrackerConfig",
    "create_kalman_filter",
    "predict_motion_with_camera_compensation",
    "update_kalman_filter",
    "calculate_center_distance_similarity",
    "calculate_combined_similarity",
    "calculate_iou",
    "calculate_scene_crowding",
    "get_adaptive_thresholds",
    "is_spatially_plausible",
    "process_folder",
    "process_video",
]
