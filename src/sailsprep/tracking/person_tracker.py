"""Motion-compensated, Kalman-based person tracking (scaffold).

This module provides a structured, notebook-friendly API for person tracking:

- Camera motion compensation using optical flow.
- Adaptive thresholds based on scene crowding.
- Depth-aware Kalman filter design (placeholders for heavy deps).

Design goals
- Lazy import heavy deps (torch, opencv, mmdet, mmpose) inside methods.
- Clear, typed interfaces: process a single video or a folder.
- Safe defaults; helpful errors if tracking stack is missing.

Example
    from sailsprep.tracking.person_tracker import process_video
    process_video("input.mp4", "output.mp4", device="cuda:0")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np


# ----------------------------- Configuration ---------------------------------


@dataclass(frozen=True)
class TrackerConfig:
    """Adaptive tracking thresholds and constraints.

    Values mirror the provided notebook script. These can be tuned per video.
    """

    base_iou_threshold: float = 0.20
    base_motion_confidence: float = 0.25
    base_center_weight: float = 0.75
    max_lost_frames: int = 150
    confidence_decay_rate: float = 0.06
    max_jump_factor: float = 2.5


# ----------------------- Camera Motion Compensation --------------------------


class CameraMotionCompensator:
    """Estimate global camera motion via optical flow.

    Uses Lucas–Kanade optical flow on feature points outside the frame center to
    infer camera translation; smooths over a short history to reduce jitter.
    """

    def __init__(self) -> None:
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.motion_history: deque[Tuple[float, float]] = deque(maxlen=5)

    def estimate_camera_motion(self, frame: np.ndarray) -> tuple[float, float]:
        """Estimate camera motion (dx, dy) between frames in pixels.

        Args:
            frame: Current BGR frame.
        Returns:
            (dx, dy): Estimated camera translation in pixels.
        """
        try:
            import cv2  # Lazy import
        except Exception as exc:  # pragma: no cover - dependency gate
            raise ImportError(
                "OpenCV (cv2) is required for camera motion compensation.\n"
                "Install with: pip install opencv-python"
            ) from exc

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0, 0.0

        if self.prev_points is None or len(self.prev_points) < 50:
            h, w = gray.shape
            mask = np.ones_like(gray, dtype=np.uint8) * 255
            mask[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = 0
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_frame_gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=15,
                mask=mask,
            )

        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_frame_gray = gray
            self.motion_history.append((0.0, 0.0))
            return 0.0, 0.0

        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, gray, self.prev_points, None, winSize=(21, 21), maxLevel=3
        )

        if curr_points is None or status is None:
            self.prev_frame_gray = gray
            self.motion_history.append((0.0, 0.0))
            return 0.0, 0.0

        good_prev = self.prev_points[status.flatten() == 1]
        good_curr = curr_points[status.flatten() == 1]

        if len(good_prev) < 10:
            self.prev_frame_gray = gray
            self.prev_points = None
            self.motion_history.append((0.0, 0.0))
            return 0.0, 0.0

        dx = float(np.median(good_curr[:, 0, 0] - good_prev[:, 0, 0]))
        dy = float(np.median(good_curr[:, 0, 1] - good_prev[:, 0, 1]))

        self.motion_history.append((dx, dy))
        smoothed_dx = float(np.mean([m[0] for m in self.motion_history]))
        smoothed_dy = float(np.mean([m[1] for m in self.motion_history]))

        self.prev_frame_gray = gray
        self.prev_points = good_curr.reshape(-1, 1, 2)
        return smoothed_dx, smoothed_dy


# ------------------------------ Similarities ---------------------------------


def calculate_iou(box1: np.ndarray | Iterable[float], box2: np.ndarray | Iterable[float]) -> float:
    """Compute IoU between two boxes: [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = map(float, box1)
    x1_2, y1_2, x2_2, y2_2 = map(float, box2)
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def calculate_center_distance_similarity(
    box1: np.ndarray | Iterable[float], box2: np.ndarray | Iterable[float]
) -> float:
    """Similarity based on bbox center distance, normalized by size."""
    b1 = list(map(float, box1))
    b2 = list(map(float, box2))
    c1_x, c1_y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    c2_x, c2_y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    size1 = np.sqrt(max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1])))
    size2 = np.sqrt(max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1])))
    avg_size = (size1 + size2) / 2
    dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
    normalized = dist / (avg_size * 0.7)
    return float(max(0.0, 1.0 - normalized))


def calculate_combined_similarity(
    box1: np.ndarray | Iterable[float], box2: np.ndarray | Iterable[float], center_weight: float
) -> float:
    """Combine IoU and center-distance similarity with a weight."""
    iou = calculate_iou(box1, box2)
    center_sim = calculate_center_distance_similarity(box1, box2)
    return float((1 - center_weight) * iou + center_weight * center_sim)


def calculate_scene_crowding(bboxes: list[list[float]] | np.ndarray) -> float:
    """Estimate crowding from pairwise center distances relative to size.

    Returns:
        0.0 (isolated) → 1.0 (very crowded)
    """
    if len(bboxes) <= 1:
        return 0.0
    centers = []
    sizes = []
    for b in bboxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        size = np.sqrt(max(1e-6, (b[2] - b[0]) * (b[3] - b[1])))
        centers.append([cx, cy])
        sizes.append(size)
    centers_a = np.array(centers, dtype=float)
    avg_size = float(np.mean(sizes))
    if avg_size <= 1e-6:
        return 0.0
    min_d = float("inf")
    for i in range(len(centers_a)):
        for j in range(i + 1, len(centers_a)):
            d = float(np.linalg.norm(centers_a[i] - centers_a[j])) / avg_size
            min_d = min(min_d, d)
    if not np.isfinite(min_d):
        return 0.0
    if min_d < 1.0:
        return 1.0
    if min_d < 2.0:
        return 0.7
    if min_d < 3.0:
        return 0.4
    return 0.0


def get_adaptive_thresholds(cfg: TrackerConfig, crowding_factor: float) -> tuple[float, float, float]:
    """Interpolate thresholds based on crowding.

    Returns:
        (iou_threshold, center_weight, motion_conf)
    """
    iou_threshold = cfg.base_iou_threshold + (crowding_factor * 0.20)
    center_weight = cfg.base_center_weight - (crowding_factor * 0.35)
    motion_conf = cfg.base_motion_confidence + (crowding_factor * 0.25)
    return float(iou_threshold), float(center_weight), float(motion_conf)


def is_spatially_plausible(
    det_bbox: Iterable[float], predicted_bbox: Iterable[float], max_jump_factor: float
) -> bool:
    """Hard constraint to avoid impossible matches between frames."""
    db = list(map(float, det_bbox))
    pb = list(map(float, predicted_bbox))
    det_cx, det_cy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
    pred_cx, pred_cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
    det_size = np.sqrt(max(1e-6, (db[2] - db[0]) * (db[3] - db[1])))
    pred_size = np.sqrt(max(1e-6, (pb[2] - pb[0]) * (pb[3] - pb[1])))
    avg_size = (det_size + pred_size) / 2
    dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy)
    return bool(dist < (avg_size * max_jump_factor))


# ----------------------------- Kalman Utilities ------------------------------


def create_kalman_filter(initial_bbox: Iterable[float]):
    """Create a Kalman filter for bbox tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    Depth-aware noise: higher for size terms to accommodate approach/retreat.
    """
    try:
        from filterpy.kalman import KalmanFilter  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency gate
        raise ImportError("Install filterpy for Kalman filtering: pip install filterpy") from exc

    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    b = list(map(float, initial_bbox))
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    w = b[2] - b[0]
    h = b[3] - b[1]
    kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)

    # Initial covariance: fairly uncertain at start
    kf.P *= 100
    # Measurement noise: moderate trust in detector
    kf.R = np.diag([3.0, 3.0, 3.0, 3.0])
    # Process noise: low on position, high on size and size velocity
    Q = np.eye(8, dtype=float)
    Q[0, 0] = 0.05
    Q[1, 1] = 0.05
    Q[2, 2] = 2.0
    Q[3, 3] = 2.0
    Q[4, 4] = 0.1
    Q[5, 5] = 0.1
    Q[6, 6] = 3.0
    Q[7, 7] = 3.0
    kf.Q = Q
    return kf


def predict_motion_with_camera_compensation(
    kalman_filter: Any, missed_updates: int, camera_motion: tuple[float, float], *, cfg: TrackerConfig | None = None
) -> tuple[np.ndarray, float]:
    """Predict next bbox and return prediction confidence.

    Adds camera motion (dx, dy) to predicted center to compensate for pan/tilt.
    Confidence decays with missed updates, bounded below by 0.1.
    """
    _cfg = cfg or TrackerConfig()
    kalman_filter.predict()
    x_center, y_center, width, height = kalman_filter.x[:4]
    dx, dy = camera_motion
    x_center = float(x_center + dx)
    y_center = float(y_center + dy)
    pred = np.array([x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2], dtype=float)
    confidence = max(0.1, 1.0 - (missed_updates * _cfg.confidence_decay_rate))
    return pred, float(confidence)


def update_kalman_filter(kalman_filter: Any, measurement_bbox: Iterable[float]) -> None:
    """Correct Kalman filter state with a measurement bbox."""
    b = list(map(float, measurement_bbox))
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    w = b[2] - b[0]
    h = b[3] - b[1]
    kalman_filter.update(np.array([cx, cy, w, h], dtype=float))

# ----------------------------- Person Tracker --------------------------------


class PersonTracker:
    """High-level tracker with lazy model initialization.

    Note: Heavy dependencies (opencv, torch, mmdet, mmpose, filterpy) are
    imported inside methods that need them.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        det_config: Optional[str] = None,
        det_checkpoint: Optional[str] = None,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self.device = device or "cuda:0"
        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.cfg = tracker_config or TrackerConfig()

        self._detector = None
        self._pose = None
        self._camera = CameraMotionCompensator()

        # Tracking state (light scaffold; full implementation left to user code)
        self.frame_count: int = 0
        self.next_track_id: int = 1
        self.active_tracks: dict[int, dict[str, Any]] = {}

    # ---------------------------- Model loading -----------------------------

    def _ensure_models(self) -> None:
        """Lazily initialize detection and pose models if available."""
        if self._detector is not None and self._pose is not None:
            return
        try:  # Heavy deps
            from mmdet.apis import init_detector as _init_det  # type: ignore
            from mmpose.apis import init_model as _init_pose  # type: ignore
            from mmengine.registry import init_default_scope  # type: ignore
            init_default_scope("mmdet")
        except Exception as exc:  # pragma: no cover - dependency gate
            raise ImportError(
                "Tracking models require OpenMMLab packages (mmdet, mmpose, mmengine).\n"
                "Install with: pip install mmdet mmpose mmengine mmcv"
            ) from exc

        if not (self.det_config and self.det_checkpoint and self.pose_config and self.pose_checkpoint):
            raise ValueError(
                "det_config/det_checkpoint and pose_config/pose_checkpoint must be provided"
            )
        self._detector = _init_det(self.det_config, self.det_checkpoint, device=self.device)
        self._pose = _init_pose(self.pose_config, self.pose_checkpoint, device=self.device)

    # ----------------------------- API methods ------------------------------

    def process_video(self, in_path: str | Path, out_path: str | Path) -> None:
        """Process a single video and write tracked output.

        This is a scaffold; plug in detection, Kalman update, and rendering.
        """
        try:
            import cv2  # Lazy import
            from tqdm import tqdm  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency gate
            raise ImportError(
                "Processing requires opencv-python and tqdm.\n"
                "Install with: pip install opencv-python tqdm"
            ) from exc

        in_path = str(in_path)
        out_path = str(out_path)
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {in_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # Optional: initialize models only when needed
        # self._ensure_models()

        self.frame_count = 0
        self.active_tracks.clear()

        try:
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Tracking")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Estimate camera motion for this frame
                dx, dy = self._camera.estimate_camera_motion(frame)
                _ = (dx, dy)  # Currently unused in scaffold

                # TODO: add detection → prediction → matching → update → render
                # For now, write frames through to preserve a working pipeline.
                out.write(frame)

                self.frame_count += 1
                pbar.update(1)
        finally:
            cap.release()
            out.release()

    def process_folder(self, in_dir: str | Path, out_dir: str | Path, pattern: str = "*.mp4") -> None:
        """Process all videos in a folder matching a glob pattern."""
        in_p = Path(in_dir)
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        for vid in sorted(in_p.glob(pattern)):
            self.process_video(vid, out_p / vid.name)


# ----------------------------- Convenience API ------------------------------


def process_video(
    in_path: str | Path,
    out_path: str | Path,
    *,
    device: Optional[str] = None,
    det_config: Optional[str] = None,
    det_checkpoint: Optional[str] = None,
    pose_config: Optional[str] = None,
    pose_checkpoint: Optional[str] = None,
) -> None:
    """Convenience wrapper to process a single video.

    Provide model configs/checkpoints to enable detection/pose if integrating
    with OpenMMLab. The scaffold currently writes a passthrough video and sets
    up motion-compensation + tracking state.
    """
    tracker = PersonTracker(
        device=device,
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
    )
    tracker.process_video(in_path, out_path)


def process_folder(
    in_dir: str | Path,
    out_dir: str | Path,
    *,
    pattern: str = "*.mp4",
    device: Optional[str] = None,
    det_config: Optional[str] = None,
    det_checkpoint: Optional[str] = None,
    pose_config: Optional[str] = None,
    pose_checkpoint: Optional[str] = None,
) -> None:
    """Convenience wrapper to process a folder of videos."""
    tracker = PersonTracker(
        device=device,
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
    )
    tracker.process_folder(in_dir, out_dir, pattern=pattern)
