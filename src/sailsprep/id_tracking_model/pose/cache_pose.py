import contextlib
import gc
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms, oks_nms
from mmpose.registry import VISUALIZERS
from tqdm import tqdm

from sailsprep.id_tracking_model.utils.cache_manager import CacheManager
from sailsprep.id_tracking_model.utils.cache_metadata import (
    CacheMetadataManager,
    create_cache_metadata_from_config,
)
from sailsprep.id_tracking_model.utils.utils import soft_nms

# ================== COCO KEYPOINTS ==================
coco_kpts = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
             "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
             "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
             "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

coco_skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

head_kpts = [coco_kpts["nose"], coco_kpts["left_eye"], coco_kpts["right_eye"],
             coco_kpts["left_ear"], coco_kpts["right_ear"]]
upper_body_kpts = [coco_kpts["left_shoulder"], coco_kpts["right_shoulder"],
                   coco_kpts["left_elbow"], coco_kpts["right_elbow"],
                   coco_kpts["left_wrist"], coco_kpts["right_wrist"],
                   coco_kpts["left_hip"], coco_kpts["right_hip"]]
lower_body_kpts = [coco_kpts["left_hip"], coco_kpts["right_hip"],
                   coco_kpts["left_knee"], coco_kpts["right_knee"],
                   coco_kpts["left_ankle"], coco_kpts["right_ankle"]]
left_kpts = [coco_kpts["left_shoulder"], coco_kpts["left_elbow"], coco_kpts["left_wrist"],
             coco_kpts["left_hip"], coco_kpts["left_knee"], coco_kpts["left_ankle"]]
right_kpts = [coco_kpts["right_shoulder"], coco_kpts["right_elbow"], coco_kpts["right_wrist"],
              coco_kpts["right_hip"], coco_kpts["right_knee"], coco_kpts["right_ankle"]]
torso_kpts = [coco_kpts["left_shoulder"], coco_kpts["right_shoulder"],
              coco_kpts["left_hip"], coco_kpts["right_hip"]]

kpts_sets = [head_kpts, upper_body_kpts, lower_body_kpts, left_kpts, right_kpts, torso_kpts]

# ================== CONFIGURATION SYSTEM ==================
@dataclass
class ModelConfig:
    """Configuration for detection and pose models"""

    detection_config: str = "/orcd/data/satra/002/models/mmdet/dino-5scale_swin-l_8xb2-36e_coco.py"
    detection_checkpoint: str = '/orcd/data/satra/002/models/mmdet/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

    # Pose model
    pose_config: str = '/orcd/data/satra/002/models/mmpose/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
    pose_checkpoint: str = '/orcd/data/satra/002/models/mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

    device: str = 'cuda:0'
    mmpose_path: str = "mmpose"

@dataclass
class ProcessingConfig:
    """Configuration for detection and pose processing"""
    # Detection settings
    detection_confidence_threshold: float = 0.5
    nms_threshold: float = 0.7
    nms_type: str = "strict"  # "strict" or "soft"
    bbox_min_height: int = 50
    bbox_min_width: int = 50

    # Pose settings
    oks_nms_threshold: float = 0.6
    kpt_threshold: float = 0.3

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    enable_visualization: bool = True
    enable_pose_drawing: bool = True
    enable_bbox_drawing: bool = True
    radius: int = 3
    line_width: int = 1

@dataclass
class CacheConfig:
    """Configuration for caching detection and pose results"""
    enable_cache: bool = False
    cache_base_path: str = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking"
    force_recompute: bool = False

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    frame_limit: int = 0  # 0 means process all frames

    # Processing mode
    detection_only: bool = False  # If True, only run detection and skip tracking/pose


# ================== DETECTION MODULE ==================
class DetResult:
    """Wrapper for detection results"""
    def __init__(self, bboxes: np.ndarray, scores: np.ndarray) -> None:
        # bboxes: (N, 4), scores: (N,)
        self.bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)  # (N, 5): [x1, y1, x2, y2, score]

    def to_dict(self) -> dict[str, np.ndarray]:
        return {"bboxes": self.bboxes}


class DetectionModule:
    """Handles person detection"""

    def __init__(self, config: ModelConfig, processing_config: ProcessingConfig) -> None:
        self.config = config
        self.processing_config = processing_config
        self.detector: Any = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize detection model"""
        det_config = os.path.join(self.config.mmpose_path, self.config.detection_config)
        self.detector = init_detector(
            det_config,
            self.config.detection_checkpoint,
            device=self.config.device
        )

    def detect_single(
        self, frame: np.ndarray, return_raw: bool = False
    ) -> "tuple[DetResult, DetResult] | DetResult":
        """
        Detect persons in single frame

        Args:
            frame: Input frame
            return_raw: If True, return tuple (filtered_result, raw_result for caching)
        """
        scope = self.detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)

        detect_result = inference_detector(self.detector, frame)
        pred_instance = detect_result.pred_instances.cpu().numpy()

        # Filter by class (person=0)
        person_mask = pred_instance.labels == 0
        raw_det_result = DetResult(
            bboxes=pred_instance.bboxes[person_mask],
            scores=pred_instance.scores[person_mask],
        )

        # Apply filtering
        filtered_det_result = self._process_detection_result(raw_det_result)

        if return_raw:
            return filtered_det_result, raw_det_result
        return filtered_det_result

    def process_cached_detection(self, cached_det: dict[str, np.ndarray]) -> DetResult:
        """Process cached detection data"""
        det_result = DetResult(cached_det['bboxes'][:, :4], cached_det['bboxes'][:, 4])
        return self._process_detection_result(det_result)

    def _process_detection_result(self, det_result: DetResult) -> DetResult:
        """Apply filtering to detection results"""
        bboxes = det_result.bboxes

        # Filter by confidence
        bboxes = bboxes[bboxes[:, 4] > self.processing_config.detection_confidence_threshold]

        # Filter by bounding box dimensions
        if len(bboxes) > 0:
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            valid_mask = np.logical_and(
                widths >= self.processing_config.bbox_min_width,
                heights >= self.processing_config.bbox_min_height
            )
            bboxes = bboxes[valid_mask]

        # Apply NMS
        if len(bboxes) > 0:
            keep_indices = (nms(bboxes, self.processing_config.nms_threshold)
                            if self.processing_config.nms_type == "strict"
                            else soft_nms(bboxes, self.processing_config.nms_threshold, method="gaussian"))  # type: ignore[no-untyped-call]
            bboxes = bboxes[keep_indices]
            return DetResult(bboxes[:, :4], bboxes[:, 4])
        else:
            return DetResult(np.empty((0, 4)), np.empty(0))


# ================== POSE ESTIMATION MODULE ==================
class PoseResult:
    """Wrapper for pose results"""
    def __init__(self, keypoints: np.ndarray, bbox: np.ndarray) -> None:
        # keypoints: (N_kpts, 3) [x, y, score]
        # bbox: (4,) [x1, y1, x2, y2]
        self.keypoints = keypoints
        self.bbox = bbox
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "keypoints": self.keypoints,
            "bbox": self.bbox,
            "metadata": self.metadata
        }


class PoseEstimationModule:
    """Handles pose estimation"""

    def __init__(self, config: ModelConfig, visualization_config: VisualizationConfig,
                 processing_config: ProcessingConfig) -> None:
        self.config = config
        self.vis_config = visualization_config
        self.processing_config = processing_config
        self.pose_estimator: Any = None
        self.visualizer: Any = None
        self._init_pose_estimator()

    def _init_pose_estimator(self) -> None:
        """Initialize pose estimation model"""
        pose_config = os.path.join(self.config.mmpose_path, self.config.pose_config)
        self.pose_estimator = init_pose_estimator(
            pose_config,
            self.config.pose_checkpoint,
            device=self.config.device,
            cfg_options=dict(
                model=dict(
                    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True)
                )
            )
        )

        # Setup visualizer
        if self.vis_config.enable_visualization and self.vis_config.enable_pose_drawing:
            self.pose_estimator.cfg.visualizer.radius = self.vis_config.radius
            self.pose_estimator.cfg.visualizer.line_width = self.vis_config.line_width
            self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
            self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta)

    def _create_pose_result(self, pose_results: list[Any]) -> list[PoseResult]:
        """Convert MMPose results to PoseResult objects"""
        unified_pose_results = []
        for pose in pose_results:
            bbox = pose.pred_instances.bboxes[0]
            if hasattr(bbox, 'cpu'):
                bbox = bbox.cpu().numpy()

            keypoints_xy = pose.pred_instances.keypoints[0]
            keypoint_scores = pose.pred_instances.keypoint_scores[0]

            # Combine keypoints and scores
            if torch.is_tensor(keypoints_xy):
                keypoints_xy = keypoints_xy.cpu().numpy()
            if torch.is_tensor(keypoint_scores):
                keypoint_scores = keypoint_scores.cpu().numpy()

            keypoints = np.concatenate([
                keypoints_xy,
                keypoint_scores.reshape(-1, 1)
            ], axis=1)

            unified_pose_results.append(PoseResult(keypoints, bbox))
        return unified_pose_results

    def estimate_single(self, frame: np.ndarray, det_result: DetResult) -> list[PoseResult]:
        """Estimate poses for single frame"""
        if len(det_result.bboxes) == 0:
            return []

        pose_results = inference_topdown(self.pose_estimator, frame, det_result.bboxes[:, :4])
        return self._create_pose_result(pose_results)

    @staticmethod
    def create_pose_results_from_cache(cached_poses: list[dict[str, np.ndarray]]) -> list[PoseResult]:
        """Convert cached pose data to PoseResult objects"""
        return [
            PoseResult(
                keypoints=p['keypoints'],
                bbox=p['bbox']
            )
            for p in cached_poses
        ]

    def apply_oks_nms(self, pose_results: list[PoseResult]) -> list[PoseResult]:
        """Apply OKS NMS to remove duplicate poses"""
        if len(pose_results) == 0:
            return []

        kdb = []
        for pose in pose_results:
            bbox = pose.bbox
            x1, y1, x2, y2 = bbox
            keypoints = pose.keypoints[:17]  # Use COCO 17 keypoints

            area = max(1.0, (x2 - x1) * (y2 - y1))
            mean_kpt = float(keypoints[:, 2].mean())
            kdb.append({
                'keypoints': keypoints,
                'area': area,
                'score': mean_kpt
            })

        keep_idx = oks_nms(
            kpts_db=kdb,
            thr=self.processing_config.oks_nms_threshold,
            vis_thr=self.processing_config.kpt_threshold,
        )

        return [pose_results[i] for i in keep_idx]

    def filter_poses_by_keypoints(self, pose_results: list[PoseResult]) -> list[PoseResult]:
        """Filter poses based on keypoint visibility"""
        for pose in pose_results:
            keypoint_scores = pose.keypoints[:17, 2]  # Get scores from first 17 keypoints

            # Check if any keypoint set is fully visible
            enough_kpts_bool = False
            for kpts_set in kpts_sets:
                scores = keypoint_scores[kpts_set]
                if np.sum(scores > self.processing_config.kpt_threshold) == len(kpts_set):
                    enough_kpts_bool = True
                    break
            pose.metadata['sufficient_keypoints'] = enough_kpts_bool

        return pose_results


# ================== VISUALIZATION MODULE ==================
class VisualizationModule:
    """Handles visualization of detection and pose results"""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def draw_results(self, frame: np.ndarray, pose_results: list[PoseResult],
                    det_result: DetResult | None = None) -> np.ndarray:
        """Draw detection boxes and poses on frame"""
        vis_frame = frame.copy()

        # Draw bounding boxes
        if self.config.enable_bbox_drawing and det_result is not None:
            for bbox in det_result.bboxes:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                score = bbox[4]
                cv2.putText(vis_frame, f"{score:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw poses
        if self.config.enable_pose_drawing:
            for pose in pose_results:
                # Draw keypoints
                for _i, (x, y, score) in enumerate(pose.keypoints[:17]):
                    if score > 0.3:
                        cv2.circle(vis_frame, (int(x), int(y)),
                                 self.config.radius, (0, 255, 255), -1)

                # Draw skeleton
                for start_idx, end_idx in coco_skeleton:
                    if start_idx < len(pose.keypoints) and end_idx < len(pose.keypoints):
                        start_pt = pose.keypoints[start_idx]
                        end_pt = pose.keypoints[end_idx]
                        if start_pt[2] > 0.3 and end_pt[2] > 0.3:
                            cv2.line(vis_frame,
                                   (int(start_pt[0]), int(start_pt[1])),
                                   (int(end_pt[0]), int(end_pt[1])),
                                   (255, 0, 0), self.config.line_width)

        return vis_frame


# ================== MAIN PIPELINE ==================
class DetectionPosePipeline:
    """Main pipeline for detection and pose estimation only"""

    def __init__(self, config: PipelineConfig, batch_signal_handler: Any = None) -> None:
        self.config = config
        self.batch_signal_handler = batch_signal_handler

        # Initialize modules
        self.detection_module = DetectionModule(config.models, config.processing)
        self.pose_module = PoseEstimationModule(config.models, config.visualization, config.processing)
        self.visualization_module: VisualizationModule | None = (VisualizationModule(
            config.visualization,
        ) if config.visualization.enable_visualization else None)

        # Performance tracking
        self.timing_stats: dict[str, list[float]] = defaultdict(list)
        self.global_start_time: float | None = None

        # Interruption handling
        self._interrupted = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._cap: cv2.VideoCapture | None = None

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown on Ctrl+C"""
        print("\n\nInterrupted! Finalizing video...")
        self._interrupted = True

        if self._proc and self._proc.stdin:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()

        if self._cap:
            with contextlib.suppress(Exception):
                self._cap.release()

        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        torch.cuda.empty_cache()
        gc.collect()

        self.print_performance_stats()

        # Chain to parent handler if provided
        if self.batch_signal_handler:
            self.batch_signal_handler(signum, frame)
        else:
            sys.exit(0)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        cached_detection: dict[str, np.ndarray] | None = None,
        cached_poses: list[dict[str, np.ndarray]] | None = None,
        saved_detections: dict[int, Any] | None = None,
        saved_poses: dict[int, Any] | None = None
    ) -> np.ndarray:
        """Process single frame"""
        # Detection
        det_start = time.time()
        if cached_detection is not None:
            # Use cached detection
            det_result: DetResult = self.detection_module.process_cached_detection(cached_detection)
        else:
            # Compute detection and optionally save for caching
            if saved_detections is not None and frame_idx is not None:
                result = self.detection_module.detect_single(frame, return_raw=True)
                det_result, raw_det_result = result  # type: ignore[misc]
                saved_detections[frame_idx] = raw_det_result.to_dict()
            else:
                det_result = self.detection_module.detect_single(frame)  # type: ignore[assignment]
        self.timing_stats['detection'].append(time.time() - det_start)

        # Detection-only mode: draw bboxes and return
        if self.config.detection_only:
            vis_frame = frame
            if self.config.visualization.enable_visualization:
                vis_frame = frame.copy()
                for bbox in det_result.bboxes:
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return vis_frame

        # Pose estimation
        pose_start = time.time()
        pose_results: list[PoseResult]
        if cached_poses is not None:
            # Use cached poses
            pose_results = self.pose_module.create_pose_results_from_cache(cached_poses)
        else:
            # Compute poses and optionally save for caching
            pose_results = self.pose_module.estimate_single(frame, det_result)
            if saved_poses is not None and frame_idx is not None:
                saved_poses[frame_idx] = [p.to_dict() for p in pose_results]
        self.timing_stats['pose'].append(time.time() - pose_start)

        # Apply OKS NMS and keypoint filtering
        oks_start = time.time()
        pose_results = self.pose_module.apply_oks_nms(pose_results)
        pose_results = self.pose_module.filter_poses_by_keypoints(pose_results)
        self.timing_stats['oks_nms'].append(time.time() - oks_start)

        # Visualization
        vis_frame = frame
        if self.visualization_module:
            vis_start = time.time()
            vis_frame = self.visualization_module.draw_results(frame, pose_results, det_result)
            self.timing_stats['visualization'].append(time.time() - vis_start)

        return vis_frame

    def process_video(self, input_path: str, output_path: str) -> None:
        """Process entire video"""
        self.global_start_time = time.time()

        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        self._cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.config.frame_limit and total_frames > self.config.frame_limit:
            total_frames = self.config.frame_limit

        # Initialize cache manager if enabled
        cache_manager: CacheManager | None = None
        detection_cache: dict[int, Any] | None = None
        pose_cache: dict[int, Any] | None = None
        detections_to_save: dict[int, Any] = {}
        poses_to_save: dict[int, Any] = {}

        if self.config.cache.enable_cache:
            cache_manager = CacheManager(
                output_video_path=output_path,
                detection_config=self.config.models.detection_config,
                pose_config=self.config.models.pose_config,
                detection_confidence_threshold=self.config.processing.detection_confidence_threshold,
                nms_type=self.config.processing.nms_type,
                nms_threshold=self.config.processing.nms_threshold,
                bbox_min_height=self.config.processing.bbox_min_height,
                bbox_min_width=self.config.processing.bbox_min_width,
                cache_base_path=self.config.cache.cache_base_path
            )

            # Load caches if they exist
            if not self.config.cache.force_recompute:
                if cache_manager.check_detection_cache():
                    print("Loading detection cache...")
                    detection_cache = cache_manager.load_all_detections()
                    if detection_cache:
                        print(f"  Loaded {len(detection_cache)} cached detections")

                if cache_manager.check_pose_cache():
                    print("Loading pose cache...")
                    pose_cache = cache_manager.load_all_poses()
                    if pose_cache:
                        print(f"  Loaded {len(pose_cache)} cached poses")

        proc: subprocess.Popen[bytes] | None = None
        if self.config.visualization.enable_visualization:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Setup ffmpeg for video encoding
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                output_path
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._proc = proc

        print(f"Processing {total_frames} frames...")
        frame_idx = 0

        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened() and not self._interrupted:
                    ret, frame = cap.read()
                    if not ret or (self.config.frame_limit and frame_idx >= self.config.frame_limit):
                        break

                    try:
                        frame_start = time.time()
                        vis_frame = self.process_frame(
                            frame,
                            frame_idx,
                            detection_cache[frame_idx] if detection_cache else None,
                            pose_cache[frame_idx] if pose_cache else None,
                            saved_detections=detections_to_save if self.config.cache.enable_cache else None,
                            saved_poses=poses_to_save if self.config.cache.enable_cache else None
                        )
                        if self.config.visualization.enable_visualization and proc is not None:
                            # Ensure frame size matches
                            if vis_frame.shape[0] != height or vis_frame.shape[1] != width:
                                vis_frame = cv2.resize(vis_frame, (width, height), interpolation=cv2.INTER_LINEAR)

                            # Write frame to ffmpeg
                            try:
                                assert proc.stdin is not None
                                proc.stdin.write(vis_frame.tobytes())
                            except BrokenPipeError as err:
                                raise RuntimeError("ffmpeg exited early") from err

                        self.timing_stats['total_frame'].append(time.time() - frame_start)

                        # Periodic cleanup
                        if frame_idx % 100 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()

                    except Exception as e:
                        import traceback
                        print(f"Error processing frame {frame_idx}: {e}")
                        traceback.print_exc()
                        # Write original frame on error
                        if self.config.visualization.enable_visualization and proc is not None:
                            try:
                                if frame.shape[0] != height or frame.shape[1] != width:
                                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                                assert proc.stdin is not None
                                proc.stdin.write(frame.tobytes())
                            except BrokenPipeError:
                                break

                    frame_idx += 1
                    pbar.update(1)

            # Save caches if enabled
            if self.config.cache.enable_cache and cache_manager is not None:
                if detections_to_save:
                    print("Saving detection cache...")
                    cache_manager.save_all_detections(detections_to_save)
                if poses_to_save:
                    print("Saving pose cache...")
                    cache_manager.save_all_poses(poses_to_save)

                # Save cache metadata for tracker_clip to discover
                if poses_to_save:
                    print("Saving cache metadata...")
                    try:
                        metadata_manager = CacheMetadataManager(self.config.cache.cache_base_path)
                        metadata = create_cache_metadata_from_config(
                            video_basename=cache_manager.video_basename,
                            detection_config=self.config.models.detection_config,
                            detection_checkpoint=self.config.models.detection_checkpoint,
                            pose_config=self.config.models.pose_config,
                            pose_checkpoint=self.config.models.pose_checkpoint,
                            processing_config={
                                'detection_confidence_threshold': self.config.processing.detection_confidence_threshold,
                                'nms_type': self.config.processing.nms_type,
                                'nms_threshold': self.config.processing.nms_threshold,
                                'bbox_min_height': self.config.processing.bbox_min_height,
                                'bbox_min_width': self.config.processing.bbox_min_width,
                                'oks_nms_threshold': self.config.processing.oks_nms_threshold,
                                'kpt_threshold': self.config.processing.kpt_threshold
                            },
                            detection_cache_path=cache_manager.detection_cache_path,
                            pose_cache_path=cache_manager.pose_cache_path,
                            cache_base_path=cache_manager.cache_base_path
                        )
                        metadata_manager.save_metadata(metadata, format="yaml")
                    except Exception as e:
                        print(f"Warning: Could not save cache metadata: {e}")
                        print("Tracking will still work if you provide cache parameters manually.")

        finally:
            cap.release()
            if self.config.visualization.enable_visualization and proc is not None:
                if proc.stdin:
                    with contextlib.suppress(BrokenPipeError):
                        proc.stdin.close()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"ffmpeg failed with code {rc}")

        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nProcessing complete. Output saved: {output_path}")
        self.print_performance_stats()

    def print_performance_stats(self) -> None:
        """Print performance statistics"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        if self.timing_stats:
            for operation, times in self.timing_stats.items():
                if times:
                    avg_time = np.mean(times)
                    total_time = np.sum(times)
                    count = len(times)
                    print(f"{operation}:")
                    print(f"  Count: {count}")
                    print(f"  Total: {total_time:.3f}s")
                    print(f"  Average: {avg_time:.3f}s")
                    print(f"  Min: {np.min(times):.3f}s")
                    print(f"  Max: {np.max(times):.3f}s")
                    print()

        if self.global_start_time:
            total_runtime = time.time() - self.global_start_time
            frames_processed = len(self.timing_stats.get('detection', []))
            fps = frames_processed / total_runtime if total_runtime > 0 else 0
            print("Overall Stats:")
            print(f"  Total runtime: {total_runtime:.2f}s")
            print(f"  Frames processed: {frames_processed}")
            print(f"  Processing FPS: {fps:.2f}")

        print("="*60)


# ================== EXAMPLE USAGE ==================
def create_default_config() -> PipelineConfig:
    """Create default configuration"""
    config = PipelineConfig()

    # Detection settings
    config.processing.detection_confidence_threshold = 0.4
    config.processing.nms_threshold = 0.7
    config.processing.nms_type = "strict"
    config.processing.bbox_min_height = 50
    config.processing.bbox_min_width = 50

    # Pose settings
    config.processing.oks_nms_threshold = 0.55
    config.processing.kpt_threshold = 0.5

    # Visualization
    config.visualization.enable_visualization = False
    config.visualization.enable_pose_drawing = True
    config.visualization.enable_bbox_drawing = True
    config.visualization.radius = 3
    config.visualization.line_width = 1

    # Cache
    config.cache.enable_cache = True

    config.detection_only = False
    return config


def main() -> None:
    config = create_default_config()

    CSV_PATH = "/home/aparnabg/orcd/scratch/videos.csv"
    TEST_NAME = "dino-5scale-swin-l"

    df = pd.read_csv(CSV_PATH)

    pipeline = DetectionPosePipeline(config)

    for source_video_path in df["video_path"].dropna():

        video_name = os.path.splitext(
            os.path.basename(source_video_path)
        )[0]

        target_video_path = (
            f"/home/aparnabg/orcd/scratch/motion_tracking_output/det/{TEST_NAME}/"
            f"{video_name}.mp4"
        )

        pipeline.process_video(source_video_path, target_video_path)


if __name__ == "__main__":
    main()