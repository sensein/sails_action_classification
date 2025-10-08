"""Motion + Appearance person tracking using sailsprep.tracking helpers.

Replicates the original notebook pipeline while delegating Kalman and camera
motion utilities to `sailsprep.tracking.person_tracker`.

This script initializes OpenMMLab detectors/pose, runs per-frame detection,
pose, feature extraction (face/upper/lower), motion-then-appearance matching,
draws results, and writes an output video. It also supports batch processing
for a folder of videos.

Run examples
- Single video:
  python notebooks/scripts/motion_appearance_reid.py --in video.mp4 --out video_tracked.mp4

- Folder:
  python notebooks/scripts/motion_appearance_reid.py --in-dir notebooks --out-dir notebooks/tracked
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _add_repo_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_repo_to_path()


# Heavy imports after sys.path tweak
import cv2  # type: ignore
import torch  # type: ignore
import contextlib
from tqdm import tqdm  # type: ignore
from scipy.optimize import linear_sum_assignment  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from facenet_pytorch import MTCNN  # type: ignore
from deepface import DeepFace  # type: ignore

from mmengine.registry import init_default_scope  # type: ignore
from mmpose.apis import inference_topdown, init_model as init_pose_estimator  # type: ignore
from mmpose.evaluation.functional import nms  # type: ignore
from mmpose.registry import VISUALIZERS  # type: ignore
from mmpose.structures import merge_data_samples  # type: ignore
from mmdet.apis import inference_detector, init_detector  # type: ignore

from sailsprep.tracking import (
    CameraMotionCompensator,
    TrackerConfig,
    calculate_iou,
    calculate_scene_crowding,
    calculate_combined_similarity,
    get_adaptive_thresholds,
    is_spatially_plausible,
    create_kalman_filter,
    predict_motion_with_camera_compensation,
    update_kalman_filter,
)


# ------------------------------ Config defaults ------------------------------


DEFAULT_DET_CONFIG = "projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py"
DEFAULT_DET_CKPT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_"
    "coco-obj365-person-235e8209.pth"
)
DEFAULT_POSE_CONFIG = (
    "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/"
    "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
)
DEFAULT_POSE_CKPT = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
)


# ------------------------------- Params/State --------------------------------


iou_threshold = 0.3
motion_confidence_threshold = 0.5
feature_update_interval = 10
max_lost_frames = 300
face_reid_threshold = 0.75
upper_reid_threshold = 0.65
lower_reid_threshold = 0.6
combined_reid_threshold = 0.7


def compute_feature_similarity(feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]) -> float:
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        sim = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
        return float(max(0.0, sim))
    except Exception:
        return 0.0


def determine_pose_type(keypoints: np.ndarray | torch.Tensor) -> str:
    try:
        kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
        if len(kpts.shape) == 3:
            kpts = kpts[0]
        if len(kpts) < 17:
            return "standing"
        vis = {}
        for name, idx in [("left_hip", 11), ("right_hip", 12), ("left_knee", 13), ("right_knee", 14), ("left_ankle", 15), ("right_ankle", 16)]:
            if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                vis[name] = kpts[idx][:2]
        if len(vis) < 3:
            return "standing"
        hip_y = [vis[k][1] for k in ("left_hip", "right_hip") if k in vis]
        ankle_y = [vis[k][1] for k in ("left_ankle", "right_ankle") if k in vis]
        if hip_y and ankle_y:
            hip_ankle = abs(np.mean(ankle_y) - np.mean(hip_y))
            if hip_ankle < 50:
                return "lying"
            if hip_ankle < 120:
                return "sitting"
            return "standing"
        return "standing"
    except Exception:
        return "standing"


def compute_lbp_histogram(gray_image: np.ndarray) -> np.ndarray:
    try:
        rows, cols = gray_image.shape
        lbp_image = np.zeros_like(gray_image)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray_image[i, j]
                code = 0
                neighbors = [
                    gray_image[i - 1, j - 1],
                    gray_image[i - 1, j],
                    gray_image[i - 1, j + 1],
                    gray_image[i, j + 1],
                    gray_image[i + 1, j + 1],
                    gray_image[i + 1, j],
                    gray_image[i + 1, j - 1],
                    gray_image[i, j - 1],
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor > center:
                        code |= (1 << k)
                lbp_image[i, j] = code
        hist, _ = np.histogram(lbp_image, bins=24, range=(0, 256))
        return hist / (hist.sum() + 1e-7)
    except Exception:
        return np.zeros(24, dtype=float)


def extract_face_region(frame: np.ndarray, keypoints: Any, bbox: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    try:
        kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
        if len(kpts.shape) == 3:
            kpts = kpts[0]
        head_idx = [0, 1, 2, 3, 4]
        pts = []
        for idx in head_idx:
            if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                pts.append(kpts[idx][:2])
        if len(pts) >= 2:
            pts = np.array(pts)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            pad = 25
            fx1, fy1 = max(0, int(x_min - pad)), max(0, int(y_min - pad))
            fx2, fy2 = min(frame.shape[1], int(x_max + pad)), min(frame.shape[0], int(y_max + pad))
        else:
            x1, y1, x2, y2 = bbox.astype(int)
            fh = int((y2 - y1) * 0.35)
            fx1, fy1, fx2, fy2 = x1, y1, x2, y1 + fh
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        roi = frame[fy1:fy2, fx1:fx2]
        if roi.shape[0] < 40 or roi.shape[1] < 30:
            return None
        return roi, (fx1, fy1, fx2, fy2)
    except Exception:
        return None


def extract_upper_body_region(frame: np.ndarray, keypoints: Any, bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    try:
        kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
        if len(kpts.shape) == 3:
            kpts = kpts[0]
        neck_point = None
        left_shoulder = kpts[5] if len(kpts) > 5 and len(kpts[5]) >= 3 and kpts[5][2] > 0.3 else None
        right_shoulder = kpts[6] if len(kpts) > 6 and len(kpts[6]) >= 3 and kpts[6][2] > 0.3 else None
        if left_shoulder is not None and right_shoulder is not None:
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2 - 15
            neck_point = np.array([neck_x, neck_y])
        hip_points = []
        for idx in [11, 12]:
            if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                hip_points.append(kpts[idx][:2])
        if neck_point is not None and hip_points:
            hip_center = np.mean(hip_points, axis=0)
            upper_y1 = int(neck_point[1])
            upper_y2 = int(hip_center[1])
            if left_shoulder is not None and right_shoulder is not None:
                x_min = min(left_shoulder[0], right_shoulder[0])
                x_max = max(left_shoulder[0], right_shoulder[0])
                padding = 20
                upper_x1 = max(0, int(x_min - padding))
                upper_x2 = min(frame.shape[1], int(x_max + padding))
            else:
                padding = 60
                upper_x1 = max(0, int(neck_point[0] - padding))
                upper_x2 = min(frame.shape[1], int(neck_point[0] + padding))
        else:
            x1, y1, x2, y2 = bbox.astype(int)
            if pose_type == "sitting":
                upper_y1 = y1 + int((y2 - y1) * 0.1)
                upper_y2 = y1 + int((y2 - y1) * 0.75)
            elif pose_type == "lying":
                upper_y1 = y1 + int((y2 - y1) * 0.2)
                upper_y2 = y1 + int((y2 - y1) * 0.8)
            else:
                upper_y1 = y1 + int((y2 - y1) * 0.15)
                upper_y2 = y1 + int((y2 - y1) * 0.65)
            upper_x1 = x1 + int((x2 - x1) * 0.1)
            upper_x2 = x2 - int((x2 - x1) * 0.1)
        upper_y1 = max(0, min(upper_y1, frame.shape[0]))
        upper_y2 = max(upper_y1, min(upper_y2, frame.shape[0]))
        upper_x1 = max(0, min(upper_x1, frame.shape[1]))
        upper_x2 = max(upper_x1, min(upper_x2, frame.shape[1]))
        if upper_y2 <= upper_y1 or upper_x2 <= upper_x1:
            return None
        roi = frame[upper_y1:upper_y2, upper_x1:upper_x2]
        if roi.shape[0] < 50 or roi.shape[1] < 30:
            return None
        return roi, (upper_x1, upper_y1, upper_x2, upper_y2)
    except Exception:
        return None


def extract_lower_body_region(frame: np.ndarray, keypoints: Any, bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    if pose_type == "lying":
        return None
    try:
        kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
        if len(kpts.shape) == 3:
            kpts = kpts[0]
        hip_points = []
        for idx in [11, 12]:
            if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                hip_points.append(kpts[idx][:2])
        ankle_points = []
        for idx in [15, 16]:
            if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                ankle_points.append(kpts[idx][:2])
        if hip_points and ankle_points:
            hip_center = np.mean(hip_points, axis=0)
            ankle_center = np.mean(ankle_points, axis=0)
            lower_y1 = int(hip_center[1])
            lower_y2 = int(ankle_center[1]) + 20
            all_points = np.array(hip_points + ankle_points)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            padding = 15
            lower_x1 = max(0, int(x_min - padding))
            lower_x2 = min(frame.shape[1], int(x_max + padding))
        else:
            x1, y1, x2, y2 = bbox.astype(int)
            if pose_type == "sitting":
                lower_y1 = y1 + int((y2 - y1) * 0.6)
                lower_y2 = y2
            else:
                lower_y1 = y1 + int((y2 - y1) * 0.55)
                lower_y2 = y2
            lower_x1 = x1 + int((x2 - x1) * 0.15)
            lower_x2 = x2 - int((x2 - x1) * 0.15)
        lower_y1 = max(0, min(lower_y1, frame.shape[0]))
        lower_y2 = max(lower_y1, min(lower_y2, frame.shape[0]))
        lower_x1 = max(0, min(lower_x1, frame.shape[1]))
        lower_x2 = max(lower_x1, min(lower_x2, frame.shape[1]))
        if lower_y2 <= lower_y1 or lower_x2 <= lower_x1:
            return None
        roi = frame[lower_y1:lower_y2, lower_x1:lower_x2]
        if roi.shape[0] < 40 or roi.shape[1] < 25:
            return None
        return roi, (lower_x1, lower_y1, lower_x2, lower_y2)
    except Exception:
        return None


def extract_face_feature(mtcnn: MTCNN, face_roi: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    try:
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(face_rgb)
        if boxes is None or probs is None or len(boxes) == 0:
            return None
        best_prob = float(np.max(probs))
        if best_prob < 0.75:
            return None
        face_resized = cv2.resize(face_roi, (112, 112))
        embedding_result = DeepFace.represent(
            face_resized, model_name="Facenet", enforce_detection=False, detector_backend="skip"
        )
        embedding = np.array(embedding_result[0]["embedding"])
        norm = np.linalg.norm(embedding)
        if norm <= 0:
            return None
        return (embedding / norm, best_prob)
    except Exception:
        return None


def extract_body_feature(roi: np.ndarray) -> Optional[np.ndarray]:
    try:
        if roi.shape[0] < 30 or roi.shape[1] < 20:
            return None
        features: List[np.ndarray] = []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        for hist in (h_hist, s_hist, v_hist):
            hist_norm = cv2.normalize(hist, hist).flatten()
            features.append(hist_norm)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (48, 64))
        lbp_hist = compute_lbp_histogram(gray_resized)
        features.append(lbp_hist)
        edges = cv2.Canny(gray_resized, 50, 150)
        edge_hist, _ = np.histogram(edges.sum(axis=1), bins=12)
        edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
        features.append(edge_hist)
        combined = np.concatenate(features)
        norm = np.linalg.norm(combined)
        if norm <= 0:
            return None
        return combined / norm
    except Exception:
        return None


def create_detection(frame: np.ndarray, pose: Any, frame_idx: int, mtcnn: MTCNN) -> Optional[Dict[str, Any]]:
    if not hasattr(pose.pred_instances, "bboxes") or len(pose.pred_instances.bboxes) == 0:
        return None
    bbox = pose.pred_instances.bboxes[0]
    if hasattr(bbox, "cpu"):
        bbox = bbox.cpu().numpy()
    if len(bbox) < 4 or bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 100:
        return None
    keypoints = pose.pred_instances.keypoints[0]
    confidence = float(bbox[4]) if len(bbox) > 4 else 1.0
    pose_type = determine_pose_type(keypoints)

    face_feature = None
    upper_feature = None
    lower_feature = None
    if frame_idx % feature_update_interval == 0:
        face_result = extract_face_region(frame, keypoints, bbox[:4])
        if face_result:
            face_roi, _ = face_result
            face_feat = extract_face_feature(mtcnn, face_roi)
            if face_feat:
                face_feature, _ = face_feat
        lower_result = extract_lower_body_region(frame, keypoints, bbox[:4], pose_type)
        if lower_result:
            lower_roi, _ = lower_result
            lower_feature = extract_body_feature(lower_roi)
    upper_result = extract_upper_body_region(frame, keypoints, bbox[:4], pose_type)
    if upper_result:
        upper_roi, _ = upper_result
        upper_feature = extract_body_feature(upper_roi)
    return {
        "bbox": bbox[:4],
        "keypoints": keypoints,
        "confidence": confidence,
        "pose_type": pose_type,
        "face_feature": face_feature,
        "upper_feature": upper_feature,
        "lower_feature": lower_feature,
    }


def match_with_motion(
    detections: List[Dict[str, Any]],
    active_tracks: Dict[int, Dict[str, Any]],
    motion_confidence_threshold: float,
    camera_motion: Tuple[float, float],
    cfg: TrackerConfig,
    *,
    iou_threshold: float,
    center_weight: float,
) -> Dict[int, int]:
    matches: Dict[int, int] = {}
    if not active_tracks or not detections:
        return matches
    track_ids = list(active_tracks.keys())
    cost = np.full((len(detections), len(track_ids)), 1.0)
    for det_idx, detection in enumerate(detections):
        for track_idx, track_id in enumerate(track_ids):
            track = active_tracks[track_id]
            predicted_bbox, motion_conf = predict_motion_with_camera_compensation(
                track["kalman"], track["missed_updates"], camera_motion, cfg=cfg
            )
            # If motion prediction is not confident, discourage assignment
            if motion_conf <= motion_confidence_threshold:
                cost[det_idx, track_idx] = 0.95
                continue
            # Spatial plausibility gate to avoid impossible jumps
            if not is_spatially_plausible(detection["bbox"], predicted_bbox, cfg.max_jump_factor):
                cost[det_idx, track_idx] = 1.0
                continue
            # Center-weighted similarity (adaptive)
            sim = calculate_combined_similarity(detection["bbox"], predicted_bbox, center_weight)
            cost[det_idx, track_idx] = 1.0 - sim
    di, ti = linear_sum_assignment(cost)
    for d, t in zip(di, ti):
        # Accept only if combined similarity meets threshold
        if cost[d, t] <= (1.0 - iou_threshold):
            matches[d] = track_ids[t]
    return matches


def compute_person_similarity(detection: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, str]:
    sims: List[float] = []
    parts: List[str] = []
    weights: List[float] = []
    if detection.get("face_feature") is not None and profile.get("face_feature") is not None:
        s = compute_feature_similarity(detection["face_feature"], profile["face_feature"])
        if s > face_reid_threshold:
            sims.append(s); parts.append("face"); weights.append(0.5)
    if detection.get("upper_feature") is not None and profile.get("upper_feature") is not None:
        s = compute_feature_similarity(detection["upper_feature"], profile["upper_feature"])
        if s > upper_reid_threshold:
            sims.append(s); parts.append("upper"); weights.append(0.35)
    if detection.get("lower_feature") is not None and profile.get("lower_feature") is not None:
        s = compute_feature_similarity(detection["lower_feature"], profile["lower_feature"])
        if s > lower_reid_threshold:
            sims.append(s); parts.append("lower"); weights.append(0.15)
    if sims and weights:
        tw = sum(weights)
        nw = [w / tw for w in weights]
        return sum(s * w for s, w in zip(sims, nw)), "+".join(parts)
    return 0.0, "none"


def update_person_profile(profile: Dict[str, Any], detection: Dict[str, Any]) -> None:
    alpha = 0.3
    for key in ("face_feature", "upper_feature", "lower_feature"):
        feat = detection.get(key)
        if feat is None:
            continue
        if profile.get(key) is None:
            profile[key] = feat.copy()
        else:
            profile[key] = alpha * feat + (1 - alpha) * profile[key]
            n = np.linalg.norm(profile[key])
            if n > 0:
                profile[key] = profile[key] / n


def draw_simple_tracking(frame: np.ndarray, pose_results: List[Any], assignments: Dict[int, int], active_tracks: Dict[int, Dict[str, Any]], frame_idx: int, visualizer: Any) -> np.ndarray:
    if not assignments:
        return frame
    filtered = []
    for det_idx, _tid in assignments.items():
        if det_idx < len(pose_results):
            filtered.append(pose_results[det_idx])
    if not filtered:
        return frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data_samples = merge_data_samples(filtered)
    visualizer.add_datasample(
        "result", img_rgb, data_sample=data_samples, draw_gt=False, draw_heatmap=False, draw_bbox=False, show=False, wait_time=0, out_file=None, kpt_thr=0.3
    )
    vis = visualizer.get_image()
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    for det_idx, track_id in assignments.items():
        if det_idx < len(pose_results):
            pose = pose_results[det_idx]
            if hasattr(pose.pred_instances, "bboxes") and len(pose.pred_instances.bboxes) > 0:
                bbox = pose.pred_instances.bboxes[0]
                if torch.is_tensor(bbox):
                    bbox = bbox.cpu().numpy()
                x1, y1, x2, y2 = bbox[:4].astype(int)
                track = active_tracks.get(track_id)
                if track and track.get("missed_updates", 0) == 0:
                    if track.get("created_frame") == frame_idx:
                        match_type = "New"; color = (0, 255, 255)
                    else:
                        match_type = "Motion"; color = (0, 255, 0)
                else:
                    match_type = "Appearance"; color = (255, 0, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                text = f"ID {track_id}: {match_type}"
                cv2.rectangle(vis, (x1, y1 - 25), (x1 + len(text) * 8, y1), color, -1)
                cv2.putText(vis, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def process_video(
    input_path: str,
    output_path: str,
    *,
    device: str = "cuda:0",
    det_config: str = DEFAULT_DET_CONFIG,
    det_checkpoint: str = DEFAULT_DET_CKPT,
    pose_config: str = DEFAULT_POSE_CONFIG,
    pose_checkpoint: str = DEFAULT_POSE_CKPT,
) -> None:
    # Configure device safely (supports CPU-only environments)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":")[-1]))
        dev_ctx = torch.cuda.device(device)
        mtcnn_device = device
    else:
        device = "cpu"
        dev_ctx = contextlib.nullcontext()
        mtcnn_device = "cpu"

    detector = init_detector(det_config, det_checkpoint, device=device)
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True))),
    )
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    mtcnn = MTCNN(keep_all=True, device=mtcnn_device, post_process=False, min_face_size=40)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open video writer")

    frame_idx = 0
    next_track_id = 1
    active_tracks: Dict[int, Dict[str, Any]] = {}
    lost_tracks: Dict[int, Dict[str, Any]] = {}
    person_profiles: Dict[int, Dict[str, Any]] = {}
    camera_comp = CameraMotionCompensator()
    cfg = TrackerConfig()

    with tqdm(total=total or None, desc="Processing frames") as pbar:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            try:
                scope = detector.cfg.get("default_scope", "mmdet")
                if scope is not None:
                    init_default_scope(scope)
                with dev_ctx:
                    detect_result = inference_detector(detector, frame)
                pred_instance = detect_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)]
                bboxes = bboxes[nms(bboxes, 0.7)][:, :4]

                with dev_ctx:
                    pose_results = inference_topdown(pose_estimator, frame, bboxes)

                detections: List[Dict[str, Any]] = []
                for pose in pose_results:
                    det = create_detection(frame, pose, frame_idx, mtcnn)
                    if det:
                        detections.append(det)

                dx, dy = camera_comp.estimate_camera_motion(frame)
                camera_motion = (dx, dy)

                # Adaptive thresholds based on scene crowding
                bboxes_for_crowding = [det["bbox"] for det in detections]
                crowding = calculate_scene_crowding(bboxes_for_crowding)
                iou_thresh, center_w, motion_conf_th = get_adaptive_thresholds(cfg, crowding)

                # 1) Motion-based matching
                motion_matches = match_with_motion(
                    detections,
                    active_tracks,
                    motion_conf_th,
                    camera_motion,
                    cfg,
                    iou_threshold=iou_thresh,
                    center_weight=center_w,
                )
                final_matches: Dict[int, int] = {}
                for det_idx, tid in motion_matches.items():
                    det = detections[det_idx]
                    track = active_tracks[tid]
                    update_kalman_filter(track["kalman"], det["bbox"])
                    track.setdefault("detections", deque(maxlen=100)).append(det)
                    track["last_seen"] = frame_idx
                    track["lost_frames"] = 0
                    track["missed_updates"] = 0
                    if frame_idx % feature_update_interval == 0:
                        prof = person_profiles.get(tid)
                        if prof:
                            update_person_profile(prof, det)
                    final_matches[det_idx] = tid

                # 2) Appearance-based matching for unmatched
                unmatched = [(i, d) for i, d in enumerate(detections) if i not in motion_matches]
                appearance_matches: Dict[int, Tuple[int, str]] = {}
                for det_idx, det in unmatched:
                    best_id, best_score, best_desc = None, 0.0, ""
                    for tid, tr in active_tracks.items():
                        # Avoid assigning the same track to multiple detections
                        already_assigned_tids = {t for (t, _) in appearance_matches.values()}
                        if tid in already_assigned_tids:
                            continue
                        prof = person_profiles.get(tid)
                        if not prof:
                            continue
                        sim, desc = compute_person_similarity(det, prof)
                        if sim > best_score and sim > combined_reid_threshold:
                            best_id, best_score, best_desc = tid, sim, desc
                    if best_id:
                        appearance_matches[det_idx] = (best_id, best_desc)
                for det_idx, (tid, _desc) in appearance_matches.items():
                    det = detections[det_idx]
                    track = active_tracks[tid]
                    update_kalman_filter(track["kalman"], det["bbox"])
                    track.setdefault("detections", deque(maxlen=100)).append(det)
                    track["last_seen"] = frame_idx
                    track["lost_frames"] = 0
                    track["missed_updates"] = 0
                    if frame_idx % feature_update_interval == 0:
                        prof = person_profiles.get(tid)
                        if prof:
                            update_person_profile(prof, det)
                    final_matches[det_idx] = tid

                # 3) Re-ID from lost tracks
                still_unmatched = [(i, d) for i, d in unmatched if i not in appearance_matches]
                reid_matches: Dict[int, Tuple[int, str]] = {}
                for det_idx, det in still_unmatched:
                    best_id, best_score, best_desc = None, 0.0, ""
                    for tid, lost in lost_tracks.items():
                        prof = person_profiles.get(tid)
                        if not prof:
                            continue
                        sim, desc = compute_person_similarity(det, prof)
                        if sim > best_score and sim > (combined_reid_threshold + 0.1):
                            best_id, best_score, best_desc = tid, sim, desc
                    if best_id:
                        reid_matches[det_idx] = (best_id, best_desc)
                for det_idx, (tid, _desc) in reid_matches.items():
                    det = detections[det_idx]
                    reactivated = lost_tracks.pop(tid)
                    reactivated["kalman"] = create_kalman_filter(det["bbox"])
                    reactivated.setdefault("detections", deque(maxlen=100)).append(det)
                    reactivated["last_seen"] = frame_idx
                    reactivated["lost_frames"] = 0
                    reactivated["missed_updates"] = 0
                    active_tracks[tid] = reactivated
                    final_matches[det_idx] = tid
                    prof = person_profiles.get(tid)
                    if prof:
                        update_person_profile(prof, det)

                # 4) New tracks for remaining unmatched
                remaining = [i for i, _ in still_unmatched if i not in reid_matches]
                for det_idx in remaining:
                    det = detections[det_idx]
                    track = {
                        "track_id": next_track_id,
                        "kalman": create_kalman_filter(det["bbox"]),
                        "detections": deque([det], maxlen=100),
                        "last_seen": frame_idx,
                        "created_frame": frame_idx,
                        "lost_frames": 0,
                        "missed_updates": 0,
                    }
                    profile = {
                        "person_id": next_track_id,
                        "creation_frame": frame_idx,
                        "face_feature": det.get("face_feature"),
                        "upper_feature": det.get("upper_feature"),
                        "lower_feature": det.get("lower_feature"),
                    }
                    active_tracks[next_track_id] = track
                    person_profiles[next_track_id] = profile
                    final_matches[det_idx] = next_track_id
                    next_track_id += 1

                # 5) Handle lost tracks
                to_remove: List[int] = []
                for tid, tr in active_tracks.items():
                    if tid not in final_matches.values():
                        tr["missed_updates"] += 1
                        tr["lost_frames"] += 1
                        if tr["lost_frames"] > max_lost_frames:
                            if len(tr.get("detections", [])) >= 10:
                                lost_tracks[tid] = tr
                            to_remove.append(tid)
                for tid in to_remove:
                    active_tracks.pop(tid, None)
                # Cleanup old lost tracks
                to_cleanup = [tid for tid, tr in lost_tracks.items() if frame_idx - tr.get("last_seen", 0) > max_lost_frames * 2]
                for tid in to_cleanup:
                    lost_tracks.pop(tid, None)

                vis_frame = draw_simple_tracking(frame, pose_results, final_matches, active_tracks, frame_idx, visualizer)
                writer.write(vis_frame)

                if frame_idx % 50 == 0:
                    if device != "cpu":
                        torch.cuda.empty_cache()
                    gc.collect()
                    print(
                        f"Frame {frame_idx}: Active={len(active_tracks)}, Lost={len(lost_tracks)}, Total={len(person_profiles)}"
                    )

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                writer.write(frame)

            frame_idx += 1
            pbar.update(1)

    cap.release(); writer.release()
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Processing complete. Output saved: {output_path}")
    print(f"Total persons tracked: {len(person_profiles)}")


def process(in_folder: str, out_folder: str, *, device: str = "cuda:0", **model_kwargs: Any) -> None:
    os.makedirs(out_folder, exist_ok=True)
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    files = [f for f in os.listdir(in_folder) if any(f.lower().endswith(ext) for ext in video_exts)]
    for i, name in enumerate(files, 1):
        print(f"\nProcessing video {i}/{len(files)}: {name}")
        in_path = os.path.join(in_folder, name)
        out_name = os.path.splitext(name)[0] + "_tracked.mp4"
        out_path = os.path.join(out_folder, out_name)
        process_video(in_path, out_path, device=device, **model_kwargs)
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()
    print(f"\nOutputs saved to {out_folder}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motion + Appearance person tracking")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="in_path", type=str, help="Input video path")
    g.add_argument("--in-dir", dest="in_dir", type=str, help="Input folder path")
    p.add_argument("--out", dest="out_path", type=str, help="Output video path (single)")
    p.add_argument("--out-dir", dest="out_dir", type=str, help="Output folder path (batch)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--det-config", default=DEFAULT_DET_CONFIG)
    p.add_argument("--det-ckpt", default=DEFAULT_DET_CKPT)
    p.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    p.add_argument("--pose-ckpt", default=DEFAULT_POSE_CKPT)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.in_path:
        if not args.out_path:
            raise SystemExit("--out is required for single video mode")
        process_video(
            args.in_path,
            args.out_path,
            device=args.device,
            det_config=args.det_config,
            det_checkpoint=args.det_ckpt,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_ckpt,
        )
    else:
        if not args.out_dir:
            raise SystemExit("--out-dir is required for folder mode")
        process(
            args.in_dir,
            args.out_dir,
            device=args.device,
            det_config=args.det_config,
            det_checkpoint=args.det_ckpt,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_ckpt,
        )


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
