import sys
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import torch
import time
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from facenet_pytorch import MTCNN
import gc
import signal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from abc import abstractmethod

from deepface import DeepFace
from sailsprep.id_tracking_model.tracking.utils.cache_manager import CacheManager
from sailsprep.id_tracking_model.tracking.utils.tracking_exporter import TrackingDataCollector


import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models

from sailsprep.id_tracking_model.tracking.tracker.person_tracker import (
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

from sailsprep.id_tracking_model.tracking.utils.utils import soft_nms, oks_nms

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
head_kpts = [coco_kpts["nose"], coco_kpts["left_eye"], coco_kpts["right_eye"], coco_kpts["left_ear"], coco_kpts["right_ear"]]
upper_body_kpts = [coco_kpts["left_shoulder"], coco_kpts["right_shoulder"], coco_kpts["left_elbow"], coco_kpts["right_elbow"], coco_kpts["left_wrist"], coco_kpts["right_wrist"], coco_kpts["left_hip"], coco_kpts["right_hip"]]
hips = [coco_kpts["left_hip"], coco_kpts["right_hip"]]
knees = [coco_kpts["left_knee"], coco_kpts["right_knee"]]
ankles = [coco_kpts["left_ankle"], coco_kpts["right_ankle"]]
lower_body_kpts = hips + knees + ankles
left_kpts = [coco_kpts["left_shoulder"], coco_kpts["left_elbow"], coco_kpts["left_wrist"], coco_kpts["left_hip"], coco_kpts["left_knee"], coco_kpts["left_ankle"]]
right_kpts = [coco_kpts["right_shoulder"], coco_kpts["right_elbow"], coco_kpts["right_wrist"], coco_kpts["right_hip"], coco_kpts["right_knee"], coco_kpts["right_ankle"]]
torso_kpts = [coco_kpts["left_shoulder"], coco_kpts["right_shoulder"], coco_kpts["left_hip"], coco_kpts["right_hip"]]

kpts_sets = [head_kpts, upper_body_kpts, lower_body_kpts, left_kpts, right_kpts, torso_kpts]

# ================== CONFIGURATION SYSTEM ==================
@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    enable_face_features: bool = True
    enable_upper_body_features: bool = True
    enable_lower_body_features: bool = True

    # Face feature settings
    no_deepface: bool = False
    face_confidence_threshold: float = 0.75
    face_min_size: int = 40

    # Body feature settings
    body_min_height: int = 50
    body_min_width: int = 30

    # Feature update frequency
    feature_update_interval: int = 10
    resnet_input_size: Tuple[int, int] = (256, 128)  # for resnet50
    resnet_feature_dim: int = 512   # for resnet50
    resnet_batch_size: int = 1   # for resnet50
    
@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    oks_nms_threshold: float = 0.9
    kpt_threshold: float = 0.3

    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    max_tracks: int = 0  # 0 means no limit

    # Re-identification thresholds
    face_reid_threshold: float = 0.75
    upper_reid_threshold: float = 0.65
    lower_reid_threshold: float = 0.6
    combined_reid_threshold: float = 0.7

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    enable_visualization: bool = True
    enable_pose_drawing: bool = True
    enable_bbox_drawing: bool = True
    enable_id_labels: bool = True

    # Visualization parameters
    radius: int = 3
    line_width: int = 1

@dataclass
class ExportConfig:
    """Configuration for data export"""
    enable_export: bool = True
    export_json: bool = True
    export_csv: bool = True
    output_path: str = "tracking_results"

@dataclass
class CacheConfig:
    """Configuration for caching detection and pose results"""
    enable_cache: bool = False
    cache_base_path: str = "/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking"
    force_recompute: bool = False  # If True, ignore existing cache and recompute

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    features: FeatureConfig = field(default_factory=FeatureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    frame_limit: int = 0  # 0 means process all frames

class PoseResult:
    """Wrapper for pose results"""
    def __init__(self, keypoints: np.ndarray, bbox: np.ndarray):
        # keypoints: (N_kpts, 3) [x, y, score]
        # bbox: (4,) [x1, y1, x2, y2]
        self.keypoints = keypoints
        self.bbox = bbox
        self.metadata = {}

    def to_dict(self):
        return {
            "keypoints": self.keypoints,
            "bbox": self.bbox,
            "metadata": self.metadata
        }

def create_pose_results_from_cache(cached_poses: List[Dict[str, np.ndarray]]) -> List[PoseResult]:
    """Convert cached pose data to PoseResult objects"""
    return [
        PoseResult(
            keypoints=p['keypoints'],
            bbox=p['bbox']
        )
        for p in cached_poses
    ]

def apply_oks_nms(pose_results: List[PoseResult], oks_nms_threshold: float, kpt_threshold: float) -> List[PoseResult]:
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
        thr=oks_nms_threshold,
        vis_thr=kpt_threshold,
    )

    return [pose_results[i] for i in keep_idx]

def filter_poses_by_keypoints(pose_results: List[PoseResult], kpt_threshold: float) -> List[PoseResult]:
    """Filter poses based on keypoint visibility"""
    for pose in pose_results:
        keypoint_scores = pose.keypoints[:17, 2]  # Get scores from first 17 keypoints

        # Check if any keypoint set is fully visible
        enough_kpts_bool = False
        for kpts_set in kpts_sets:
            scores = keypoint_scores[kpts_set]
            if np.sum(scores > kpt_threshold) == len(kpts_set):
                enough_kpts_bool = True
                break
        pose.metadata['sufficient_keypoints'] = enough_kpts_bool

    return pose_results

# ================== FEATURE EXTRACTION MODULE ==================

class RegionExtractor:
    """Extracts different body regions from frames"""

    @staticmethod
    def extract_face_region(frame: np.ndarray, kpts: np.ndarray, bbox: np.ndarray, kpt_threshold: float = 0.3) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract face region using head keypoints"""
        try:
            # Get head keypoints (nose, eyes, ears)
            head_points = []
            for idx in head_kpts:
                conf = kpts[idx][2]
                if conf > kpt_threshold:
                    head_points.append(kpts[idx][:2])

            if len(head_points) >= 2:
                head_points = np.array(head_points)
                x_min, y_min = np.min(head_points, axis=0)
                x_max, y_max = np.max(head_points, axis=0)

                padding = 25
                face_x1, face_y1 = max(0, int(x_min - padding)), max(0, int(y_min - padding)),
                face_x2, face_y2 = min(frame.shape[1], int(x_max + padding)), min(frame.shape[0], int(y_max + padding))
            else:
                # Fallback to upper bbox region
                x1, y1, x2, y2 = bbox[:4].astype(int)
                face_h = int((y2 - y1) * 0.35)
                face_x1, face_y1 = x1, y1
                face_x2, face_y2 = x2, y1 + face_h
            
            if face_x2 <= face_x1 or face_y2 <= face_y1:
                return None

            face_roi = frame[face_y1:face_y2, face_x1:face_x2]

            if face_roi.shape[0] < 40 or face_roi.shape[1] < 30:
                return None
            
            return face_roi, (face_x1, face_y1, face_x2, face_y2)
        except Exception as e:
            print(f"Error extracting face region: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def extract_upper_body_region(frame: np.ndarray, kpts: np.ndarray, bbox: np.ndarray, pose_type: str, kpt_threshold: float = 0.3) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract upper body region"""
        try:

            # Get neck point from shoulders
            neck_point = None
            left_shoulder = kpts[coco_kpts["left_shoulder"]] if kpts[coco_kpts["left_shoulder"]][2] > kpt_threshold else None
            right_shoulder = kpts[coco_kpts["right_shoulder"]] if kpts[coco_kpts["right_shoulder"]][2] > kpt_threshold else None

            if left_shoulder is not None and right_shoulder is not None:
                neck_x, neck_y = (left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2 - 15
                neck_point = np.array([neck_x, neck_y])

            # Get hip points
            hip_points = []
            for idx in hips:  # left_hip, right_hip
                if kpts[idx][2] > kpt_threshold:
                    hip_points.append(kpts[idx][:2])

            if neck_point is not None and hip_points:
                hip_center = np.mean(hip_points, axis=0)
                upper_y1, upper_y2 = int(neck_point[1]), int(hip_center[1])

                # Use shoulders for width
                if left_shoulder is not None and right_shoulder is not None:
                    x_min, x_max = min(left_shoulder[0], right_shoulder[0]), max(left_shoulder[0], right_shoulder[0])
                    padding = 20
                    upper_x1, upper_x2 = max(0, int(x_min - padding)), min(frame.shape[1], int(x_max + padding))
                else:
                    padding = 60
                    upper_x1, upper_x2 = max(0, int(neck_point[0] - padding)), min(frame.shape[1], int(neck_point[0] + padding))
            else:
                # Fallback to bbox-based region
                x1, y1, x2, y2 = bbox.astype(int)
                pose_mult = {"sitting": (0.1, 0.75), "lying": (0.2, 0.8), "standing": (0.15, 0.65)}
                y1_m, y2_m = pose_mult.get(pose_type, (0.15, 0.65))
                upper_y1, upper_y2 = y1 + int((y2 - y1) * y1_m), y1 + int((y2 - y1) * y2_m)
                upper_x1, upper_x2 = x1 + int((x2 - x1) * 0.1), x2 - int((x2 - x1) * 0.1)

            # Ensure valid region
            upper_y1 = max(0, min(upper_y1, frame.shape[0]))
            upper_y2 = max(upper_y1, min(upper_y2, frame.shape[0]))
            upper_x1 = max(0, min(upper_x1, frame.shape[1]))
            upper_x2 = max(upper_x1, min(upper_x2, frame.shape[1]))

            if upper_y2 <= upper_y1 or upper_x2 <= upper_x1:
                return None

            upper_roi = frame[upper_y1:upper_y2, upper_x1:upper_x2]

            if upper_roi.shape[0] < 50 or upper_roi.shape[1] < 30:
                return None

            return upper_roi, (upper_x1, upper_y1, upper_x2, upper_y2)
        except Exception as e:
            print(f"Error extracting upper body region: {e}")
            return None

    @staticmethod
    def extract_lower_body_region(frame: np.ndarray, kpts: np.ndarray, bbox: np.ndarray, pose_type: str, kpt_threshold: float = 0.3) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract lower body region"""
        if pose_type == "lying":
            return None

        try:
            # Get hip points
            hip_points = []
            for idx in hips:  # left_hip, right_hip
                if kpts[idx][2] > kpt_threshold:
                    hip_points.append(kpts[idx][:2])

            # Get ankle points
            ankle_points = []
            for idx in ankles:  # left_ankle, right_ankle
                if kpts[idx][2] > kpt_threshold:
                    ankle_points.append(kpts[idx][:2])

            if hip_points and ankle_points:
                hip_center = np.mean(hip_points, axis=0)
                ankle_center = np.mean(ankle_points, axis=0)

                lower_y1, lower_y2 = int(hip_center[1]), int(ankle_center[1]) + 20

                all_points = hip_points + ankle_points
                all_points = np.array(all_points)
                x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
                padding = 15
                lower_x1, lower_x2 = max(0, int(x_min - padding)), min(frame.shape[1], int(x_max + padding))
            else:
                # Fallback to bbox-based region
                x1, y1, x2, y2 = bbox.astype(int)
                pose_mult = {"sitting": 0.6, "standing": 0.55}
                y1_m = pose_mult.get(pose_type, 0.55)
                lower_y1, lower_y2 = y1 + int((y2 - y1) * y1_m), y2
                lower_x1, lower_x2 = x1 + int((x2 - x1) * 0.15), x2 - int((x2 - x1) * 0.15)

            # Ensure valid region
            lower_y1 = max(0, min(lower_y1, frame.shape[0]))
            lower_y2 = max(lower_y1, min(lower_y2, frame.shape[0]))
            lower_x1 = max(0, min(lower_x1, frame.shape[1]))
            lower_x2 = max(lower_x1, min(lower_x2, frame.shape[1]))

            if lower_y2 <= lower_y1 or lower_x2 <= lower_x1:
                return None

            lower_roi = frame[lower_y1:lower_y2, lower_x1:lower_x2]

            if lower_roi.shape[0] < 40 or lower_roi.shape[1] < 25:
                return None

            return lower_roi, (lower_x1, lower_y1, lower_x2, lower_y2)
        except Exception as e:
            print(f"Error extracting lower body region: {e}")
            return None

class FeatureExtractor:
    """Base class for feature extractors"""

    @abstractmethod
    def extract(self, roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        pass

class FaceFeatureExtractor(FeatureExtractor):
    """Extracts face features using DeepFace"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.mtcnn = MTCNN(
            keep_all=True,
            device='cuda:0',
            post_process=False,
            min_face_size=config.face_min_size
        )
        # self.resnet_extractor = ResNetBodyFeatureExtractor(config) if config.no_deepface else None

    def extract(self, face_roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Extract face embedding"""
        if not self.config.enable_face_features:
            return None
        # if self.config.no_deepface:
        #     return self.resnet_extractor.extract(face_roi)
        try:
            # Validate with MTCNN
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(face_rgb)

            if boxes is None or probs is None or len(boxes) == 0:
                return None

            best_prob = float(np.max(probs))
            if best_prob < self.config.face_confidence_threshold:
                return None

            # Extract DeepFace embedding
            face_resized = cv2.resize(face_roi, (112, 112))
            embedding_result = DeepFace.represent(
                face_resized,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip'
            )
            embedding = np.array(embedding_result[0]['embedding'])

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                return embedding

        except Exception as e:
            print(f"Error extracting face features with DeepFace/MTCNN: {e}")

        return None


class ResNetBodyFeatureExtractor(FeatureExtractor):
    """ResNet50 body feature extractor"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50 and remove final FC layer
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.eval()
        self.resnet_model.to(self.device)
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, roi: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """Extract ResNet-based appearance feature"""
        try:
            if roi.shape[0] < 30 or roi.shape[1] < 20:
                return None
            
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            tensor_roi = self.transform(roi_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet_model(tensor_roi)
                features = features.squeeze().cpu().numpy()
            
            # L2 normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        except Exception as e:
            print(f"ResNet feature extraction error: {e}")
            return None


class FeatureExtractionModule:
    """Updated feature extraction module with simplified ResNet"""

    def __init__(self, config: FeatureConfig, kpt_threshold: float = 0.3):
        self.config = config
        self.kpt_threshold = kpt_threshold

        self.body_extractor = None
        if config.enable_upper_body_features or config.enable_lower_body_features or (config.no_deepface and config.enable_face_features):
            self.body_extractor = ResNetBodyFeatureExtractor(config)

        # Use Resnet for face features if no_deepface is True
        self.face_extractor = None
        if config.enable_face_features:
            if config.no_deepface:
                self.face_extractor = self.body_extractor
            else:
                self.face_extractor = FaceFeatureExtractor(config)

        self.region_extractor = RegionExtractor()

    def extract_features(self, frame: np.ndarray, pose_results: List[PoseResult], frame_count: int) -> List[Dict]:
        """Extract features for all detections in frame"""
        detections = []

        for pose in pose_results:
            detection = self._create_detection(frame, pose, frame_count)
            if detection:
                detections.append(detection)

        return detections

    def _create_detection(self, frame: np.ndarray, pose: PoseResult, frame_count: int) -> Optional[Dict]:
        """Create detection with simplified ResNet features from pose result"""
        bbox = pose.bbox
        keypoints = pose.keypoints

        confidence = float(bbox[4]) if len(bbox) > 4 else 1.0
        pose_type = self._determine_pose_type(keypoints)

        # Extract features
        face_feature = None
        upper_feature = None
        lower_feature = None

        if frame_count % self.config.feature_update_interval == 0:
            # Face feature
            if self.config.enable_face_features and self.face_extractor:
                face_result = self.region_extractor.extract_face_region(frame, keypoints, bbox[:4], self.kpt_threshold)
                if face_result:
                    face_roi, _ = face_result
                    face_feature = self.face_extractor.extract(face_roi)

            # Lower body feature
            if self.config.enable_lower_body_features and self.body_extractor:
                lower_result = self.region_extractor.extract_lower_body_region(frame, keypoints, bbox[:4], pose_type, self.kpt_threshold)
                if lower_result:
                    lower_roi, _ = lower_result
                    lower_feature = self.body_extractor.extract(lower_roi)

        # Upper body feature 
        if self.config.enable_upper_body_features and self.body_extractor:
            upper_result = self.region_extractor.extract_upper_body_region(frame, keypoints, bbox[:4], pose_type, self.kpt_threshold)
            if upper_result:
                upper_roi, _ = upper_result
                upper_feature = self.body_extractor.extract(upper_roi)

        return {
            'bbox': bbox[:4],
            'keypoints': keypoints,
            'confidence': confidence,
            'pose_type': pose_type,
            'face_feature': face_feature,
            'upper_feature': upper_feature,
            'lower_feature': lower_feature,
            'sufficient_keypoints': getattr(pose, 'metadata', {}).get('sufficient_keypoints', True)
        }

    def _determine_pose_type(self, kpts) -> str:
        """Determine pose type from keypoints"""
        try:
            if len(kpts) < 17:
                return "standing"

            visible_points = 0
            for idx in lower_body_kpts:
                conf = kpts[idx][2]
                if conf > self.kpt_threshold:
                    visible_points += 1

            if visible_points < 3:
                return "standing"

            hip_y = []
            for hip in hips:
                if kpts[hip][2] > self.kpt_threshold:
                    hip_y.append(kpts[hip][1])

            ankle_y = []
            for ankle in ankles:
                if kpts[ankle][2] > self.kpt_threshold:
                    ankle_y.append(kpts[ankle][1])

            if hip_y and ankle_y:
                hip_ankle_dist = abs(np.mean(ankle_y) - np.mean(hip_y))

                if hip_ankle_dist < 50:
                    return "lying"
                elif hip_ankle_dist < 120:
                    return "sitting"
                else:
                    return "standing"

            return "standing"
        except Exception as e:
            print(f"Error determining pose type: {e}")
            return "standing"

    def _filter_out(bbox: np.ndarray, keypoints: np.ndarray) -> bool:
        """Filter out detections with insufficient keypoints"""
        # Unimplemented
        return True

# ================== TRACKING MODULE ==================

class TrackingModule:
    """Handles multi-person tracking with re-identification"""

    GEOMETRY_EMA_ALPHA = 0.25

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tracker_config = config.tracker_config
        self.frame_count = 0
        self.next_track_id = 1
        self.active_tracks = {}
        self.lost_tracks = {}
        self.person_profiles = {}
        self.camera_compensator = CameraMotionCompensator() 
        self.similarity_info = {"face": {"feat": "face_feature", "weight": 0.5, "threshold": self.config.face_reid_threshold},
                           "upper": {"feat": "upper_feature", "weight": 0.35, "threshold": self.config.upper_reid_threshold},
                           "lower": {"feat": "lower_feature", "weight": 0.15, "threshold": self.config.lower_reid_threshold }}

    def update(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, int]:
        """Update tracking system with new detections"""
        final_matches = {}
        self.frame_count += 1
        
        # Estimate camera motion
        camera_motion = self.camera_compensator.estimate_camera_motion(frame)
        
        # Calculate scene crowding for adaptive thresholds
        det_bboxes = [d['bbox'] for d in detections]
        crowding = calculate_scene_crowding(det_bboxes)
        iou_thresh, center_weight, motion_conf_thresh = get_adaptive_thresholds(
            self.tracker_config, crowding
        )

        # 1. Motion-based matching
        motion_matches = self._match_with_motion(detections, camera_motion, iou_thresh, center_weight, motion_conf_thresh)

        # Check geometry consistency and drop suspect matches (SUBJECT TO CHANGE)
        suspect_detections = self._check_geometry_consistency(motion_matches, detections)
        motion_matches = {i: j for i, j in motion_matches.items() if i not in suspect_detections}

        self._update_matches(motion_matches, detections, final_matches, match_type="motion")

        # Discard detections with insufficient keypoints if not matched by motion
        unmatched_detections = [(i, det) for i, det in enumerate(detections) if i not in motion_matches and det['sufficient_keypoints']]

        # 2. Appearance-based matching
        appearance_matches = self._match_with_appearance(unmatched_detections)
        self._update_matches(appearance_matches, detections, final_matches, match_type="appearance")
        unmatched_detections = [(i, det) for i, det in unmatched_detections if i not in appearance_matches]

        # 3. Re-identification with lost tracks
        reid_matches = self._match_with_lost_tracks(unmatched_detections)
        self._update_matches(reid_matches, detections, final_matches, match_type="appearance_reid")
        remaining_unmatched = [i for i, det in unmatched_detections if i not in reid_matches]

        # 4. Create new tracks
        for det_idx in remaining_unmatched:
            detection = detections[det_idx]
            new_id = self._create_new_track(detection)
            final_matches[det_idx] = new_id

        # 5. Handle lost tracks
        self._handle_lost_tracks(final_matches)
        return final_matches

    def _match_with_motion(self, detections: List[Dict], camera_motion: tuple, 
                       iou_thresh: float, center_weight: float, motion_conf_thresh: float) -> Dict[int, int]:
        """Match detections with tracks using motion prediction"""
        matches = {}
    
        if not self.active_tracks:
            return matches
    
        track_ids = list(self.active_tracks.keys())

        # No detections this frame → still advance every track once so the
        # state stays aligned with the video timeline.
        if not detections:
            for track_id in track_ids:
                track = self.active_tracks[track_id]
                predict_motion_with_camera_compensation(
                    track['kalman'],
                    track['missed_updates'],
                    camera_motion,
                    cfg=self.tracker_config,
                )
            return matches
    
        cost_matrix = np.full((len(detections), len(track_ids)), 1.0)
    
        # Predict once per track and reuse the results while building the cost matrix
        track_predictions: Dict[int, Tuple[np.ndarray, float]] = {}
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            track_predictions[track_id] = predict_motion_with_camera_compensation(
                track['kalman'],
                track['missed_updates'],
                camera_motion,
                cfg=self.tracker_config,
            )

        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                predicted_bbox, motion_confidence = track_predictions[track_id]
                if motion_confidence > motion_conf_thresh:
                    if is_spatially_plausible(detection['bbox'], predicted_bbox, 
                                             self.tracker_config.max_jump_factor):
                        sim = calculate_combined_similarity(detection['bbox'], predicted_bbox, center_weight)
                        cost_matrix[det_idx, track_idx] = 1.0 - sim
                    else:
                        cost_matrix[det_idx, track_idx] = 0.99
                else:
                    cost_matrix[det_idx, track_idx] = 0.95
    
        # Hungarian assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
    
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < (1.0 - iou_thresh):
                track_id = track_ids[track_idx]
                matches[det_idx] = track_id
        return matches
        
    def _match_with_appearance_hungarian(self, unmatched_detections: List[Tuple[int, Dict]],
                                         tracks_dict: Dict, threshold: float) -> Dict[int, Tuple[int, str]]:
        """Match detections with tracks using appearance features via Hungarian assignment

        Args:
            unmatched_detections: List of (det_idx, detection) tuples
            tracks_dict: Dictionary of tracks to match against (active_tracks or lost_tracks)
            threshold: Minimum similarity threshold for matching
        """
        matches = {}

        if not unmatched_detections:
            return matches

        # Get eligible tracks (not seen this frame and have profile)
        eligible_tracks = [(track_id, track) for track_id, track in tracks_dict.items()
                          if track['last_seen'] != self.frame_count and self.person_profiles.get(track_id)]

        if not eligible_tracks:
            return matches

        # Build cost matrix (detections x tracks)
        det_indices = [det_idx for det_idx, _ in unmatched_detections]
        track_ids = [track_id for track_id, _ in eligible_tracks]

        cost_matrix = np.full((len(unmatched_detections), len(eligible_tracks)), 1.0)
        match_types = {}

        for i, (det_idx, detection) in enumerate(unmatched_detections):
            for j, (track_id, track) in enumerate(eligible_tracks):
                profile = self.person_profiles[track_id]
                similarity, match_type = self._compute_person_similarity(detection, profile)
                cost_matrix[i, j] = 1.0 - similarity  # Convert similarity to cost
                match_types[(i, j)] = match_type

        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            similarity = 1.0 - cost

            if similarity > threshold:
                det_idx = det_indices[row_idx]
                track_id = track_ids[col_idx]
                match_type = match_types[(row_idx, col_idx)]
                matches[det_idx] = (track_id, match_type)

        return matches

    def _match_with_appearance(self, unmatched_detections: List[Tuple[int, Dict]]) -> Dict[int, Tuple[int, str]]:
        """Match with active tracks using appearance"""
        return self._match_with_appearance_hungarian(
            unmatched_detections,
            self.active_tracks,
            self.config.combined_reid_threshold
        )

    def _match_with_lost_tracks(self, unmatched_detections: List[Tuple[int, Dict]]) -> Dict[int, Tuple[int, str]]:
        """Re-identification with lost tracks (higher threshold)"""
        return self._match_with_appearance_hungarian(
            unmatched_detections,
            self.lost_tracks,
            self.config.combined_reid_threshold + 0.1
        )

    def _compute_person_similarity(self, detection: Dict, profile: Dict) -> Tuple[float, str]:
        """Compute similarity between detection and person profile"""
        similarities = []
        matching_components = []
        weights = []
        for body_section in self.similarity_info:
            feature_name = self.similarity_info[body_section]["feat"]
            weight = self.similarity_info[body_section]["weight"]
            threshold = self.similarity_info[body_section]["threshold"]
            if detection[feature_name] is not None and profile.get(feature_name) is not None:
                sim = self._compute_feature_similarity(detection[feature_name], profile[feature_name])
                if sim > threshold:
                    similarities.append(sim)
                    matching_components.append(body_section)
                    weights.append(weight)

        if similarities and weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            combined_sim = sum(s * w for s, w in zip(similarities, normalized_weights))
            match_description = "+".join(matching_components)
            return combined_sim, match_description

        return 0.0, "none"

    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between features"""
        try:
            similarity = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
            return max(0.0, similarity)
        except Exception as e:
            print(f"Error computing feature similarity: {e}")
            return 0.0

    def _update_matches(self, matches: Dict[int, int], detections: List[Dict], final_matches: Dict[int, int], match_type: str) -> None:
        """Update tracks and profiles based on matches"""
        for det_idx, track_id in matches.items():
            app_match_type = None
            if type(track_id) is tuple:
                track_id, app_match_type = track_id
            detection = detections[det_idx]
            track = self.lost_tracks.pop(track_id) if match_type == "appearance_reid" else self.active_tracks[track_id]

            # Update Kalman filter
            update_kalman_filter(track['kalman'], detection['bbox'])

            # Update track
            if match_type == "appearance_reid":
                track['kalman'] = create_kalman_filter(detection['bbox'])
            track['detections'].append(detection)
            track['last_seen'] = self.frame_count
            track['lost_frames'] = 0
            track['missed_updates'] = 0
            track["match_type"] = match_type
            track["app_match_type"] = app_match_type if app_match_type else None
            self._update_track_geometry(track, detection, clean=not detection.get('group', False))

            if match_type == "appearance_reid":
                self.active_tracks[track_id] = track
            # Update profile
            profile = self.person_profiles.get(track_id)
            if profile:
                self._update_person_profile(profile, detection)

            final_matches[det_idx] = track_id

    def _update_person_profile(self, profile: Dict, detection: Dict):
        """Update person profile with new features"""
        # Update features with exponential moving average
        alpha = 0.3

        for body_section in self.similarity_info:
            feature_name = self.similarity_info[body_section]["feat"]
            if detection[feature_name] is not None:
                if profile.get(feature_name) is None:
                    profile[feature_name] = detection[feature_name].copy()
                else:
                    profile[feature_name] = alpha * detection[feature_name] + (1 - alpha) * profile[feature_name]
                    norm = np.linalg.norm(profile[feature_name])
                    if norm > 0:
                        profile[feature_name] = profile[feature_name] / norm

    def _create_new_track(self, detection: Dict) -> int:
        """Create new track and person profile"""
        track = {
            'track_id': self.next_track_id,
            'kalman': create_kalman_filter(detection['bbox']),
            'detections': deque([detection], maxlen=100),
            'last_seen': self.frame_count,
            'created_frame': self.frame_count,
            'lost_frames': 0,
            'missed_updates': 0,
            'match_type': "new"
        }
        self._ensure_track_geometry(track, detection['bbox'])

        profile = {
            'person_id': self.next_track_id,
            'creation_frame': self.frame_count,
            'face_feature': detection['face_feature'].copy() if detection['face_feature'] is not None else None,
            'upper_feature': detection['upper_feature'].copy() if detection['upper_feature'] is not None else None,
            'lower_feature': detection['lower_feature'].copy() if detection['lower_feature'] is not None else None
        }

        self.active_tracks[self.next_track_id] = track
        self.person_profiles[self.next_track_id] = profile

        current_id = self.next_track_id
        self.next_track_id += 1

        return current_id

    def _handle_lost_tracks(self, final_matches: Dict[int, int]):
        """Handle lost tracks and cleanup"""
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track_id not in final_matches.values():
                track['missed_updates'] += 1
                track['lost_frames'] += 1

                if track['lost_frames'] > self.tracker_config.max_lost_frames:
                    if len(track['detections']) >= 10:
                        self.lost_tracks[track_id] = track
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]

        # Cleanup old lost tracks
        tracks_to_cleanup = []
        for track_id, track in self.lost_tracks.items():
            if self.frame_count - track['last_seen'] > self.tracker_config.max_lost_frames * 2: 
                tracks_to_cleanup.append(track_id)

        for track_id in tracks_to_cleanup:
            del self.lost_tracks[track_id]

    def _check_geometry_consistency(self, motion_matches: Dict[int, int], detections: List[Dict]) -> List[int]:
        """
        Check geometry consistency for motion-matched tracks

        Returns list of detection indices with suspect geometry changes.
        
        """
        suspect_matches = []
        for det_idx, track_id in motion_matches.items():
            track = self.active_tracks[track_id]
            if track['missed_updates'] > 5:
                continue
            detection = detections[det_idx]
            det_area, det_aspect = self._compute_bbox_geometry(detection['bbox'])
            track = self.active_tracks[track_id]
            pred_area, pred_aspect = track['geometry']['area_ema'], track['geometry']['aspect_ema']

            # Check if the geometric properties are consistent
            if not self._is_geometry_consistent((det_area, det_aspect), (pred_area, pred_aspect)):
                suspect_matches.append(det_idx)
        return suspect_matches

    def _is_geometry_consistent(self, det_geom: Tuple[float, float], pred_geom: Tuple[float, float]) -> bool:
        """Check if detection geometry is consistent with predicted geometry (bbox area and aspect)"""
        det_area, det_aspect = det_geom
        pred_area, pred_aspect = pred_geom

        area_ratio = det_area / pred_area if pred_area > 0 else 0
        aspect_ratio = det_aspect / pred_aspect if pred_aspect > 0 else 0

        area_consistent = 0.5 <= area_ratio <= 2
        aspect_consistent = 0.7 <= aspect_ratio <= 1.5

        return area_consistent and aspect_consistent

    @staticmethod
    def _compute_bbox_geometry(bbox: np.ndarray) -> Tuple[float, float]:
        """Return (area, aspect_ratio) for the first four bbox coordinates."""
        x1, y1, x2, y2 = map(float, bbox[:4])
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area = width * height
        aspect = width / max(height, 1e-3)
        return area, aspect

    def _ensure_track_geometry(self, track: Dict, bbox: np.ndarray) -> None:
        """Create the geometry stats bundle if this track lacks one."""
        if 'geometry' not in track:
            area, aspect = self._compute_bbox_geometry(bbox)
            track['geometry'] = {
                'area_ema': area,
                'aspect_ema': aspect,
                'last_clean_frame': self.frame_count,
                'status': 'stable',
                'suspect_frames': 0,
            }

    def _update_track_geometry(self, track: Dict, detection: Dict, *, clean: bool = True) -> None:
        """Update (area, aspect) EMA and status flags for a track."""
        self._ensure_track_geometry(track, detection['bbox'])
        geom = track['geometry']
        area, aspect = self._compute_bbox_geometry(detection['bbox'])

        if clean:
            alpha = self.GEOMETRY_EMA_ALPHA
            geom['area_ema'] = alpha * area + (1 - alpha) * geom['area_ema']
            geom['aspect_ema'] = alpha * aspect + (1 - alpha) * geom['aspect_ema']
            geom['last_clean_frame'] = self.frame_count
            geom['status'] = 'stable'
            geom['suspect_frames'] = 0
        else:
            geom['suspect_frames'] = geom.get('suspect_frames', 0) + 1
            geom['status'] = 'suspect'


# ================== VISUALIZATION MODULE ==================

class VisualizationModule:
    """Handles visualization of tracking results"""

    def __init__(self, config: VisualizationConfig, kpt_threshold: float = 0.3):
        self.config = config
        self.kpt_threshold = kpt_threshold

    def draw_tracking_results(self, frame: np.ndarray, pose_results: List[PoseResult], person_assignments: Dict[int, int], active_tracks: Dict) -> np.ndarray:
        """Draw tracking results on frame"""
        if not self.config.enable_visualization or not person_assignments:
            return frame

        # Retrieve poses for tracked detections
        detected_poses: List[PoseResult] = []
        for det_idx, track_id in person_assignments.items():
            if det_idx < len(pose_results):
                pose: PoseResult = pose_results[det_idx]
                detected_poses.append(pose_results[det_idx])

        if not detected_poses:
            return frame

        vis_result = frame.copy()

        # Draw poses if enabled
        if self.config.enable_pose_drawing:
            for pose in detected_poses:
                keypoints: np.ndarray = pose.keypoints[:17]
                keypoint_scores: np.ndarray = pose.keypoints[:, 2]

                # Draw keypoints
                for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if score > self.kpt_threshold:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(vis_result, (x, y), self.config.radius, (0, 255, 0), -1)

                # Draw skeleton connections (COCO format)
                for pt1_idx, pt2_idx in coco_skeleton:
                    if (pt1_idx < len(keypoint_scores) and pt2_idx < len(keypoint_scores) and
                        keypoint_scores[pt1_idx] > self.kpt_threshold and
                        keypoint_scores[pt2_idx] > self.kpt_threshold):
                        pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                        pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                        cv2.line(vis_result, pt1, pt2, (255, 200, 0), self.config.line_width)

        # Draw tracking info
        if self.config.enable_bbox_drawing or self.config.enable_id_labels:
            for det_idx, track_id in person_assignments.items():
                if det_idx < len(pose_results):
                    pose = pose_results[det_idx]
                    if len(pose.bbox) > 0:
                        bbox =pose.bbox

                        x1, y1, x2, y2 = bbox[:4].astype(int)

                        # Determine match type and color
                        track = active_tracks.get(track_id)
                        match_color = {"new": (0, 255, 255), "motion": (0, 255, 0),
                                       "appearance": (255, 0, 0), "appearance_reid": (0, 0, 255)}

                        # Draw bounding box
                        if self.config.enable_bbox_drawing:
                            cv2.rectangle(vis_result, (x1, y1), (x2, y2), match_color[track["match_type"]], 2)

                        # Draw ID and match type
                        if self.config.enable_id_labels:
                            app_match_type = track.get('app_match_type', None)
                            text = f"ID {track_id}: {track['match_type'] + ('(' + app_match_type + ')' if app_match_type else '')}"
                            cv2.rectangle(vis_result, (x1, y1 - 25), (x1 + len(text) * 12, y1), match_color[track["match_type"]], -1)
                            cv2.putText(vis_result, text, (x1 + 2, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis_result


# ================== MAIN PIPELINE ORCHESTRATOR ==================

class MultiPersonTrackingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: PipelineConfig, data_collector: TrackingDataCollector = None, batch_signal_handler=None):
        self.config = config
        self.batch_signal_handler = batch_signal_handler

        # Only initialize pose/tracking modules if not in detection-only mode
        self.feature_module = FeatureExtractionModule(config.features, config.processing.kpt_threshold)
        self.tracking_module = TrackingModule(config.processing)
        self.visualization_module = VisualizationModule(
            config.visualization,
            config.processing.kpt_threshold
        ) if config.visualization.enable_visualization else None

        # Data export
        self.data_collector = (data_collector or TrackingDataCollector()) if config.export.enable_export else None

        # Performance tracking
        self.timing_stats = defaultdict(list)
        self.global_start_time = None

        # Interruption handling
        self._interrupted = False
        self._proc = None
        self._cap = None

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print("\n\nInterrupted! Finalizing video and printing metrics...")
        self._interrupted = True

        # Close ffmpeg stdin to let it finalize the video
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception as e:
                print(f"Error closing ffmpeg stdin during interrupt: {e}")

        # Release video capture
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                print(f"Error releasing video capture during interrupt: {e}")

        # Wait for ffmpeg to finish
        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        if self.data_collector:
            total_runtime = time.time() - self.global_start_time

            # Create output directory if needed
            os.makedirs(os.path.dirname(self.config.export.output_path), exist_ok=True)


            self.data_collector.export_data(self.config.export.output_path, total_runtime)

        # Print performance stats
        print(f"\nPartial processing complete after {self.tracking_module.frame_count} frames")
        print(f"Total persons tracked: {len(self.tracking_module.person_profiles)}")
        self.print_performance_stats()

        # Call batch signal handler if provided
        if self.batch_signal_handler:
            self.batch_signal_handler(signum, frame)
        else:
            sys.exit(0)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        cached_poses: List[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Process single frame with optional caching"""
        # Pose estimation
        pose_start = time.time()
        pose_results: List[PoseResult] = create_pose_results_from_cache(cached_poses)

        # Keypoint Filtering
        oks_start = time.time()
        pose_results = apply_oks_nms(pose_results, self.config.processing.oks_nms_threshold, self.config.processing.kpt_threshold)
        filter_poses_by_keypoints(pose_results, self.config.processing.kpt_threshold)
        self.timing_stats['oks_nms'].append(time.time() - oks_start)

        # Feature extraction
        feat_start = time.time()
        detections = self.feature_module.extract_features(frame, pose_results, self.tracking_module.frame_count + 1)
        self.timing_stats['features'].append(time.time() - feat_start)

        # Tracking
        track_start = time.time()
        person_assignments = self.tracking_module.update(detections, frame)
        self.timing_stats['tracking'].append(time.time() - track_start)

        # Data collection
        if self.data_collector:
            data_collect_start = time.time()
            self.data_collector.collect_frame_data(
                self.tracking_module.frame_count, detections, person_assignments
            )
            self.timing_stats['data_collection'].append(time.time() - data_collect_start)

        # Visualization
        vis_frame = frame
        if self.visualization_module:
            vis_start = time.time()
            vis_frame = self.visualization_module.draw_tracking_results(
                frame, pose_results, person_assignments, self.tracking_module.active_tracks
            )
            self.timing_stats['visualization'].append(time.time() - vis_start)

        return vis_frame

    def process_video(self, input_path: str, output_path: str, segment: Tuple[int, int] = None):
        """Process entire video"""
        self.global_start_time = time.time()

        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Store references for signal handler
        self._cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.config.frame_limit and total_frames > self.config.frame_limit:
            total_frames = self.config.frame_limit

        # Load pose cache using metadata
        pose_cache = None

        # Create CacheManager from metadata
        cache_manager: CacheManager = CacheManager.from_metadata(
            video_filename=output_path,
            cache_base_path=self.config.cache.cache_base_path,
            metadata_format="yaml"
        )

        if not cache_manager:
            raise RuntimeError(
                f"Could not find cache metadata for video: {Path(output_path).stem}\n"
                f"Please run cache_pose.py first to generate detection and pose caches."
            )

        if cache_manager.check_pose_cache():
            print(f"Loading pose cache for: {Path(input_path).stem}")
            pose_cache = cache_manager.load_all_poses()
            if pose_cache:
                print(f"  Loaded {len(pose_cache)} cached poses")
        else:
            raise RuntimeError(
                f"Pose cache file not found at: {cache_manager.pose_cache_path}\n"
                f"Please run cache_pose.py first to generate caches."
            )

        # Initialize data collector with video metadata
        if self.data_collector:
            self.data_collector.set_video_info(input_path, total_frames, fps, width, height)

        # Setup ffmpeg for h264 encoding
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18",
            output_path
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Store reference for signal handler
        self._proc = proc

        lower_frame_bound = segment[0] * fps if segment else 0
        upper_frame_bound = segment[1] * fps if segment else total_frames
        print(f"Processing frames {lower_frame_bound} to {upper_frame_bound}...")
        frame_ix = -1
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened() and not self._interrupted:
                    ret, frame = cap.read()
                    frame_ix += 1
                    if not ret:
                        break

                    try:
                        frame_start = time.time()
                        if frame_ix < lower_frame_bound or frame_ix > upper_frame_bound:
                            pbar.update(1)
                            continue
                        if len(pose_cache) <= frame_ix:
                            print(f"Warning: No cached pose for frame {frame_ix}, skipping.")
                            pbar.update(1)
                            continue
                        vis_frame = self.process_frame(
                            frame, 
                            frame_ix, 
                            pose_cache[frame_ix], 
                        )

                        # Ensure frame size matches what we told ffmpeg
                        if vis_frame.shape[0] != height or vis_frame.shape[1] != width:
                            vis_frame = cv2.resize(vis_frame, (width, height), interpolation=cv2.INTER_LINEAR)

                        # Write frame to ffmpeg stdin
                        try:
                            proc.stdin.write(vis_frame.tobytes())
                        except BrokenPipeError:
                            # ffmpeg died early; print its error and stop
                            raise RuntimeError(f"ffmpeg exited early.")

                        self.timing_stats['total_frame'].append(time.time() - frame_start)

                        # Progress update
                        if self.tracking_module.frame_count % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                            print(f"Frame {self.tracking_module.frame_count}: "
                                  f"Active={len(self.tracking_module.active_tracks)}, "
                                  f"Lost={len(self.tracking_module.lost_tracks)}, "
                                  f"Total={len(self.tracking_module.person_profiles)}")

                    except Exception as e:
                        import traceback
                        frame_num = self.tracking_module.frame_count
                        print(f"Error processing frame {frame_num}: {e}")
                        traceback.print_exc()
                        # Write original frame on error
                        try:
                            if frame.shape[0] != height or frame.shape[1] != width:
                                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            proc.stdin.write(frame.tobytes())
                        except BrokenPipeError as e:
                            print(f"Error writing original frame to ffmpeg (broken pipe): {e}")
                            break

                    pbar.update(1)
        finally:
            cap.release()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError as e:
                    print(f"Error closing ffmpeg stdin in finally block (broken pipe): {e}")
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode(errors="ignore")
                raise RuntimeError(f"ffmpeg failed (code {rc}).\n{err}")

        torch.cuda.empty_cache()
        gc.collect()

        if self.data_collector:
            total_runtime = time.time() - self.global_start_time

            # Create output directory if needed
            os.makedirs(os.path.dirname(self.config.export.output_path), exist_ok=True)
            self.data_collector.export_data(self.config.export.output_path, total_runtime)

        print(f"Processing complete. Output saved: {output_path}")
        print(f"Total persons tracked: {len(self.tracking_module.person_profiles)}")
        self.print_performance_stats()

    def reset_for_next_video(self):
        """Reset state between videos while keeping models loaded"""
        # Reset tracking state
        if self.tracking_module:
            self.tracking_module.frame_count = 0
            self.tracking_module.next_track_id = 1
            self.tracking_module.active_tracks.clear()
            self.tracking_module.lost_tracks.clear()
            self.tracking_module.person_profiles.clear()

            # Reset camera motion compensator
            if self.tracking_module.camera_compensator:
                self.tracking_module.camera_compensator.prev_frame_gray = None
                self.tracking_module.camera_compensator.prev_points = None
                self.tracking_module.camera_compensator.motion_history.clear()

        # Reset timing stats
        self.timing_stats.clear()

        # Reset interruption state
        self._interrupted = False
        self._proc = None
        self._cap = None
        self.global_start_time = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def print_performance_stats(self):
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
            frames_processed = self.tracking_module.frame_count
            fps = frames_processed / total_runtime if total_runtime > 0 else 0
            print(f"Overall Stats:")
            print(f"  Total runtime: {total_runtime:.2f}s")
            print(f"  Frames processed: {frames_processed}")
            print(f"  Processing FPS: {fps:.2f}")

        print("="*60)


# ================== EXAMPLE USAGE ==================

def create_batch_config(json_output_path: str = " ") -> 'PipelineConfig':
    """Create configuration for batch processing"""
    config = PipelineConfig()
    # feature extraction
    config.features.enable_face_features = True
    config.features.no_deepface = True
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True
    config.features.feature_update_interval = 1
    config.features.face_confidence_threshold = 0.4
    config.features.face_min_size = 40
    config.features.body_min_height = 50
    config.features.body_min_width = 30
    # detection
    config.processing.oks_nms_threshold = 0.55             # Keypoint OKS NMS threshold (0.5-0.95) - higher= more overlapping poses removal
    config.processing.kpt_threshold = 0.5

    # motion tracking
    tracker_cfg = TrackerConfig(
        base_iou_threshold=0.40,           # IoU threshold for matching (0.1-0.5) - higher=stricter matching
        base_motion_confidence=0.2,       # Motion prediction confidence (0.1-0.5) - higher=trust motion more
        base_center_weight=0.75,           # Weight for center distance vs IoU (0.0-1.0) - higher=more weight on center
        max_lost_frames=500,               # Frames before track is lost (50-600) - higher=keep tracks alive longer
        confidence_decay_rate=0.04,        # How fast confidence decays (0.01-0.1) - higher=faster decay
        max_jump_factor=2.0               # Max allowed position jump (1.5-3.5) - higher=allow bigger jumps
    )
    config.processing.tracker_config = tracker_cfg
    # reid
    config.processing.face_reid_threshold = 0.75             # Face matching threshold (0.5-0.9) - higher=stricter
    config.processing.upper_reid_threshold = 0.5           # Upper body matching (0.5-0.9) - higher=stricter
    config.processing.lower_reid_threshold = 0.5           # Lower body matching (0.5-0.9) - higher=stricter
    config.processing.combined_reid_threshold = 0.7          # Combined ReID threshold (0.3-0.9) - lower=more matches
    # save
    config.visualization.enable_visualization = True
    config.visualization.enable_pose_drawing = True
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True
    config.visualization.radius = 3
    config.visualization.line_width = 1
    config.export.enable_export = True
    config.export.output_path = json_output_path

    config.cache.enable_cache = True

    return config

def main():
    """Main function demonstrating usage"""
    # Create configuration
    config = create_batch_config("/orcd/data/satra/001/users/brukew/motion_tracking_output/tracking_results_reid_test_new_params.json")
    DATA_DIR = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized"

    passing = [
        "/M.J._Home_Videos_AMES_S9B3J7I4R1/12-16 month videos/20170802_084834.mp4", # (very easy case) single child alone walking
        "/T.M._Home_Videos_AMES_M0P5V8K9N0/34-38 month videos/IMG_9006.mp4", # (easy case) child and mom interacting but stays in some place with some bbox intersections
        "/S.V._Home_Videos_AMES_B1L0B3F6F1/34-38 month videos/September 10 2022.mp4", # (medium hard case) girl running into fathers arms and getting lifted
        "/S.V._Home_Videos_AMES_B1L0B3F6F1/34-38 month videos/August 15 2022.mp4", # STIL HAS ISSUE WITH non-human detection (medium case) child in box with limbs poking out - pops out at end - picture in back gets confused as person
        "/M.J._Home_Videos_AMES_S9B3J7I4R1/12-16 month videos/20170729_091714.mp4", # (medium case) 2 small children walking around living room with adults leg poking out
    ]
    appearance_tests = [
        "/M.J._Home_Videos_AMES_S9B3J7I4R1/12-16 month videos/20170729_091714.mp4", # (medium case) 2 small children walking around living room with adults leg poking out
        "/A.SR._Home_Videos_AMES_Y4W4H9A8X5/34-38 month videos/07-12-2021.mp4", # (hard case) adult and 2 children walking, adult and children occlude one another frequently
        "/S.V._Home_Videos_AMES_B1L0B3F6F1/34-38 month videos/September 10 2022.mp4", # (medium hard case) girl running into fathers arms and getting lifted
        "/A.M._Home_Videos_AMES_B4Q3G8H2N3/34-38 month videos/IMG_1845.mp4" # (medium case) reid test with child in chair and camera panning
        "/A.B._Home_Videos_AMES_S2C4T1Y7V7/34-38 month videos/IMG_4610.MOV" # (medium easy case) 2 children not in the same frame jumping around in similar clothing (appearance test)
        "/D.B._Home_Videos_AMES_V7D7K3H8B4/IMG_5398.mov" # (medium hard case) 2 children moving around occlusing one another frequently (appearance test)
    ]
    motion_tests = [
        "/N.S._Home_Videos_AMES_D4Y7P4G2V4/20211202_214033.mp4", # (medium easy case) child moving in front of adult (appearance / motion)
        "/B.C._Home_Videos_AMES_L0B0Q5O3Q3/07-10-2021.mp4",
        "/D.B._Home_Videos_AMES_V7D7K3H8B4/IMG_5398.mp4", # (medium hard case) 2 children moving around occluding one another frequently (appearance test)

    ]
    insanely_hard = [
        "/I.F._Home_Videos_AMES_V7G3E2O2E6/12-16 Month Home Videos/Screen_Recording_20230511_163000_Facebook.mp4", # (medium case) bad quality vid - 2 very similar looking children interacting
    ]
    active_tests = [
        "/A.SR._Home_Videos_AMES_Y4W4H9A8X5/34-38 month videos/07-12-2021.mp4", # (hard case) adult and 2 children walking, adult and children occlude one another frequently
        "/A.M._Home_Videos_AMES_B4Q3G8H2N3/34-38 month videos/IMG_1845.mp4" # (medium case) reid test with child in chair and camera panning
    ]
    new_tests = [
        "/H.T._Home_Videos_AMES_A5X1S7S1E7/34-38 month videos/078A361D-9F42-4FD2-9A10-8169514646CA2018-07-18_08-45-02_000.mp4", # (hard case) child behind adult for whole video
        "/A.B._Home_Videos_AMES_S2C4T1Y7V7/34-38 month videos/IMG_4610.mp4", # (medium easy case) 2 children not in the same frame jumping around in similar clothing (appearance test)
        "/D.B._Home_Videos_AMES_V7D7K3H8B4/IMG_5398.mp4", # (medium hard case) 2 children moving around occluding one another frequently (appearance test)
        "/N.S._Home_Videos_AMES_D4Y7P4G2V4/20211202_214033.mp4", # (medium easy case) child moving in front of adult (appearance / motion)
        "/D.B-A._Home_Videos_AMES_G3A6S5R9F7/34-38 month videos/June 4 2018.mp4", # (medium case) parent and child in train with many in background and camera panning (appearance / motion)
        "/B.C._Home_Videos_AMES_L0B0Q5O3Q3/07-10-2021.mp4", # (easy medium case) parent holding child on swings (appearance / motion)
        "/N.S._Home_Videos_AMES_D4Y7P4G2V4/VID_20200412_173309.mp4", # (medium hard case) 2 children and adult gathered around - camera panning around them
    ]

    tests = new_tests + passing
    # Process video
    # VID_LOCAL_PATH = "/B.C._Home_Videos_AMES_L0B0Q5O3Q3/07-10-2021.mp4"
    # TARGET_VIDEO_PATH = "/orcd/data/satra/001/users/brukew/motion_tracking_output/cache_test/caching_test.mp4"
    # SOURCE_VIDEO_PATH = DATA_DIR + VID_LOCAL_PATH
    # SOURCE_VIDEO_PATH = '/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external/D.B-A._Home_Videos_AMES_G3A6S5R9F7/34-38 month videos/June 4 2018.mp4'
    # pipeline = MultiPersonTrackingPipeline(config)
    # pipeline.process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH)

    test_name = "face_features_base"
    for VID_LOCAL_PATH in tests:
        SOURCE_VIDEO_PATH = DATA_DIR + VID_LOCAL_PATH
        TARGET_VIDEO_PATH = f"/orcd/data/satra/001/users/brukew/motion_tracking_output/resnet/{test_name}/" + VID_LOCAL_PATH[:-4].replace("/", "_") + ".mp4"
        pipeline = MultiPersonTrackingPipeline(config)
        pipeline.process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH)

if __name__ == "__main__":
    main()