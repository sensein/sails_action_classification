import sys
import os
from pathlib import Path

# Import local utils before adding CLIP-ReID
from sailsprep.id_tracking_model.utils.utils import oks_nms

clip_path: str = "sailsprep/feature_processing/tracker/clip/CLIP-ReID"
if os.path.exists(clip_path) and clip_path not in sys.path:
    sys.path.insert(0, clip_path)
else:
    raise ImportError(f"CLIP-ReID path '{clip_path}' does not exist.")

import numpy as np
import cv2
from tqdm import tqdm
import torch
import time
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import gc
import signal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from abc import abstractmethod
import tempfile

# from deepface import DeepFace
from sailsprep.id_tracking_model.utils.cache_manager import CacheManager
from sailsprep.id_tracking_model.utils.tracking_exporter_new import TrackingDataCollector


import torchvision.transforms as transforms
import torchvision.models as models

from sailsprep.id_tracking_model.tracker.person_tracker import (
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
class CLIPReIDConfig:
    """CLIP-ReID model configuration."""
    config_path: str = 'configs/person/vit_clipreid.yml'
    checkpoint_path: str = '/orcd/data/satra/002/models/clip/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'
    num_classes: int = 1041
    camera_num: int = 15
    view_num: int = 1
    device: str = "cuda"

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
    combined_reid_threshold: float = 0.6

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
    export_hdf5: bool = True  # Export per-track HDF5 files with embeddings
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
    clip: CLIPReIDConfig = field(default_factory=CLIPReIDConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    frame_limit: int = 0  # 0 means process all frames


# ================== DETECTION AND POSE HELPER MODULE ==================

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
            # embedding_result = DeepFace.represent(
            #     face_resized,
            #     model_name='Facenet',
            #     enforce_detection=False,
            #     detector_backend='skip'
            # )
            embedding_result = [[1]]
            embedding = np.array(embedding_result[0]['embedding'])

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                return embedding

        except Exception as e:
            print(f"Error extracting face features with DeepFace/MTCNN: {e}")

        return None


class CLIPReIDFeatureExtractor:
    """
    Extracts appearance features using CLIP-ReID model.
    
    Transforms person crops into normalized embedding vectors for re-identification.
    """
    
    def __init__(self, config: CLIPReIDConfig):
        """
        Initialize CLIP-ReID feature extractor.
        
        Args:
            config: CLIPReIDConfig with model checkpoint and parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self.input_size = (256, 128)
        self._init_model()
    
    def _init_model(self):
        """Initialize CLIP-ReID model from checkpoint."""                
        from config.defaults import _C as cfg
        from model.make_model_clipreid import make_model
        from torchvision import transforms
        
        config_file = os.path.join(clip_path, self.config.config_path)

        with open(config_file, 'r') as f:
            config_str = f.read()
        
        config_str = config_str.replace(
            "DATASETS:\n#   NAMES: ('market1501')\n#   ROOT_DIR: ('')\n# OUTPUT_DIR: ''",
            "DATASETS:\n  NAMES: 'market1501'\n  ROOT_DIR: ''\n "
        )
        
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_file:
            temp_file.write(config_str)
            temp_file.flush()
            temp_config_path = temp_file.name
        
        cfg.merge_from_file(temp_config_path)
        cfg.defrost()
        cfg.MODEL.DEVICE = str(self.device)
        cfg.MODEL.Transformer_TYPE = 'ViT-B-16'
        cfg.MODEL.NAME = cfg.MODEL.Transformer_TYPE
        cfg.MODEL.PRETRAIN_PATH = self.config.checkpoint_path
        cfg.INPUT.SIZE_TRAIN = [256, 128]
        cfg.INPUT.SIZE_TEST = [256, 128]
        cfg.MODEL.SIE_CAMERA = True
        cfg.MODEL.SIE_VIEW = True
        cfg.MODEL.STRIDE_SIZE = [12, 12]
        cfg.TEST.WEIGHT = self.config.checkpoint_path
        cfg.freeze()
        
        self.model = make_model(cfg, num_class=self.config.num_classes,
                               camera_num=self.config.camera_num,
                               view_num=self.config.view_num)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.to(self.device)
        self.model.eval()
        
        self.normalize = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        self.to_tensor = transforms.ToTensor()
        
        print(f"CLIP-ReID loaded from: {cfg.TEST.WEIGHT}")
    
    def extract(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized appearance feature from image region.
        
        Args:
            roi: Image region of interest as numpy array (H, W, 3) in BGR format
            
        Returns:
            Normalized feature vector of shape (feature_dim,) or None if extraction fails
        """
        try:
            if roi.shape[0] < 40 or roi.shape[1] < 20:
                return None
            
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(roi_rgb, (self.input_size[1], self.input_size[0]),
                                interpolation=cv2.INTER_LINEAR)
            
            tensor = self.to_tensor(resized)
            tensor = self.normalize(tensor)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):
                    features = self.model(tensor)
            
            feature_vector = features.cpu().numpy()[0]
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                return feature_vector / norm
            
        except Exception as e:
            print(f"CLIP-ReID extraction error: {e}")
        
        return None

class FeatureExtractionModule:
    """Updated feature extraction module with simplified ResNet"""

    def __init__(self, config: FeatureConfig, clip_config: CLIPReIDConfig, kpt_threshold: float = 0.3):
        self.config = config
        self.kpt_threshold = kpt_threshold
        self.clip_config = clip_config

        self.body_extractor = None
        if config.enable_upper_body_features or config.enable_lower_body_features or (config.no_deepface and config.enable_face_features):
            self.body_extractor = CLIPReIDFeatureExtractor(self.clip_config)

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
            'sufficient_keypoints': getattr(pose, 'metadata', {}).get('sufficient_keypoints', True),
            'frame_idx': frame_count  
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
    
# ================== Detection Validator ==================

class DetectionValidator:
    """
    Validates if detection is a real person, not photo/screen/TV
    
    Uses motion naturalness and texture analysis to identify:
    - Real people: Natural varying motion, rich texture details
    - Photos/Screens: Uniform motion (moves with camera), low texture variance
    """
    
    def __init__(self):
        """Initialize validator with tracking history"""
        self.motion_history = {}
        self.texture_history = {}
        self.size_history = {}
    
    def is_real_person(self, track_id: int, bbox: np.ndarray, frame: np.ndarray,
                       min_frames: int = 10) -> bool:
        """
        Check if track represents real person (not photo/TV/screen)
        
        Strategy:
        1. Motion analysis - photos don't move naturally
        2. Texture variance - screens/photos have different texture patterns
        3. Size consistency - photos often have unnatural scaling
        
        Args:
            track_id: track id to validate
            bbox: Bounding box [x1, y1, x2, y2]
            frame: Current frame image
            min_frames: Minimum frames needed for validation
            
        Returns:
            bool: True if real person, False if likely photo/screen
        """
        # Extract roi
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return False  # Too small to analyze (remove samll detection)

        # 1. Texture variance analysis 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (to measure of texture detail)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = np.var(laplacian)
        
        # Compute edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if track_id not in self.texture_history:
            self.texture_history[track_id] = deque(maxlen=30)
        self.texture_history[track_id].append({
            'variance': texture_var,
            'edge_density': edge_density
        })
        
        # Wait for enough history
        if len(self.texture_history[track_id]) < min_frames:
            return True  # Assume valid initially
        
        # Analyze texture consistency
        texture_data = list(self.texture_history[track_id])
        avg_texture = np.mean([t['variance'] for t in texture_data])
        texture_std = np.std([t['variance'] for t in texture_data])
        avg_edge_density = np.mean([t['edge_density'] for t in texture_data])
        
        # Photos/screens: Very consistent, low texture variance
        if avg_texture < 50 and texture_std < 10:
            return False
        
        # Photos/screens: Often have very uniform edge patterns
        if avg_edge_density < 0.02 or avg_edge_density > 0.5:
            return False
        
        # 2.Motion naturalness analysis
        if track_id in self.motion_history and len(self.motion_history[track_id]) >= min_frames:
            motions = list(self.motion_history[track_id])
            
            # Real people have varying motion patterns
            motion_variance = np.var(motions)
            motion_range = max(motions) - min(motions)
            
            # Photos: Near-zero motion variance (moves uniformly with camera)
            if motion_variance < 2.0 and motion_range < 5.0:
                return False

        # 3.Size consistency analysis
        bbox_size = (x2 - x1) * (y2 - y1)
        
        if track_id not in self.size_history:
            self.size_history[track_id] = deque(maxlen=30)
        self.size_history[track_id].append(bbox_size)
        
        if len(self.size_history[track_id]) >= min_frames:
            sizes = list(self.size_history[track_id])
            size_std = np.std(sizes)
            avg_size = np.mean(sizes)
            
            # Photos: Unnatural size variation
            if avg_size > 0:
                cv = size_std / avg_size
                if cv > 0.4:  # More than 40% variation
                    return False
        
        return True
    
    def update_motion(self, track_id: int, motion_score: float):
        """Update motion history for track."""
        if track_id not in self.motion_history:
            self.motion_history[track_id] = deque(maxlen=30)
        self.motion_history[track_id].append(motion_score)
    
    def reset_track(self, track_id: int):
        """Reset validation history for track."""
        if track_id in self.motion_history:
            del self.motion_history[track_id]
        if track_id in self.texture_history:
            del self.texture_history[track_id]
        if track_id in self.size_history:
            del self.size_history[track_id]
            
# ================== TRACKING MODULE ==================

class TrackingModule:
    """Handles multi-person tracking with re-identification"""

    GEOMETRY_EMA_ALPHA = 0.25

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tracker_config = config.tracker_config
        self.frame_count = 0
        self.next_track_id = 1
        
        # Track storage
        self.active_tracks = {}
        self.lost_tracks = {}
        self.person_profiles = {}
        
        # Candidate tracking
        self.candidate_tracks = {}
        self.next_candidate_id = -1
        self.confirmation_frames_required = 3
        
        self.camera_compensator = CameraMotionCompensator() 
        
        self.validator = DetectionValidator()
        self.invalid_track_ids = set()  # For the tracks marked as photos/screens
        
    def _calculate_motion_score(self, track: Dict) -> float:
        """Calculate motion score for validator from bbox movement."""
        if len(track['detections']) < 2:
            return 0.0
        
        # Use last 10 detections
        recent = list(track['detections'])[-10:]
        centers = []
        for det in recent:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))
        
        # Calculate frame-to-frame movement
        movements = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            movements.append(dist)
        
        return np.mean(movements) if movements else 0.0
   
   
   
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

        # 1. Motion-based matching (include candidates)
        all_tracks_for_motion = {**self.active_tracks, **self.candidate_tracks}
        motion_matches = self._match_with_motion(
            detections, camera_motion, iou_thresh, center_weight, 
            motion_conf_thresh, all_tracks_for_motion
        )

        # Check geometry consistency and drop suspect matches
        suspect_detections = self._check_geometry_consistency(motion_matches, detections)
        suspect_detections += self._check_appearance_consistency(motion_matches, detections)
        motion_matches = {i: j for i, j in motion_matches.items() if i not in suspect_detections}

        # Update matched tracks (handle candidates vs active)
        for det_idx, track_id in motion_matches.items():
            detection = detections[det_idx]
            
            if track_id > 0:  # Active track
                track = self.active_tracks[track_id]
                update_kalman_filter(track['kalman'], detection['bbox'])
                track['detections'].append(detection)
                track['last_seen'] = self.frame_count
                track['lost_frames'] = 0
                track['missed_updates'] = 0
                track["match_type"] = "motion"
                track["app_match_type"] = None
                self._update_track_geometry(track, detection, clean=not detection.get('group', False))
                
                # Update motion history for validator
                motion_score = self._calculate_motion_score(track)
                self.validator.update_motion(track_id, motion_score)
                
                profile = self.person_profiles.get(track_id)
                if profile:
                    self._update_person_profile(profile, detection)
            
            else:  # Candidate track (negative ID)
                self._update_candidate_track(detection, track_id)
            
            final_matches[det_idx] = track_id

        # Discard detections with insufficient keypoints if not matched by motion
        unmatched_detections = [(i, det) for i, det in enumerate(detections) 
                            if i not in motion_matches and det['sufficient_keypoints']]

        # 2. Appearance-based matching (include candidates)
        all_tracks_for_appearance = {**self.active_tracks, **self.candidate_tracks}
        appearance_matches = self._match_with_appearance(
            unmatched_detections, all_tracks_for_appearance
        )
        
        for det_idx, (track_id, app_match_type) in appearance_matches.items():
            detection = detections[det_idx]
            
            if track_id > 0:  # Active track
                track = self.active_tracks[track_id]
                update_kalman_filter(track['kalman'], detection['bbox'])
                track['detections'].append(detection)
                track['last_seen'] = self.frame_count
                track['lost_frames'] = 0
                track['missed_updates'] = min(track['missed_updates'] + 1, 3)
                track["match_type"] = "appearance"
                track["app_match_type"] = app_match_type
                self._update_track_geometry(track, detection, clean=not detection.get('group', False))
                
                motion_score = self._calculate_motion_score(track)
                self.validator.update_motion(track_id, motion_score)
                
                profile = self.person_profiles.get(track_id)
                if profile:
                    self._update_person_profile(profile, detection)
            
            else:  # Candidate track
                self._update_candidate_track(detection, track_id)
            
            final_matches[det_idx] = track_id
        
        unmatched_detections = [(i, det) for i, det in unmatched_detections 
                            if i not in appearance_matches]

        # 3. Re-identification with lost tracks (only confirmed tracks)
        reid_matches = self._match_with_lost_tracks(unmatched_detections)
        
        for det_idx, (track_id, match_type) in reid_matches.items():
            detection = detections[det_idx]
            track = self.lost_tracks.pop(track_id)
            
            track['kalman'] = create_kalman_filter(detection['bbox'])
            track['detections'].append(detection)
            track['last_seen'] = self.frame_count
            track['lost_frames'] = 0
            track['missed_updates'] = 0
            track["match_type"] = "appearance_reid"
            track["app_match_type"] = match_type
            self._update_track_geometry(track, detection, clean=not detection.get('group', False))
            
            self.active_tracks[track_id] = track
            
            motion_score = self._calculate_motion_score(track)
            self.validator.update_motion(track_id, motion_score)
            
            profile = self.person_profiles.get(track_id)
            if profile:
                self._update_person_profile(profile, detection)
            
            final_matches[det_idx] = track_id
        
        remaining_unmatched = [i for i, det in unmatched_detections if i not in reid_matches]

        # 4. Create candidate tracks (not permanent tracks)
        for det_idx in remaining_unmatched:
            detection = detections[det_idx]
            candidate_id = self._create_candidate_track(detection)
            if candidate_id is not None:
                final_matches[det_idx] = candidate_id

        # 5. promote candidates that reached confirmation threshold
        self._promote_confirmed_candidates()
        
        # Print tracking stats
        if self.frame_count % 50 == 0:
            print(f"\nFrame {self.frame_count}:")
            print(f"  Active tracks: {len(self.active_tracks)}")
            print(f"  Candidate tracks: {len(self.candidate_tracks)}")
            print(f"  Lost tracks: {len(self.lost_tracks)}")
            print(f"  Confirmed persons: {len(self.person_profiles)}")
        # 6. Cleanup failed candidates
        self._cleanup_failed_candidates()

        # Validate tracks (only active, not candidates)
        for track_id, track in self.active_tracks.items():
            if track.get('needs_validation', False) and not track.get('validated', False):
                last_detection = track['detections'][-1]
                is_valid = self.validator.is_real_person(
                    track_id,
                    last_detection['bbox'],
                    frame,
                    min_frames=10
                )
                
                if not is_valid:
                    self.invalid_track_ids.add(track_id)
                    print(f"Track {track_id} marked as photo/screen")
                
                track['validated'] = True
                track['needs_validation'] = False

        # 7. Handle lost tracks (only active tracks)
        self._handle_lost_tracks_with_candidates(final_matches)
        
        return final_matches

     
    def _match_with_motion(self, detections: List[Dict], camera_motion: tuple, 
                                    iou_thresh: float, center_weight: float, 
                                    motion_conf_thresh: float, all_tracks: Dict) -> Dict[int, int]:
        """
        Match detections with tracks (active + candidates) using motion prediction.
        
        Args:
            detections: List of detections
            camera_motion: Estimated camera motion
            iou_thresh: IoU threshold for matching
            center_weight: Weight for center distance
            motion_conf_thresh: Motion confidence threshold
            all_tracks: Combined dict of active_tracks and candidate_tracks
            
        Returns:
            Dict mapping detection index to track_id (positive or negative)
        """
        matches = {}

        if not all_tracks:
            return matches

        track_ids = list(all_tracks.keys())

        if not detections:
            # Advance all tracks even with no detections
            for track_id in track_ids:
                track = all_tracks[track_id]
                predict_motion_with_camera_compensation(
                    track['kalman'],
                    track['missed_updates'],
                    camera_motion,
                    cfg=self.tracker_config,
                )
            return matches

        cost_matrix = np.full((len(detections), len(track_ids)), 1.0)

        # Predict once per track
        track_predictions = {}
        for track_id in track_ids:
            track = all_tracks[track_id]
            track_predictions[track_id] = predict_motion_with_camera_compensation(
                track['kalman'],
                track['missed_updates'],
                camera_motion,
                cfg=self.tracker_config,
            )

        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                predicted_bbox, motion_confidence = track_predictions[track_id]
                
                # Adjust threshold for candidates (more lenient)
                effective_thresh = motion_conf_thresh
                track = all_tracks[track_id]
                if track.get('is_candidate', False):
                    effective_thresh *= 0.8  # 20% more lenient for candidates
                
                if motion_confidence > effective_thresh:
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
        
        unmatched_det_indices = set(range(len(detections))) - set(matches.keys())
        #if unmatched_det_indices and len(all_tracks) > 0:
            #print(f"\n  Frame {self.frame_count}: {len(unmatched_det_indices)} detections NOT matched by motion")
            #print(f"   Active tracks: {[tid for tid in all_tracks.keys() if tid > 0]}")
            #print(f"   Candidate tracks: {[tid for tid in all_tracks.keys() if tid < 0]}")
        
        return matches


    def _match_with_appearance(self, unmatched_detections: List[Tuple[int, Dict]],
                                        all_tracks: Dict) -> Dict[int, Tuple[int, str]]:
        """
        Lower threshold when face unavailable in either detection or profile
        """
        matches = {}
        
        if not unmatched_detections:
            return matches
        
        # Get eligible tracks
        eligible_tracks = []
        for track_id, track in all_tracks.items():
            if track['last_seen'] == self.frame_count:
                continue
            
            # Active tracks: check profile
            if track_id > 0 and track_id in self.person_profiles:
                eligible_tracks.append((track_id, track))
            # Candidates: check features in last detection
            elif track_id < 0 and len(track['detections']) > 0:
                last_det = track['detections'][-1]
                if (last_det.get('upper_feature') is not None or 
                    last_det.get('lower_feature') is not None):
                    eligible_tracks.append((track_id, track))
        
        if not eligible_tracks:
            return matches
        
        det_indices = [det_idx for det_idx, _ in unmatched_detections]
        track_ids = [track_id for track_id, _ in eligible_tracks]
        
        cost_matrix = np.full((len(unmatched_detections), len(eligible_tracks)), 1.0)
        match_types = {}
        
        for i, (det_idx, detection) in enumerate(unmatched_detections):
            for j, (track_id, track) in enumerate(eligible_tracks):
                # Get profile or last detection features
                if track_id > 0:
                    profile = self.person_profiles[track_id]
                else:
                    last_det = track['detections'][-1]
                    profile = {
                        'upper_feature': last_det.get('upper_feature'),
                        'lower_feature': last_det.get('lower_feature'),
                        'face_feature': last_det.get('face_feature')
                    }
                
                similarity, match_type = self._compute_person_similarity(detection, profile)
                
                # Threshold based on face availability
                base_threshold = self.config.combined_reid_threshold
                
                det_has_face = detection.get('face_feature') is not None
                prof_has_face = profile.get('face_feature') is not None
                
                # Calculate effective threshold
                if not det_has_face or not prof_has_face:
                    # Body only match Lower threshold significantly
                    effective_threshold = base_threshold * 0.85  # 15% more lenient
                else:
                    # Full match Use standard threshold
                    effective_threshold = base_threshold
                
                # Additional leniency for candidates
                if track_id < 0:
                    effective_threshold *= 0.90
                
                cost_matrix[i, j] = 1.0 - similarity
                match_types[(i, j)] = (match_type, effective_threshold)
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            similarity = 1.0 - cost
            match_type, effective_threshold = match_types[(row_idx, col_idx)]
            
            if similarity > effective_threshold:
                det_idx = det_indices[row_idx]
                track_id = track_ids[col_idx]
                matches[det_idx] = (track_id, match_type)
        
        return matches
    
    def _match_with_lost_tracks(self, unmatched_detections: List[Tuple[int, Dict]]) -> Dict[int, Tuple[int, str]]:
        """
        Re-identification with lost tracks (higher threshold but still adaptive).
        """
        matches = {}
        
        if not unmatched_detections:
            return matches
        
        eligible_tracks = [(track_id, track) for track_id, track in self.lost_tracks.items()
                          if self.person_profiles.get(track_id)]
        
        if not eligible_tracks:
            return matches
        
        det_indices = [det_idx for det_idx, _ in unmatched_detections]
        track_ids = [track_id for track_id, _ in eligible_tracks]
        
        cost_matrix = np.full((len(unmatched_detections), len(eligible_tracks)), 1.0)
        match_types = {}
        
        for i, (det_idx, detection) in enumerate(unmatched_detections):
            for j, (track_id, track) in enumerate(eligible_tracks):
                profile = self.person_profiles[track_id]
                similarity, match_type = self._compute_person_similarity(detection, profile)
                
                # ADAPTIVE THRESHOLD for lost tracks too
                base_threshold = self.config.combined_reid_threshold + 0.1  # Stricter for ReID
                
                det_has_face = detection.get('face_feature') is not None
                prof_has_face = profile.get('face_feature') is not None
                
                if not det_has_face or not prof_has_face:
                    effective_threshold = base_threshold * 0.90  # Still lenient for body-only
                else:
                    effective_threshold = base_threshold
                
                cost_matrix[i, j] = 1.0 - similarity
                match_types[(i, j)] = (match_type, effective_threshold)
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            similarity = 1.0 - cost
            match_type, effective_threshold = match_types[(row_idx, col_idx)]
            
            if similarity > effective_threshold:
                det_idx = det_indices[row_idx]
                track_id = track_ids[col_idx]
                matches[det_idx] = (track_id, match_type)
        
        return matches

    def _compute_person_similarity(self, detection: Dict, profile: Dict) -> Tuple[float, str]:
        """
        1. Body features weighted higher than face
        2. Adaptive thresholds when face missing
        3. Accept matches on body alone if strong enough
        """
        similarities = []
        matching_components = []
        weights = []
        
        # PRIORITY: Body features over face
        feature_configs = [
            {'name': 'upper', 'feat': 'upper_feature', 'weight': 0.50, 'threshold': 0.65},  # PRIMARY
            {'name': 'lower', 'feat': 'lower_feature', 'weight': 0.30, 'threshold': 0.60},  # SECONDARY  
            {'name': 'face', 'feat': 'face_feature', 'weight': 0.20, 'threshold': 0.75}     # SUPPLEMENTARY
        ]
        
        det_has_face = detection.get('face_feature') is not None
        prof_has_face = profile.get('face_feature') is not None
        face_available = det_has_face and prof_has_face
        
        for config in feature_configs:
            det_feat = detection.get(config['feat'])
            prof_feat = profile.get(config['feat'])
            
            if det_feat is not None and prof_feat is not None:
                sim = self._compute_feature_similarity(det_feat, prof_feat)
                
                # ADAPTIVE THRESHOLD: Lower if face not available
                threshold = config['threshold']
                if not face_available and config['name'] != 'face':
                    threshold *= 0.90  # 10% more lenient for body-only matching
                
                if sim > threshold:
                    similarities.append(sim)
                    matching_components.append(config['name'])
                    weights.append(config['weight'])
        
        if similarities and weights:
            # Normalize weights (critical when some features missing)
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            combined_sim = sum(s * w for s, w in zip(similarities, normalized_weights))
            
            # BOOST similarity if body features match strongly (even without face)
            if not face_available and 'upper' in matching_components:
                # Body-only match bonus
                combined_sim *= 1.05  # 5% boost
            
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

    
    def _update_person_profile(self, profile: Dict, detection: Dict):
        """
        Update profile with EMA AND add newly discovered features.
        
        Profiles can GROW to include face when it appears later.
        """
        alpha = 0.3
        
        feature_names = ['face_feature', 'upper_feature', 'lower_feature']
        
        for feature_name in feature_names:
            new_feat = detection.get(feature_name)
            
            if new_feat is not None:
                old_feat = profile.get(feature_name)
                
                if old_feat is None:
                    # NEW FEATURE DISCOVERED - add it to profile
                    profile[feature_name] = new_feat.copy()
                    print(f"Track {profile['person_id']}: Added {feature_name} "
                          f"at frame {self.frame_count}")
                else:
                    # Update existing feature with EMA
                    profile[feature_name] = alpha * new_feat + (1 - alpha) * old_feat
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

    def _handle_lost_tracks_with_candidates(self, final_matches: Dict[int, int]):
        """Handle lost tracks and cleanup (separate logic for active vs candidates)."""
        
        # Handle ACTIVE tracks
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
                self.validator.reset_track(track_id)

        # Cleanup old lost tracks
        tracks_to_cleanup = []
        for track_id, track in self.lost_tracks.items():
            if self.frame_count - track['last_seen'] > self.tracker_config.max_lost_frames * 2: 
                tracks_to_cleanup.append(track_id)

        for track_id in tracks_to_cleanup:
            del self.lost_tracks[track_id]
            self.validator.reset_track(track_id)
        
        # Handle candidate tracks (increment lost frames for cleanup)
        for candidate_id, track in list(self.candidate_tracks.items()):
            if candidate_id not in final_matches.values():
                track['lost_frames'] += 1
                # Actual cleanup happens in _cleanup_failed_candidates()

    
    
    def _create_candidate_track(self, detection: Dict) -> int:
        """Create candidate track with detailed logging"""
        # Check if we should create new track
        if self.config.max_tracks > 0:
            total_tracks = len(self.person_profiles) + len(self.candidate_tracks)
            if total_tracks >= self.config.max_tracks:
                return None
        
        # Assign negative ID
        candidate_id = self.next_candidate_id
        self.next_candidate_id -= 1
        
        # Log why new candidate was created
        print(f"\n Frame {self.frame_count}: Creating candidate {candidate_id}")
        print(f"   Active tracks: {len(self.active_tracks)}")
        print(f"   Lost tracks: {len(self.lost_tracks)}")
        print(f"   Detection bbox: {detection['bbox']}")
        print(f"   Has face: {detection.get('face_feature') is not None}")
        print(f"   Has upper: {detection.get('upper_feature') is not None}")
        print(f"   Has lower: {detection.get('lower_feature') is not None}")
        
        # Create track structure
        track = {
            'track_id': candidate_id,
            'kalman': create_kalman_filter(detection['bbox']),
            'detections': deque([detection], maxlen=100),
            'last_seen': self.frame_count,
            'created_frame': self.frame_count,
            'lost_frames': 0,
            'missed_updates': 0,
            'match_type': "candidate",
            'confirmation_count': 1,
            'is_candidate': True
        }
        
        self._ensure_track_geometry(track, detection['bbox'])
        self.candidate_tracks[candidate_id] = track
        
        return candidate_id

    def _update_candidate_track(self, detection: Dict, candidate_id: int):
        """
        Update candidate track and increment confirmation counter.
        
        Args:
            candidate_id: Negative candidate track ID
            detection: New detection
        """
        if candidate_id not in self.candidate_tracks:
            return
        
        track = self.candidate_tracks[candidate_id]
        
        # Update Kalman filter
        update_kalman_filter(track['kalman'], detection['bbox'])
        
        # Update track state
        track['detections'].append(detection)
        track['last_seen'] = self.frame_count
        track['lost_frames'] = 0
        track['missed_updates'] = 0
        
        # confirmation counter 
        track['confirmation_count'] += 1
        
        # Update geometry
        self._update_track_geometry(track, detection, clean=True)


    def _promote_confirmed_candidates(self):
        """
        Promote candidate tracks that have been matched for 3+ consecutive frames.
        """
        candidates_to_promote = []
        
        # Find candidates ready for promotion
        for candidate_id, track in self.candidate_tracks.items():
            if track['confirmation_count'] >= self.confirmation_frames_required:
                candidates_to_promote.append(candidate_id)
        
        # Promote each confirmed candidate
        for candidate_id in candidates_to_promote:
            # Remove from candidates
            track = self.candidate_tracks.pop(candidate_id)
            print(f"   Confirmed over {track['confirmation_count']} frames")
            print(f"   Created at frame: {track['created_frame']}")
            print(f"   Total detections: {len(track['detections'])}")
            
            # Check if this might be a duplicate
            if len(self.person_profiles) > 0:
                last_det = track['detections'][-1]
                print(f"   Checking for duplicates")
                
                for existing_id, profile in self.person_profiles.items():
                    sim, match_type = self._compute_person_similarity(last_det, profile)
                    #if sim > 0.5:  # Potential duplicate
                    #    print(f"  High similarity ({sim:.2f}, {match_type}) with Track {existing_id}")
            # Assign PERMANENT ID
            permanent_id = self.next_track_id
            self.next_track_id += 1
            track['promoted_to'] = permanent_id
            track['original_candidate_id'] = candidate_id
            # Update track metadata
            track['track_id'] = permanent_id
            track['is_candidate'] = False
            track['match_type'] = "confirmed"
            
            # Move to active tracks
            self.active_tracks[permanent_id] = track
            
            # NOW create person profile (using best detection)
            profile_detection = None
            for det in reversed(track['detections']):
                if det.get('upper_feature') is not None or det.get('lower_feature') is not None:
                    profile_detection = det
                    break
            
            if profile_detection is None:
                profile_detection = track['detections'][-1]
            
            profile = {
                'person_id': permanent_id,
                'creation_frame': track['created_frame'],
                'face_feature': profile_detection.get('face_feature').copy() if profile_detection.get('face_feature') is not None else None,
                'upper_feature': profile_detection.get('upper_feature').copy() if profile_detection.get('upper_feature') is not None else None,
                'lower_feature': profile_detection.get('lower_feature').copy() if profile_detection.get('lower_feature') is not None else None,
            }
            self.person_profiles[permanent_id] = profile
            
            print(f" Promoted candidate {candidate_id} → Track {permanent_id} "
                f"(confirmed over {track['confirmation_count']} frames)")



    def _cleanup_failed_candidates(self):
        """
        Remove candidate tracks that failed to confirm.
        """
        candidates_to_remove = []
        
        for candidate_id, track in self.candidate_tracks.items():
            age = self.frame_count - track['created_frame']
            
            # Remove if lost for too long
            if track['lost_frames'] > 10:
                candidates_to_remove.append(candidate_id)
            
            # Remove if too old but not confirmed
            elif age > 30 and track['confirmation_count'] < self.confirmation_frames_required:
                candidates_to_remove.append(candidate_id)
        
        for candidate_id in candidates_to_remove:
            del self.candidate_tracks[candidate_id]
    
    def _check_temporal_overlap(self, track1: Dict, track2: Dict) -> bool:
        """
        Check if two tracks appear in the same frame at any point.
        
        Returns:
            bool: True if tracks overlap temporally (appear together), False otherwise
        """
        # Get frame numbers for each track
        frames1 = set()
        frames2 = set()
        
        for det in track1['detections']:
            frame_idx = det.get('frame_idx', track1['created_frame'])
            frames1.add(frame_idx)
        
        for det in track2['detections']:
            frame_idx = det.get('frame_idx', track2['created_frame'])
            frames2.add(frame_idx)
        
        # Check for intersection
        return len(frames1 & frames2) > 0
    
    
    def _merge_non_overlapping_tracks(self, all_tracks: Dict[int, Dict], 
                                      appearance_threshold: float = 0.75) -> Dict[int, int]:
        """
        Merge tracks with high appearance similarity that never appear in same frame.
        
        Args:
            all_tracks: Dictionary of all tracks (active + lost + candidates)
            appearance_threshold: Similarity threshold for merging (default: 0.75)
            
        Returns:
            Dict mapping old track_id -> new track_id for merged tracks
        """
        merge_map = {}  # old_id -> new_id
        merged_into = {}  # track_id -> list of track_ids merged into it
        
        # Only consider tracks with profiles (confirmed tracks)
        eligible_tracks = [
            (tid, track) for tid, track in all_tracks.items()
            if tid in self.person_profiles and tid > 0  # Only permanent tracks
        ]
        
        if len(eligible_tracks) < 2:
            return merge_map
        
        print(f"\nChecking {len(eligible_tracks)} tracks for post-hoc merging")
        
        # Compare all pairs
        for i, (tid1, track1) in enumerate(eligible_tracks):
            if tid1 in merge_map:  # Already merged
                continue
                
            profile1 = self.person_profiles[tid1]
            
            for tid2, track2 in eligible_tracks[i+1:]:
                if tid2 in merge_map:  # Already merged
                    continue
                
                # Check if tracks overlap temporally
                if self._check_temporal_overlap(track1, track2):
                    continue  # They coexist, can't be the same person
                
                # Check appearance similarity
                profile2 = self.person_profiles[tid2]
                
                # Create pseudo-detection from profile for similarity check
                pseudo_det = {
                    'face_feature': profile2.get('face_feature'),
                    'upper_feature': profile2.get('upper_feature'),
                    'lower_feature': profile2.get('lower_feature')
                }
                
                similarity, match_type = self._compute_person_similarity(pseudo_det, profile1)
                
                # Use same logic as appearance matching
                effective_threshold = appearance_threshold
                
                det_has_face = profile2.get('face_feature') is not None
                prof_has_face = profile1.get('face_feature') is not None
                
                if not det_has_face or not prof_has_face:
                    effective_threshold = 0.8  # More lenient for body-only
                
                if similarity > effective_threshold:
                    # Merge tid2 into tid1 (keep lower ID)
                    merge_map[tid2] = tid1
                    
                    if tid1 not in merged_into:
                        merged_into[tid1] = []
                    merged_into[tid1].append(tid2)
                    
                    print(f"  Merging Track {tid2} -> Track {tid1}")
                    print(f"    Similarity: {similarity:.3f} ({match_type})")
                    print(f"    Track {tid1}: frames {track1['created_frame']}-{track1['last_seen']}")
                    print(f"    Track {tid2}: frames {track2['created_frame']}-{track2['last_seen']}")
        
        if merge_map:
            print(f"\nMerged {len(merge_map)} tracks into {len(merged_into)} base tracks")
        
        return merge_map

    
    def _apply_track_merging(self, all_tracks: Dict[int, Dict], merge_map: Dict[int, int]):
        """
        Apply track merging by combining detections and updating profiles.
        
        Args:
            all_tracks: Dictionary of all tracks
            merge_map: Mapping of old_id -> new_id for tracks to merge
        """
        if not merge_map:
            return
        
        for old_id, new_id in merge_map.items():
            if old_id not in all_tracks or new_id not in all_tracks:
                continue
            
            old_track = all_tracks[old_id]
            new_track = all_tracks[new_id]
            
            # Merge detections (append old track's detections to new track)
            new_track['detections'].extend(old_track['detections'])
            
            # Update metadata
            new_track['last_seen'] = max(new_track['last_seen'], old_track['last_seen'])
            new_track['created_frame'] = min(new_track['created_frame'], old_track['created_frame'])
            
            # Merge profiles with weighted average based on detection count
            if old_id in self.person_profiles and new_id in self.person_profiles:
                old_profile = self.person_profiles[old_id]
                new_profile = self.person_profiles[new_id]
                
                old_count = len(old_track['detections'])
                new_count = len(new_track['detections']) - old_count  # Subtract just-added detections
                total = old_count + new_count
                
                # Weighted average for each feature
                for feat_name in ['face_feature', 'upper_feature', 'lower_feature']:
                    old_feat = old_profile.get(feat_name)
                    new_feat = new_profile.get(feat_name)
                    
                    if old_feat is not None and new_feat is not None:
                        # Weighted average
                        merged_feat = (new_feat * new_count + old_feat * old_count) / total
                        norm = np.linalg.norm(merged_feat)
                        if norm > 0:
                            new_profile[feat_name] = merged_feat / norm
                    elif old_feat is not None and new_feat is None:
                        # Only old has feature, adopt it
                        new_profile[feat_name] = old_feat.copy()
                
                # Remove old profile
                del self.person_profiles[old_id]
            
            # Remove old track from all dictionaries
            if old_id in self.active_tracks:
                del self.active_tracks[old_id]
            if old_id in self.lost_tracks:
                del self.lost_tracks[old_id]
            if old_id in all_tracks:
                del all_tracks[old_id]

    def _check_appearance_consistency(self, motion_matches: Dict[int, int], detections: List[Dict]) -> List[int]:
        """Check appearance consistency for motion-matched tracks (active + candidates)"""
        suspect_matches = []
        for det_idx, track_id in motion_matches.items():
            # Get track from correct dictionary
            if track_id > 0:
                track = self.active_tracks.get(track_id)
                profile = self.person_profiles.get(track_id)
            else:
                track = self.candidate_tracks.get(track_id)
                # For candidates, skip appearance check (they don't have profiles yet)
                continue
            
            if not track or not profile:
                continue
                
            detection = detections[det_idx]
            similarity, _ = self._compute_person_similarity(detection, profile)
            if similarity < self.config.combined_reid_threshold/2:
                suspect_matches.append(det_idx)
        return suspect_matches

    def _check_geometry_consistency(self, motion_matches: Dict[int, int], detections: List[Dict]) -> List[int]:
        """Check geometry consistency for motion-matched tracks (active + candidates)"""
        suspect_matches = []
        for det_idx, track_id in motion_matches.items():
            # Get track from correct dictionary
            if track_id > 0:
                track = self.active_tracks.get(track_id)
            else:
                track = self.candidate_tracks.get(track_id)
            
            if not track:
                continue
                
            if track['missed_updates'] > 5:
                continue
                
            detection = detections[det_idx]
            
            # Check if geometry exists (might not for new candidates)
            if 'geometry' not in track:
                continue
                
            det_area, det_aspect = self._compute_bbox_geometry(detection['bbox'])
            pred_area, pred_aspect = track['geometry']['area_ema'], track['geometry']['aspect_ema']

            if not self._is_geometry_consistent((det_area, det_aspect), (pred_area, pred_aspect)):
                print(f"  Frame {self.frame_count}: Track {track_id} geometry mismatch:")
                print(f"   Area ratio: {det_area/pred_area:.2f} (expected 0.5-2.0)")
                print(f"   Aspect ratio: {det_aspect/pred_aspect:.2f} (expected 0.7-1.5)")
                suspect_matches.append(det_idx)
        return suspect_matches
    
    
    def _is_geometry_consistent(self, det_geom, pred_geom):   #  Up from 1.5
        """More forgiving geometry check"""
        det_area, det_aspect = det_geom
        pred_area, pred_aspect = pred_geom
        
        area_ratio = det_area / pred_area if pred_area > 0 else 0
        aspect_ratio = det_aspect / pred_aspect if pred_aspect > 0 else 0
        
        area_consistent = 0.5 <= area_ratio <= 2
        aspect_consistent =  0.7 <= aspect_ratio <= 1.5
        
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

    def draw_tracking_results(self, frame: np.ndarray, pose_results: List[PoseResult], 
                        person_assignments: Dict[int, int], active_tracks: Dict) -> np.ndarray:
        """Draw tracking results on frame"""
        if not self.config.enable_visualization:
            return frame
    
        vis_result = frame.copy()
    
        # Draw ALL poses (including unmatched ones) 
        if self.config.enable_pose_drawing:
            for pose in pose_results:  # Use pose_results directly
                keypoints: np.ndarray = pose.keypoints[:17]
                keypoint_scores: np.ndarray = pose.keypoints[:, 2]
    
                for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                    if score > self.kpt_threshold:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(vis_result, (x, y), self.config.radius, (0, 255, 0), -1)
    
                for pt1_idx, pt2_idx in coco_skeleton:
                    if (pt1_idx < len(keypoint_scores) and pt2_idx < len(keypoint_scores) and
                        keypoint_scores[pt1_idx] > self.kpt_threshold and
                        keypoint_scores[pt2_idx] > self.kpt_threshold):
                        pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                        pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                        cv2.line(vis_result, pt1, pt2, (255, 200, 0), self.config.line_width)
    
        # Draw tracking info only for matched detections
        if self.config.enable_bbox_drawing or self.config.enable_id_labels:
            for det_idx, track_id in person_assignments.items():
                # Validate index 
                if det_idx >= len(pose_results):
                    continue
                    
                pose = pose_results[det_idx] 
                if len(pose.bbox) == 0:
                    continue
                    
                bbox = pose.bbox
                x1, y1, x2, y2 = bbox[:4].astype(int)
    
                # Determine color based on track type
                if track_id < 0:  # Candidate track
                    color = (255, 255, 0)  # Yellow
                    text = f"Candidate {track_id}"
                else:  # Active track
                    # Get track from combined sources 
                    track = active_tracks.get(track_id)
                    if not track:
                        # Try candidate_tracks if active lookup fails
                        continue
                        
                    match_color_map = {
                        "new": (0, 255, 255),
                        "motion": (0, 255, 0),
                        "appearance": (255, 0, 0),
                        "appearance_reid": (0, 0, 255),
                        "confirmed": (0, 255, 0),
                        "candidate": (255, 255, 0)
                    }
                    match_type = track.get("match_type", "new")
                    color = match_color_map.get(match_type, (0, 255, 0))
                    
                    app_match_type = track.get('app_match_type', None)
                    text = f"ID {track_id}: {match_type}"
                    if app_match_type:
                        text += f"({app_match_type})"
    
                # Draw bounding box
                if self.config.enable_bbox_drawing:
                    cv2.rectangle(vis_result, (x1, y1), (x2, y2), color, 2)
    
                # Draw ID label
                if self.config.enable_id_labels:
                    cv2.rectangle(vis_result, (x1, y1 - 25), (x1 + len(text) * 12, y1), color, -1)
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
        self.feature_module = FeatureExtractionModule(config.features, config.clip, config.processing.kpt_threshold)
        self.tracking_module = TrackingModule(config.processing)
        self.visualization_module = VisualizationModule(
            config.visualization,
            config.processing.kpt_threshold
        ) if config.visualization.enable_visualization else None

        # Data export
        self.data_collector = (data_collector or TrackingDataCollector(enable_hdf5=config.export.export_hdf5)) if config.export.enable_export else None

        # Performance tracking
        self.timing_stats = defaultdict(list)
        self.global_start_time = None

        # Interruption handling
        self._interrupted = False
        self._proc = None
        self._cap = None

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print("\n\nInterrupted Finalizing video and printing metrics ...")
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

            # Export with person profiles for HDF5 embeddings
            self.data_collector.export_data(
                self.config.export.output_path,
                total_runtime,
                person_profiles=self.tracking_module.person_profiles
            )

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

    def _check_extremely_stationary_track(self, track: Dict, threshold_pixels: float = 2.0,min_keypoint_confidence: float = 0.6) -> bool:
        """
        Check if track represents extremely stationary object with low confidence or few visible keypoints
        
        This only for obvious non-movers.
        Real humans will almost always exceed x pixels due to:
        - Breathing (chest/shoulder movement)
        - Head movements
        - Micro-adjustments
        - Camera noise/jitter
        
        Args:
            track: Track dictionary with detections
            threshold_pixels: Maximum average movement (default: 2.0 pixels) and low confidence or few visible keypoints
            
        Returns:
            bool: True if track is extremely stationary (likely artifact)
        """
        detections = track['detections']
        if len(detections) < 10:  # No enough frames to judge. 
            return True # Need to check this (to give true or false need to watch more outputs)
        
        centers = []
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append((center_x, center_y))
        
        movements = []
        for i in range(1, len(centers)):
            prev_x, prev_y = centers[i-1]
            curr_x, curr_y = centers[i]
            dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(dist)
        
        avg_movement = np.mean(movements)
        max_movement = np.max(movements)
        is_stationary = avg_movement < threshold_pixels
        keypoint_confidences = []
        visible_keypoint_counts = []
        
        for det in detections:
            if 'keypoints' in det:
                kpts = det['keypoints'][:17]  # COCO 17 keypoints
                confidences = kpts[:, 2]
                
                # Count visible keypoints (confidence > 0.3)
                visible_count = np.sum(confidences > 0.3)
                visible_keypoint_counts.append(visible_count)
                
                # Get average confidence of visible keypoints
                visible_confidences = confidences[confidences > 0.3]
                if len(visible_confidences) > 0:
                    keypoint_confidences.append(np.mean(visible_confidences))

        
        avg_kpt_confidence = np.mean(keypoint_confidences)
        avg_visible_kpts = np.mean(visible_keypoint_counts)
        
        # Artifact if: stationary AND (low confidence or few visible keypoints)
        low_confidence = avg_kpt_confidence < min_keypoint_confidence
        few_keypoints = avg_visible_kpts < 8  # Less than half of 17 keypoints
        
        is_artifact = is_stationary and (low_confidence or few_keypoints)
        
        if is_artifact:
            print(f"Stationary artifact: avg_motion={avg_movement:.2f}px, "
                  f"max_motion={max_movement:.2f}px, "
                  f"avg_kpt_conf={avg_kpt_confidence:.3f}, "
                  f"avg_visible_kpts={avg_visible_kpts:.1f}")
        return is_artifact
    
    def _filter_artifacts(self, all_tracks: Dict[int, Dict]) -> Tuple[Dict[int, Dict], List[int]]:
        """
        Filter out photo/screen/stationary from tracks.
        
        Args:
            all_tracks: Combined dictionary of all tracks
            
        Returns:
            Tuple of (filtered_tracks, removed_track_ids)
        """
        removed_ids = []
        
        print("POST-HOC FILTERING")

        # Filter 1: Remove tracks marked as photos/screens by validator
        for track_id in self.tracking_module.invalid_track_ids:
            if track_id in all_tracks:
                removed_ids.append(track_id)
                print(f"  Removing Track {track_id}: Photo/screen detected (texture + motion analysis)")
        
        # Filter 2: Remove extremely stationary tracks
        for track_id, track in all_tracks.items():
            if track_id in removed_ids:
                continue
                
            if self._check_extremely_stationary_track(track, threshold_pixels=2.0):
                removed_ids.append(track_id)
                print(f"  Removing Track {track_id}: Extremely stationary (frozen detection)")
        
        # Create filtered tracks
        filtered_tracks = {tid: t for tid, t in all_tracks.items() 
                        if tid not in removed_ids}
        
        print(f"  Total tracks: {len(all_tracks)}")
        print(f"  Removed: {len(removed_ids)}")
        print(f"  Remaining: {len(filtered_tracks)}")
        return filtered_tracks, removed_ids
    
    
    def process_video(self, input_path: str, output_path: str, segment: Tuple[int, int] = None):
        """Process video with deferred rendering """
        self.global_start_time = time.time()
        
        # PASS 1: Tracking
        print("\nPASS 1: Tracking and collecting data")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cache_manager = CacheManager.from_metadata(
            video_filename=output_path,
            cache_base_path=self.config.cache.cache_base_path,
            metadata_format="yaml"
        )
        pose_cache = cache_manager.load_all_poses()
        
        # Set video metadata for export
        if self.data_collector:
            self.data_collector.set_video_info(input_path, total_frames, fps, width, height)
        
        minimal_frame_data = {}  # Store assignments for visualization ONLY
        frame_ix = -1
        
        with tqdm(total=total_frames, desc="Tracking") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                frame_ix += 1
                
                if not ret:
                    break
                
                if len(pose_cache) <= frame_ix:
                    pbar.update(1)
                    continue
                
                pose_results = create_pose_results_from_cache(pose_cache[frame_ix])
                pose_results = apply_oks_nms(pose_results, 
                                            self.config.processing.oks_nms_threshold, 
                                            self.config.processing.kpt_threshold)
                filter_poses_by_keypoints(pose_results, self.config.processing.kpt_threshold)
                
                detections = self.feature_module.extract_features(
                    frame, pose_results, self.tracking_module.frame_count + 1
                )
                
                person_assignments = self.tracking_module.update(detections, frame)
                
                # Collect data once during tracking (with original IDs)
                if self.data_collector:
                    self.data_collector.collect_frame_data(
                        frame_ix, detections, person_assignments
                    )
                
                # Store only assignments for visualization (not detections)
                minimal_frame_data[frame_ix] = {
                    'det_to_track': person_assignments.copy(),
                    'num_detections': len(detections)
                }
                
                pbar.update(1)
        
        cap.release()
        
        # POST-PROCESSING: FILTERING, MERGING, AND ID REMAPPING
        print("\nPOST-HOC PROCESSING")
        
        # Step 1: Collect all tracks
        all_tracks = {
            **self.tracking_module.active_tracks,
            **self.tracking_module.lost_tracks,
            **self.tracking_module.candidate_tracks
        }
        print(f"Total tracks before processing: {len(all_tracks)}")
        
        # Step 2: Filter artifacts
        print("\n[1/4] Filtering artifacts")
        filtered_tracks, removed_ids = self._filter_artifacts(all_tracks)
        
        # Update tracking dictionaries
        self.tracking_module.active_tracks = {
            tid: t for tid, t in filtered_tracks.items() 
            if tid in self.tracking_module.active_tracks
        }
        self.tracking_module.lost_tracks = {
            tid: t for tid, t in filtered_tracks.items() 
            if tid in self.tracking_module.lost_tracks
        }
        self.tracking_module.candidate_tracks = {
            tid: t for tid, t in filtered_tracks.items() 
            if tid in self.tracking_module.candidate_tracks
        }
        
        # Remove filtered tracks from data collector
        if self.data_collector:
            for track_id in removed_ids:
                self.data_collector.remove_track(track_id)
        
        # Remove profiles of filtered tracks
        for track_id in removed_ids:
            if track_id in self.tracking_module.person_profiles:
                del self.tracking_module.person_profiles[track_id]
        
        print(f"  Removed {len(removed_ids)} artifact tracks")
        print(f"  Remaining tracks: {len(filtered_tracks)}")
        
        # Step 3: Merge non-overlapping tracks
        print("\n[2/4] Merging similar non-overlapping tracks")
        merge_map = self.tracking_module._merge_non_overlapping_tracks(
            filtered_tracks, appearance_threshold=0.9
        )
        
        # Apply merging
        self.tracking_module._apply_track_merging(filtered_tracks, merge_map)
        print(f"  Merged {len(merge_map)} tracks")
        print(f"  Final unique persons: {len(self.tracking_module.person_profiles)}")
        
        # Step 4: Build ID remapping
        print("\n[3/4] Building ID remapping")
        
        # Candidate promotions
        candidate_to_permanent = {}
        for track_id, track in {**self.tracking_module.active_tracks, 
                               **self.tracking_module.lost_tracks}.items():
            if 'original_candidate_id' in track:
                candidate_to_permanent[track['original_candidate_id']] = track_id
        
        # Combine all remapping
        full_remap = {}
        
        # Apply candidate promotions
        for cand_id, perm_id in candidate_to_permanent.items():
            final_id = merge_map.get(perm_id, perm_id)
            full_remap[cand_id] = final_id
        
        # Apply merge mapping
        for old_id, new_id in merge_map.items():
            full_remap[old_id] = new_id
        
        # Ensure all remaining valid IDs map to themselves
        valid_track_ids = (set(self.tracking_module.active_tracks.keys()) | 
                          set(self.tracking_module.lost_tracks.keys()) | 
                          set(self.tracking_module.candidate_tracks.keys()))
        
        for tid in valid_track_ids:
            if tid not in full_remap:
                full_remap[tid] = tid
        
        print(f"  Candidate promotions: {len(candidate_to_permanent)}")
        print(f"  Track merges: {len(merge_map)}")
        print(f"  Total ID mappings: {len(full_remap)}")
        
        # Apply ID remapping to data collector
        if self.data_collector:
            print("\n[3.5/4] Remapping IDs in data collector")
            self._remap_collector_ids(full_remap, removed_ids)
        
        # Step 5: Apply ID remapping to minimal_frame_data (for visualization)
        print("\n[4/4] Remapping frame assignments (for visualization)")
        remapping_stats = {
            'removed_artifact': 0,
            'remapped_candidate': 0,
            'remapped_merged': 0,
            'invalid': 0
        }
        
        for frame_idx in minimal_frame_data:
            det_to_track = minimal_frame_data[frame_idx]['det_to_track']
            remapped = {}
            
            for det_idx, track_id in det_to_track.items():
                # Check if removed as artifact
                if track_id in removed_ids:
                    remapping_stats['removed_artifact'] += 1
                    continue
                
                # Apply full remapping
                if track_id in full_remap:
                    final_id = full_remap[track_id]
                    
                    # Verify final ID is valid
                    if final_id in valid_track_ids:
                        remapped[det_idx] = final_id
                        
                        # Track what happened
                        if track_id < 0:
                            remapping_stats['remapped_candidate'] += 1
                        elif track_id != final_id:
                            remapping_stats['remapped_merged'] += 1
                    else:
                        remapping_stats['invalid'] += 1
                        print(f"  WARNING: Frame {frame_idx}, track {track_id} -> {final_id} (INVALID)")
                else:
                    # Track ID not in mapping
                    if track_id in valid_track_ids:
                        remapped[det_idx] = track_id
                    else:
                        remapping_stats['invalid'] += 1
            
            minimal_frame_data[frame_idx]['det_to_track'] = remapped
        
        print(f"\nRemapping statistics:")
        print(f"  Remapped (merged): {remapping_stats['remapped_merged']}")
        print(f"  Removed (artifacts): {remapping_stats['removed_artifact']}")
        print(f"  Invalid: {remapping_stats['invalid']}")
        
        # Validation
        print("\n[Validation] Checking ID consistency")
        all_final_ids = set()
        for frame_data in minimal_frame_data.values():
            all_final_ids.update(frame_data['det_to_track'].values())
        
        invalid_ids = all_final_ids - valid_track_ids
        if invalid_ids:
            print(f"  ERROR: Found {len(invalid_ids)} invalid track IDs: {invalid_ids}")
        else:
            print(f"  All {len(all_final_ids)} track IDs are valid")
        
        # PASS 2: Visualization only (no data collection)
        print("\nPASS 2: Rendering video")
        cap = cv2.VideoCapture(input_path)
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18", output_path
        ]
        
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                               stderr=subprocess.DEVNULL)
        
        # Combine active + lost for visualization
        all_final_tracks = {
            **self.tracking_module.active_tracks,
            **self.tracking_module.lost_tracks,
            **self.tracking_module.candidate_tracks
        }
        
        frame_ix = -1
        with tqdm(total=total_frames, desc="Rendering") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                frame_ix += 1
                
                if not ret:
                    break
                
                #  Only visualize, don't collect data again
                if (frame_ix in minimal_frame_data and 
                    len(pose_cache) > frame_ix and 
                    self.visualization_module):
                    
                    pose_results = create_pose_results_from_cache(pose_cache[frame_ix])
                    pose_results = apply_oks_nms(pose_results,
                                                self.config.processing.oks_nms_threshold,
                                                self.config.processing.kpt_threshold)
                    
                    assignments = minimal_frame_data[frame_ix]['det_to_track']
                    
                    vis_frame = self.visualization_module.draw_tracking_results(
                        frame, pose_results, assignments, all_final_tracks
                    )
                else:
                    vis_frame = frame
                
                if vis_frame.shape[:2] != (height, width):
                    vis_frame = cv2.resize(vis_frame, (width, height))
                
                proc.stdin.write(vis_frame.tobytes())
                pbar.update(1)
        
        cap.release()
        proc.stdin.close()
        proc.wait()
        
        # EXPORT
        if self.data_collector:
            total_runtime = time.time() - self.global_start_time
            os.makedirs(os.path.dirname(self.config.export.output_path), exist_ok=True)
            
            self.data_collector.export_data(
                self.config.export.output_path,
                total_runtime,
                person_profiles=self.tracking_module.person_profiles
            )
        
        print(f"\n Output: {output_path}")
        print(f"  Final tracked persons: {len(self.tracking_module.person_profiles)}")
        print(f"  Active tracks: {len(self.tracking_module.active_tracks)}")
        print(f"  Lost tracks: {len(self.tracking_module.lost_tracks)}")
    
    def _remap_collector_ids(self, full_remap: Dict[int, int], removed_ids: set):
        """
        Apply ID remapping to the data collector's stored tracking data.
        This ensures exported JSON/HDF5 files use final merged IDs.
        """
        if not self.data_collector:
            return
        
        exporter = self.data_collector.exporter
        old_tracking_data = dict(exporter.tracking_data)
        
        # Clear existing data
        exporter.tracking_data.clear()
        
        remapping_stats = {'remapped': 0, 'removed': 0, 'kept': 0}
        
        for old_id, track_data in old_tracking_data.items():
            # Skip removed tracks
            if old_id in removed_ids:
                remapping_stats['removed'] += 1
                continue
            
            # Get final ID
            final_id = full_remap.get(old_id, old_id)
            
            # Check if this ID was already remapped (merge case)
            if final_id in exporter.tracking_data:
                # Merge frame data into existing track
                existing_frames = exporter.tracking_data[final_id]['frames']
                for frame_num, frame_data in track_data['frames'].items():
                    if frame_num not in existing_frames:
                        existing_frames[frame_num] = frame_data
                
                # Update start/end frames
                exporter.tracking_data[final_id]['start_frame'] = min(
                    exporter.tracking_data[final_id]['start_frame'],
                    track_data['start_frame']
                )
                exporter.tracking_data[final_id]['end_frame'] = max(
                    exporter.tracking_data[final_id]['end_frame'],
                    track_data['end_frame']
                )
                
                remapping_stats['remapped'] += 1
            else:
                # New ID, copy track data
                exporter.tracking_data[final_id] = track_data
                
                if old_id != final_id:
                    remapping_stats['remapped'] += 1
                else:
                    remapping_stats['kept'] += 1
        
        print(f"  Data collector remapping:")
        print(f"    Remapped: {remapping_stats['remapped']}")
        print(f"    Removed: {remapping_stats['removed']}")
        print(f"    Kept unchanged: {remapping_stats['kept']}")
        print(f"    Final tracks in collector: {len(exporter.tracking_data)}")

    def reset_for_next_video(self):
        """Reset state between videos while keeping models loaded"""
        # Reset tracking state
        if self.tracking_module:
            self.tracking_module.frame_count = 0
            self.tracking_module.next_track_id = 1
            self.tracking_module.active_tracks.clear()
            self.tracking_module.lost_tracks.clear()
            self.tracking_module.person_profiles.clear()
            self.tracking_module.candidate_tracks.clear()
            self.tracking_module.next_candidate_id = -1
            # Reset invalid track history
            self.tracking_module.invalid_track_ids.clear()
            self.tracking_module.validator.motion_history.clear()
            self.tracking_module.validator.texture_history.clear()
            self.tracking_module.validator.size_history.clear()

            # Reset camera motion compensator
            if self.tracking_module.camera_compensator:
                self.tracking_module.camera_compensator.prev_frame_gray = None
                self.tracking_module.camera_compensator.prev_points = None
                self.tracking_module.camera_compensator.motion_history.clear()
                self.tracking_module.camera_compensator.orb = None

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
        print("PERFORMANCE METRICS")
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



def create_batch_config(json_output_path: str = " ") -> 'PipelineConfig':
    """Create configuration for batch processing"""
    config = PipelineConfig()
    # feature extraction
    config.features.enable_face_features = True
    config.features.no_deepface = True
    config.features.enable_upper_body_features = True
    config.features.enable_lower_body_features = True
    config.features.feature_update_interval = 1
    config.features.face_confidence_threshold = 0.7
    config.features.face_min_size = 40
    config.features.body_min_height = 80
    config.features.body_min_width = 40
    # detection
    config.processing.oks_nms_threshold = 0.6   # Keypoint OKS NMS threshold (0.5-0.95) - higher= more overlapping poses removal

    # motion tracking
    tracker_cfg = TrackerConfig(
        base_iou_threshold=0.50,           # IoU threshold for matching (0.1-0.5) - higher=stricter matching
        base_motion_confidence=0.4,       # Motion prediction confidence (0.1-0.5) - higher=trust motion more
        base_center_weight=0.5,           # Weight for center distance vs IoU (0.0-1.0) - higher=more weight on center
        max_lost_frames=200,               # Frames before track is lost (50-600) - higher=keep tracks alive longer
        confidence_decay_rate=0.06,        # How fast confidence decays (0.01-0.1) - higher=faster decay
        max_jump_factor=1.2               # Max allowed position jump (1.5-3.5) - higher=allow bigger jumps
    )
    config.processing.tracker_config = tracker_cfg
    # reid
    config.processing.face_reid_threshold = 0.75             # Face matching threshold (0.5-0.9) - higher=stricter
    config.processing.upper_reid_threshold = 0.75          # Upper body matching (0.5-0.9) - higher=stricter
    config.processing.lower_reid_threshold = 0.65         # Lower body matching (0.5-0.9) - higher=stricter
    config.processing.combined_reid_threshold = 0.8         # Combined ReID threshold (0.3-0.9) - lower=more matches
    # save
    config.visualization.enable_visualization = True
    config.visualization.enable_pose_drawing = True
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True
    config.visualization.radius = 3
    config.visualization.line_width = 1
    config.export.enable_export = True
    config.export.output_path = json_output_path
    config.export.export_hdf5 = True
    config.cache.enable_cache = True

    return config

