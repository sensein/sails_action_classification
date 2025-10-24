
#use no face (deepface is not good most of the time)

import sys
import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import time
import subprocess
import signal
import gc
import argparse
import logging
import tempfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tracking_exporter import TrackingDataCollector
from person_tracker import (
    CameraMotionCompensator,
    TrackerConfig as PersonTrackerConfig,
    calculate_iou,
    calculate_scene_crowding,
    calculate_combined_similarity,
    get_adaptive_thresholds,
    is_spatially_plausible,
    create_kalman_filter,
    predict_motion_with_camera_compensation,
    update_kalman_filter,
)

# ============================================================================
# ENHANCED TRACKING MODULE WITH 3-FRAME CONFIRMATION
# ============================================================================

@dataclass
class CLIPReIDConfig:
    """CLIP-ReID model configuration."""
    config_path: str = 'configs/person/vit_clipreid.yml'
    checkpoint_path: str = '/home/aparnabg/reid/models_pth/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'
    num_classes: int = 1041
    camera_num: int = 15
    view_num: int = 1
    device: str = "cuda"


@dataclass
class FaceDetectionConfig:
    """DeepFace verification configuration."""
    enable_face_features: bool = True
    deepface_backend: str = "retinaface"
    deepface_model: str = "Facenet512"
    face_confidence_threshold: float = 0.7
    face_keypoint_threshold: float = 0.6
    min_face_keypoints: int = 50
    appearance_similarity_low: float = 0.45
    appearance_similarity_high: float = 0.75


@dataclass
class TrackerConfig:
    """Core tracking parameters."""
    max_lost_frames: int = 1500
    iou_threshold_low_crowd: float = 0.35
    iou_threshold_high_crowd: float = 0.20
    center_weight_low_crowd: float = 0.3
    center_weight_high_crowd: float = 0.5
    motion_confidence_threshold: float = 0.5
    max_jump_factor: float = 1.5
    new_track_confirmation_frames: int = 3  # NEW: Require 3 frames before confirming


@dataclass
class FeatureConfig:
    """Feature extraction settings."""
    enable_upper_body: bool = True
    enable_lower_body: bool = True
    feature_update_interval: int = 1


@dataclass
class DetectionQualityConfig:
    """Initial detection quality thresholds."""
    min_keypoint_confidence: float = 0.5
    min_keypoints_required: int = 15
    enable_quality_check: bool = True


@dataclass
class TrackingConfig:
    """Re-identification thresholds."""
    upper_reid_threshold: float = 0.7
    lower_reid_threshold: float = 0.7
    face_reid_threshold: float = 0.7
    combined_reid_threshold: float = 0.65
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    new_track_threshold_increase: float = 0.15
    max_persons = None
    person_tracker_config: PersonTrackerConfig = field(default_factory=lambda: PersonTrackerConfig(
        base_iou_threshold=0.3,
        base_motion_confidence=0.3,
        base_center_weight=0.4,
        max_lost_frames=1500,
        confidence_decay_rate=0.06,
        max_jump_factor=1.5
    ))
    enable_detection_validation: bool = True  # NEW: Enable photo/screen filtering

@dataclass
class FilteringConfig:
    """Track filtering parameters with granular control."""
    # Master switches
    enable_posthoc_filtering: bool = True
    enable_id_reassignment: bool = True  # Control ID remapping
    
    # Individual filter toggles
    enable_validation_filter: bool = True      # Remove invalid tracks (photos/screens)
    enable_size_filter: bool = True            # Remove tracks with wrong size/aspect ratio
    enable_quality_filter: bool = True         # Remove low keypoint confidence tracks
    enable_person_classification: bool = True  # Classify main vs background persons
    enable_duration_filter: bool = True        # Remove short tracks
    
    # Filter parameters
    min_track_length: int = 5
    stationary_threshold_pixels: float = 20
    min_duration_seconds: float = 1.5 
    
    # Size constraints (NEW)
    min_avg_height: int = 100
    min_avg_width: int = 60
    min_aspect_ratio: float = 1.2
    max_aspect_ratio: float = 4.0
    
    # Quality thresholds (NEW)
    min_avg_keypoint_confidence: float = 0.5
    
    # Old parameters (keep for backwards compatibility)
    filter_short_tracks: bool = True


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    enable_visualization: bool = True
    show_keypoints: bool = False


@dataclass
class ExportConfig:
    """Export settings."""
    enable_export: bool = True


@dataclass
class Stage2Config:
    """Main configuration container."""
    clip_reid: CLIPReIDConfig = field(default_factory=CLIPReIDConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    track_filtering: FilteringConfig = field(default_factory=FilteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    detection_quality: DetectionQualityConfig = field(default_factory=DetectionQualityConfig)


# ============================================================================
# DETECTION VALIDATOR - Filters Photos/TV/Screens
# ============================================================================

class DetectionValidator:
    """
    Validates if detection is a real person, not photo/screen/TV.
    
    Uses motion naturalness and texture analysis to distinguish:
    - Real people: Natural varying motion, rich texture details
    - Photos/Screens: Uniform motion (moves with camera), low texture variance
    """
    
    def __init__(self):
        """Initialize validator with history tracking."""
        self.motion_history = {}  # track_id -> deque of motion scores
        self.texture_history = {}  # track_id -> deque of texture variance
        self.size_history = {}  # track_id -> deque of bbox sizes
    
    def is_real_person(self, track_id: int, bbox: np.ndarray, frame: np.ndarray,
                       min_frames: int = 10) -> bool:
        """
        Check if track represents real person (not photo/TV/screen).
        
        Strategy:
        1. Motion analysis - photos don't move naturally, move uniformly with camera
        2. Texture variance - screens/photos have different texture patterns
        3. Size consistency - photos often have unnatural scaling
        
        Args:
            track_id: Track ID to validate
            bbox: Bounding box [x1, y1, x2, y2]
            frame: Current frame image
            min_frames: Minimum frames needed for validation
            
        Returns:
            bool: True if real person, False if likely photo/screen
        """
        # Extract ROI
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return True  # Too small to analyze
        
        # ====================================================================
        # 1. TEXTURE VARIANCE ANALYSIS
        # Photos/screens often have less texture variance and edge detail
        # ====================================================================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (measure of texture detail)
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
        
        # ====================================================================
        # ANALYSIS: Check texture consistency
        # ====================================================================
        texture_data = list(self.texture_history[track_id])
        avg_texture = np.mean([t['variance'] for t in texture_data])
        texture_std = np.std([t['variance'] for t in texture_data])
        avg_edge_density = np.mean([t['edge_density'] for t in texture_data])
        
        # Photos/screens: Very consistent, low texture variance
        if avg_texture < 50 and texture_std < 10:
            return False
        
        # Photos/screens: Often have very uniform edge patterns
        if avg_edge_density < 0.02 or avg_edge_density > 0.5:
            # Too few edges (blurry photo) or too many edges (digital artifact)
            return False
        
        # ====================================================================
        # 2. MOTION NATURALNESS ANALYSIS
        # Real people: Varying motion with articulated body parts
        # Photos: Move uniformly with camera motion, no internal variation
        # ====================================================================
        if track_id in self.motion_history and len(self.motion_history[track_id]) >= min_frames:
            motions = list(self.motion_history[track_id])
            
            # Real people have varying motion patterns
            motion_variance = np.var(motions)
            motion_range = max(motions) - min(motions)
            
            # Photos: Near-zero motion variance (moves uniformly with camera)
            if motion_variance < 2.0 and motion_range < 5.0:
                return False
            
            # Check for unnatural motion patterns (sudden jerks in photos)
            if len(motions) >= 5:
                motion_changes = np.diff(motions)
                if np.max(np.abs(motion_changes)) > 50:  # Sudden jump
                    return False
        
        # ====================================================================
        # 3. SIZE CONSISTENCY ANALYSIS
        # Photos often have unnatural size changes (perspective issues)
        # ====================================================================
        bbox_size = (x2 - x1) * (y2 - y1)
        
        if track_id not in self.size_history:
            self.size_history[track_id] = deque(maxlen=30)
        self.size_history[track_id].append(bbox_size)
        
        if len(self.size_history[track_id]) >= min_frames:
            sizes = list(self.size_history[track_id])
            size_std = np.std(sizes)
            avg_size = np.mean(sizes)
            
            # Photos: Unnatural size variation (coefficient of variation)
            if avg_size > 0:
                cv = size_std / avg_size
                if cv > 0.4:  # More than 40% variation
                    return False
        
        return True
    
    def update_motion(self, track_id: int, motion_score: float):
        """
        Update motion history for track.
        
        Args:
            track_id: Track ID
            motion_score: Motion magnitude/score for current frame
        """
        if track_id not in self.motion_history:
            self.motion_history[track_id] = deque(maxlen=30)
        self.motion_history[track_id].append(motion_score)
    
    def reset_track(self, track_id: int):
        """Reset validation history for track (when reactivated)."""
        if track_id in self.motion_history:
            del self.motion_history[track_id]
        if track_id in self.texture_history:
            del self.texture_history[track_id]
        if track_id in self.size_history:
            del self.size_history[track_id]


# ============================================================================
# POST-HOC FILTERING MODULE
# ============================================================================
class PostHocTrackFilter:
    """Post-processing filter with GRANULAR control."""
    
    def __init__(self, fps: float, config: FilteringConfig):
        self.fps = fps
        self.config = config
        self.min_frames = int(fps * config.min_duration_seconds)
        
        print(f"Post-hoc filter initialized:")
        print(f"  FPS: {fps}")
        print(f"  Validation filter: {config.enable_validation_filter}")
        print(f"  Size filter: {config.enable_size_filter}")
        print(f"  Quality filter: {config.enable_quality_filter}")
        print(f"  Person classification: {config.enable_person_classification}")
        print(f"  Duration filter: {config.enable_duration_filter}")
        print(f"  ID reassignment: {config.enable_id_reassignment}")
    
    def filter_tracks(self, tracks: Dict[int, Dict], person_profiles: Dict[int, Dict],
                     max_main_persons: Optional[int] = None,
                     invalid_track_ids: List[int] = None) -> Tuple[Dict[int, Dict], List[int]]:
        """Filter tracks with configurable filtering steps."""
        removed_ids = []
        removal_reasons = {}
        
        # ====================================================================
        # STEP 1: Remove invalid tracks (photos/screens/TVs)
        # ====================================================================
        if self.config.enable_validation_filter and invalid_track_ids:
            for track_id in invalid_track_ids:
                if track_id in tracks:
                    removed_ids.append(track_id)
                    removal_reasons[track_id] = "Failed validation (photo/screen/TV)"
            
            if removed_ids:
                print(f"\n  Validation filter: Removed {len(removed_ids)} invalid tracks")
        
        # ====================================================================
        # STEP 2: Size filtering (height, width, aspect ratio)
        # ====================================================================
        if self.config.enable_size_filter:
            size_removed = []
            
            for track_id in list(tracks.keys()):
                if track_id in removed_ids:
                    continue
                
                track = tracks[track_id]
                detections = track['detections']
                
                if len(detections) == 0:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = "No detections"
                    continue
                
                heights = []
                widths = []
                aspect_ratios = []
                
                for det in detections:
                    bbox = det['bbox']
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    
                    if w > 0 and h > 0:
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(h / w)
                
                if not heights or not widths:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = "Invalid bbox dimensions"
                    continue
                
                avg_height = np.mean(heights)
                avg_width = np.mean(widths)
                avg_aspect = np.mean(aspect_ratios)
                
                if avg_height < self.config.min_avg_height:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = f"Too short (avg {avg_height:.0f}px < {self.config.min_avg_height}px)"
                    continue
                
                if avg_width < self.config.min_avg_width:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = f"Too narrow (avg {avg_width:.0f}px < {self.config.min_avg_width}px)"
                    continue
                
                if avg_aspect < self.config.min_aspect_ratio:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = f"Wrong aspect ratio (too wide: {avg_aspect:.2f})"
                    continue
                
                if avg_aspect > self.config.max_aspect_ratio:
                    size_removed.append(track_id)
                    removal_reasons[track_id] = f"Wrong aspect ratio (too tall: {avg_aspect:.2f})"
                    continue
            
            if size_removed:
                removed_ids.extend(size_removed)
                print(f"\n  Size filter: Removed {len(size_removed)} tracks")
        
        # ====================================================================
        # STEP 3: Quality filtering (keypoint confidence)
        # ====================================================================
        if self.config.enable_quality_filter:
            quality_removed = []
            
            for track_id in list(tracks.keys()):
                if track_id in removed_ids:
                    continue
                
                track = tracks[track_id]
                detections = track['detections']
                
                avg_keypoint_conf = []
                for det in detections:
                    kpts = det['keypoints']
                    if len(kpts) > 0:
                        confs = [kp[2] for kp in kpts if len(kp) >= 3]
                        if confs:
                            avg_keypoint_conf.append(np.mean(confs))
                
                if avg_keypoint_conf:
                    overall_avg_conf = np.mean(avg_keypoint_conf)
                    if overall_avg_conf < self.config.min_avg_keypoint_confidence:
                        quality_removed.append(track_id)
                        removal_reasons[track_id] = f"Low keypoint confidence ({overall_avg_conf:.2f})"
            
            if quality_removed:
                removed_ids.extend(quality_removed)
                print(f"\n  Quality filter: Removed {len(quality_removed)} tracks")
        
        # ====================================================================
        # STEP 4: Person classification (main vs background)
        # ====================================================================
        if self.config.enable_person_classification and max_main_persons is not None and max_main_persons > 0:
            remaining_tracks = {tid: t for tid, t in tracks.items() 
                              if tid not in removed_ids}
            
            print(f"\n  Person classification: {len(remaining_tracks)} tracks, max allowed: {max_main_persons}")
            
            if len(remaining_tracks) > max_main_persons:
                track_scores = {}
                
                for track_id, track in remaining_tracks.items():
                    duration_frames = track['last_seen'] - track['created_frame']
                    duration_seconds = duration_frames / self.fps
                    num_detections = len(track['detections'])
                    avg_confidence = np.mean([d['confidence'] for d in track['detections']])
                    
                    if len(track['detections']) > 1:
                        centers_x = [(d['bbox'][0] + d['bbox'][2])/2 for d in track['detections']]
                        centers_y = [(d['bbox'][1] + d['bbox'][3])/2 for d in track['detections']]
                        
                        motion_variance_x = np.var(centers_x)
                        motion_variance_y = np.var(centers_y)
                        motion_variance = np.sqrt(motion_variance_x**2 + motion_variance_y**2)
                        
                        total_distance = 0
                        for i in range(1, len(centers_x)):
                            dx = centers_x[i] - centers_x[i-1]
                            dy = centers_y[i] - centers_y[i-1]
                            total_distance += np.sqrt(dx**2 + dy**2)
                        
                        avg_movement = total_distance / len(centers_x)
                    else:
                        motion_variance = 0
                        avg_movement = 0
                    
                    score = (
                        duration_seconds * 2.0 +
                        avg_movement * 0.1 +
                        motion_variance * 0.05 +
                        num_detections * 0.5 +
                        avg_confidence * 50 * 0.3
                    )
                    
                    track_scores[track_id] = {
                        'score': score,
                        'duration': duration_seconds,
                        'movement': avg_movement,
                        'variance': motion_variance,
                        'detections': num_detections,
                        'confidence': avg_confidence
                    }
                
                # Sort tracks by score
                sorted_tracks = sorted(track_scores.items(), 
                                     key=lambda x: x[1]['score'], 
                                     reverse=True)
                
                # Identify main persons (top N) vs background persons
                main_person_ids = [tid for tid, info in sorted_tracks[:max_main_persons]]
                background_person_ids = [tid for tid, info in sorted_tracks[max_main_persons:]]
                
                # Mark tracks as main/background but KEEP ALL OF THEM
                for track_id in remaining_tracks.keys():
                    if track_id in main_person_ids:
                        tracks[track_id]['is_main_person'] = True
                        tracks[track_id]['is_background_person'] = False
                    else:
                        tracks[track_id]['is_main_person'] = False
                        tracks[track_id]['is_background_person'] = True
                
                # Print classification
                print(f"    Main persons ({len(main_person_ids)}):")
                for rank, (track_id, info) in enumerate(sorted_tracks[:max_main_persons], 1):
                    print(f"      #{rank:2d} Track {track_id:3d} [MAIN] - "
                          f"Score={info['score']:7.1f}, Duration={info['duration']:5.1f}s")
                
                if background_person_ids:
                    print(f"    Background persons ({len(background_person_ids)}):")
                    for rank, (track_id, info) in enumerate(sorted_tracks[max_main_persons:], 1):
                        print(f"      #{rank:2d} Track {track_id:3d} [BACKGROUND] - "
                              f"Score={info['score']:7.1f}, Duration={info['duration']:5.1f}s")
        
        # ====================================================================
        # STEP 5: Duration filtering (remove short tracks from BACKGROUND only)
        # ====================================================================
        if self.config.enable_duration_filter:
            duration_removed = []
            
            print(f"\n  Duration filter (min {self.config.min_duration_seconds}s) - BACKGROUND PERSONS ONLY:")
            
            for track_id in list(tracks.keys()):
                if track_id in removed_ids:
                    continue
                    
                track = tracks[track_id]
                duration_frames = track['last_seen'] - track['created_frame']
                duration_seconds = duration_frames / self.fps
                
                # Only apply duration filter to BACKGROUND persons
                if track.get('is_background_person', False):
                    if duration_seconds < self.config.min_duration_seconds:
                        duration_removed.append(track_id)
                        removal_reasons[track_id] = f"Too short ({duration_seconds:.2f}s) [BACKGROUND]"
            
            if duration_removed:
                removed_ids.extend(duration_removed)
                print(f"    Removed {len(duration_removed)} short BACKGROUND tracks")
            
            print(f"    Main persons: kept all durations")
        
        # ====================================================================
        # FINAL: Print summary
        # ====================================================================
        print(f"\n  Post-hoc filtering summary:")
        print(f"    Total tracks removed: {len(set(removed_ids))}")
        
        if removal_reasons:
            reason_counts = {}
            for reason in removal_reasons.values():
                reason_type = reason.split('(')[0].strip()
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason_type, count in sorted(reason_counts.items()):
                print(f"      {reason_type}: {count}")
        
        filtered_tracks = {tid: t for tid, t in tracks.items() if tid not in removed_ids}
        
        # Count remaining main vs background
        remaining_main = sum(1 for t in filtered_tracks.values() if t.get('is_main_person', False))
        remaining_background = sum(1 for t in filtered_tracks.values() if t.get('is_background_person', False))
        
        print(f"\n  Final track counts:")
        print(f"    Main persons: {remaining_main}")
        print(f"    Background persons: {remaining_background}")
        print(f"    Total: {len(filtered_tracks)}")
        
        return filtered_tracks, list(set(removed_ids))
    
    def reassign_track_ids(self, tracks: Dict[int, Dict], 
                          person_profiles: Dict[int, Dict]) -> Tuple[Dict[int, int], Dict[int, Dict], Dict[int, Dict]]:
        """Reassign track IDs to be consecutive (1, 2, 3, ...) - OPTIONAL."""
        
        if not self.config.enable_id_reassignment:
            print(f"\n  ID reassignment: DISABLED - keeping original IDs")
            # Return identity mapping
            old_to_new = {tid: tid for tid in tracks.keys()}
            return old_to_new, tracks, person_profiles
        
        print(f"\n  ID reassignment: ENABLED - remapping to consecutive IDs")
        
        sorted_track_ids = sorted(tracks.keys(), 
                                 key=lambda tid: tracks[tid]['created_frame'])
        
        old_to_new = {}
        new_tracks = {}
        new_profiles = {}
        
        for new_id, old_id in enumerate(sorted_track_ids, start=1):
            old_to_new[old_id] = new_id
            
            track = tracks[old_id].copy()
            track['track_id'] = new_id
            new_tracks[new_id] = track
            
            if old_id in person_profiles:
                profile = person_profiles[old_id].copy()
                profile['person_id'] = new_id
                new_profiles[new_id] = profile
        
        print(f"    Remapped: {len(old_to_new)} IDs (1 to {len(new_tracks)})")
        
        return old_to_new, new_tracks, new_profiles



class DeepFaceVerifier:
    """
    Face detection and verification using DeepFace library.
    
    Handles face detection and feature extraction for re-identification.
    """
    
    def __init__(self, config: FaceDetectionConfig):
        """
        Initialize DeepFace verifier.
        
        Args:
            config: FaceDetectionConfig containing backend, model, and threshold settings
        """
        self.config = config
        self.backend = config.deepface_backend
        self.model_name = config.deepface_model
        self.confidence_threshold = config.face_confidence_threshold
        
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            print(f"DeepFace initialized: backend={self.backend}, model={self.model_name}")
        except ImportError:
            print("WARNING: DeepFace not installed. Face verification disabled.")
            self.DeepFace = None
    
    def extract_face_feature(self, bbox_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face feature vector from bounding box crop.
        
        Args:
            bbox_crop: Cropped image region as numpy array (H, W, 3)
            
        Returns:
            Normalized face embedding vector if face found with sufficient confidence, else None
        """
        if self.DeepFace is None:
            return None
        
        if bbox_crop.shape[0] < 50 or bbox_crop.shape[1] < 30:
            return None
        
        try:
            face_objs = self.DeepFace.extract_faces(
                img_path=bbox_crop,
                detector_backend=self.backend,
                enforce_detection=False,
                align=True
            )
            
            if not face_objs:
                return None
            
            face_obj = face_objs[0]
            confidence = face_obj.get('confidence', 0.0)
            
            if confidence < self.confidence_threshold:
                return None
            
            face_region = face_obj['face']
            if face_region.max() <= 1.0:
                face_region = (face_region * 255).astype(np.uint8)
            
            embedding = self.DeepFace.represent(
                img_path=face_region,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding:
                face_feature = np.array(embedding[0]['embedding'], dtype=np.float32)
                norm = np.linalg.norm(face_feature)
                if norm > 0:
                    face_feature = face_feature / norm
                return face_feature
            
            return None
            
        except Exception:
            return None
    
    def check_face_keypoints_condition(self, keypoints: np.ndarray) -> bool:
        """
        Check if sufficient high-confidence face keypoints are detected.
        
        Args:
            keypoints: Pose keypoints array, shape (N, 3) where N >= 91
                      Format: [x, y, confidence] for each keypoint
                      Indices 23-90 correspond to face keypoints
            
        Returns:
            bool: True if sufficient face keypoints meet confidence threshold
        """
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            if len(kpts) < 91:
                return False
            
            face_keypoints = kpts[23:91]
            
            high_conf_count = sum(1 for kp in face_keypoints 
                                 if len(kp) >= 3 and kp[2] > self.config.face_keypoint_threshold)
            
            return high_conf_count > self.config.min_face_keypoints
            
        except Exception:
            return False



class PoseDataLoader:
    """
    Loads and manages pose detection data from JSON files.
    
    Provides frame-by-frame access to pose detections with metadata.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize pose data loader.
        
        Args:
            json_path: Path to JSON file containing pose detections
        """
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video_metadata = self.data['video_metadata']
        self.frames_data = self.data['frames']
        print(f"Loaded {len(self.frames_data)} frames from {json_path}")
    
    def get_video_metadata(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dict containing fps, width, height, total_frames
        """
        return self.video_metadata
    
    def get_frame_data(self, frame_number: int) -> List[Dict]:
        """
        Get pose detections for specific frame.
        
        Args:
            frame_number: Zero-indexed frame number
            
        Returns:
            List of detection dictionaries containing bbox, keypoints, confidence
        """
        frame_key = str(frame_number)
        if frame_key in self.frames_data:
            return self.frames_data[frame_key]['detections']
        return []
    
    def get_total_frames(self) -> int:
        """Get total number of frames in video."""
        return len(self.frames_data)

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
        print("Initializing CLIP-ReID model...")
        
        CLIP_REID_PATH = '/home/aparnabg/reid/CLIP-ReID'
        sys.path.append(CLIP_REID_PATH)
        
        from config.defaults import _C as cfg
        from model.make_model_clipreid import make_model
        from torchvision import transforms
        
        config_file = os.path.join(CLIP_REID_PATH, self.config.config_path)
        
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

class RegionExtractor:
    """
    Extracts upper and lower body regions from person detections.
    
    Uses pose keypoints to intelligently crop body parts for feature extraction.
    Handles different pose types: standing, sitting, lying.
    """
    
    @staticmethod
    def determine_pose_type(keypoints) -> str:
        """
        Determine person's pose type from keypoints.
        
        Args:
            keypoints: Pose keypoints array with confidence scores
            
        Returns:
            str: One of "standing", "sitting", "lying"
        """
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts) < 17:
                return "standing"
            
            visible_points = {}
            for name, idx in [('left_hip', 11), ('right_hip', 12),
                            ('left_knee', 13), ('right_knee', 14),
                            ('left_ankle', 15), ('right_ankle', 16)]:
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    visible_points[name] = kpts[idx][:2]
            
            if len(visible_points) < 3:
                return "standing"
            
            hip_y = [visible_points[h][1] for h in ['left_hip', 'right_hip'] if h in visible_points]
            ankle_y = [visible_points[a][1] for a in ['left_ankle', 'right_ankle'] if a in visible_points]
            
            if hip_y and ankle_y:
                hip_ankle_dist = abs(np.mean(ankle_y) - np.mean(hip_y))
                
                if hip_ankle_dist < 50:
                    return "lying"
                elif hip_ankle_dist < 120:
                    return "sitting"
            
            return "standing"
            
        except Exception:
            return "standing"
    
    def extract_upper_body_region(self, frame: np.ndarray, keypoints: np.ndarray,
                                  bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Extract upper body region from frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            keypoints: Pose keypoints array
            bbox: Bounding box [x1, y1, x2, y2]
            pose_type: One of "standing", "sitting", "lying"
            
        Returns:
            Tuple of (cropped_region, (x1, y1, x2, y2)) or None if extraction fails
        """
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
            
            hip_points = [kpts[idx][:2] for idx in [11, 12]
                         if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            
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
            
            upper_roi = frame[upper_y1:upper_y2, upper_x1:upper_x2]
            
            if upper_roi.shape[0] < 50 or upper_roi.shape[1] < 30:
                return None
            
            return upper_roi, (upper_x1, upper_y1, upper_x2, upper_y2)
            
        except Exception:
            return None
    
    def extract_lower_body_region(self, frame: np.ndarray, keypoints: np.ndarray,
                                  bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Extract lower body region from frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            keypoints: Pose keypoints array
            bbox: Bounding box [x1, y1, x2, y2]
            pose_type: One of "standing", "sitting", "lying"
            
        Returns:
            Tuple of (cropped_region, (x1, y1, x2, y2)) or None if extraction fails
            Note: Returns None for lying pose type
        """
        if pose_type == "lying":
            return None
        
        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            hip_points = [kpts[idx][:2] for idx in [11, 12]
                         if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            ankle_points = [kpts[idx][:2] for idx in [15, 16]
                           if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            
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
            
            lower_roi = frame[lower_y1:lower_y2, lower_x1:lower_x2]
            
            if lower_roi.shape[0] < 40 or lower_roi.shape[1] < 25:
                return None
            
            return lower_roi, (lower_x1, lower_y1, lower_x2, lower_y2)
            
        except Exception:
            return None

class FeatureExtractionModule:
    """
    Orchestrates feature extraction from video frames.
    
    Validates detection quality and extracts appearance features for tracking.
    Implements initial keypoint quality filtering.
    """
    
    def __init__(self, config: Stage2Config, clip_reid: CLIPReIDFeatureExtractor):
        """
        Initialize feature extraction module.
        
        Args:
            config: Stage2Config with feature extraction settings
            clip_reid: Initialized CLIP-ReID feature extractor
        """
        self.config = config
        self.clip_reid = clip_reid
        self.region_extractor = RegionExtractor()
    
    def validate_detection_quality(self, keypoints: np.ndarray) -> bool:
        """
        Validate detection has sufficient high-confidence keypoints.
        
        Args:
            keypoints: Pose keypoints array with confidence scores
            
        Returns:
            bool: True if detection meets quality threshold
        """
        if not self.config.detection_quality.enable_quality_check:
            return True
        
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            high_conf_count = sum(
                1 for kp in kpts 
                if len(kp) >= 3 and kp[2] > self.config.detection_quality.min_keypoint_confidence
            )
            
            return high_conf_count >= self.config.detection_quality.min_keypoints_required
            
        except Exception:
            return False
    
    def extract_features(self, frame: np.ndarray, detections_data: List[Dict],
                        frame_count: int) -> List[Dict]:
        """
        Extract features for all detections in frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            detections_data: List of detection dictionaries from pose detector
            frame_count: Current frame number for interval-based extraction
            
        Returns:
            List of detection dictionaries with extracted features
        """
        detections = []
        
        for det_data in detections_data:
            if not self.validate_detection_quality(det_data['keypoints']):
                continue
            
            detection = self._create_detection(frame, det_data, frame_count)
            if detection:
                detections.append(detection)
        
        return detections
    
    def _create_detection(self, frame: np.ndarray, det_data: Dict, frame_count: int) -> Optional[Dict]:
        """
        Create detection dictionary with features.
        
        Args:
            frame: Full frame image
            det_data: Detection data from pose detector
            frame_count: Current frame number
            
        Returns:
            Detection dictionary with bbox, keypoints, features or None if invalid
        """
        bbox = np.array(det_data['bbox'], dtype=np.float32) if isinstance(det_data['bbox'], list) else det_data['bbox']
        keypoints = np.array(det_data['keypoints'], dtype=np.float32) if isinstance(det_data['keypoints'], list) else det_data['keypoints']
        confidence = det_data['confidence']
        
        pose_type = self.region_extractor.determine_pose_type(keypoints)
        
        upper_feature = None
        lower_feature = None
        
        if frame_count % self.config.features.feature_update_interval == 0:
            if self.config.features.enable_lower_body:
                lower_result = self.region_extractor.extract_lower_body_region(
                    frame, keypoints, bbox, pose_type
                )
                if lower_result:
                    lower_roi, _ = lower_result
                    lower_feature = self.clip_reid.extract(lower_roi)
        
        if self.config.features.enable_upper_body:
            upper_result = self.region_extractor.extract_upper_body_region(
                frame, keypoints, bbox, pose_type
            )
            if upper_result:
                upper_roi, _ = upper_result
                upper_feature = self.clip_reid.extract(upper_roi)
        
        return {
            'bbox': bbox,
            'keypoints': keypoints,
            'confidence': confidence,
            'pose_type': pose_type,
            'face_feature': None,
            'upper_feature': upper_feature,
            'lower_feature': lower_feature
        }       


def parse_count_value(value) -> int:
    """
    Parse count value from CSV, handling special cases.
    
    Examples:
        '5' -> 5
        '10+' -> 10
        'x+' -> 999 (unknown/large crowd)
        'many' -> 999
    
    Args:
        value: Value from CSV (int, float, or string)
        
    Returns:
        Parsed integer count
    """
    if pd.isna(value):
        return 0
    
    value_str = str(value).strip().lower()
    
    # Handle unknown/large crowd indicators
    if value_str in ['x+', 'x', '?+', '?', 'many', 'crowd']:
        return 999
    
    # Remove '+' suffix: '10+' becomes '10'
    value_str = value_str.rstrip('+')
    
    try:
        return int(float(value_str))
    except ValueError:
        return 0


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
        print("Initializing CLIP-ReID model...")
        
        CLIP_REID_PATH = '/home/aparnabg/reid/CLIP-ReID'
        sys.path.append(CLIP_REID_PATH)
        
        from config.defaults import _C as cfg
        from model.make_model_clipreid import make_model
        from torchvision import transforms
        
        config_file = os.path.join(CLIP_REID_PATH, self.config.config_path)
        
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


class RegionExtractor:
    """
    Extracts upper and lower body regions from person detections.
    
    Uses pose keypoints to intelligently crop body parts for feature extraction.
    Handles different pose types: standing, sitting, lying.
    """
    
    @staticmethod
    def determine_pose_type(keypoints) -> str:
        """
        Determine person's pose type from keypoints.
        
        Args:
            keypoints: Pose keypoints array with confidence scores
            
        Returns:
            str: One of "standing", "sitting", "lying"
        """
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts) < 17:
                return "standing"
            
            visible_points = {}
            for name, idx in [('left_hip', 11), ('right_hip', 12),
                            ('left_knee', 13), ('right_knee', 14),
                            ('left_ankle', 15), ('right_ankle', 16)]:
                if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3:
                    visible_points[name] = kpts[idx][:2]
            
            if len(visible_points) < 3:
                return "standing"
            
            hip_y = [visible_points[h][1] for h in ['left_hip', 'right_hip'] if h in visible_points]
            ankle_y = [visible_points[a][1] for a in ['left_ankle', 'right_ankle'] if a in visible_points]
            
            if hip_y and ankle_y:
                hip_ankle_dist = abs(np.mean(ankle_y) - np.mean(hip_y))
                
                if hip_ankle_dist < 50:
                    return "lying"
                elif hip_ankle_dist < 120:
                    return "sitting"
            
            return "standing"
            
        except Exception:
            return "standing"
    
    def extract_upper_body_region(self, frame: np.ndarray, keypoints: np.ndarray,
                                  bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Extract upper body region from frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            keypoints: Pose keypoints array
            bbox: Bounding box [x1, y1, x2, y2]
            pose_type: One of "standing", "sitting", "lying"
            
        Returns:
            Tuple of (cropped_region, (x1, y1, x2, y2)) or None if extraction fails
        """
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
            
            hip_points = [kpts[idx][:2] for idx in [11, 12]
                         if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            
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
            
            upper_roi = frame[upper_y1:upper_y2, upper_x1:upper_x2]
            
            if upper_roi.shape[0] < 50 or upper_roi.shape[1] < 30:
                return None
            
            return upper_roi, (upper_x1, upper_y1, upper_x2, upper_y2)
            
        except Exception:
            return None
    
    def extract_lower_body_region(self, frame: np.ndarray, keypoints: np.ndarray,
                                  bbox: np.ndarray, pose_type: str) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Extract lower body region from frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            keypoints: Pose keypoints array
            bbox: Bounding box [x1, y1, x2, y2]
            pose_type: One of "standing", "sitting", "lying"
            
        Returns:
            Tuple of (cropped_region, (x1, y1, x2, y2)) or None if extraction fails
            Note: Returns None for lying pose type
        """
        if pose_type == "lying":
            return None
        
        try:
            kpts = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            hip_points = [kpts[idx][:2] for idx in [11, 12]
                         if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            ankle_points = [kpts[idx][:2] for idx in [15, 16]
                           if idx < len(kpts) and len(kpts[idx]) >= 3 and kpts[idx][2] > 0.3]
            
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
            
            lower_roi = frame[lower_y1:lower_y2, lower_x1:lower_x2]
            
            if lower_roi.shape[0] < 40 or lower_roi.shape[1] < 25:
                return None
            
            return lower_roi, (lower_x1, lower_y1, lower_x2, lower_y2)
            
        except Exception:
            return None


class FeatureExtractionModule:
    """
    Orchestrates feature extraction from video frames.
    
    Validates detection quality and extracts appearance features for tracking.
    Implements initial keypoint quality filtering.
    """
    
    def __init__(self, config: Stage2Config, clip_reid: CLIPReIDFeatureExtractor):
        """
        Initialize feature extraction module.
        
        Args:
            config: Stage2Config with feature extraction settings
            clip_reid: Initialized CLIP-ReID feature extractor
        """
        self.config = config
        self.clip_reid = clip_reid
        self.region_extractor = RegionExtractor()
    
    def validate_detection_quality(self, keypoints: np.ndarray) -> bool:
        """
        Validate detection has sufficient high-confidence keypoints.
        
        Args:
            keypoints: Pose keypoints array with confidence scores
            
        Returns:
            bool: True if detection meets quality threshold
        """
        if not self.config.detection_quality.enable_quality_check:
            return True
        
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            high_conf_count = sum(
                1 for kp in kpts 
                if len(kp) >= 3 and kp[2] > self.config.detection_quality.min_keypoint_confidence
            )
            
            return high_conf_count >= self.config.detection_quality.min_keypoints_required
            
        except Exception:
            return False
    
    def extract_features(self, frame: np.ndarray, detections_data: List[Dict],
                        frame_count: int) -> List[Dict]:
        """
        Extract features for all detections in frame.
        
        Args:
            frame: Full frame image (H, W, 3)
            detections_data: List of detection dictionaries from pose detector
            frame_count: Current frame number for interval-based extraction
            
        Returns:
            List of detection dictionaries with extracted features
        """
        detections = []
        
        for det_data in detections_data:
            if not self.validate_detection_quality(det_data['keypoints']):
                continue
            
            detection = self._create_detection(frame, det_data, frame_count)
            if detection:
                detections.append(detection)
        
        return detections
    
    def _create_detection(self, frame: np.ndarray, det_data: Dict, frame_count: int) -> Optional[Dict]:
        """
        Create detection dictionary with features.
        
        Args:
            frame: Full frame image
            det_data: Detection data from pose detector
            frame_count: Current frame number
            
        Returns:
            Detection dictionary with bbox, keypoints, features or None if invalid
        """
        bbox = np.array(det_data['bbox'], dtype=np.float32) if isinstance(det_data['bbox'], list) else det_data['bbox']
        keypoints = np.array(det_data['keypoints'], dtype=np.float32) if isinstance(det_data['keypoints'], list) else det_data['keypoints']
        confidence = det_data['confidence']
        
        pose_type = self.region_extractor.determine_pose_type(keypoints)
        
        upper_feature = None
        lower_feature = None
        
        if frame_count % self.config.features.feature_update_interval == 0:
            if self.config.features.enable_lower_body:
                lower_result = self.region_extractor.extract_lower_body_region(
                    frame, keypoints, bbox, pose_type
                )
                if lower_result:
                    lower_roi, _ = lower_result
                    lower_feature = self.clip_reid.extract(lower_roi)
        
        if self.config.features.enable_upper_body:
            upper_result = self.region_extractor.extract_upper_body_region(
                frame, keypoints, bbox, pose_type
            )
            if upper_result:
                upper_roi, _ = upper_result
                upper_feature = self.clip_reid.extract(upper_roi)
        
        return {
            'bbox': bbox,
            'keypoints': keypoints,
            'confidence': confidence,
            'pose_type': pose_type,
            'face_feature': None,
            'upper_feature': upper_feature,
            'lower_feature': lower_feature
        }


class EnhancedTrackingModule:
    """
    Enhanced tracking with 3-frame confirmation and detection validation.
    
    Key improvements:
    1. Candidate tracks: New detections get temporary IDs
    2. 3-frame confirmation: Must match consistently for 3 frames before confirmed
    3. Detection validation: Filter photos/TV/screens
    4. Better ID stability
    """
    
    def __init__(self, config: Stage2Config, deepface_verifier: Optional['DeepFaceVerifier'] = None):
        """Initialize enhanced tracking module."""
        self.config = config
        self.tracker_config = config.tracking.tracker_config
        self.person_tracker_config = config.tracking.person_tracker_config
        self.deepface_verifier = deepface_verifier
        
        self.frame_count = 0
        self.next_track_id = 1
        self.next_candidate_id = -1  # Negative IDs for candidates
        
        # Track storage
        self.active_tracks = {}  # Confirmed tracks
        self.candidate_tracks = {}  # Unconfirmed tracks (awaiting 3-frame validation)
        self.lost_tracks = {}
        self.person_profiles = {}
        
        # Tracking state
        self.camera_compensator = CameraMotionCompensator()
        self.match_history = {}
        self.track_motion_stats = {}
        self.track_stability = {}
        self.max_persons = config.tracking.max_persons
        
        # NEW: Detection validator
        self.detection_validator = DetectionValidator() if config.tracking.enable_detection_validation else None
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, int]:
        """
        Update tracking with 3-frame confirmation logic.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame image
            
        Returns:
            Dict mapping detection index to track_id (positive=confirmed, negative=candidate)
        """
        self.frame_count += 1
        
        camera_motion = self.camera_compensator.estimate_camera_motion(frame)
        
        det_bboxes = [d['bbox'] for d in detections]
        crowding = calculate_scene_crowding(det_bboxes)
        iou_thresh, center_weight, motion_conf_thresh = get_adaptive_thresholds(
            self.person_tracker_config, crowding
        )
        
        # ====================================================================
        # STEP 1: Motion matching (active + candidate tracks)
        # ====================================================================
        all_tracks = {**self.active_tracks, **self.candidate_tracks}
        motion_matches = self._match_with_motion(
            detections, camera_motion, iou_thresh, center_weight, 
            motion_conf_thresh, all_tracks
        )
        
        final_matches = {}
        for det_idx, track_id in motion_matches.items():
            if track_id in self.active_tracks:
                self._update_track(detections[det_idx], track_id, motion_match=True)
            else:  # Candidate track
                self._update_candidate_track(detections[det_idx], track_id, frame)
            final_matches[det_idx] = track_id
        
        unmatched_detections = [
            (i, det) for i, det in enumerate(detections) if i not in motion_matches
        ]
        
        # ====================================================================
        # STEP 2: Re-ID with lost tracks
        # ====================================================================
        reid_matches = self._match_with_lost_tracks(unmatched_detections, frame)
        
        for det_idx, (track_id, match_type) in reid_matches.items():
            self._reactivate_track(detections[det_idx], track_id)
            final_matches[det_idx] = track_id
        
        unmatched_detections = [
            (i, det) for i, det in unmatched_detections if i not in reid_matches
        ]
        
        # ====================================================================
        # STEP 3: Appearance matching with active tracks
        # ====================================================================
        appearance_matches = self._match_with_appearance(
            unmatched_detections, set(motion_matches.values()), frame
        )
        
        for det_idx, (track_id, match_type) in appearance_matches.items():
            if track_id in self.active_tracks:
                self._update_track(detections[det_idx], track_id, motion_match=False)
            else:  # Candidate track
                self._update_candidate_track(detections[det_idx], track_id, frame)
            final_matches[det_idx] = track_id
        
        unmatched_detections = [
            (i, det) for i, det in unmatched_detections if i not in appearance_matches
        ]
        
        # ====================================================================
        # STEP 4: Create CANDIDATE tracks for unmatched detections
        # NEW: Don't assign permanent ID immediately!
        # ====================================================================
        for det_idx, detection in unmatched_detections:
            candidate_id = self._create_candidate_track(detection, frame)
            if candidate_id is not None:
                final_matches[det_idx] = candidate_id
        
        # ====================================================================
        # STEP 5: Promote confirmed candidates to active tracks
        # ====================================================================
        self._promote_confirmed_candidates()
        
        # ====================================================================
        # STEP 6: Clean up failed candidates
        # ====================================================================
        self._cleanup_failed_candidates()
        
        self._record_match_history(final_matches, motion_matches, appearance_matches, reid_matches)
        self._handle_lost_tracks(final_matches)
        
        # Update detection validator
        if self.detection_validator:
            for det_idx, track_id in final_matches.items():
                if track_id > 0:  # Only for confirmed tracks
                    bbox = detections[det_idx]['bbox']
                    
                    # Calculate motion score
                    if track_id in self.track_motion_stats and len(self.track_motion_stats[track_id]) >= 2:
                        prev_center = self.track_motion_stats[track_id][-2]
                        curr_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                        motion_score = np.sqrt(
                            (curr_center[0] - prev_center[0])**2 + 
                            (curr_center[1] - prev_center[1])**2
                        )
                        self.detection_validator.update_motion(track_id, motion_score)
        
        return final_matches
    
    def _create_candidate_track(self, detection: Dict, frame: np.ndarray) -> Optional[int]:
        """
        Create CANDIDATE track (unconfirmed, needs 3-frame validation).
        
        Args:
            detection: Detection dictionary
            frame: Current frame
            
        Returns:
            Negative candidate ID or None
        """
        if not self._should_create_new_track():
            return None
        
        candidate_id = self.next_candidate_id
        self.next_candidate_id -= 1
        
        track = {
            'track_id': candidate_id,
            'kalman': create_kalman_filter(detection['bbox']),
            'detections': deque([detection], maxlen=100),
            'last_seen': self.frame_count,
            'created_frame': self.frame_count,
            'lost_frames': 0,
            'missed_updates': 0,
            'confirmation_count': 1,  # NEW: Count consecutive matches
            'is_candidate': True
        }
        
        self.candidate_tracks[candidate_id] = track
        
        return candidate_id
    
    def _update_candidate_track(self, detection: Dict, candidate_id: int, frame: np.ndarray):
        """
        Update candidate track and increment confirmation counter.
        
        Args:
            candidate_id: Negative candidate track ID
            detection: New detection
            frame: Current frame
        """
        if candidate_id not in self.candidate_tracks:
            return
        
        track = self.candidate_tracks[candidate_id]
        
        update_kalman_filter(track['kalman'], detection['bbox'])
        track['detections'].append(detection)
        track['last_seen'] = self.frame_count
        track['lost_frames'] = 0
        track['confirmation_count'] += 1
    
    def _promote_confirmed_candidates(self):
        """
        Promote candidate tracks that have been matched for 3+ consecutive frames.
        """
        candidates_to_promote = []
        
        for candidate_id, track in self.candidate_tracks.items():
            # Check if confirmed for required frames
            if track['confirmation_count'] >= self.tracker_config.new_track_confirmation_frames:
                candidates_to_promote.append(candidate_id)
        
        for candidate_id in candidates_to_promote:
            track = self.candidate_tracks.pop(candidate_id)
            
            # Assign permanent ID
            permanent_id = self.next_track_id
            self.next_track_id += 1
            
            track['track_id'] = permanent_id
            track['is_candidate'] = False
            
            self.active_tracks[permanent_id] = track
            
            # Create person profile
            # Find detection with best features
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
            
            print(f"Promoted candidate {candidate_id} to Track {permanent_id} "
                  f"(confirmed over {track['confirmation_count']} frames)")
    
    def _cleanup_failed_candidates(self):
        """Remove candidate tracks that failed confirmation."""
        candidates_to_remove = []
        
        for candidate_id, track in self.candidate_tracks.items():
            # Remove if lost for too long or failed to confirm
            age = self.frame_count - track['created_frame']
            
            if track['lost_frames'] > 10:  # Lost candidate
                candidates_to_remove.append(candidate_id)
            elif age > 30 and track['confirmation_count'] < self.tracker_config.new_track_confirmation_frames:
                # Too old but not confirmed
                candidates_to_remove.append(candidate_id)
        
        for candidate_id in candidates_to_remove:
            del self.candidate_tracks[candidate_id]
    
    def _should_create_new_track(self) -> bool:
        """Check if new track can be created."""
        if self.max_persons is None:
            return True
        
        # Count confirmed tracks only (not candidates)
        current_person_count = len(self.person_profiles)
        return current_person_count < self.max_persons
    
    def _match_with_motion(self, detections: List[Dict], camera_motion: tuple,
                          iou_thresh: float, center_weight: float, 
                          motion_conf_thresh: float, tracks: Dict) -> Dict[int, int]:
        """Match detections with tracks using motion prediction."""
        matches = {}
        
        if not tracks or not detections:
            return matches
        
        track_ids = list(tracks.keys())
        cost_matrix = np.full((len(detections), len(track_ids)), 1.0)
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                track = tracks[track_id]
                
                predicted_bbox, motion_confidence = predict_motion_with_camera_compensation(
                    track['kalman'], track['missed_updates'], camera_motion, cfg=self.person_tracker_config
                )
                
                # Adjust threshold based on track type
                effective_motion_thresh = motion_conf_thresh
                if track.get('is_candidate', False):
                    # More lenient for candidates
                    effective_motion_thresh *= 0.7
                elif 0 < track['missed_updates'] <= 3:
                    effective_motion_thresh *= 0.6
                
                if motion_confidence > effective_motion_thresh:
                    if is_spatially_plausible(detection['bbox'], predicted_bbox,
                                             self.tracker_config.max_jump_factor * 1.3):
                        sim = calculate_combined_similarity(detection['bbox'], predicted_bbox, center_weight)
                        cost_matrix[det_idx, track_idx] = 1.0 - sim
                    else:
                        cost_matrix[det_idx, track_idx] = 0.98
                else:
                    cost_matrix[det_idx, track_idx] = 0.90
        
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < 0.85:
                track_id = track_ids[track_idx]
                matches[det_idx] = track_id
        
        return matches
    
    def _match_with_appearance(self, unmatched_detections: List[Tuple[int, Dict]],
                               motion_matched_track_ids: set, frame: np.ndarray) -> Dict[int, Tuple[int, str]]:
        """Match with appearance features (active tracks + candidates)."""
        matches = {}
        
        all_tracks = {**self.active_tracks, **self.candidate_tracks}
        
        for det_idx, detection in unmatched_detections:
            best_match_id = None
            best_score = 0.0
            best_match_type = ""
            
            for track_id, track in all_tracks.items():
                if track_id in motion_matched_track_ids:
                    continue
                
                if track_id in [match[0] for match in matches.values()]:
                    continue
                
                # Stability-based threshold
                stability = self.track_stability.get(track_id, 0)
                
                if stability >= 5:
                    required_threshold = self.config.tracking.combined_reid_threshold + 0.20
                elif stability >= 3:
                    required_threshold = self.config.tracking.combined_reid_threshold + 0.10
                else:
                    required_threshold = self.config.tracking.combined_reid_threshold
                
                # Lower threshold for candidates
                if track.get('is_candidate', False):
                    required_threshold *= 0.8
                
                profile = self.person_profiles.get(track_id)
                if not profile:
                    # For candidates, use last detection features
                    if len(track['detections']) > 0:
                        last_det = track['detections'][-1]
                        profile = {
                            'upper_feature': last_det.get('upper_feature'),
                            'lower_feature': last_det.get('lower_feature'),
                            'face_feature': last_det.get('face_feature')
                        }
                    else:
                        continue
                
                similarity, match_type = self._compute_person_similarity(
                    detection, profile, frame, det_idx, track_id
                )
                
                if similarity > best_score and similarity > required_threshold:
                    best_score = similarity
                    best_match_id = track_id
                    best_match_type = match_type
            
            if best_match_id:
                matches[det_idx] = (best_match_id, best_match_type)
        
        return matches
    
    def _match_with_lost_tracks(self, unmatched_detections: List[Tuple[int, Dict]],
                                frame: np.ndarray) -> Dict[int, Tuple[int, str]]:
        """Match with lost tracks for re-identification."""
        matches = {}
        
        sorted_lost_tracks = sorted(
            self.lost_tracks.items(),
            key=lambda x: x[1]['last_seen'],
            reverse=True
        )
        
        for det_idx, detection in unmatched_detections:
            best_match_id = None
            best_score = 0.0
            best_match_type = ""
            
            for track_id, lost_track in sorted_lost_tracks:
                profile = self.person_profiles.get(track_id)
                if not profile:
                    continue
                
                similarity, match_type = self._compute_person_similarity(
                    detection, profile, frame, det_idx, track_id
                )
                
                frames_since_lost = self.frame_count - lost_track['last_seen']
                
                if frames_since_lost < 30:
                    reid_threshold = self.config.tracking.combined_reid_threshold + 0.05
                elif frames_since_lost < 100:
                    reid_threshold = self.config.tracking.combined_reid_threshold + 0.10
                else:
                    reid_threshold = self.config.tracking.combined_reid_threshold + 0.15
                
                if similarity > best_score and similarity > reid_threshold:
                    best_score = similarity
                    best_match_id = track_id
                    best_match_type = match_type
            
            if best_match_id:
                matches[det_idx] = (best_match_id, best_match_type)
        
        return matches
    
    def _compute_person_similarity(self, detection: Dict, profile: Dict,
                                   frame: np.ndarray, det_idx: int, track_id: int) -> Tuple[float, str]:
        """Compute appearance similarity between detection and profile."""
        similarities = []
        matching_components = []
        weights = []
        
        if detection['upper_feature'] is not None and profile.get('upper_feature') is not None:
            upper_sim = self._compute_feature_similarity(detection['upper_feature'], profile['upper_feature'])
            if upper_sim > self.config.tracking.upper_reid_threshold:
                similarities.append(upper_sim)
                matching_components.append("upper")
                weights.append(0.6)
        
        if detection['lower_feature'] is not None and profile.get('lower_feature') is not None:
            lower_sim = self._compute_feature_similarity(detection['lower_feature'], profile['lower_feature'])
            if lower_sim > self.config.tracking.lower_reid_threshold:
                similarities.append(lower_sim)
                matching_components.append("lower")
                weights.append(0.3)
        
        if similarities and weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            preliminary_sim = sum(s * w for s, w in zip(similarities, normalized_weights))
        else:
            preliminary_sim = 0.0
        
        face_feature = None
        
        if (self.config.face_detection.enable_face_features and
            self.deepface_verifier is not None and
            self.config.face_detection.appearance_similarity_low <= preliminary_sim <= self.config.face_detection.appearance_similarity_high):
            
            if self.deepface_verifier.check_face_keypoints_condition(detection['keypoints']):
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(b) for b in bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                bbox_crop = frame[y1:y2, x1:x2]
                face_feature = self.deepface_verifier.extract_face_feature(bbox_crop)
                
                if face_feature is not None:
                    if profile.get('face_feature') is not None:
                        face_sim = self._compute_feature_similarity(face_feature, profile['face_feature'])
                        if face_sim > self.config.tracking.face_reid_threshold:
                            similarities.append(face_sim)
                            matching_components.append("face")
                            weights.append(0.1)
        
        if similarities and weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            combined_sim = sum(s * w for s, w in zip(similarities, normalized_weights))
            match_description = "+".join(matching_components)
            
            if face_feature is not None:
                detection['face_feature'] = face_feature
            
            return combined_sim, match_description
        
        return 0.0, "none"
    
    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between features."""
        if feat1 is None or feat2 is None:
            return 0.0
        try:
            similarity = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
            return max(0.0, similarity)
        except Exception:
            return 0.0
    
    def _update_track(self, detection: Dict, track_id: int, motion_match: bool):
        """Update active track with new detection."""
        track = self.active_tracks[track_id]
        
        update_kalman_filter(track['kalman'], detection['bbox'])
        track['detections'].append(detection)
        track['last_seen'] = self.frame_count
        track['lost_frames'] = 0
        
        if motion_match:
            track['missed_updates'] = 0
        else:
            track['missed_updates'] = min(track['missed_updates'] + 1, 3)
        
        self._update_track_stability(track_id, motion_match)
        self._update_motion_stats(track_id, detection['bbox'])
        
        profile = self.person_profiles.get(track_id)
        if profile:
            self._update_person_profile(profile, detection)
    
    def _update_person_profile(self, profile: Dict, detection: Dict):
        """Update person profile with EMA."""
        alpha = 0.3
        
        if detection['upper_feature'] is not None:
            if profile.get('upper_feature') is None:
                profile['upper_feature'] = detection['upper_feature'].copy()
            else:
                profile['upper_feature'] = alpha * detection['upper_feature'] + (1 - alpha) * profile['upper_feature']
                norm = np.linalg.norm(profile['upper_feature'])
                if norm > 0:
                    profile['upper_feature'] = profile['upper_feature'] / norm
        
        if detection['lower_feature'] is not None:
            if profile.get('lower_feature') is None:
                profile['lower_feature'] = detection['lower_feature'].copy()
            else:
                profile['lower_feature'] = alpha * detection['lower_feature'] + (1 - alpha) * profile['lower_feature']
                norm = np.linalg.norm(profile['lower_feature'])
                if norm > 0:
                    profile['lower_feature'] = profile['lower_feature'] / norm
        
        if detection.get('face_feature') is not None:
            if profile.get('face_feature') is None:
                profile['face_feature'] = detection['face_feature'].copy()
            else:
                profile['face_feature'] = alpha * detection['face_feature'] + (1 - alpha) * profile['face_feature']
                norm = np.linalg.norm(profile['face_feature'])
                if norm > 0:
                    profile['face_feature'] = profile['face_feature'] / norm
    
    def _update_track_stability(self, track_id: int, motion_match: bool):
        """Track stability: consecutive motion matches."""
        if track_id not in self.track_stability:
            self.track_stability[track_id] = 0
        
        if motion_match:
            self.track_stability[track_id] = min(self.track_stability[track_id] + 1, 20)
        else:
            self.track_stability[track_id] = max(0, self.track_stability[track_id] - 1)
    
    def _update_motion_stats(self, track_id: int, bbox: np.ndarray):
        """Update motion statistics."""
        if track_id not in self.track_motion_stats:
            self.track_motion_stats[track_id] = []
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.track_motion_stats[track_id].append((center_x, center_y))
    
    def _reactivate_track(self, detection: Dict, track_id: int):
        """Reactivate lost track."""
        reactivated_track = self.lost_tracks.pop(track_id)
        reactivated_track['kalman'] = create_kalman_filter(detection['bbox'])
        reactivated_track['detections'].append(detection)
        reactivated_track['last_seen'] = self.frame_count
        reactivated_track['lost_frames'] = 0
        reactivated_track['missed_updates'] = 0
        
        self.active_tracks[track_id] = reactivated_track
        self._update_motion_stats(track_id, detection['bbox'])
        
        profile = self.person_profiles.get(track_id)
        if profile:
            self._update_person_profile(profile, detection)
        
        # Reset validator history
        if self.detection_validator:
            self.detection_validator.reset_track(track_id)
    
    def _record_match_history(self, final_matches: Dict[int, int],
                              motion_matches: Dict[int, int],
                              appearance_matches: Dict[int, Tuple[int, str]],
                              reid_matches: Dict[int, Tuple[int, str]]):
        """Record match history."""
        for det_idx, track_id in final_matches.items():
            if track_id < 0:  # Skip candidates
                continue
            
            if track_id not in self.match_history:
                self.match_history[track_id] = deque(maxlen=50)
            
            if det_idx in motion_matches:
                match_type = 'M'
            elif det_idx in reid_matches:
                match_type = f'R({reid_matches[det_idx][1]})'
            elif det_idx in appearance_matches:
                match_type = f'A({appearance_matches[det_idx][1]})'
            else:
                match_type = 'N'
            
            self.match_history[track_id].append(match_type)
    
    def _handle_lost_tracks(self, final_matches: Dict[int, int]):
        """Handle unmatched tracks."""
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
        
        # Handle lost candidates
        for candidate_id, track in list(self.candidate_tracks.items()):
            if candidate_id not in final_matches.values():
                track['lost_frames'] += 1
    
    def get_match_history_summary(self, track_id: int, last_n: int = 10) -> str:
        """Get match history string."""
        if track_id not in self.match_history:
            return "No history"
        
        history = list(self.match_history[track_id])[-last_n:]
        return "->".join(history) if history else "No history"
    
    def validate_tracks_with_detector(self, frame: np.ndarray) -> List[int]:
        """
        Validate all active tracks using detection validator.
        
        Returns:
            List of track IDs that failed validation (photos/screens)
        """
        if not self.detection_validator:
            return []
        
        invalid_tracks = []
        
        for track_id, track in self.active_tracks.items():
            if len(track['detections']) < 10:
                continue  # Need enough history
            
            last_detection = track['detections'][-1]
            bbox = last_detection['bbox']
            
            is_valid = self.detection_validator.is_real_person(
                track_id, bbox, frame, min_frames=10
            )
            
            if not is_valid:
                invalid_tracks.append(track_id)
                print(f"Track {track_id} failed validation (likely photo/screen)")
        
        return invalid_tracks
    
    def apply_posthoc_filtering(self, fps: float, 
                               max_main_persons: Optional[int] = None,
                               invalid_track_ids: List[int] = None) -> Dict[int, int]:
        """
        Apply post-hoc filtering to all tracks.
        
        Args:
            fps: Video frame rate
            max_main_persons: Max main persons (adults + children from CSV)
            invalid_track_ids: Tracks that failed validation (NEW)
            
        Returns:
            Dict mapping old track IDs to new track IDs
        """
        # Combine ALL tracks (active + lost)
        all_tracks = {**self.active_tracks, **self.lost_tracks}
        
        print(f"\nPre-filtering stats:")
        print(f"  Total tracks: {len(all_tracks)}")
        print(f"  Active: {len(self.active_tracks)}")
        print(f"  Lost: {len(self.lost_tracks)}")
        print(f"  Profiles: {len(self.person_profiles)}")
        
        # Create filter with full config
        post_filter = PostHocTrackFilter(
            fps=fps,
            config=self.config.track_filtering  # Pass entire FilteringConfig
        )
        
        # Filter all tracks
        filtered_tracks, removed_ids = post_filter.filter_tracks(
            all_tracks, 
            self.person_profiles, 
            max_main_persons,
            invalid_track_ids=invalid_track_ids or []
        )
        
        print(f"\nPost-filtering stats:")
        print(f"  Remaining tracks: {len(filtered_tracks)}")
        print(f"  Removed: {len(removed_ids)}")
        
        # Reassign consecutive IDs (optional based on config)
        old_to_new, new_tracks, new_profiles = post_filter.reassign_track_ids(
            filtered_tracks, self.person_profiles
        )
        
        # Update ALL tracking state with new IDs
        self.active_tracks = {tid: t for tid, t in new_tracks.items() 
                             if t['last_seen'] >= self.frame_count - self.tracker_config.max_lost_frames}
        self.lost_tracks = {tid: t for tid, t in new_tracks.items() 
                           if t['last_seen'] < self.frame_count - self.tracker_config.max_lost_frames}
        self.person_profiles = new_profiles
        
        # Remap internal state dictionaries
        self._remap_tracking_state(old_to_new)
        
        print(f"\nFinal state:")
        print(f"  Active tracks: {len(self.active_tracks)}")
        print(f"  Lost tracks: {len(self.lost_tracks)}")
        print(f"  Total profiles: {len(self.person_profiles)}")
        
        return old_to_new

    def _is_stationary_track(self, track_id: int) -> bool:
        """
        Check if track represents stationary person.
        
        Args:
            track_id: Track ID to check
            
        Returns:
            bool: True if track is stationary
        """
        if track_id not in self.track_motion_stats:
            return False
        
        centers = self.track_motion_stats[track_id]
        if len(centers) < 2:
            return False
        
        movements = []
        for i in range(1, len(centers)):
            prev_x, prev_y = centers[i-1]
            curr_x, curr_y = centers[i]
            dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(dist)
        
        avg_movement = np.mean(movements)
        return avg_movement < self.config.track_filtering.stationary_threshold_pixels    
    
    def filter_short_stationary_tracks(self):
        """
        Remove short-lived stationary tracks.
        
        Filters tracks that are too short in duration and stationary.
        """
        if not self.config.track_filtering.filter_short_tracks:
            return
        
        tracks_to_remove = []
        
        for track_id in list(self.person_profiles.keys()):
            track = self.active_tracks.get(track_id) or self.lost_tracks.get(track_id)
            
            if track is None:
                continue
            
            lifespan = track['last_seen'] - track['created_frame']
            
            if lifespan <= self.config.track_filtering.min_track_length:
                if track_id not in self.lost_tracks or track_id in self.active_tracks:
                    if self._is_stationary_track(track_id):
                        tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
            if track_id in self.person_profiles:
                del self.person_profiles[track_id]
            if track_id in self.match_history:
                del self.match_history[track_id]
            if track_id in self.track_motion_stats:
                del self.track_motion_stats[track_id]
        
        if tracks_to_remove:
            print(f"Filtered {len(tracks_to_remove)} tracks: {tracks_to_remove}")
    def _remap_tracking_state(self, old_to_new: Dict[int, int]):
        """Remap all tracking state to new IDs."""
        new_match_history = {}
        for old_id, new_id in old_to_new.items():
            if old_id in self.match_history: 
                new_match_history[new_id] = self.match_history[old_id]  
        
        # Clear all old histories (including unmapped ones)
        self.match_history.clear() 
        self.match_history = new_match_history  
        
        new_track_stability = {}
        for old_id, new_id in old_to_new.items():
            if old_id in self.track_stability: 
                new_track_stability[new_id] = self.track_stability[old_id]  
        self.track_stability = new_track_stability  
        new_motion_stats = {}
        for old_id, new_id in old_to_new.items():
            if old_id in self.track_motion_stats:  
                new_motion_stats[new_id] = self.track_motion_stats[old_id]  
        self.track_motion_stats = new_motion_stats  
    # ============================================================================
# ENHANCED TRACKING PIPELINE
# ============================================================================

class DeepFaceVerifier:
    """
    Face detection and verification using DeepFace library.
    
    Handles face detection and feature extraction for re-identification.
    """
    
    def __init__(self, config: FaceDetectionConfig):
        """
        Initialize DeepFace verifier.
        
        Args:
            config: FaceDetectionConfig containing backend, model, and threshold settings
        """
        self.config = config
        self.backend = config.deepface_backend
        self.model_name = config.deepface_model
        self.confidence_threshold = config.face_confidence_threshold
        
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            print(f"DeepFace initialized: backend={self.backend}, model={self.model_name}")
        except ImportError:
            print("WARNING: DeepFace not installed. Face verification disabled.")
            self.DeepFace = None
    
    def extract_face_feature(self, bbox_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face feature vector from bounding box crop.
        
        Args:
            bbox_crop: Cropped image region as numpy array (H, W, 3)
            
        Returns:
            Normalized face embedding vector if face found with sufficient confidence, else None
        """
        if self.DeepFace is None:
            return None
        
        if bbox_crop.shape[0] < 50 or bbox_crop.shape[1] < 30:
            return None
        
        try:
            face_objs = self.DeepFace.extract_faces(
                img_path=bbox_crop,
                detector_backend=self.backend,
                enforce_detection=False,
                align=True
            )
            
            if not face_objs:
                return None
            
            face_obj = face_objs[0]
            confidence = face_obj.get('confidence', 0.0)
            
            if confidence < self.confidence_threshold:
                return None
            
            face_region = face_obj['face']
            if face_region.max() <= 1.0:
                face_region = (face_region * 255).astype(np.uint8)
            
            embedding = self.DeepFace.represent(
                img_path=face_region,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding:
                face_feature = np.array(embedding[0]['embedding'], dtype=np.float32)
                norm = np.linalg.norm(face_feature)
                if norm > 0:
                    face_feature = face_feature / norm
                return face_feature
            
            return None
            
        except Exception:
            return None
    
    def check_face_keypoints_condition(self, keypoints: np.ndarray) -> bool:
        """
        Check if sufficient high-confidence face keypoints are detected.
        
        Args:
            keypoints: Pose keypoints array, shape (N, 3) where N >= 91
                      Format: [x, y, confidence] for each keypoint
                      Indices 23-90 correspond to face keypoints
            
        Returns:
            bool: True if sufficient face keypoints meet confidence threshold
        """
        try:
            if isinstance(keypoints, list):
                kpts = np.array(keypoints, dtype=np.float32)
            elif torch.is_tensor(keypoints):
                kpts = keypoints.cpu().numpy()
            else:
                kpts = keypoints
            
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            
            if len(kpts) < 91:
                return False
            
            face_keypoints = kpts[23:91]
            
            high_conf_count = sum(1 for kp in face_keypoints 
                                 if len(kp) >= 3 and kp[2] > self.config.face_keypoint_threshold)
            
            return high_conf_count > self.config.min_face_keypoints
            
        except Exception:
            return False


class PoseDataLoader:
    """
    Loads and manages pose detection data from JSON files.
    
    Provides frame-by-frame access to pose detections with metadata.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize pose data loader.
        
        Args:
            json_path: Path to JSON file containing pose detections
        """
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.video_metadata = self.data['video_metadata']
        self.frames_data = self.data['frames']
        print(f"Loaded {len(self.frames_data)} frames from {json_path}")
    
    def get_video_metadata(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dict containing fps, width, height, total_frames
        """
        return self.video_metadata
    
    def get_frame_data(self, frame_number: int) -> List[Dict]:
        """
        Get pose detections for specific frame.
        
        Args:
            frame_number: Zero-indexed frame number
            
        Returns:
            List of detection dictionaries containing bbox, keypoints, confidence
        """
        frame_key = str(frame_number)
        if frame_key in self.frames_data:
            return self.frames_data[frame_key]['detections']
        return []
    
    def get_total_frames(self) -> int:
        """Get total number of frames in video."""
        return len(self.frames_data)
        
class EnhancedTrackingPipeline:
    """Enhanced pipeline with all new features."""
    
            
    def __init__(self, config: Stage2Config = None):
        if config is None:
            config = Stage2Config()
        
        self.config = config
        self.clip_reid = CLIPReIDFeatureExtractor(config.clip_reid)
        
        self.deepface_verifier = None
        if config.face_detection.enable_face_features:
            self.deepface_verifier = DeepFaceVerifier(config.face_detection)
        
        self.feature_module = FeatureExtractionModule(config, self.clip_reid)
        self.tracking_module = EnhancedTrackingModule(config, self.deepface_verifier)
        
        self._interrupted = False
        self._proc = None
        self._cap = None
    
    def _signal_handler(self, signum, frame):
        """
        Handle graceful shutdown on interrupt.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print("\n\nInterrupted! Finalizing...")
        self._interrupted = True
        
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(0)
    # before post filter video dont knwo the other function is correct or not need to check even the post filtering itself 
    '''def process_video(self, pose_json_path: str, video_path: str,
                     output_video_path: Optional[str] = None,
                     output_json_path: Optional[str] = None):
        """Process video with enhanced tracking."""
        print(f"Processing: {pose_json_path}")
        print(f"Video: {video_path}")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        
        pose_loader = PoseDataLoader(pose_json_path)
        metadata = pose_loader.get_video_metadata()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        self._cap = cap
        
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']
        total_frames = pose_loader.get_total_frames()
        
        proc = None
        if self.config.visualization.enable_visualization and output_video_path:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", f"{fps}", "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_video_path
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self._proc = proc
        
        start_time = time.time()
        frame_number = 0
        
        try:
            with tqdm(total=total_frames, desc="Tracking") as pbar:
                while frame_number < total_frames and not self._interrupted:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    try:
                        detections_data = pose_loader.get_frame_data(frame_number)
                        
                        detections = self.feature_module.extract_features(
                            frame, detections_data, self.tracking_module.frame_count + 1
                        )
                        
                        person_assignments = self.tracking_module.update(detections, frame)
                        
                        vis_frame = frame
                        if self.config.visualization.enable_visualization:
                            vis_frame = self._draw_tracking(frame, detections, person_assignments)
                        
                        if proc:
                            if vis_frame.shape[0] != height or vis_frame.shape[1] != width:
                                vis_frame = cv2.resize(vis_frame, (width, height))
                            try:
                                proc.stdin.write(vis_frame.tobytes())
                            except BrokenPipeError:
                                break
                        
                        if self.tracking_module.frame_count % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                            print(f"\nFrame {self.tracking_module.frame_count}: "
                                  f"Active={len(self.tracking_module.active_tracks)}, "
                                  f"Lost={len(self.tracking_module.lost_tracks)}, "
                                  f"Total={len(self.tracking_module.person_profiles)}")
                            
                            recent_tracks = sorted(self.tracking_module.active_tracks.keys())[-3:]
                            if recent_tracks:
                                print("Recent track patterns (last 10 frames):")
                                for tid in recent_tracks:
                                    history = self.tracking_module.get_match_history_summary(tid, 10)
                                    print(f"  Track {tid}: {history}")
                    
                    except Exception as e:
                        print(f"Error processing frame {frame_number}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    frame_number += 1
                    pbar.update(1)
        
        finally:
            cap.release()
            if proc:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                rc = proc.wait()
                if rc != 0:
                    err = proc.stderr.read().decode(errors="ignore")
                    print(f"FFmpeg error: {err}")
        
        processing_time = time.time() - start_time
        
        self._print_match_statistics()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\nProcessing complete.")
        if output_video_path and self.config.visualization.enable_visualization:
            print(f"Output video: {output_video_path}")
        if output_json_path:
            print(f"Output JSON: {output_json_path}")
        print(f"Total persons tracked: {len(self.tracking_module.person_profiles)}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"FPS: {frame_number / processing_time:.2f}")        

        
        # After processing all frames, apply post-hoc filtering
        if self.config.track_filtering.enable_posthoc_filtering:
            metadata = pose_loader.get_video_metadata()
            fps = metadata['fps']
            
            max_main_persons = self.config.tracking.max_persons
            if max_main_persons and max_main_persons >= 999:
                max_main_persons = None  # Disable for large crowds
            
            self.tracking_module.apply_posthoc_filtering(
                fps=fps,
                max_main_persons=max_main_persons
            )
        
        print("\nProcessing complete.")
        print(f"Final person count: {len(self.tracking_module.person_profiles)}")
    
    def _draw_tracking(self, frame: np.ndarray, detections: List[Dict],
                      person_assignments: Dict[int, int]) -> np.ndarray:
        """
        Draw tracking visualization on frame.
        
        Args:
            frame: Original frame image
            detections: List of detections with features
            person_assignments: Dict mapping detection index to track_id
            
        Returns:
            np.ndarray: Frame with tracking visualization
        """
        vis_frame = frame.copy()
        
        for det_idx, track_id in person_assignments.items():
            if det_idx < len(detections):
                detection = detections[det_idx]
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(b) for b in bbox]
                
                track = self.tracking_module.active_tracks.get(track_id)
                if track:
                    if track['missed_updates'] == 0:
                        if track['created_frame'] == track['last_seen']:
                            match_type = "New"
                            color = (0, 255, 255)
                        else:
                            match_type = "Motion"
                            color = (0, 255, 0)
                    else:
                        match_type = "Appearance"
                        color = (255, 0, 0)
                else:
                    match_type = "Re-ID"
                    color = (0, 0, 255)
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                history_str = ""
                if track_id in self.tracking_module.match_history:
                    recent = list(self.tracking_module.match_history[track_id])[-3:]
                    history_str = f" [{','.join(recent)}]"
                
                text = f"ID {track_id}: {match_type}{history_str}"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(vis_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                cv2.putText(vis_frame, text, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_frame'''
    def process_video(self, pose_json_path: str, video_path: str,
                 output_video_path: Optional[str] = None,
                 output_json_path: Optional[str] = None):
        """
        Process video with two-pass approach:
        Pass 1: Extract all tracking data
        Pass 2: Apply post-processing, then render video with final IDs
        """
        print(f"Processing: {pose_json_path}")
        print(f"Video: {video_path}")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        
        pose_loader = PoseDataLoader(pose_json_path)
        metadata = pose_loader.get_video_metadata()
        
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']
        total_frames = pose_loader.get_total_frames()
        
        print(f"\n{'='*70}")
        print(f"PASS 1: TRACKING")
        print(f"{'='*70}")
        
        # ====================================================================
        # PASS 1: Process all frames and collect tracking data
        # ====================================================================
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        self._cap = cap
        
        # Store frame-by-frame assignments for later visualization
        frame_tracking_data = []  # List of (frame, detections, assignments)
        
        start_time = time.time()
        frame_number = 0
        
        try:
            with tqdm(total=total_frames, desc="Pass 1: Tracking") as pbar:
                while frame_number < total_frames and not self._interrupted:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    try:
                        detections_data = pose_loader.get_frame_data(frame_number)
                        
                        detections = self.feature_module.extract_features(
                            frame, detections_data, self.tracking_module.frame_count + 1
                        )
                        
                        person_assignments = self.tracking_module.update(detections, frame)
                        
                        # Store tracking data for this frame (with ORIGINAL IDs)
                        frame_tracking_data.append({
                            'frame_number': frame_number,
                            'detections': detections.copy(),
                            'assignments': person_assignments.copy()
                        })
                        
                        if self.tracking_module.frame_count % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                            print(f"\nFrame {self.tracking_module.frame_count}: "
                                  f"Active={len(self.tracking_module.active_tracks)}, "
                                  f"Lost={len(self.tracking_module.lost_tracks)}, "
                                  f"Total={len(self.tracking_module.person_profiles)}")
                    
                    except Exception as e:
                        print(f"Error processing frame {frame_number}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    frame_number += 1
                    pbar.update(1)
        
        finally:
            cap.release()
        
        pass1_time = time.time() - start_time
        print(f"\nPass 1 complete: {pass1_time:.2f}s")
        print(f"Frames processed: {len(frame_tracking_data)}")
        print(f"Tracks before filtering: {len(self.tracking_module.person_profiles)}")
        
        # ====================================================================
        # POST-PROCESSING: Filter tracks and reassign IDs
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"POST-PROCESSING")
        print(f"{'='*70}")
        
        old_to_new_mapping = {}
        invalid_track_ids = []
        
        # Validate tracks with detector
        if self.tracking_module.detection_validator:
            print("\nValidating tracks (checking for photos/screens)...")
            cap_validation = cv2.VideoCapture(video_path)
            # Sample validation on last frame for each track
            cap_validation.set(cv2.CAP_PROP_POS_FRAMES, len(frame_tracking_data) - 1)
            ret, last_frame = cap_validation.read()
            if ret:
                invalid_track_ids = self.tracking_module.validate_tracks_with_detector(last_frame)
            cap_validation.release()
        
        if self.config.track_filtering.enable_posthoc_filtering:
            max_main_persons = self.config.tracking.max_persons
            if max_main_persons and max_main_persons >= 999:
                max_main_persons = None  # Disable for large crowds
            
            old_to_new_mapping = self.tracking_module.apply_posthoc_filtering(
                fps=fps,
                max_main_persons=max_main_persons,
                invalid_track_ids=invalid_track_ids
            )
            
            print(f"\nID Mapping: {len(old_to_new_mapping)} tracks")
            if len(old_to_new_mapping) <= 20:
                print("  Old ID -> New ID:")
                for old_id, new_id in sorted(old_to_new_mapping.items()):
                    print(f"    {old_id:3d} -> {new_id:3d}")
        else:
            # No filtering: identity mapping
            old_to_new_mapping = {
                tid: tid for tid in self.tracking_module.person_profiles.keys()
            }
        
        print(f"\nFinal track count: {len(self.tracking_module.person_profiles)}")
        
        # ====================================================================
        # PASS 2: Render video with final track IDs
        # ====================================================================
        if self.config.visualization.enable_visualization and output_video_path:
            print(f"\n{'='*70}")
            print(f"PASS 2: RENDERING VIDEO")
            print(f"{'='*70}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not re-open video {video_path}")
                return
            
            # Setup FFmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", f"{fps}", "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_video_path
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self._proc = proc
            
            try:
                with tqdm(total=len(frame_tracking_data), desc="Pass 2: Rendering") as pbar:
                    for frame_data in frame_tracking_data:
                        if self._interrupted:
                            break
                        
                        frame_num = frame_data['frame_number']
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        
                        if not ret:
                            print(f"Warning: Could not read frame {frame_num}")
                            continue
                        
                        # Remap assignments to new IDs
                        original_assignments = frame_data['assignments']
                        remapped_assignments = {}
                        
                        for det_idx, old_track_id in original_assignments.items():
                            # Skip candidate tracks (negative IDs)
                            if old_track_id < 0:
                                continue
                            
                            # Map to new ID (or skip if filtered out)
                            new_track_id = old_to_new_mapping.get(old_track_id)
                            if new_track_id is not None:
                                remapped_assignments[det_idx] = new_track_id
                        
                        # Draw with FINAL track IDs
                        vis_frame = self._draw_tracking_final(
                            frame, 
                            frame_data['detections'], 
                            remapped_assignments
                        )
                        
                        if vis_frame.shape[0] != height or vis_frame.shape[1] != width:
                            vis_frame = cv2.resize(vis_frame, (width, height))
                        
                        try:
                            proc.stdin.write(vis_frame.tobytes())
                        except BrokenPipeError:
                            break
                        
                        pbar.update(1)
            
            finally:
                cap.release()
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                rc = proc.wait()
                if rc != 0:
                    err = proc.stderr.read().decode(errors="ignore")
                    print(f"FFmpeg error: {err}")
            
            pass2_time = time.time() - start_time - pass1_time
            print(f"\nPass 2 complete: {pass2_time:.2f}s")
        
        processing_time = time.time() - start_time
        
        self._print_match_statistics()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        if output_video_path and self.config.visualization.enable_visualization:
            print(f"Output video: {output_video_path}")
        if output_json_path:
            print(f"Output JSON: {output_json_path}")
        print(f"Final persons tracked: {len(self.tracking_module.person_profiles)}")
        print(f"Total processing time: {processing_time:.2f}s")
        print(f"  Pass 1 (tracking): {pass1_time:.2f}s")
        if self.config.visualization.enable_visualization and output_video_path:
            print(f"  Pass 2 (rendering): {pass2_time:.2f}s")
        print(f"Average FPS: {frame_number / processing_time:.2f}")
        print("="*70)
    
    
    def _draw_tracking_final(self, frame: np.ndarray, detections: List[Dict],
                            person_assignments: Dict[int, int]) -> np.ndarray:
        """
        Draw tracking visualization with FINAL track IDs (after post-processing).
        
        Args:
            frame: Original frame image
            detections: List of detections with features
            person_assignments: Dict mapping detection index to FINAL track_id
            
        Returns:
            np.ndarray: Frame with tracking visualization
        """
        vis_frame = frame.copy()
        
        for det_idx, track_id in person_assignments.items():
            if det_idx < len(detections):
                detection = detections[det_idx]
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(b) for b in bbox]
                
                # Check if track is main person or background
                track = self.tracking_module.active_tracks.get(track_id)
                if not track:
                    track = self.tracking_module.lost_tracks.get(track_id)
                
                is_main = False
                is_background = False
                if track:
                    is_main = track.get('is_main_person', False)
                    is_background = track.get('is_background_person', False)
                
                # Color coding:
                # Green: Main person
                # Yellow: Background person
                # Blue: Unknown (shouldn't happen after filtering)
                if is_main:
                    color = (0, 255, 0)  # Green
                    label_suffix = " [MAIN]"
                elif is_background:
                    color = (0, 255, 255)  # Yellow
                    label_suffix = " [BG]"
                else:
                    color = (255, 0, 0)  # Blue
                    label_suffix = ""
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"ID {track_id}{label_suffix}"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(vis_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                cv2.putText(vis_frame, text, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    def _print_match_statistics(self):
        """
        Print detailed match statistics for all tracks.
        """
        for track_id in sorted(self.tracking_module.match_history.keys()):
            history = list(self.tracking_module.match_history[track_id])
            if len(history) < 5:
                continue
            
            motion_count = sum(1 for h in history if h == 'M')
            appearance_count = sum(1 for h in history if h.startswith('A'))
            reid_count = sum(1 for h in history if h.startswith('R'))
            new_count = sum(1 for h in history if h == 'N')
            
            total = len(history)
            motion_pct = (motion_count / total) * 100
            appearance_pct = (appearance_count / total) * 100
            
            switches = sum(1 for i in range(1, len(history))
                          if (history[i-1] == 'M' and history[i].startswith('A')) or
                             (history[i-1].startswith('A') and history[i] == 'M'))
            
            is_stationary = self.tracking_module._is_stationary_track(track_id)
            
            print(f"\nTrack {track_id} ({total} frames):")
            print(f"  Motion: {motion_count} ({motion_pct:.1f}%)")
            print(f"  Appearance: {appearance_count} ({appearance_pct:.1f}%)")
            print(f"  Re-ID: {reid_count}")
            print(f"  Switches: {switches}")
            print(f"  Stationary: {'Yes' if is_stationary else 'No'}")
            print(f"  Last 15: {self.tracking_module.get_match_history_summary(track_id, 15)}")
    
    def reset_for_next_video(self):
        """
        Reset pipeline state for processing next video.
        """
        self.tracking_module.frame_count = 0
        self.tracking_module.next_track_id = 1
        self.tracking_module.active_tracks.clear()
        self.tracking_module.lost_tracks.clear()
        self.tracking_module.person_profiles.clear()
        self.tracking_module.match_history.clear()
        self.tracking_module.track_motion_stats.clear()
        self.tracking_module.candidate_tracks.clear()
        self.tracking_module.next_candidate_id = -1
        if self.tracking_module.detection_validator:
            self.tracking_module.detection_validator.motion_history.clear()
            self.tracking_module.detection_validator.texture_history.clear()
            self.tracking_module.detection_validator.size_history.clear()
        self._interrupted = False
        self._proc = None
        self._cap = None
        
        torch.cuda.empty_cache()
        gc.collect()




def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"tracking_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_max_persons_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load max persons configuration and video paths from CSV file.
    
    CSV Format:
        SourceFile,FileName,#_adults,#_children,#_people_background
        /path/to/video1.mp4,video1.mp4,2,1,3
        /path/to/video2.mov,video2.mov,5,0,10+
        /path/to/video3.avi,video3.avi,1,2,x+
    
    Calculation:
        total_persons = adults + children + background
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dict mapping video filename (without extension) to dict containing:
            - 'max_persons': int (total person count)
            - 'source_path': str (full path to video file from SourceFile column)
        
    Example:
        {
            'video1': {'max_persons': 6, 'source_path': '/path/to/video1.mp4'},
            'video2': {'max_persons': 15, 'source_path': '/path/to/video2.mov'}
        }
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Map column names (case-insensitive)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'sourcefile' in col_lower or 'source_file' in col_lower or 'source' in col_lower:
            column_mapping['SourceFile'] = col
        elif 'filename' in col_lower:
            column_mapping['FileName'] = col
        elif 'adult' in col_lower:
            column_mapping['#_adults'] = col
        elif 'child' in col_lower:
            column_mapping['#_children'] = col
        elif 'background' in col_lower or 'bg' in col_lower:
            column_mapping['#_people_background'] = col
    
    # Validate required columns
    required = ['SourceFile', 'FileName', '#_adults', '#_children', '#_people_background']
    missing = [req for req in required if req not in column_mapping]
    if missing:
        raise ValueError(
            f"CSV must contain columns: {', '.join(required)}\n"
            f"Found columns: {', '.join(df.columns)}\n"
            f"Missing mappings for: {', '.join(missing)}"
        )
    
    max_persons_map = {}
    for _, row in df.iterrows():
        filename = row[column_mapping['FileName']]
        source_path = row[column_mapping['SourceFile']]
        
        if pd.isna(filename) or pd.isna(source_path):
            continue
        
        # Use filename without extension as key
        video_stem = Path(str(filename)).stem
        
        # Parse counts
        adults = parse_count_value(row[column_mapping['#_adults']])
        children = parse_count_value(row[column_mapping['#_children']])
        background = parse_count_value(row[column_mapping['#_people_background']])
        
        # Calculate total
        total_count = adults + children + background
        # Convert SourceFile path to .mp4 extension
        source_path_obj = Path(str(source_path).strip())
        source_path_mp4 = str(source_path_obj.with_suffix('.mp4'))
        max_persons_map[video_stem] = {
            'max_persons': total_count,
            'source_path': source_path_mp4
        }
    
    return max_persons_map


def get_video_info_for_json(json_stem: str, max_persons_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get video information (max persons and source path) for a specific JSON file.
    
    Args:
        json_stem: Stem of JSON filename (without '_pose_data' suffix and extension)
        max_persons_map: Dictionary from load_max_persons_from_csv()
        
    Returns:
        Dict with 'max_persons' and 'source_path' or None if not found in CSV
    """
    return max_persons_map.get(json_stem, None)


def process_videos_batch(pose_json_dir: str, output_dir: str,
                        config: Stage2Config, max_persons_csv: str,
                        max_videos: Optional[int] = None, skip_existing: bool = True):
    """
    Process multiple videos in batch mode using CSV for max persons and video paths.
    
    Args:
        pose_json_dir: Directory containing pose JSON files
        output_dir: Output directory for results
        config: Stage2Config with tracking settings
        max_persons_csv: Path to CSV file with max persons and SourceFile paths (REQUIRED)
        max_videos: Optional limit on number of videos to process
        skip_existing: If True, skip already processed videos
    
    Note: video_dir parameter is removed - video paths come from CSV SourceFile column
    """
    logger = setup_logging(os.path.join(output_dir, "logs"))
    
    logger.info("=" * 70)
    logger.info("MULTI-PERSON TRACKING SYSTEM - BATCH PROCESSING")
    logger.info("=" * 70)
    logger.info(f"Pose JSON directory: {pose_json_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max persons CSV: {max_persons_csv}")
    
    # Load CSV configuration (REQUIRED)
    try:
        max_persons_map = load_max_persons_from_csv(max_persons_csv)
        logger.info(f"Loaded configuration from CSV")
        logger.info(f"  Found configurations for {len(max_persons_map)} videos")
        
        if max_persons_map:
            counts = [info['max_persons'] for info in max_persons_map.values()]
            logger.info(f"  Max persons range: {min(counts)} to {max(counts)}")
            logger.info(f"  Average: {sum(counts) / len(counts):.1f}")
            
            large_crowd_count = sum(1 for c in counts if c >= 999)
            if large_crowd_count > 0:
                logger.info(f"  Videos with large/unknown crowds: {large_crowd_count}")
            
            # Show sample
            sample_items = list(max_persons_map.items())[:3]
            logger.info(f"  Sample entries:")
            for video_stem, info in sample_items:
                logger.info(f"    {video_stem}:")
                logger.info(f"      Max persons: {info['max_persons']}")
                logger.info(f"      Source: {info['source_path']}")
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        logger.error("Cannot proceed without CSV configuration")
        return
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Visualization: {'Enabled' if config.visualization.enable_visualization else 'Disabled'}")
    logger.info(f"  Face verification: {'Enabled' if config.face_detection.enable_face_features else 'Disabled'}")
    logger.info(f"  Track filtering: {'Enabled' if config.track_filtering.filter_short_tracks else 'Disabled'}")
    logger.info(f"  Quality check: {'Enabled' if config.detection_quality.enable_quality_check else 'Disabled'}")
    
    # Find pose JSON files
    logger.info("\nFinding pose JSON files...")
    json_files = []
    for root, dirs, files in os.walk(pose_json_dir):
        for file in files:
            if file.endswith('_pose_data.json'):
                json_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(json_files)} pose JSON files")
    
    if not json_files:
        logger.error("No pose JSON files found!")
        return
    
    if max_videos:
        json_files = json_files[:max_videos]
        logger.info(f"Limited to first {max_videos} files")
    
    # Initialize pipeline
    logger.info("\nInitializing tracking pipeline...")
    pipeline = EnhancedTrackingPipeline(config)
    
    successful = 0
    failed = 0
    skipped = 0
    not_in_csv = 0
    video_not_found = 0
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING VIDEO PROCESSING")
    logger.info("=" * 70)
    
    for i, json_path in enumerate(json_files, 1):
        try:
            logger.info(f"\n[{i}/{len(json_files)}] Processing video")
            logger.info(f"JSON: {os.path.basename(json_path)}")
            
            json_stem = Path(json_path).stem.replace('_pose_data', '')
            
            # Get video info from CSV
            video_info = get_video_info_for_json(json_stem, max_persons_map)
            
            if video_info is None:
                logger.warning(f"Video not found in CSV: {json_stem}")
                logger.warning(f"  Skipping this video (no configuration)")
                not_in_csv += 1
                continue
            
            video_path = video_info['source_path']
            video_max_persons = video_info['max_persons']
            
            # Validate video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                logger.error(f"  Path specified in CSV SourceFile column")
                video_not_found += 1
                failed += 1
                continue
            
            logger.info(f"Video: {video_path}")
            logger.info(f"Max persons (from CSV): {video_max_persons}")
            
            # Set max persons for this video
            config.tracking.max_persons = video_max_persons
            
            # Setup output paths
            output_video_path = None
            if config.visualization.enable_visualization:
                output_video_path = os.path.join(output_dir, "videos", f"{json_stem}_tracked.mp4")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            output_json_path = os.path.join(output_dir, "tracking_data", f"{json_stem}_tracking.json")
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            
            # Check if already processed
            if skip_existing:
                video_exists = os.path.exists(output_video_path) if config.visualization.enable_visualization else True
                json_exists = os.path.exists(output_json_path)
                
                if video_exists and json_exists:
                    logger.info("SKIPPED - Already processed")
                    skipped += 1
                    continue
            
            # Process video
            pipeline.process_video(json_path, video_path, output_video_path, output_json_path)
            
            logger.info(f"SUCCESS")
            if output_video_path:
                logger.info(f"  Output video: {output_video_path}")
            logger.info(f"  Output JSON: {output_json_path}")
            
            successful += 1
            
            # Reset for next video
            pipeline.reset_for_next_video()
            
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user")
            break
            
        except Exception as e:
            logger.error(f"FAILED - Error processing")
            logger.error(f"  Error: {str(e)}", exc_info=True)
            failed += 1
            
            try:
                pipeline.reset_for_next_video()
            except Exception:
                pass
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files found:       {len(json_files)}")
    logger.info(f"Successfully processed:  {successful}")
    logger.info(f"Failed:                  {failed}")
    logger.info(f"  - Video not found:     {video_not_found}")
    logger.info(f"Skipped (existing):      {skipped}")
    logger.info(f"Not in CSV:              {not_in_csv}")
    logger.info("=" * 70)


def main():
    """Main entry point - CSV with SourceFile column is REQUIRED."""
    parser = argparse.ArgumentParser(
        description='Multi-Person Tracking System with CSV-Based Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format (REQUIRED - must include SourceFile column):
  SourceFile,FileName,#_adults,#_children,#_people_background
  /path/to/video1.mp4,video1.mp4,2,1,3
  /path/to/video2.mov,video2.mov,5,0,10+
  /path/to/video3.avi,video3.avi,1,2,x+

Notes:
  - SourceFile: Full path to the video file (can be any extension)
  - FileName: Video filename (used to match with pose JSON files)
  - The video extension in SourceFile can differ from FileName
  
Calculation:
  total_persons = adults + children + background
  
Special values:
  '10+'  10 (removes + suffix)
  'x+', 'many', 'crowd'  999 (large/unknown crowd)

Examples:
  Basic usage (no video-dir needed - paths from CSV):
    python tracking_system.py -p pose_data/ -o output/ -c counts.csv

  Without visualization (JSON only):
    python tracking_system.py -p pose_data/ -o output/ -c counts.csv --no-video

  Without face verification:
    python tracking_system.py -p pose_data/ -o output/ -c counts.csv --no-face
        """
    )
    
    # Required arguments (video-dir removed)
    parser.add_argument('-p', '--pose-json-dir', type=str, required=True,
                       help='Directory containing pose JSON files')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output directory for tracking results')
    parser.add_argument('-c', '--csv', type=str, required=True,
                       help='CSV file with SourceFile paths and max persons (REQUIRED)')
    
    # Optional arguments
    parser.add_argument('-n', '--max-videos', type=int, default=None,
                       help='Maximum number of videos to process')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess videos even if they already exist')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation, only generate tracking JSON')
    parser.add_argument('--no-face', action='store_true',
                       help='Disable face verification with DeepFace')
    parser.add_argument('--no-filter', action='store_true',
                       help='Disable short track filtering')
    parser.add_argument('--no-quality-check', action='store_true',
                       help='Disable initial detection quality filtering')
    
    # Model configuration
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to CLIP-ReID checkpoint file')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Number of classes in CLIP-ReID model')
    parser.add_argument('--camera-num', type=int, default=None,
                       help='Number of cameras in training dataset')
    parser.add_argument('--max-lost-frames', type=int, default=1500,
                       help='Maximum frames a track can be lost')
    parser.add_argument('--min-keypoints', type=int, default=15,
                       help='Minimum number of confident keypoints required')
    
    parser.add_argument('--no-id-remap', action='store_true',
                       help='Keep original track IDs (no reassignment)')
    parser.add_argument('--no-validation-filter', action='store_true',
                       help='Skip validation filtering (photos/screens)')
    parser.add_argument('--no-size-filter', action='store_true',
                       help='Skip size/aspect ratio filtering')
    parser.add_argument('--no-quality-filter', action='store_true',
                       help='Skip keypoint quality filtering')
    parser.add_argument('--no-person-classification', action='store_true',
                       help='Skip main vs background classification')
    parser.add_argument('--no-duration-filter', action='store_true',
                       help='Skip duration filtering')
    
    # Size constraint arguments
    parser.add_argument('--min-height', type=int, default=80,
                       help='Minimum average bbox height (default: 80)')
    parser.add_argument('--min-width', type=int, default=40,
                       help='Minimum average bbox width (default: 40)')
    parser.add_argument('--min-aspect-ratio', type=float, default=1.2,
                       help='Minimum height/width ratio (default: 1.2)')
    parser.add_argument('--max-aspect-ratio', type=float, default=4.0,
                       help='Maximum height/width ratio (default: 4.0)')
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.pose_json_dir):
        print(f"Error: Pose JSON directory does not exist: {args.pose_json_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file does not exist: {args.csv}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Create configuration
    config = Stage2Config()
    
    if args.checkpoint:
        config.clip_reid.checkpoint_path = args.checkpoint
    
    if args.num_classes:
        config.clip_reid.num_classes = args.num_classes
    
    if args.camera_num:
        config.clip_reid.camera_num = args.camera_num
    
    if args.max_lost_frames:
        config.tracking.tracker_config.max_lost_frames = args.max_lost_frames
    
    if args.min_keypoints:
        config.detection_quality.min_keypoints_required = args.min_keypoints
    
    if args.no_video:
        config.visualization.enable_visualization = False
    
    if args.no_face:
        config.face_detection.enable_face_features = False
    
    if args.no_filter:
        config.track_filtering.filter_short_tracks = False
    
    if args.no_quality_check:
        config.detection_quality.enable_quality_check = False
    if args.no_id_remap:
        config.track_filtering.enable_id_reassignment = False
    
    if args.no_validation_filter:
        config.track_filtering.enable_validation_filter = False
    if args.no_size_filter:
        config.track_filtering.enable_size_filter = False
    if args.no_quality_filter:
        config.track_filtering.enable_quality_filter = False
    if args.no_person_classification:
        config.track_filtering.enable_person_classification = False
    if args.no_duration_filter:
        config.track_filtering.enable_duration_filter = False
    config.track_filtering.min_avg_height = args.min_height
    config.track_filtering.min_avg_width = args.min_width
    config.track_filtering.min_aspect_ratio = args.min_aspect_ratio
    config.track_filtering.max_aspect_ratio = args.max_aspect_ratio
    
    process_videos_batch(
        pose_json_dir=args.pose_json_dir,
        output_dir=args.output,
        config=config,
        max_persons_csv=args.csv,
        max_videos=args.max_videos,
        skip_existing=not args.no_skip
    )


if __name__ == "__main__":
    main()