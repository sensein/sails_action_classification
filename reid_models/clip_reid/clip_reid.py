"""
Person Re-Identification Video Processing Pipeline

Combines YOLO detection, ByteTrack tracking, and CLIP-ReID for person re-identification in videos.
"""

import os
import sys
import cv2
import time
import random
import tempfile
import warnings
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import subprocess
import ffmpeg
warnings.filterwarnings("ignore")

sys.path.append('/home/aparnabg/reid/CLIP-ReID')

from config.defaults import _C as cfg
from model.make_model_clipreid import make_model


# CONFIGURATION

class Config:
    """Centralized configuration for the entire pipeline."""
    # Paths
    REPO_PATH = '/home/aparnabg/reid/CLIP-ReID'
    CHECKPOINT_PATH = '/home/aparnabg/reid/Market1501_clipreid_12x12sie_ViT-B-16_60.pth'
    VIDEO_PATH = '/home/aparnabg/reid/xxx.mp4'
    OUTPUT_VIDEO_PATH = '/home/aparnabg/reid/output.mp4'
    CONFIG_FILE = 'configs/person/vit_clipreid.yml'
    
    # Model parameters
    NUM_CLASSES = 751 #,msm 1041 occ 702 market 751
    CAMERA_NUM = 6 #msm 15 occ 8 market 6
    VIEW_NUM = 1
    EXPECTED_FEATURE_DIM = 768  # 768 for ViT-B-16, otherwise 1280
    
    # Re-ID parameters
    SIMILARITY_THRESHOLD = 0.8
    REID_INTERVAL = 5  # Process re-ID every N frames
    
    # Detection parameters
    DETECTION_CONF = 0.5
    DETECTION_IOU = 0.5
    DETECTION_IMGSZ = 640
    PERSON_CLASS_ID = 0
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DETECTION_DEVICE = 'cuda:0'
    
    @staticmethod
    def print_config():
        """Print current configuration settings to the console."""
        print("CONFIGURATION")
        print(f"Device: {Config.DEVICE}")
        print(f"Detection Device: {Config.DETECTION_DEVICE}")
        print(f"Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
        print(f"ReID Interval: {Config.REID_INTERVAL} frames")



# MODEL INITIALIZATION

class ModelLoader:
    """Handles loading and configuring the CLIP-ReID model and its settings."""
    
    @staticmethod
    def load_config() -> None:
        """
        Load and configure CLIP-ReID model settings from a YAML file.
        
        It reads the base configuration, fixes a formatting issue, saves it to a temporary file,
        merges it with global config, and updates settings based on Config class constants.
        """
        config_file_path = os.path.join(Config.REPO_PATH, Config.CONFIG_FILE)
        
        # Read and preprocess config file
        with open(config_file_path, 'r') as f:
            config_str = f.read()
        
        # Fix config format
        config_str = config_str.replace(
            "DATASETS:\n#   NAMES: ('market1501')\n#   ROOT_DIR: ('')\n# OUTPUT_DIR: ''",
            "DATASETS:\n  NAMES: 'market1501'\n  ROOT_DIR: ''\n "
        )
        
        # Save to temporary file and merge
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_file:
            temp_file.write(config_str)
            temp_file.flush()
            temp_config_path = temp_file.name
        
        cfg.merge_from_file(temp_config_path)
        
        # Update configuration
        cfg.defrost()
        cfg.MODEL.DEVICE = str(Config.DEVICE)
        cfg.MODEL.Transformer_TYPE = 'ViT-B-16'
        cfg.MODEL.NAME = cfg.MODEL.Transformer_TYPE
        cfg.MODEL.PRETRAIN_PATH = Config.CHECKPOINT_PATH
        cfg.INPUT.SIZE_TRAIN = [256, 128]
        cfg.INPUT.SIZE_TEST = [256, 128]
        cfg.MODEL.SIE_CAMERA = True
        cfg.MODEL.SIE_VIEW = True
        cfg.MODEL.STRIDE_SIZE = [12, 12]
        cfg.TEST.WEIGHT = Config.CHECKPOINT_PATH
        cfg.freeze()
        
        print(f"Config loaded from: {config_file_path}")
    
    @staticmethod
    def build_reid_model():
        """
        Build and load the ReID model with pretrained weights.
        
        The model is moved to the configured device and set to evaluation mode.
        It attempts to compile the model for performance optimization on PyTorch 2.0+.
        
        Returns:
            torch.nn.Module: The loaded and configured ReID model.
        """
        reid_model = make_model(
            cfg,
            num_class=Config.NUM_CLASSES,
            camera_num=Config.CAMERA_NUM,
            view_num=Config.VIEW_NUM
        )
        
        try:
            reid_model.load_param(cfg.TEST.WEIGHT)
            print(f"Checkpoint loaded from: {cfg.TEST.WEIGHT}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
        
        reid_model.to(cfg.MODEL.DEVICE)
        reid_model.eval()
        
        # Compile model for optimization (PyTorch 2.0+)
        pt_version = torch.__version__.split('+')[0]
        major, minor = map(int, pt_version.split('.')[:2])
        if major >= 2:
            reid_model = torch.compile(reid_model, mode="reduce-overhead")
            print("Model compiled with torch.compile()")
        
        return reid_model
    
    @staticmethod
    def get_transforms():
        """
        Get the image transformation pipeline components for ReID inference.
        
        Returns:
            Tuple[transforms.Normalize, transforms.ToTensor, Tuple[int, int]]: 
                A tuple containing the normalize transform, the ToTensor transform, 
                and the expected input size (height, width).
        """
        normalize = transforms.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD
        )
        to_tensor = transforms.ToTensor()
        input_size = tuple(cfg.INPUT.SIZE_TEST)
        
        print(f"ReID Input Size (H, W): {input_size}")
        return normalize, to_tensor, input_size


# FEATURE EXTRACTION


class FeatureExtractor:
    """Handles feature extraction for re-identification using the loaded model."""
    
    def __init__(self, model, normalize_transform, to_tensor_transform, input_size, device):
        """
        Initializes the FeatureExtractor.
        
        Args:
            model (torch.nn.Module): The loaded ReID model.
            normalize_transform (transforms.Normalize): Image normalization transform.
            to_tensor_transform (transforms.ToTensor): Image to tensor conversion transform.
            input_size (Tuple[int, int]): Expected input size (height, width).
            device (torch.device): The device (CPU/CUDA) for running the model.
        """
        self.model = model
        self.normalize = normalize_transform
        self.to_tensor = to_tensor_transform
        self.input_size_wh = (input_size[1], input_size[0])  # (width, height) for cv2
        self.device = device
    
    def extract_batch(self, np_crops_rgb: List[np.ndarray]) -> Optional[torch.Tensor]:
        """
        Extract feature vectors for a batch of cropped NumPy RGB images.
        
        Args:
            np_crops_rgb (List[np.ndarray]): List of NumPy arrays (H, W, C) in RGB format.
            
        Returns:
            Optional[torch.Tensor]: Feature tensor of shape (N, feature_dim) or None if error or empty input.
        """
        if not np_crops_rgb:
            return None
        
        try:
            tensors = []
            target_w, target_h = self.input_size_wh
            
            for crop in np_crops_rgb:
                # Resize using cv2 (faster than PIL)
                resized_crop = cv2.resize(
                    crop,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Convert to tensor and normalize
                tensor = self.to_tensor(resized_crop)
                tensor = self.normalize(tensor)
                tensors.append(tensor.unsqueeze(0))
            
            # Concatenate and move to GPU with non-blocking transfer
            input_batch = torch.cat(tensors, dim=0).to(self.device, non_blocking=True)
            
            # Use mixed precision for faster inference
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    features = self.model(input_batch)
            
            return features
            
        except Exception as e:
            print(f"Error in batch feature extraction: {e}")
            return None



# DATABASE MANAGEMENT


class ReIDDatabase:
    """
    Manages the gallery of known person features.
    
    Supports multi-shot storage (multiple features per person ID).
    """
    
    def __init__(self, feature_dim: int, device: torch.device, max_features_per_person: int = 5):
        """
        Initializes the ReIDDatabase.
        
        Args:
            feature_dim (int): The dimensionality of the feature vectors.
            device (torch.device): The device (CPU/CUDA) where features should be stored.
            max_features_per_person (int): Maximum number of features to store per person ID.
        """
        self.feature_dim = feature_dim
        self.device = device
        self.max_features_per_person = max_features_per_person
        
        # Multi-shot storage: person_id -> list of feature tensors
        self.person_features = {}  # {id: [feature1, feature2, ...]}
        self.person_frame_counts = {}  # {id: frame_count} for feature update logic
        self.ids = []  # List of all person IDs, determines gallery order
        self.next_id = 0
        
    def get_unique_id(self) -> int:
        """
        Generate a unique person ID.
        
        Returns:
            int: A new, unique person identifier.
        """
        id_set = set(self.ids)
        while self.next_id in id_set:
            self.next_id += 1
        return self.next_id
    
    def add_person(self, feature: torch.Tensor, frame_count: int = 0) -> int:
        """
        Add a new person to the database with an initial feature.
        
        Args:
            feature (torch.Tensor): The feature vector (1, D) for the new person.
            frame_count (int): The frame number when the person was first seen.
            
        Returns:
            int: The newly assigned unique person ID.
        """
        person_id = self.get_unique_id()
        
        # Ensure feature has correct shape (1, D)
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        
        # Initialize with first feature
        self.person_features[person_id] = [feature.clone()]
        self.person_frame_counts[person_id] = frame_count
        self.ids.append(person_id)
        self.next_id = person_id + 1
        
        print(f"    Added person ID {person_id} (initial feature)")
        return person_id
    
    def add_feature_to_person(self, person_id: int, feature: torch.Tensor, frame_count: int):
        """
        Add a new feature observation for an existing person, managing the max feature limit.
        
        If the limit is exceeded, the oldest feature is removed (FIFO).
        
        Args:
            person_id (int): The ID of the person to update.
            feature (torch.Tensor): The new feature vector (1, D).
            frame_count (int): The frame number of the new observation.
        """
        if person_id not in self.person_features:
            print(f"    Warning: Person ID {person_id} not in database")
            return
        
        # Ensure feature has correct shape (1, D)
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        
        features_list = self.person_features[person_id]
        
        # Strategy: Keep most recent features (remove oldest if max reached)
        if len(features_list) >= self.max_features_per_person:
            # Remove oldest feature (at index 0)
            features_list.pop(0)
        
        # Add new feature
        features_list.append(feature.clone())
        self.person_frame_counts[person_id] = frame_count
        
        print(f"    Updated ID {person_id} features (now {len(features_list)}/{self.max_features_per_person})")
    
    # NOTE: get_gallery_tensor and get_feature_to_id_mapping were only used by the removed PersonMatcher.match method,
    #       but are kept as they represent core database functionality that might be useful for debugging or future extension.
    def get_gallery_tensor(self) -> torch.Tensor:
        """
        Get all stored features as a single tensor for batch comparison.
        
        Returns:
            torch.Tensor: Tensor of shape (total_features, feature_dim).
        """
        all_features = []
        for person_id in self.ids:
            all_features.extend(self.person_features[person_id])
        
        if not all_features:
            return torch.empty((0, self.feature_dim), device=self.device)
        
        return torch.cat(all_features, dim=0)
    
    def get_feature_to_id_mapping(self) -> list:
        """
        Get mapping from gallery index (in get_gallery_tensor) to person ID.
        
        Returns:
            list: List where index is gallery position, value is person_id.
        """
        mapping = []
        for person_id in self.ids:
            num_features = len(self.person_features[person_id])
            mapping.extend([person_id] * num_features)
        return mapping
    
    def size(self) -> int:
        """
        Return the number of unique persons (IDs) in the database.
        
        Returns:
            int: Number of unique person IDs.
        """
        return len(self.ids)
    
    def total_features(self) -> int:
        """
        Return the total number of stored features across all persons.
        
        Returns:
            int: Total number of features in the gallery.
        """
        return sum(len(features) for features in self.person_features.values())
    
    def ensure_dtype_match(self, query_dtype: torch.dtype):
        """
        Ensure all stored features match the query dtype (e.g., for mixed precision).
        
        Args:
            query_dtype (torch.dtype): The desired data type for the stored features.
        """
        for person_id in self.person_features:
            for i, feature in enumerate(self.person_features[person_id]):
                if feature.dtype != query_dtype:
                    self.person_features[person_id][i] = feature.to(dtype=query_dtype)



# PERSON MATCHING


class PersonMatcher:
    """
    Handles person matching logic, including similarity computation and assignment.
    
    Uses Hungarian algorithm for optimal assignment between current tracks and database IDs.
    """
    
    def __init__(self, similarity_threshold: float):
        """
        Initializes the PersonMatcher.
        
        Args:
            similarity_threshold (float): The minimum cosine similarity required for a match.
        """
        self.similarity_threshold = similarity_threshold
    
    def compute_person_similarity(
        self,
        query_feature: torch.Tensor,
        person_features_list: List[torch.Tensor]
    ) -> float:
        """
        Compute similarity between a query feature and a person ID with multiple stored features.
        
        This uses a max-pooling approach, returning the highest cosine similarity
        against any stored feature for that person.
        
        Args:
            query_feature (torch.Tensor): Single query feature (1, D).
            person_features_list (List[torch.Tensor]): List of features for one person.
            
        Returns:
            float: Maximum similarity score found.
        """
        if not person_features_list:
            return 0.0
        
        # Stack all person features
        person_features_tensor = torch.cat(person_features_list, dim=0)  # (N, D)
        
        # Compute similarity with all features
        similarities = F.cosine_similarity(
            query_feature,
            person_features_tensor,
            dim=1
        )
        
        # Return max similarity (best match among all stored features)
        return similarities.max().item()
    
    def match_batch_with_hungarian(
        self,
        query_features: torch.Tensor,
        track_ids: List[int],
        database: ReIDDatabase,
        frame_count: int,
    ) -> Dict[int, int]:
        """
        Match a batch of features (current tracks) to the database IDs using the 
        Hungarian algorithm (optimal assignment) based on maximum similarity.
        
        Assignments below the similarity threshold are treated as no-match.
        
        Args:
            query_features (torch.Tensor): Batch of feature vectors (N, D) from current tracks.
            track_ids (List[int]): List of track IDs corresponding to the query features.
            database (ReIDDatabase): ReID database with multi-feature storage.
            frame_count (int): Current frame number.
            
        Returns:
            Dict[int, int]: Dictionary mapping {track_id: reid_id}.
        """
        num_queries = query_features.shape[0]
        assignments = {}
        
        # Handle empty database
        if database.size() == 0:
            print(f"  Database empty. Adding {num_queries} new persons...")
            for i, track_id in enumerate(track_ids):
                reid_id = database.add_person(query_features[i:i+1], frame_count)
                assignments[track_id] = reid_id
            return assignments
        
        # Ensure dtype matches
        query_features = query_features.to(device=database.device)
        database.ensure_dtype_match(query_features.dtype)
        
        print(f"Computing similarity matrix ({num_queries} queries vs {database.size()} persons)...")
        
        # Build similarity matrix: (num_queries, num_persons)
        similarity_matrix = torch.zeros((num_queries, database.size()))
        
        for q_idx in range(num_queries):
            query = query_features[q_idx:q_idx+1]
            
            for p_idx, person_id in enumerate(database.ids):
                person_features = database.person_features[person_id]
                sim = self.compute_person_similarity(query, person_features)
                similarity_matrix[q_idx, p_idx] = sim
        
        print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # Convert similarity to cost for Hungarian algorithm
        cost_matrix = 1.0 - similarity_matrix.cpu().numpy()
        
        # Apply threshold: set cost to a very high value for similarities below threshold
        cost_matrix[similarity_matrix.cpu().numpy() < self.similarity_threshold] = 1e6

        
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Hungarian algorithm finds minimum cost assignment
            query_indices, person_indices = linear_sum_assignment(cost_matrix)
            
            # Process assignments
            
            for q_idx, p_idx in zip(query_indices, person_indices):
                track_id = track_ids[q_idx]
                similarity = similarity_matrix[q_idx, p_idx].item()
                
                # Check if assignment is valid (passes threshold 1e6 check)
                if similarity >= self.similarity_threshold:
                    reid_id = database.ids[p_idx]
                    assignments[track_id] = reid_id
                    
                    # Update person's feature gallery
                    database.add_feature_to_person(reid_id, query_features[q_idx:q_idx+1], frame_count)
                    
                    print(f" Track {track_id}: Matched to ID {reid_id}, sim={similarity:.4f}")
            
            # Assign new IDs to unmatched queries
            for q_idx in range(num_queries):
                track_id = track_ids[q_idx]
                if track_id not in assignments:
                    reid_id = database.add_person(query_features[q_idx:q_idx+1], frame_count)
                    assignments[track_id] = reid_id
                    
                    best_sim = similarity_matrix[q_idx].max().item()
                    print(f"  Track {track_id}: No match (best_sim={best_sim:.4f}). New ID: {reid_id}")
        
        except ImportError:
            print("  scipy not available. Falling back to greedy assignment...")
            assignments = self._greedy_assignment(
                similarity_matrix, query_features, track_ids, database, frame_count
            )
        
        return assignments
    
    def _greedy_assignment(
        self,
        similarity_matrix: torch.Tensor,
        query_features: torch.Tensor,
        track_ids: List[int],
        database: ReIDDatabase,
        frame_count: int
    ) -> Dict[int, int]:
        """
        Fallback greedy assignment logic if scipy is not available for Hungarian algorithm.
        
        It prioritizes the highest similarity matches first.
        
        Args:
            similarity_matrix (torch.Tensor): Similarity matrix (N_queries, N_persons).
            query_features (torch.Tensor): Batch of feature vectors (N, D).
            track_ids (List[int]): List of track IDs corresponding to the query features.
            database (ReIDDatabase): ReID database.
            frame_count (int): Current frame number.
            
        Returns:
            Dict[int, int]: Dictionary mapping {track_id: reid_id}.
        """
        assignments = {}
        num_queries, num_persons = similarity_matrix.shape
        
        # Create (similarity, query_idx, person_idx) tuples
        candidates = []
        for q in range(num_queries):
            for p in range(num_persons):
                sim = similarity_matrix[q, p].item()
                if sim >= self.similarity_threshold:
                    candidates.append((sim, q, p))
        
        # Sort by similarity (highest first)
        candidates.sort(reverse=True)
        
        assigned_queries = set()
        assigned_persons = set()
        
        # Greedy assignment
        for sim, q_idx, p_idx in candidates:
            if q_idx not in assigned_queries and p_idx not in assigned_persons:
                track_id = track_ids[q_idx]
                reid_id = database.ids[p_idx]
                assignments[track_id] = reid_id
                assigned_queries.add(q_idx)
                assigned_persons.add(p_idx)
                
                database.add_feature_to_person(reid_id, query_features[q_idx:q_idx+1], frame_count)
                print(f"  Track {track_id}: Matched to ID {reid_id}, sim={sim:.4f}")
        
        # Assign new IDs to unmatched
        for q_idx, track_id in enumerate(track_ids):
            if q_idx not in assigned_queries:
                reid_id = database.add_person(query_features[q_idx:q_idx+1], frame_count)
                assignments[track_id] = reid_id
                best_sim = similarity_matrix[q_idx].max().item()
                print(f"  Track {track_id}: No match (best_sim={best_sim:.4f}). New ID: {reid_id}")
        
        return assignments


# VIDEO PROCESSING


class VideoProcessor:
    """Main video processing pipeline orchestrating detection, tracking, and ReID."""
    
    def __init__(self):
        """Initializes the processor with necessary components set to None."""
        self.detection_model = None
        self.reid_model = None
        self.feature_extractor = None
        self.database = None
        self.matcher = None
        self.track_to_reid = {}  # Map track_id (from ByteTrack) -> reid_id (from database)
        self.id_colors = {}      # Map reid_id -> color
        
    def setup(self):
        """Initialize all models (YOLO, CLIP-ReID) and components (extractor, database, matcher)."""
        print("INITIALIZING SYSTEM")

        
        # Load configuration
        ModelLoader.load_config()
        Config.print_config()
        
        # Build ReID model
        print("Building ReID model...")
        self.reid_model = ModelLoader.build_reid_model()
        normalize, to_tensor, input_size = ModelLoader.get_transforms()
        
        # Setup feature extractor
        self.feature_extractor = FeatureExtractor(
            self.reid_model,
            normalize,
            to_tensor,
            input_size,
            Config.DEVICE
        )
        
        # Initialize database
        self.database = ReIDDatabase(Config.EXPECTED_FEATURE_DIM, Config.DEVICE)
        print(f"Database initialized (feature_dim={Config.EXPECTED_FEATURE_DIM})")
        
        # Initialize matcher
        # Removed K1, K2, LAMBDA_VALUE from initialization
        self.matcher = PersonMatcher(
            Config.SIMILARITY_THRESHOLD
        )
        print("Matcher initialized")
        
        # Load detection model
        print("Loading YOLO detection model...")
        self.detection_model = YOLO('/home/aparnabg/reid/yolo11m.pt')
        self.detection_model.to(Config.DETECTION_DEVICE)
        print("Detection model loaded")
        
        # Setup color mapping
        random.seed(42)
        print("\nSystem ready\n")
    
    def get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        """
        Get a consistent BGR color for a given ReID person ID.
        
        Args:
            person_id (int): The unique ReID identifier.
            
        Returns:
            Tuple[int, int, int]: A BGR color tuple.
        """
        if person_id not in self.id_colors:
            self.id_colors[person_id] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
        return self.id_colors[person_id]
    
    
    def process_video(self):
        """
        video processing loop.
        
        Reads the video frame by frame, runs YOLO tracking, performs ReID periodically,
        and writes the annotated frames to an output video file using FFmpeg for proper encoding.
        """
        # Open video
        cap = cv2.VideoCapture(Config.VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video {Config.VIDEO_PATH}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {Config.VIDEO_PATH}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}")
        print(f"Output: {Config.OUTPUT_VIDEO_PATH}\n")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",              # input from stdin
            "-an",                  # no audio
            "-c:v", "libx264",      # H.264 codec
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "18",
            Config.OUTPUT_VIDEO_PATH
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
        frame_count = 0
        last_time = time.time()
        reid_fps_display = 0
    
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                loop_time = current_time - last_time
                last_time = current_time
                fps_display = 1.0 / loop_time if loop_time > 0 else 0
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                tracking_results = self.detection_model.track(
                    frame_rgb,
                    imgsz=Config.DETECTION_IMGSZ,
                    conf=Config.DETECTION_CONF,
                    iou=Config.DETECTION_IOU,
                    classes=[Config.PERSON_CLASS_ID],
                    device=Config.DETECTION_DEVICE,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )[0]
    
                reid_processed = False
                current_tracks = {}
    
                if tracking_results.boxes.id is not None:
                    boxes = tracking_results.boxes.xyxy.cpu().numpy()
                    track_ids = tracking_results.boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        current_tracks[track_id] = box
    
                    if frame_count % Config.REID_INTERVAL == 0:
                        reid_processed = True
                        reid_start = time.time()
                        crops, track_ids_list = [], []
    
                        for track_id, box in current_tracks.items():
                            x1, y1, x2, y2 = map(int, box)
                            crop = frame_rgb[y1:y2, x1:x2]
                            if crop.shape[0] < 1 or crop.shape[1] < 1:
                                continue
                            crops.append(crop)
                            track_ids_list.append(track_id)
    
                        if crops:
                            features = self.feature_extractor.extract_batch(crops)
                            if features is not None:
                                self.database.ensure_dtype_match(features.dtype)
                                assignments = self.matcher.match_batch_with_hungarian(
                                    features, track_ids_list, self.database, frame_count=frame_count
                                )
                                for track_id, reid_id in assignments.items():
                                    self.track_to_reid[track_id] = reid_id
    
                        reid_time = time.time() - reid_start
                        reid_fps_display = 1.0 / reid_time if reid_time > 0 else 0
    
                if tracking_results.boxes.id is not None:
                    for box, track_id in zip(boxes, track_ids):
                        if track_id in self.track_to_reid:
                            reid_id = self.track_to_reid[track_id]
                            x1, y1, x2, y2 = map(int, box)
                            color = self.get_color_for_id(reid_id)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"ID {reid_id}"
                            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            label_y = max(y1 - 10, label_size[1] + 10)
                            cv2.rectangle(frame, (x1, label_y - label_size[1] - baseline),
                                          (x1 + label_size[0], label_y + baseline), color, cv2.FILLED)
                            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    

    
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break
    
                frame_count += 1
    
        finally:
            cap.release()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode(errors="ignore")
                raise RuntimeError(f"FFmpeg failed (code {rc}).\n{err}")
    
        print("PROCESSING COMPLETE")
        print(f"Output saved: {Config.OUTPUT_VIDEO_PATH}")
        print(f"Total frames processed: {frame_count}")
        print(f"Unique persons identified: {len(self.database.ids)}")
        print(f"Database size: {self.database.total_features()} features")
    



# MAIN FUNCTION

'''video_chunks = [
    {'input_path': '/path/to/video1.mp4', 'output_path': '/output/video1.mp4'},
    {'input_path': '/path/to/video2.mp4', 'output_path': '/output/video2.mp4'},
]'''

csv_file = "/home/aparnabg/orcd/scratch/tracking/clip_reid2/muti_child_Facebodyhighvisi_andlight_1ormoreadult.csv"
muti_child_df = pd.read_csv(csv_file)
old_base = '/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/'
new_input_base = '/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external/'
output_base = '/home/aparnabg/orcd/scratch/tracking/clip_reid2/output_Market1501_clipreid'

video_chunks = []
for _, row in muti_child_df.iterrows():
    source_file = row['SourceFile']
    filename = row['FileName']
    
    input_path = source_file.replace(old_base, new_input_base)
    

    video_chunks.append({
        'input_path': input_path,
        'output_path': os.path.join(output_base, f"{os.path.splitext(filename)[0]}.mp4")
    })


os.makedirs(output_base, exist_ok=True)
for i, video_info in enumerate(video_chunks, 1):
    print(f"PROCESSING VIDEO {i}/{len(video_chunks)}")
    
    Config.VIDEO_PATH = video_info['input_path']
    Config.OUTPUT_VIDEO_PATH = video_info['output_path']
    
    processor = VideoProcessor()
    processor.setup()
    
    try:
        processor.process_video()
        print(f"\n Video {i} completed")
    except Exception as e:
        print(f"\n Error: {e}")