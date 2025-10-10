import os
import glob
import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional

import sys
mmpose_path: str = "mmpose"
if os.path.exists(mmpose_path) and mmpose_path not in sys.path:
    sys.path.insert(0, mmpose_path)

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import cv2
import ffmpeg
from tqdm import tqdm
import torch
import time
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from facenet_pytorch import MTCNN
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import signal
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

from deepface import DeepFace
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from filterpy.kalman import KalmanFilter
from tracking_exporter import TrackingDataCollector


from motion_pipeline import MultiPersonTrackingPipeline, PipelineConfig

def clean_data(df):
    """Clean common data inconsistencies"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" and col != "FileName":
            df[col] = df[col].astype(str).str.strip().str.lower()

    df['Location'] = df['Location'].replace({
        'outside oublic': 'outside public',
        'outside public ': 'outside public',
        'outside private ': 'outside private'
    })
    df['Context'] = df['Context'].replace({
        'other ': 'other',
        'toy play ': 'toy play',
        'general social interaction ': 'general social communication interaction'
    })
    df['Child_of_interest_clear'] = df['Child_of_interest_clear'].replace({
        'yes ': 'yes'
    })
    return df

def get_filename_list(df_subset):
    """Return filenames from a dataframe subset"""
    return df_subset['FileName'].tolist()

def create_video_groups(df):
    """Create video groups based on criteria"""
    df = clean_data(df)
    groups = {}

    # Ensure numeric columns
    df['#_children'] = pd.to_numeric(df['#_children'], errors='coerce')
    df['#_adults'] = pd.to_numeric(df['#_adults'], errors='coerce')
    df['Video_Quality_Child_Face_Visibility'] = pd.to_numeric(df['Video_Quality_Child_Face_Visibility'], errors='coerce')
    df['Video_Quality_Child_Body_Visibility'] = pd.to_numeric(df['Video_Quality_Child_Body_Visibility'], errors='coerce')
    df['Video_Quality_Lighting'] = pd.to_numeric(df['Video_Quality_Lighting'], errors='coerce')

    # gp2
    groups['children_morethan_1_condition'] = get_filename_list(
        df[(df['Video_Quality_Child_Face_Visibility'] >= 9) &
           (df['Video_Quality_Child_Body_Visibility'] >= 9) &
           (df['Video_Quality_Lighting'] >= 8) &   
           (df['#_children'] > 1)]  
    )
    
    # gp1
    groups['face_perfect'] = get_filename_list(df[df['Video_Quality_Child_Face_Visibility'] >= 9])
    groups['single_child_ideal'] = get_filename_list(
        df[(df['Video_Quality_Child_Face_Visibility'] >= 9) &
           (df['Video_Quality_Child_Body_Visibility'] >= 9) &
           (df['Video_Quality_Lighting'] >= 8) &   
           (df['#_children'] == 1)]
    )
    groups['toy_play_good_quality'] = get_filename_list(
        df[(df['Context'] == 'toy play') &
           (df['Video_Quality_Child_Face_Visibility'] >= 7) &
           (df['Video_Quality_Child_Body_Visibility'] >= 7)]
    )
    #add the gps
    return groups

def find_video_paths(video_filenames: List[str], base_directory: str) -> List[str]:
    """Find full paths for video filenames in directory structure"""
    found_videos = []
    
    for filename in video_filenames:
        # Search recursively for the filename
        pattern = os.path.join(base_directory, '**', filename)
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            found_videos.append(matches[0])  
        else:
            print(f"Warning: Video {filename} not found in {base_directory}")
    
    return found_videos

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging for batch processing"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"filtered_batch_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_paths(input_video_path: str, base_output_dir: str, group_name: str) -> tuple:
    """Create output paths maintaining directory structure"""
    input_path = Path(input_video_path)
    video_name = input_path.stem
    
    # Create group-specific directories
    output_video_dir = os.path.join(base_output_dir, "videos", group_name)
    output_json_dir = os.path.join(base_output_dir, "tracking_data", group_name)
    
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_video_dir, f"{video_name}_tracked.mp4")
    output_json_path = os.path.join(output_json_dir, f"{video_name}_tracking.json")
    
    return output_video_path, output_json_path

def create_batch_config(json_output_path: str) -> 'PipelineConfig':
    """Create configuration for batch processing"""
    config = PipelineConfig()
    
    # Optimize for batch processing
    config.features.enable_face_features = True
    config.features.feature_update_interval = 10
    
    config.processing.detection_confidence_threshold = 0.6
    config.processing.combined_reid_threshold = 0.7
    config.processing.max_lost_frames = 500
    
    # Visualization settings
    config.visualization.enable_visualization = True
    config.visualization.enable_pose_drawing = False
    config.visualization.enable_bbox_drawing = True
    config.visualization.enable_id_labels = True
    
    # Export settings
    config.export.enable_export = True
    config.export.output_path = json_output_path
    
    return config

def process_filtered_videos(excel_path: str, video_base_dir: str, output_dir: str, 
                          group_name: str = 'ideal_condition', max_videos: Optional[int] = None):
    """Process videos from a specific group"""
    
    logger = setup_logging(os.path.join(output_dir, "logs"))
    
    # Load Excel data and create groups
    logger.info(f"Loading video metadata from: {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"Excel shape: {df.shape}")
    
    # Create video groups
    video_groups = create_video_groups(df)
    
    if group_name not in video_groups:
        logger.error(f"Group '{group_name}' not found. Available groups: {list(video_groups.keys())}")
        return
    
    selected_filenames = video_groups[group_name]
    logger.info(f"Selected group '{group_name}' contains {len(selected_filenames)} videos")
    
    if max_videos:
        selected_filenames = selected_filenames[:max_videos]
        logger.info(f"Limited to first {max_videos} videos")
    
    # Find full paths for selected videos
    logger.info("Finding video file paths...")
    video_paths = find_video_paths(selected_filenames, video_base_dir)
    logger.info(f"Found {len(video_paths)} video files out of {len(selected_filenames)} requested")
    
    if not video_paths:
        logger.error("No video files found!")
        return
    
    # Process each video
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, video_path in enumerate(video_paths, 1):
        try:
            logger.info(f"Processing video {i}/{len(video_paths)}: {os.path.basename(video_path)}")
            logger.info(f"Full path: {video_path}")
            
            # Create output paths
            output_video_path, output_json_path = create_output_paths(
                video_path, output_dir, group_name
            )
            
            # Check if already processed
            if os.path.exists(output_video_path) and os.path.exists(output_json_path):
                logger.info(f"Skipping - already processed")
                skipped_count += 1
                continue
            
            # Create configuration
            config = create_batch_config(output_json_path)
            
            # Initialize pipeline
            logger.info("Initializing pipeline...")
            pipeline = MultiPersonTrackingPipeline(config)
            
            # Process video
            start_time = time.time()
            pipeline.process_video(video_path, output_video_path)
            processing_time = time.time() - start_time
            
            logger.info(f"uccessfully processed in {processing_time:.2f}s")
            logger.info(f"   Output video: {output_video_path}")
            logger.info(f"   Output JSON: {output_json_path}")
            
            successful_count += 1
            
            # Clean up memory
            del pipeline
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            break
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}")
            logger.error(f"   Error: {str(e)}")
            failed_count += 1
            continue
    
    # Final summary
    logger.info("FILTERED BATCH PROCESSING SUMMARY")
    logger.info(f"Group processed: {group_name}")
    logger.info(f"Videos in group: {len(selected_filenames)}")
    logger.info(f"Video files found: {len(video_paths)}")
    logger.info(f"Successfully processed: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Skipped (already processed): {skipped_count}")
    logger.info(f"Output directory: {output_dir}")

def main_filtered():
    """Main function """
    
    # paths
    EXCEL_PATH = "/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx"
    VIDEO_BASE_DIR = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external"
    OUTPUT_DIR = "/home/aparnabg/orcd/scratch/tracking/motion_pipeline"
    
    # Which group to process
    GROUP_NAME = 'children_morethan_1_condition' 
    
    # Optional: limit for testing
    MAX_VIDEOS = 15  # Set to None to process all videos in the group
    
    print(f"Starting filtered batch processing...")
    print(f"Excel file: {EXCEL_PATH}")
    print(f"Video directory: {VIDEO_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing group: {GROUP_NAME}")
    
    if MAX_VIDEOS:
        print(f"Processing limit: {MAX_VIDEOS} videos")
    
    process_filtered_videos(
        excel_path=EXCEL_PATH,
        video_base_dir=VIDEO_BASE_DIR,
        output_dir=OUTPUT_DIR,
        group_name=GROUP_NAME,
        max_videos=MAX_VIDEOS
    )

if __name__ == "__main__":
    main_filtered()