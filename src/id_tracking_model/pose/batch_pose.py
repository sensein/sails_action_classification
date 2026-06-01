#!/usr/bin/env python3
"""
Batch processing script for running cache_pose.py (DetectionPosePipeline) on multiple videos from CSV files.
Supports graceful shutdown and resuming from where it left off.
"""



import sys
import os
import csv
import signal
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import json
from datetime import datetime

from sailsprep.feature_processing.pose.cache_pose import DetectionPosePipeline, PipelineConfig, create_default_config


class BatchProcessor:
    """Batch processor for running detection/pose pipeline on multiple videos"""

    def __init__(self, csv_path: str, output_base_dir: str, base_video_dir: str, exp_id: Optional[str] = None, reuse_pipeline: bool = True, rmm: bool = False, start_row: int = 0, end_row: Optional[int] = None):
        self.csv_path = csv_path
        self.output_base_dir = output_base_dir
        self.base_video_dir = base_video_dir
        self.exp_id = exp_id
        self.reuse_pipeline = reuse_pipeline
        self.rmm = rmm
        self.start_row = start_row
        self.end_row = end_row
        self.config: PipelineConfig = create_default_config()

        # Configure for batch processing
        self.config.cache.enable_cache = True
        self.config.visualization.enable_visualization = False  # No video output by default
        self.config.frame_limit = 9000  # Process up to 9,000 frames per video (about 5 minutes at 30 FPS)
        self.config.detection_only = False
        
        self.interrupted = False
        self.current_video = None
        self.progress_file = None
        self.completed_videos = set()

        # Create subset name from CSV filename
        csv_name = Path(csv_path).stem
        self.subset_name = csv_name

        parent_dir = self.subset_name
        if self.exp_id:
            parent_dir += f"_{self.exp_id}"

        # Add row range to directory name to avoid conflicts between parallel jobs
        if self.start_row > 0 or self.end_row is not None:
            row_suffix = f"_rows{self.start_row}-{self.end_row if self.end_row is not None else 'end'}"
            parent_dir += row_suffix

        self.output_dir = os.path.join(output_base_dir, parent_dir)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Progress tracking file
        self.progress_file = os.path.join(self.output_dir, "processing_progress.json")
        self._load_progress()

        # Initialize pipeline once if reusing
        self.pipeline = None
        if self.reuse_pipeline:
            print("Initializing shared pipeline (models will be loaded once)...")
            self.pipeline = DetectionPosePipeline(self.config, batch_signal_handler=self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print(f"\n\nBatch processing interrupted! Cleaning up and saving progress...")
        self.interrupted = True

        # Clean up partial files from current video
        if self.current_video:
            print(f"Cleaning up partial cache files for: {self.current_video}")

            # Find the video info for current video
            videos = self._read_video_list()
            current_video_info = None
            for v in videos:
                if v['filename'] == self.current_video:
                    current_video_info = v
                    break

            if current_video_info:
                # Note: Cache cleanup would be handled by CacheManager
                # For now, just mark as not completed
                if self.current_video in self.completed_videos:
                    self.completed_videos.remove(self.current_video)

        # Save current progress
        self._save_progress()

        print(f"\nProgress saved to: {self.progress_file}")
        print(f"Completed videos: {len(self.completed_videos)}")
        if self.current_video:
            print(f"Interrupted during: {self.current_video}")
            print("You can resume processing by running the script again with the same parameters.")

        sys.exit(0)

    def _load_progress(self):
        """Load processing progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.completed_videos = set(progress_data.get('completed_videos', []))
                print(f"Resumed from previous session. {len(self.completed_videos)} videos already completed.")
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
                self.completed_videos = set()
        else:
            self.completed_videos = set()

    def _save_progress(self):
        """Save processing progress to file"""
        progress_data = {
            'completed_videos': list(self.completed_videos),
            'last_updated': datetime.now().isoformat(),
            'csv_path': self.csv_path,
            'output_dir': self.output_dir
        }

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def _read_video_list(self) -> List[Dict]:
        """Read video list from CSV file"""
        videos = []

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract video path info
                    source_file = row.get('SourceFile', '')
                    filename = row.get('FileName', '')
                    video_id = row.get('ID', '')
                    coder = row.get('Coder', '')

                    if source_file and filename:
                        videos.append({
                            'source_file': source_file,
                            'filename': filename,
                            'video_id': video_id,
                            'row_data': row,
                            'coder': coder,
                        })

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

        return videos

    def _convert_path(self, source_path: str) -> str:
        """Convert source path to actual file system path"""
        # Remove the /Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/ prefix
        # and replace with base_video_dir

        if self.rmm:
           return os.path.join(self.base_video_dir, source_path)

        if source_path.startswith('/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/'):
            relative_path = source_path.replace('/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/', '')
            full_path = os.path.join(self.base_video_dir, relative_path)
            return full_path
        else:
            # If path doesn't match expected format, try to extract relative part
            parts = source_path.split('/')
            if len(parts) >= 2:
                # Take last two parts as relative path
                relative_path = '/'.join(parts[-2:])
                full_path = os.path.join(self.base_video_dir, relative_path)
                return full_path

        return source_path

    def _create_output_path(self, video_info: Dict) -> str:
        """Create output path for cache (dummy path, actual cache managed by CacheManager)"""
        filename = video_info['filename']
        video_id = video_info['video_id']
        coder = video_info['coder']

        # Remove extension
        base_name = os.path.splitext(filename)[0]
        base = f"{video_id}_{coder}_{base_name}"

        # This is a dummy path - actual cache location is determined by CacheManager
        # based on video hash. The basename is used for cache metadata lookup.
        dummy_output = os.path.join(self.output_dir, "cache", f"{base}.mp4")

        return dummy_output

    def _check_cache_exists(self, source_path: str) -> bool:
        """Check if cache already exists for this video"""
        from sailsprep.feature_processing.utils.cache_manager import CacheManager

        # Create a temporary cache manager to check if cache exists
        dummy_output = os.path.join(self.output_dir, "cache", "temp.mp4")
        cache_mgr = CacheManager(
            output_video_path=dummy_output,
            detection_config=self.config.models.detection_config,
            pose_config=self.config.models.pose_config,
            detection_confidence_threshold=self.config.processing.detection_confidence_threshold,
            nms_type=self.config.processing.nms_type,
            nms_threshold=self.config.processing.nms_threshold,
            bbox_min_height=self.config.processing.bbox_min_height,
            bbox_min_width=self.config.processing.bbox_min_width,
            cache_base_path=self.config.cache.cache_base_path
        )

        # Check if both detection and pose caches exist
        return cache_mgr.check_detection_cache() and cache_mgr.check_pose_cache()

    def _process_single_video(self, video_info: Dict) -> bool:
        """Process a single video"""
        source_path = self._convert_path(video_info['source_file'])
        output_path = self._create_output_path(video_info)

        print(f"\nProcessing: {video_info['filename']}")
        print(f"Source: {source_path}")
        print(f"Cache will be stored in: {self.config.cache.cache_base_path}")

        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"ERROR: Source file not found: {source_path}")
            return False

        # Check if cache already exists
        if self._check_cache_exists(source_path):
            print(f"Cache already exists, skipping: {video_info['filename']}")
            return True

        try:
            # Setup pipeline
            if self.reuse_pipeline:
                # Pipeline is already initialized, just process
                pass
            else:
                # Initialize fresh pipeline for each video
                self.pipeline = DetectionPosePipeline(self.config, batch_signal_handler=self._signal_handler)

            # Process the video (this will create cache files)
            self.pipeline.process_video(source_path, output_path)

            print(f"✅ Successfully processed: {video_info['filename']}")
            return True

        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to be caught by signal handler
            raise
        except Exception as e:
            print(f"❌ Error processing {video_info['filename']}: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up pipeline to free memory (only if not reusing)
            if not self.reuse_pipeline and self.pipeline:
                del self.pipeline
                self.pipeline = None

                # Force garbage collection
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def process_all(self):
        """Process all videos in the CSV file"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"Starting batch processing for subset: {self.subset_name}")
        print(f"CSV file: {self.csv_path}")
        print(f"Cache directory: {self.config.cache.cache_base_path}")
        print(f"Base video directory: {self.base_video_dir}")

        # Read video list
        videos = self._read_video_list()
        if not videos:
            print("No videos found in CSV file!")
            return

        print(f"Found {len(videos)} videos in CSV")

        # Apply row range filtering
        if self.start_row > 0 or self.end_row is not None:
            original_count = len(videos)
            videos = videos[self.start_row:self.end_row]
            print(f"Row range filter [{self.start_row}:{self.end_row}]: {original_count} -> {len(videos)} videos")

        # Filter out already completed videos
        remaining_videos = [v for v in videos if v['filename'] not in self.completed_videos]
        print(f"Remaining to process: {len(remaining_videos)}")

        if not remaining_videos:
            print("All videos have already been processed!")
            return

        # Process each video
        start_time = time.time()
        processed_count = 0
        failed_count = 0

        for i, video_info in enumerate(remaining_videos):
            if self.interrupted:
                break

            self.current_video = video_info['filename']

            print(f"\n{'='*60}")
            print(f"Processing video {i+1}/{len(remaining_videos)}")
            print(f"Total progress: {len(self.completed_videos)}/{len(videos)}")
            print(f"{'='*60}")

            success = self._process_single_video(video_info)

            if success:
                self.completed_videos.add(video_info['filename'])
                processed_count += 1
            else:
                failed_count += 1

            # Save progress periodically
            if (i + 1) % 5 == 0:
                self._save_progress()

        # Final save
        self._save_progress()

        # Print summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Videos processed this session: {processed_count}")
        print(f"Videos failed this session: {failed_count}")
        print(f"Total completed videos: {len(self.completed_videos)}")
        print(f"Cache directory: {self.config.cache.cache_base_path}")
        print(f"Progress file: {self.progress_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch process videos to create detection/pose caches')
    parser.add_argument('csv_file', help='CSV file containing video list')
    parser.add_argument('--output-dir', default='/orcd/scratch/bcs/001/sensein/sails/feature_processing/pipeline_outputs/pose_cache',
                       help='Base output directory for progress tracking (default: pose_cache)')
    parser.add_argument('--video-dir', default='/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external',
                       help='Base directory containing source videos')
    parser.add_argument('--cache-dir', default='/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking',
                       help='Base directory for cache storage')
    parser.add_argument('--exp-id',
                       help='Optional experiment identifier to append to output directory name')
    parser.add_argument('--no-reuse-pipeline', action='store_true',
                       help='Create new pipeline for each video instead of reusing (slower but safer)')
    parser.add_argument('--rmm', action='store_true', default=True,
                       help='Use RMM dataset path conversion logic')
    parser.add_argument('--start-row', type=int, default=0,
                       help='Start processing from this row index in CSV (0-based, inclusive)')
    parser.add_argument('--end-row', type=int, default=None,
                       help='End processing at this row index in CSV (exclusive, None means process to end)')

    args = parser.parse_args()

    print("Batch Pose Cache Processing")
    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize and run batch processor
    processor = BatchProcessor(
        args.csv_file,
        args.output_dir,
        args.video_dir,
        args.exp_id,
        reuse_pipeline=not args.no_reuse_pipeline,
        rmm=args.rmm,
        start_row=args.start_row,
        end_row=args.end_row
    )

    # Set cache directory from command line
    processor.config.cache.cache_base_path = args.cache_dir

    processor.process_all()


if __name__ == "__main__":
    main()
