#!/usr/bin/env python3
"""
Batch processing script for running tracker_clip_new.py (MultiPersonTrackingPipeline) on multiple videos from CSV files.
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

from sailsprep.id_tracking_model.tracker.clip.tracker_clip_new import MultiPersonTrackingPipeline, PipelineConfig, create_batch_config
from sailsprep.id_tracking_model.utils.tracking_exporter_new import TrackingDataCollector


class BatchTracker:
    """Batch processor for running tracking pipeline on multiple videos"""

    def __init__(self, csv_path: str, output_base_dir: str, base_video_dir: str,
                 exp_id: Optional[str] = None, reuse_pipeline: bool = True,
                 rmm: bool = False, enable_visualization: bool = True,
                 filter_ids: Optional[List[str]] = None, start_row: int = 0,
                 end_row: Optional[int] = None):
        self.csv_path = csv_path
        self.output_base_dir = output_base_dir
        self.base_video_dir = base_video_dir
        self.exp_id = exp_id
        self.reuse_pipeline = reuse_pipeline
        self.rmm = rmm
        self.filter_ids = set(filter_ids) if filter_ids else None
        self.start_row = start_row
        self.end_row = end_row

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

        # Configure pipeline
        json_output_path = os.path.join(self.output_dir, "tracking_results.json")
        self.config: PipelineConfig = create_batch_config(json_output_path)
        self.config.visualization.enable_visualization = enable_visualization

        self.interrupted = False
        self.current_video = None
        self.progress_file = None
        self.completed_videos = set()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Progress tracking file
        self.progress_file = os.path.join(self.output_dir, "processing_progress.json")
        self._load_progress()

        # Initialize pipeline once if reusing
        self.pipeline = None
        if self.reuse_pipeline:
            print("Initializing shared pipeline (models will be loaded once)...")
            self.pipeline = MultiPersonTrackingPipeline(
                self.config,
                batch_signal_handler=self._signal_handler
            )

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print(f"\n\nBatch processing interrupted! Cleaning up and saving progress...")
        self.interrupted = True

        # Clean up partial files from current video
        if self.current_video:
            print(f"Cleaning up partial files for: {self.current_video}")

            # Find the video info for current video
            videos = self._read_video_list()
            current_video_info = None
            for v in videos:
                if v['filename'] == self.current_video:
                    current_video_info = v
                    break

            if current_video_info:
                vid_output_path, tracking_output_path = self._create_output_paths(current_video_info)

                cleanup_files = []
                if tracking_output_path and os.path.exists(tracking_output_path):
                    cleanup_files.append(tracking_output_path)
                if vid_output_path and os.path.exists(vid_output_path):
                    cleanup_files.append(vid_output_path)

                for filepath in cleanup_files:
                    try:
                        os.remove(filepath)
                        print(f"  Removed partial file: {filepath}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {filepath}: {e}")

                # Ensure current video is NOT in completed list
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
        """Read video list from CSV file, optionally filtered by IDs"""
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
                    if not coder:
                        coder = row.get("Original_Coder", "")

                    # Skip if filtering by IDs and this ID is not in the filter list
                    if self.filter_ids and video_id not in self.filter_ids:
                        continue

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

    def _create_output_paths(self, video_info: Dict) -> tuple:
        """Create output paths for tracked video and tracking data

        Note: Basename must match batch_pose.py cache metadata naming.
        CacheManager.from_metadata() uses Path(output_path).stem to find cached data.
        """
        filename = video_info['filename']
        video_id = video_info['video_id']
        coder = video_info['coder']

        # Remove extension
        base_name = os.path.splitext(filename)[0]
        # This pattern MUST match batch_pose.py for cache lookup
        base = f"{video_id}_{coder}_{base_name}"

        output_filename = f"{base}.mp4"
        tracking_basename = f"{base}_tracking"

        # Create subdirectories
        tracking_dir = os.path.join(self.output_dir, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)

        # Only create video directory if visualization is enabled
        video_path = None
        if self.config.visualization.enable_visualization:
            video_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, output_filename)

        tracking_path = os.path.join(tracking_dir, tracking_basename)

        return video_path, tracking_path

    def _check_output_exists(self, output_path: str) -> bool:
        """Check if tracked video already exists"""
        return os.path.exists(output_path)

    def _process_single_video(self, video_info: Dict) -> bool:
        """Process a single video"""
        source_path = self._convert_path(video_info['source_file'])
        vid_output_path, tracking_output_path = self._create_output_paths(video_info)

        print(f"\nProcessing: {video_info['filename']}")
        print(f"Source: {source_path}")
        if vid_output_path:
            print(f"Video Output: {vid_output_path}")
        print(f"Tracking Output: {tracking_output_path}")

        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"ERROR: Source file not found: {source_path}")
            # Try with .mp4 extension
            source_path = source_path[:-4] + ".mp4"
            if os.path.exists(source_path):
                print(f"Found converted video file: {source_path}")
            else:
                print(f"ERROR: Converted video file not found: {source_path}")
                return False

        # Check if tracking output already exists
        if os.path.exists(tracking_output_path):
            print(f"Tracking file already exists, skipping: {tracking_output_path}")
            return True

        try:
            # Setup pipeline
            if self.reuse_pipeline:
                # Reset state for new video
                self.pipeline.reset_for_next_video()
                self.config.export.output_path = tracking_output_path
                # Create fresh data collector for this video
                self.pipeline.data_collector = TrackingDataCollector(enable_hdf5=self.config.export.export_hdf5) if self.config.export.enable_export else None
            else:
                # Initialize fresh pipeline for each video
                self.config.export.output_path = tracking_output_path
                self.pipeline = MultiPersonTrackingPipeline(
                    self.config,
                    batch_signal_handler=self._signal_handler
                )

            # Process the video
            self.pipeline.process_video(source_path, vid_output_path)

            print(f"✅ Successfully processed: {video_info['filename']}")
            return True

        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to be caught by signal handler
            raise
        except Exception as e:
            print(f"❌ Error processing {video_info['filename']}: {e}")
            import traceback
            traceback.print_exc()

            # Clean up partial output files
            cleanup_files = [tracking_output_path]
            if vid_output_path:
                cleanup_files.append(vid_output_path)

            for filepath in cleanup_files:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"Cleaned up partial output file: {filepath}")
                    except:
                        pass

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

        print(f"Starting batch tracking for subset: {self.subset_name}")
        print(f"CSV file: {self.csv_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Cache directory: {self.config.cache.cache_base_path}")
        print(f"Base video directory: {self.base_video_dir}")
        if self.filter_ids:
            print(f"Filtering by IDs: {sorted(self.filter_ids)}")

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
        print("BATCH TRACKING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Videos processed this session: {processed_count}")
        print(f"Videos failed this session: {failed_count}")
        print(f"Total completed videos: {len(self.completed_videos)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Progress file: {self.progress_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch process videos with person tracking and re-identification')
    parser.add_argument('csv_file', help='CSV file containing video list')
    parser.add_argument('--output-dir', default='/orcd/scratch/bcs/001/sensein/sails/feature_processing/pipeline_outputs',
                       help='Base output directory for tracked videos and results')
    parser.add_argument('--video-dir', default='/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized',
                       help='Base directory containing source videos')
    parser.add_argument('--cache-dir', default='/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking',
                       help='Base directory for pose cache (must already exist from cache_pose.py)')
    parser.add_argument('--exp-id',
                       help='Optional experiment identifier to append to output directory name')
    parser.add_argument('--no-reuse-pipeline', action='store_true',
                       help='Create new pipeline for each video instead of reusing (slower but safer)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable visualization (no video output, only JSON tracking results)')
    parser.add_argument('--rmm', action='store_true', default=True,
                       help='Use RMM dataset path conversion logic')
    parser.add_argument('--ids', nargs='+',
                       help='Filter by specific video IDs (space-separated list)')
    parser.add_argument('--start-row', type=int, default=0,
                       help='Start processing from this row index in CSV (0-based, inclusive)')
    parser.add_argument('--end-row', type=int, default=None,
                       help='End processing at this row index in CSV (exclusive, None means process to end)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        sys.exit(1)

    if not os.path.exists(args.cache_dir):
        print(f"Error: Cache directory not found: {args.cache_dir}")
        print("Please run batch_pose.py first to generate pose caches!")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and run batch tracker
    processor = BatchTracker(
        args.csv_file,
        args.output_dir,
        args.video_dir,
        args.exp_id,
        reuse_pipeline=not args.no_reuse_pipeline,
        rmm=args.rmm,
        enable_visualization=not args.no_visualization,
        filter_ids=args.ids,
        start_row=args.start_row,
        end_row=args.end_row
    )

    # Set cache directory from command line
    processor.config.cache.cache_base_path = args.cache_dir

    processor.process_all()


if __name__ == "__main__":
    main()
