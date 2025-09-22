#!/usr/bin/env python3
"""
Batch processing script for running modular_pipeline.py on multiple videos from CSV files.
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

# Add current directory to path for importing modular_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from modular_pipeline import MultiPersonTrackingPipeline, create_custom_config


class BatchProcessor:
    """Batch processor for running pipeline on multiple videos"""

    def __init__(self, csv_path: str, output_base_dir: str, base_video_dir: str):
        self.csv_path = csv_path
        self.output_base_dir = output_base_dir
        self.base_video_dir = base_video_dir
        self.interrupted = False
        self.current_video = None
        self.progress_file = None
        self.completed_videos = set()

        # Create subset name from CSV filename
        csv_name = Path(csv_path).stem
        self.subset_name = csv_name
        self.output_dir = os.path.join(output_base_dir, self.subset_name)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Progress tracking file
        self.progress_file = os.path.join(self.output_dir, "processing_progress.json")
        self._load_progress()

        # Initialize pipeline
        self.pipeline = None

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print(f"\n\nBatch processing interrupted! Saving progress...")
        self.interrupted = True

        # Save current progress
        self._save_progress()

        print(f"Progress saved to: {self.progress_file}")
        print(f"Completed videos: {len(self.completed_videos)}")
        if self.current_video:
            print(f"Current video was: {self.current_video}")
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
        """Create output path for processed video"""
        filename = video_info['filename']
        video_id = video_info['video_id']
        coder = video_info['coder']

        # Remove extension and add .mp4
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{video_id}_{coder}_{base_name}_processed.mp4"

        return os.path.join(self.output_dir, output_filename)

    def _process_single_video(self, video_info: Dict) -> bool:
        """Process a single video"""
        source_path = self._convert_path(video_info['source_file'])
        output_path = self._create_output_path(video_info)

        print(f"\nProcessing: {video_info['filename']}")
        print(f"Source: {source_path}")
        print(f"Output: {output_path}")

        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"ERROR: Source file not found: {source_path}")
            return False

        # Check if output already exists
        if os.path.exists(output_path):
            print(f"Output file already exists, skipping: {output_path}")
            return True

        try:
            # Initialize fresh pipeline for each video to avoid memory issues
            config = create_custom_config()
            self.pipeline = MultiPersonTrackingPipeline(config)

            # Process the video
            self.pipeline.process_video(source_path, output_path)

            print(f"✅ Successfully processed: {video_info['filename']}")
            return True

        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to be caught by signal handler
            raise
        except Exception as e:
            print(f"❌ Error processing {video_info['filename']}: {e}")

            # Clean up partial output file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"Cleaned up partial output file: {output_path}")
                except:
                    pass

            return False
        finally:
            # Clean up pipeline to free memory
            if self.pipeline:
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

        print(f"Starting batch processing for subset: {self.subset_name}")
        print(f"CSV file: {self.csv_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Base video directory: {self.base_video_dir}")

        # Read video list
        videos = self._read_video_list()
        if not videos:
            print("No videos found in CSV file!")
            return

        print(f"Found {len(videos)} videos in CSV")

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
            print(f"Total progress: {len(self.completed_videos) + i}/{len(videos)}")
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
        print(f"Output directory: {self.output_dir}")
        print(f"Progress file: {self.progress_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch process videos using modular pipeline')
    parser.add_argument('csv_file', help='CSV file containing video list')
    parser.add_argument('--output-dir', default='your_output_directory',
                       help='Base output directory (default: pipeline_outputs)')
    parser.add_argument('--video-dir', default='/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external',
                       help='Base directory containing source videos')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and run batch processor
    processor = BatchProcessor(args.csv_file, args.output_dir, args.video_dir)

    processor.process_all()


if __name__ == "__main__":
    main()