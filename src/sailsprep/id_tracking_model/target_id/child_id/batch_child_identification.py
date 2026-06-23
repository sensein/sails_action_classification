#!/usr/bin/env python3
"""
Batch Child Identification and Video Generation

This script processes all tracking JSON files, runs child identification,
and generates videos with bounding boxes around identified children.

Usage:
    python batch_child_identification.py [--test] [--max-files N]
"""

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sailsprep.id_tracking_model.target_id.child_id.single_child_identification import (
    AnnotationInfo,
    ChildIdentificationConfig,
    ChildResult,
    Track,
    identify_single_child,
)

# Configuration
BASE_DIR = Path("/orcd/data/satra/002/projects/SAILS/feature_processing/pipeline_outputs")


class ChildIdentificationProcessor:
    def __init__(self, config: ChildIdentificationConfig, input_dir: Path, output_video_dir: Path, output_log_dir: Path) -> None:
        self.config = config
        self.input_dir = input_dir
        self.output_video_dir = output_video_dir
        self.output_log_dir = output_log_dir
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging to both file and console"""
        log_file = self.output_log_dir / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch processing log started: {log_file}")

    def convert_tracking_json_to_tracks(self, tracking_data: dict[str, Any]) -> list[Track]:
        """Convert tracking JSON results to Track objects"""
        tracks = []

        fps = tracking_data['video_metadata']['fps']
        video_path = tracking_data['video_metadata']['input_path']

        for track_id_str, track_data in tracking_data['tracking_results'].items():
            track_id = int(track_id_str)
            start_frame = track_data['start_frame']
            end_frame = track_data['end_frame']
            frames_data = track_data['frames']

            # Extract keypoints and bboxes for all frames
            keypoints_list = []
            bboxes_list = []
            frame_numbers = []

            # Get all frame numbers and sort them
            sorted_frame_nums = sorted([int(f) for f in frames_data])  # SIM118: no .keys()

            for frame_num in sorted_frame_nums:
                frame_str = str(frame_num)
                if frame_str in frames_data:
                    frame_info = frames_data[frame_str]

                    # Store keypoints as list (keeping original format)
                    keypoints = frame_info['keypoints']
                    keypoints_list.append(keypoints)

                    # Store bbox as tuple
                    bbox = tuple(frame_info['bbox'])
                    bboxes_list.append(bbox)

                    # Store frame number
                    frame_numbers.append(frame_num)

            # Create Track object
            track = Track(
                id=track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                keypoints=keypoints_list,
                bboxes=bboxes_list,
                face_crops=None,
                video_path=video_path,
                frame_numbers=frame_numbers,
                meta={
                    'total_detections': len(frame_numbers),
                    'frame_numbers': frame_numbers
                }
            )

            tracks.append(track)

        return tracks

    def estimate_child_age_from_filename(self, filename: str) -> float:
        """Estimate child age in months from filename patterns"""
        filename_lower = filename.lower()

        age_patterns = {
            '12-16': 14.0,
            '16-20': 18.0,
            '14m': 14.0,
            '18m': 18.0,
            '12m': 12.0,
            '24m': 24.0,
        }

        for pattern, age in age_patterns.items():
            if pattern in filename_lower:
                return age

        return 18.0  # 18 months default

    def create_child_video(
        self,
        video_path: str,
        child_result: ChildResult,
        tracking_data: dict[str, Any],
        output_path: Path,
        max_frames: int | None = None
    ) -> bool:
        """Create video with bounding boxes around identified child"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Cannot open video: {video_path}")
                return False

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            use_ffmpeg = True
            proc: subprocess.Popen[bytes] | None = None
            out: cv2.VideoWriter | None = None

            try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                    "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output_path)
                ]
                proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.logger.info("Using ffmpeg h264 encoding")
            except FileNotFoundError:
                use_ffmpeg = False
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                self.logger.info("ffmpeg not found, using cv2 video writer")

            # Create frame-to-bbox mapping for child segments
            child_frame_bboxes: dict[int, Any] = {}

            for segment in child_result.segments:
                track_key = str(segment.id)
                if track_key in tracking_data['tracking_results']:
                    track_data = tracking_data['tracking_results'][track_key]
                    for frame_str, frame_data in track_data['frames'].items():
                        frame_num = int(frame_str)
                        if segment.start_frame <= frame_num <= segment.end_frame:
                            child_frame_bboxes[frame_num] = frame_data['bbox']

            frame_count = 0
            processed_frames = 0

            self.logger.info(f"Processing video: {total_frames} frames at {fps:.1f} fps")

            video_failed = False  # B012: avoid return inside finally
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if max_frames and processed_frames >= max_frames:
                        break

                    if frame_count in child_frame_bboxes:
                        bbox = child_frame_bboxes[frame_count]
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        label = f"CHILD (conf: {child_result.confidence:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    if use_ffmpeg and proc is not None and proc.stdin is not None:
                        if frame.shape[0] != height or frame.shape[1] != width:
                            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        try:
                            proc.stdin.write(frame.tobytes())
                        except BrokenPipeError:
                            err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
                            self.logger.error(f"ffmpeg exited early: {err}")
                            break
                    elif out is not None:
                        out.write(frame)

                    processed_frames += 1

                    if processed_frames % 1000 == 0:
                        self.logger.info(f"Processed {processed_frames} frames...")

            finally:
                cap.release()

                if use_ffmpeg and proc is not None:
                    if proc.stdin is not None:
                        with contextlib.suppress(BrokenPipeError):  # SIM105
                            proc.stdin.close()

                    rc = proc.wait()
                    if rc != 0:
                        err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
                        self.logger.error(f"ffmpeg failed (code {rc}): {err}")
                        video_failed = True  # B012: flag instead of returning inside finally
                elif out is not None:
                    out.release()

            if video_failed:
                return False

            self.logger.info(f"Video created: {output_path} ({processed_frames} frames)")
            return True

        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            traceback.print_exc()
            return False

    def save_detailed_log(
        self,
        filename: str,
        child_result: ChildResult,
        tracking_data: dict[str, Any],
        processing_time: float,
    ) -> None:
        """Save detailed analysis log for this video"""
        log_path = self.output_log_dir / f"{filename}_analysis.json"

        diagnostics = child_result.diagnostics

        log_data = {
            "video_info": {
                "filename": filename,
                "source_video": tracking_data['video_metadata']['input_path'],
                "fps": tracking_data['video_metadata']['fps'],
                "total_frames": tracking_data['video_metadata']['total_frames'],
                "width": tracking_data['video_metadata'].get('width', 'unknown'),
                "height": tracking_data['video_metadata'].get('height', 'unknown'),
                "processing_time_seconds": round(processing_time, 2)
            },

            "child_identification": {
                "selected_track_ids": child_result.child_track_id_sequence,
                "confidence": round(child_result.confidence, 4),
                "uncertainty": child_result.uncertainty,
                "num_segments": len(child_result.segments),
                "total_duration_seconds": sum(seg.duration_seconds() for seg in child_result.segments),
                "segments": [
                    {
                        "track_id": seg.id,
                        "start_frame": seg.start_frame,
                        "end_frame": seg.end_frame,
                        "duration_seconds": round(seg.duration_seconds(), 2),
                        "duration_frames": seg.duration_frames()
                    }
                    for seg in child_result.segments
                ]
            },

            "detailed_analysis": {
                "total_nodes": len(diagnostics['nodes']),
                "total_edges": len(diagnostics['edges']),
                "selected_path_length": len(diagnostics['path_indices']),

                "nodes": [
                    {
                        "index": i,
                        "track_id": node.tracklet.id,
                        "score": round(node.score, 4),
                        "weight": round(node.weight, 2),
                        "duration_seconds": round(node.tracklet.duration_seconds(), 2),
                        "selected": i in diagnostics['path_indices'],
                        "evidence_flags": node.evidence.flags,
                        "age_prob": round(node.evidence.p_age, 4) if node.evidence.p_age is not None else None,
                        "skeleton_prob": round(node.evidence.p_skeleton, 4) if node.evidence.p_skeleton is not None else None
                    }
                    for i, node in enumerate(diagnostics['nodes'])
                ],

                "edges": [
                    {
                        "from_track": diagnostics['nodes'][edge.src_index].tracklet.id,
                        "to_track": diagnostics['nodes'][edge.dst_index].tracklet.id,
                        "score": round(edge.score, 4),
                        "reasons": {k: round(v, 4) for k, v in edge.reasons.items()}
                    }
                    for edge in diagnostics['edges']
                ]
            },

            "configuration": {
                "age_estimation_method": self.config.age_estimation_method,
                "enable_body_visibility_filter": self.config.enable_body_visibility_filter,
                "min_visible_keypoints": self.config.min_visible_keypoints,
                "min_track_frames": self.config.min_track_frames,
                "sampling_percentage": self.config.sampling_percentage,
                "sampling_max_frames": self.config.sampling_max_frames_per_track,
                "age_child_years_threshold": self.config.age_child_years_threshold
            }
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"Analysis log saved: {log_path}")

    def process_single_file(self, json_path: Path, skip_existing: bool = True) -> bool:
        """Process a single tracking JSON file"""
        filename = json_path.stem.replace('_tracking', '')
        self.logger.info(f"\n=== Processing: {filename} ===")

        if skip_existing:
            output_video_path = self.output_video_dir / f"{filename}_child_identified.mp4"
            if output_video_path.exists():
                self.logger.info(f"⊘ Skipping {filename} (already processed)")
                return True

        try:
            with open(json_path) as f:
                tracking_data = json.load(f)

            video_path = tracking_data['video_metadata']['input_path']
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path}")
                return False

            tracks = self.convert_tracking_json_to_tracks(tracking_data)
            self.logger.info(f"Loaded {len(tracks)} tracks")

            estimated_age = self.estimate_child_age_from_filename(filename)
            annotations = AnnotationInfo(
                age_in_months=estimated_age,
                quality_flags={}
            )

            start_time = datetime.now()
            child_result = identify_single_child(tracks, annotations, self.config)
            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Child identification completed in {processing_time:.2f}s")
            self.logger.info(f"Selected tracks: {child_result.child_track_id_sequence}")
            self.logger.info(f"Confidence: {child_result.confidence:.4f}")

            self.save_detailed_log(filename, child_result, tracking_data, processing_time)

            output_video_path = self.output_video_dir / f"{filename}_child_identified.mp4"
            video_success = self.create_child_video(
                video_path, child_result, tracking_data, output_video_path, max_frames=5000
            )

            if video_success:
                self.logger.info(f"✓ Successfully processed {filename}")
                return True
            else:
                self.logger.error(f"✗ Video creation failed for {filename}")
                return False

        except Exception as e:
            self.logger.error(f"✗ Error processing {filename}: {e}")
            traceback.print_exc()
            return False

    def process_batch(
        self,
        max_files: int | None = None,
        test_mode: bool = False,
        max_workers: int | None = None,
        aggressive_sampling: bool = False,
        skip_existing: bool = True,
    ) -> None:
        """Process all tracking files in batch"""
        json_files = list(self.input_dir.glob("*_tracking.json"))

        if test_mode:
            json_files = json_files[:3]
            self.logger.info("TEST MODE: Processing only first 3 files")

        if max_files:
            json_files = json_files[:max_files]
            self.logger.info(f"Limited to {max_files} files")

        self.logger.info(f"Found {len(json_files)} tracking files to process")

        if aggressive_sampling:
            self.config.sampling_percentage = 0.05
            self.config.sampling_max_frames_per_track = 8
            self.logger.info("AGGRESSIVE SAMPLING: 5% frames, max 8 per track")

        success_count = 0
        failed_count = 0

        if max_workers and max_workers > 1:
            self.logger.info(f"Using parallel processing with {max_workers} workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_single_file, json_path, skip_existing): json_path
                    for json_path in json_files
                }

                for i, future in enumerate(as_completed(future_to_path), 1):
                    json_path = future_to_path[future]
                    filename = json_path.stem.replace('_tracking', '')
                    self.logger.info(f"\n--- Progress: {i}/{len(json_files)} ({filename}) ---")

                    try:
                        if future.result():
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        self.logger.error(f"✗ Worker exception for {filename}: {e}")
                        failed_count += 1
        else:
            for i, json_path in enumerate(json_files, 1):
                self.logger.info(f"\n--- Progress: {i}/{len(json_files)} ---")
                if self.process_single_file(json_path, skip_existing=skip_existing):
                    success_count += 1
                else:
                    failed_count += 1

        self.logger.info("\n=== BATCH PROCESSING COMPLETE ===")
        self.logger.info(f"Total files: {len(json_files)}")
        self.logger.info(f"Successful: {success_count}")
        self.logger.info(f"Failed: {failed_count}")
        self.logger.info(f"Success rate: {success_count/len(json_files)*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch child identification and video generation')
    parser.add_argument('target_folder', help='Directory within pipeline outputs to process')
    parser.add_argument('--output-dir', help='Optional parent directory of pipeline outputs')
    parser.add_argument('--test', action='store_true', help='Test mode: process only first 3 files')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: sequential)')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive sampling (5%% frames, max 8)')
    parser.add_argument('--no-skip', action='store_true', help='Reprocess all files (do not skip existing)')

    args = parser.parse_args()

    target_dir = Path(BASE_DIR) / args.target_folder

    input_dir = target_dir / "tracking"
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    parent = f"child_classifications/{args.output_dir}" if args.output_dir else "child_classifications"
    output_video_dir = target_dir / (parent + "/videos")
    output_log_dir = target_dir / (parent + "/logs")

    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    config = ChildIdentificationConfig(
        age_estimation_method="siglip",
        enable_body_visibility_filter=True,
        min_visible_keypoints=4,
        enable_roi_size_filter=False,
        sampling_percentage=0.25,
        sampling_max_frames_per_track=30,
        min_track_frames=10,
        sampling_mode="smart",
        min_pose_confidence=0.7,
        age_child_years_threshold=10.0,
        age_tau=2.5,
        enable_skeleton_ratios=False,
        skeleton_min_confidence=0.3,
        skeleton_min_visible_for_ratio=2,
        w_age_default=1.0,
        w_skel_default=0.0,
        continuity_gap_seconds=6.0,
        intra_id_gamma=0.3,
        intra_id_tau=1.0
    )

    processor = ChildIdentificationProcessor(config, input_dir, output_video_dir, output_log_dir)
    processor.process_batch(
        max_files=args.max_files,
        test_mode=args.test,
        max_workers=args.workers,
        aggressive_sampling=args.aggressive,
        skip_existing=not args.no_skip
    )


if __name__ == "__main__":
    main()