#!/usr/bin/env python3
"""
Batch Target Identification Script

Identifies the target child across multiple videos using:
- Reference profiles from solo videos (#_children=1)
- Multi-cue embedding similarity (face, upper body, lower body)
- Metadata-based filtering and confidence scoring
- Cross-video consensus validation

Usage:
    python batch_identify_target.py annotations.csv \
        --embeddings-dir /path/to/pipeline_outputs \
        --ids CHILD_ID1 CHILD_ID2 \
        --output-dir /path/to/output
"""

import sys
import os
import csv
import argparse
import h5py
import numpy as np
import json
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import re

import cv2
import pandas as pd
from pandas.api.types import is_integer_dtype

# Configure logging for SLURM output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from sailsprep.feature_processing.target_id.child_id.single_child_identification import (
    AnnotationInfo,
    ChildIdentificationConfig,
    SigLipModel,
)
from sailsprep.feature_processing.target_id.child_id.single_child_track_selector import (
    SingleTrackSelection,
    load_track_from_h5,
    select_from_directory,
)


def _json_default(obj):
    """Fallback serializer for NumPy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_for_path(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    safe = re.sub(r"[^\w\-]+", "_", str(value).strip())
    return safe or "unknown"


@dataclass
class EmbeddingProfile:
    """Container for averaged embeddings"""
    face_feature: Optional[np.ndarray] = None
    upper_feature: Optional[np.ndarray] = None
    lower_feature: Optional[np.ndarray] = None
    num_observations: int = 0
    source_videos: List[str] = None

    def __post_init__(self):
        if self.source_videos is None:
            self.source_videos = []


@dataclass
class TrackMatch:
    """Container for track matching results"""
    video_id: str
    track_id: int
    similarity_score: float
    face_score: float
    upper_score: float
    lower_score: float
    num_frames: int
    start_frame: int
    end_frame: int
    confidence: str  # 'high', 'medium', 'low'
    timepoint: Optional[str] = None
    is_reference: bool = False

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
POSE_CONF_THRESHOLD = 0.65


class TargetIdentifier:
    """Identifies target child across multiple videos"""

    def __init__(self, csv_path: str, embeddings_base_dir: str, output_dir: str,
                 filter_ids: Optional[List[str]] = None,
                 render: bool = False,
                 video_base_dir: Optional[str] = None,
                 rmm: bool = False,
                 face_only: bool = False,
                 min_score: float = 0.5):
        """
        Initialize target identifier

        Args:
            csv_path: Path to annotations CSV
            embeddings_base_dir: Base directory containing tracks_hdf5 subdirectories
            output_dir: Output directory for results
            filter_ids: Optional list of child IDs to process
            render: Whether to render visualization videos
        """
        self.csv_path = csv_path
        self.embeddings_base_dir = Path(embeddings_base_dir)
        self.output_dir = Path(output_dir)
        self.filter_ids = set(filter_ids) if filter_ids else None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Load annotations
        logger.info(f"Loading annotations from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} video annotations")
        self._normalize_numeric_columns()
        self.child_timepoint_months, self.global_timepoint_months = self._build_timepoint_index()

        # Storage for results
        self.reference_videos = {}  # {child_id: {video_path: track_id}}
        self.reference_profiles = {}  # {child_id: {timepoint: EmbeddingProfile}}
        self.match_results = defaultdict(list)  # {child_id: [TrackMatch]}
        self.reference_track_diagnostics = defaultdict(dict)  # {child_id: {source_file: diagnostics}}
        self.child_metrics = defaultdict(lambda: {
            'total_videos': 0,
            'successes': 0,
            'failures': 0,
            'failure_reasons': Counter(),
        })
        self.global_metrics = {
            'total_videos': 0,
            'successes': 0,
            'failures': 0,
            'failure_reasons': Counter(),
            'children_processed': 0,
        }
        self.selector_config = ChildIdentificationConfig()
        self.siglip_model = SigLipModel() if self.selector_config.age_estimation_method == "siglip" else None
        if self.siglip_model:
            self.siglip_model.load_siglip_model()
        # self.selector_config.age_estimation_method = "none"
        # self.selector_config.enable_body_visibility_filter = False
        self.selector_config.enable_rigidity_detection = False
        self.render = render
        self.video_base_dir = Path(video_base_dir) if video_base_dir else None
        self.rmm = rmm
        self.face_only = face_only
        self.min_score = float(min_score)
        if face_only:
            self.similarity_weights = {'face': 1.0, 'upper': 0.0, 'lower': 0.0}
        else:
            self.similarity_weights = {'face': 0.5, 'upper': 0.3, 'lower': 0.2}

    def _video_basename(self, video_info: Dict) -> str:
        filename = video_info['SourceFile']
        video_id = video_info['ID']
        coder = video_info['Coder']

        base_name = Path(filename).stem
        return f"{video_id}_{coder}_{base_name}"

    def _get_embedding_path(self, video_info: Dict) -> Path:
        """
        Get path to HDF5 embeddings directory for a video

        Args:
            video_info: Video metadata dict

        Returns:
            Path to tracks_hdf5/{video_basename}/ directory
        """
        video_basename = self._video_basename(video_info)

        candidates = [
            self.embeddings_base_dir / 'tracks_hdf5' / f"{video_basename}_tracking",
            self.embeddings_base_dir / f"{video_basename}_tracking",
            self.embeddings_base_dir / f"{video_basename}_tracking.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fall back to the last candidate even if it does not exist so caller can warn.
        return candidates[-1]
 
    def _normalize_numeric_columns(self):
        """Convert int-like columns to nullable integers for robust comparisons."""
        if self.df.empty:
            return

        candidate_columns = set(
            col for col in self.df.columns
            if col in {'#_children', 'num_children', 'num_child', 'num_people', 'num_adults'}
            or col.lower().startswith('num_')
            or col.lower().endswith('_count')
        )

        for col in candidate_columns:
            try:
                converted = pd.to_numeric(self.df[col], errors='coerce')
            except Exception:
                continue

            if converted.isnull().all() and not is_integer_dtype(self.df[col]):
                continue

            try:
                self.df[col] = converted.astype('Int64')
            except (TypeError, ValueError):
                self.df[col] = converted

        if 'Age' in self.df.columns:
            self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')

    def _build_timepoint_index(self) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, float]]]:
        """Collect available timepoints (child-specific and global) with numeric month values."""
        child_map: Dict[str, Dict[str, float]] = defaultdict(dict)
        global_map: Dict[str, float] = {}

        if 'timepoint' not in self.df.columns:
            return {}, []

        for _, row in self.df.iterrows():
            tp = row.get('timepoint')
            if pd.isna(tp):
                continue
            tp_str = str(tp)
            months = self._parse_timepoint_to_months(tp_str)
            if months is None:
                continue
            child_id = row.get('ID')
            if child_id:
                child_map[str(child_id)][tp_str] = months
            global_map[tp_str] = months

        child_timepoint_months = {
            child: sorted(entries.items(), key=lambda item: item[1])
            for child, entries in child_map.items()
        }
        global_timepoint_months = sorted(global_map.items(), key=lambda item: item[1])
        return child_timepoint_months, global_timepoint_months

    @staticmethod
    def _parse_timepoint_to_months(timepoint: str) -> Optional[float]:
        match = re.search(r'(\d+(?:\.\d+)?)', timepoint)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _infer_timepoint_from_age(self, child_id: str, age_value) -> Tuple[Optional[str], Optional[float]]:
        """Infer the timepoint label from age (in years) by snapping to nearest known timepoint."""
        try:
            age_float = float(age_value)
        except (TypeError, ValueError):
            return None, None

        if not np.isfinite(age_float):
            return None, None

        age_months = age_float * 12.0

        candidates = self.child_timepoint_months.get(child_id)
        if not candidates:
            candidates = self.global_timepoint_months

        if not candidates:
            return None, age_months

        best_timepoint, _ = min(candidates, key=lambda item: abs(item[1] - age_months))
        return best_timepoint, age_months

    def _get_output_subdir(self, base_dir: Path, child_id: str, timepoint: Optional[str], video_name: Optional[str] = None) -> Path:
        """Get output subdirectory, optionally creating a video-specific folder."""
        child_dir = base_dir / _sanitize_for_path(child_id)
        tp_dir = child_dir / _sanitize_for_path(timepoint)

        if video_name:
            # Create folder for this specific video
            video_dir = tp_dir / _sanitize_for_path(video_name)
            video_dir.mkdir(parents=True, exist_ok=True)
            return video_dir
        else:
            tp_dir.mkdir(parents=True, exist_ok=True)
            return tp_dir

    def _store_video_result(self,
                            child_id: str,
                            timepoint: Optional[str],
                            video_info: Dict,
                            match: Optional[TrackMatch],
                            miss_reason: Optional[str]) -> None:
        raw_source = video_info.get('SourceFile', '')
        source_file = str(raw_source) if raw_source is not None else ''
        video_stem = Path(source_file).stem if source_file else match.video_id if match else "video"

        # Create a folder for this specific video
        video_dir = self._get_output_subdir(self.output_dir, child_id, timepoint, video_stem)
        file_name = "result.json"
        video_id = match.video_id if match else f"{video_info.get('ID', 'unknown')}_{video_info.get('Coder', 'unknown')}_{video_stem}"

        result_data: Dict[str, Any] = {
            'child_id': child_id,
            'timepoint': timepoint,
            'video_id': video_id,
            'source_file': source_file,
            'coder': video_info.get('Coder'),
            'age_years': float(video_info.get('Age')) if pd.notna(video_info.get('Age')) else None,
            'match_found': match is not None,
        }

        if match:
            result_data.update({
                'track_id': match.track_id,
                'similarity_score': float(match.similarity_score),
                'face_score': float(match.face_score),
                'upper_score': float(match.upper_score),
                'lower_score': float(match.lower_score),
                'num_frames': int(match.num_frames) if match.num_frames is not None else None,
                'start_frame': int(match.start_frame) if match.start_frame is not None else None,
                'end_frame': int(match.end_frame) if match.end_frame is not None else None,
                'confidence': match.confidence,
                'is_reference': match.is_reference,
            })
            if match.is_reference:
                diagnostics = self.reference_track_diagnostics.get(child_id, {}).get(str(source_file))
                if diagnostics:
                    result_data['reference_track_selector'] = diagnostics
        else:
            result_data['reason'] = miss_reason

        with open(video_dir / file_name, 'w') as f:
            json.dump(result_data, f, indent=2, default=_json_default)

    def _categorize_failure(self, reason: Optional[str]) -> str:
        if not reason:
            return 'unknown'
        text = reason.lower()
        if 'reference profile' in text:
            return 'reference_profile_missing'
        if 'timepoint' in text:
            return 'timepoint_missing'
        if 'embeddings' in text:
            return 'embeddings_missing'
        if 'track files' in text:
            return 'no_track_files'
        if 'threshold' in text:
            return 'score_below_threshold'
        if 'select reference track' in text or 'could not select reference track' in text:
            return 'reference_selector_failed'
        if 'no solo videos' in text:
            return 'no_reference_videos'
        return 'other'

    def _record_match_outcome(self, child_id: str, match_found: bool, miss_reason: Optional[str]) -> None:
        metrics = self.child_metrics[child_id]
        metrics['total_videos'] += 1
        self.global_metrics['total_videos'] += 1

        if match_found:
            metrics['successes'] += 1
            self.global_metrics['successes'] += 1
            return

        metrics['failures'] += 1
        self.global_metrics['failures'] += 1
        category = self._categorize_failure(miss_reason)
        metrics['failure_reasons'][category] += 1
        self.global_metrics['failure_reasons'][category] += 1

    def _format_failure_breakdown(self, counter: Counter, total: int) -> Dict[str, Dict[str, float]]:
        breakdown: Dict[str, Dict[str, float]] = {}
        if total <= 0:
            return breakdown

        for reason, count in counter.items():
            breakdown[reason] = {
                'count': int(count),
                'rate': float(count) / float(total),
            }
        return breakdown

    def _convert_source_path(self, source_path: str) -> Optional[Path]:
        """Convert a SourceFile entry into an actual filesystem path."""
        if not source_path:
            return None

        source_candidate = Path(source_path)
        if source_candidate.exists():
            return source_candidate

        if self.rmm and self.video_base_dir:
            return self.video_base_dir / source_path

        if self.video_base_dir:
            rel_candidate = self.video_base_dir / source_candidate
            if rel_candidate.exists():
                return rel_candidate
            prefix = '/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/'
            if source_path.startswith(prefix):
                relative_path = source_path.replace(prefix, '', 1)
                return self.video_base_dir / relative_path

            parts = source_path.split('/')
            if len(parts) >= 2:
                relative_path = Path(*parts[-2:])
                return self.video_base_dir / relative_path

        return source_candidate

    def _infer_video_path(self, video_info: Dict) -> Optional[Path]:
        """
        Infer the source video path associated with a tracking directory.
        """
        video_basename = self._video_basename(video_info)
        candidates = []

        # Candidate from CSV metadata if it already points to an existing file
        source_file = video_info.get('SourceFile') or video_info.get('source_file')
        if source_file:
            converted = self._convert_source_path(source_file)
            if converted:
                if converted.exists():
                    print(f"Chosen video path: {converted}")
                    return converted
                converted = converted.with_suffix('.mp4')
                if converted.exists():
                    print(f"Chosen video path: {converted}")
                    return converted

        # Candidate relative to embeddings base directory (../videos/<basename>.mp4)
        parent_dir = self.embeddings_base_dir.parent
        candidates.append(parent_dir / 'videos' / f"{video_basename}.mp4")
        candidates.append(self.embeddings_base_dir / 'videos' / f"{video_basename}.mp4")
        if self.video_base_dir:
            path = self.video_base_dir / f"{video_basename}.mp4"
            if path.exists():
                print(f"Chosen video path: {path}")
                return path

        return None

    def _select_reference_track(self,
                                tracking_dir: Path,
                                video_info: Dict,
                                include_details: bool = False):
        """Run single-child scoring to pick the best track from a solo video."""
        annotations = AnnotationInfo()
        age_months = video_info.get('Age_in_months') or video_info.get('age_in_months')

        if age_months is not None and not pd.isna(age_months):
            try:
                annotations.age_in_months = float(age_months)
            except Exception:
                pass

        video_path = self._infer_video_path(video_info)

        return select_from_directory(
            tracking_dir,
            video_path=video_path,
            annotations=annotations,
            cfg=self.selector_config,
            siglip_model=self.siglip_model,
            include_diagnostics=include_details,
        )

    def _resolve_track_path(self, tracking_dir: Path, track_id: int) -> Optional[Path]:
        """Find the HDF5 file for a given track ID inside a tracking directory."""
        candidates = [
            tracking_dir / f"track_{track_id:04d}.h5",
            tracking_dir / f"track_{track_id:05d}.h5",
            tracking_dir / f"track_{track_id}.h5",
        ]

        for cand in candidates:
            if cand.exists():
                return cand

        glob_matches = sorted(tracking_dir.glob(f"track_{track_id:04d}*.h5"))
        if glob_matches:
            return glob_matches[0]

        glob_matches = sorted(tracking_dir.glob(f"track_{track_id}*.h5"))
        if glob_matches:
            return glob_matches[0]

        return None

    def _render_target_track(self, video_info: Dict, match: TrackMatch):
        """Render visualization for the target track (bbox + pose) if enabled."""
        if not self.render:
            return

        try:
            self._render_target_track_impl(video_info, match)
        except Exception as e:
            logger.error(f"  Error rendering track for {match.video_id}: {e}")
            import traceback
            traceback.print_exc()

    def _render_target_track_impl(self, video_info: Dict, match: TrackMatch):
        """Internal implementation of track rendering with full error handling."""
        tracking_dir = self._get_embedding_path(video_info)
        if not tracking_dir.exists():
            logger.warning(f"  Tracking directory missing for rendering: {tracking_dir}")
            return

        track_path = self._resolve_track_path(tracking_dir, match.track_id)
        if not track_path:
            logger.warning(f"  Could not locate track file for ID {match.track_id} in {tracking_dir}")
            return

        video_path = self._infer_video_path(video_info)
        if not video_path:
            logger.warning(f"  Could not infer video path for rendering {match.video_id}")
            return

        logger.info(f"  Attempting to render video: {video_path}")

        try:
            loaded = load_track_from_h5(track_path, video_path=video_path)
            track = loaded.track
        except Exception as e:
            logger.warning(f"  Failed to load track for rendering: {e}")
            return

        if not track.frame_numbers:
            logger.warning(f"  Track {match.track_id} has no frame data, skipping render")
            return

        frame_data: Dict[int, Dict[str, np.ndarray]] = {}
        total_frames = len(track.frame_numbers)
        for idx in range(total_frames):
            frame_num = track.frame_numbers[idx]
            if frame_num is None:
                continue
            data: Dict[str, np.ndarray] = {}
            if track.bboxes and idx < len(track.bboxes):
                bbox = track.bboxes[idx]
                if bbox is not None:
                    data['bbox'] = np.asarray(bbox, dtype=float)
            if track.keypoints and idx < len(track.keypoints):
                keypoints = track.keypoints[idx]
                if keypoints is not None:
                    data['keypoints'] = np.asarray(keypoints, dtype=float)

            if data:
                frame_data[int(frame_num)] = data

        if not frame_data:
            logger.warning(f"  No drawable data for track {match.track_id}, skipping render")
            return

        # Wrap OpenCV operations in try-catch to prevent segfaults
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"  Cannot open video for rendering: {video_path}")
                if cap:
                    cap.release()
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not width or not height or width <= 0 or height <= 0:
                logger.warning(f"  Invalid video dimensions: {width}x{height}, skipping render")
                cap.release()
                return

            # Try to read first frame to verify video is readable
            test_ret, test_frame = cap.read()
            if not test_ret or test_frame is None:
                logger.warning(f"  Cannot read frames from video, skipping render: {video_path}")
                cap.release()
                return

            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        except Exception as e:
            logger.error(f"  Error opening/testing video: {e}")
            if cap:
                cap.release()
            return

        timepoint_label = match.timepoint or video_info.get('timepoint')
        source_file = video_info.get('SourceFile', '')
        video_stem = Path(source_file).stem if source_file else match.video_id

        # Save video in the same folder as the JSON result
        video_dir = self._get_output_subdir(self.output_dir, video_info['ID'], timepoint_label, video_stem)
        output_path = video_dir / f"{match.video_id}_target.mp4"
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        use_ffmpeg = True
        proc = None
        writer = None

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", f"{fps}",
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "20",
                "-movflags", "+faststart",
                str(output_path)
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            use_ffmpeg = False
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.warning(f"  Failed to open video writer for {output_path}")
                cap.release()
                return

        try:
            frame_count = 0
            max_frames = 100000  # Safety limit to prevent infinite loops

            while frame_count < max_frames:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                except Exception as e:
                    logger.warning(f"  Error reading frame {frame_count}: {e}")
                    break

                frame_count += 1
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                # if current_frame > max_frame:
                #     break

                # Add [REFERENCE] label at top-left corner for reference videos
                try:
                    if match.is_reference:
                        cv2.putText(frame, "[REFERENCE]", (10, 30), font, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
                except Exception as e:
                    logger.warning(f"  Error adding reference label to frame {current_frame}: {e}")

                data = frame_data.get(current_frame)
                if data:
                    try:
                        bbox = data.get('bbox')
                        if bbox is not None and bbox.size >= 4:
                            x1, y1, x2, y2 = bbox.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label_y = max(y1 - 10, 20)
                            cv2.putText(frame, f"TARGET (Track {match.track_id})", (x1, label_y), font, 0.7, color, 2, cv2.LINE_AA)

                        keypoints = data.get('keypoints')
                        if keypoints is not None and keypoints.size >= 3:
                            num_points = keypoints.shape[0]

                            for idx in range(min(num_points, 17)):
                                x, y, conf = keypoints[idx]
                                if conf >= POSE_CONF_THRESHOLD:
                                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)

                            for a, b in COCO_SKELETON:
                                if a < num_points and b < num_points:
                                    x1, y1, c1 = keypoints[a]
                                    x2, y2, c2 = keypoints[b]
                                    if c1 >= POSE_CONF_THRESHOLD and c2 >= POSE_CONF_THRESHOLD:
                                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    except Exception as e:
                        logger.warning(f"  Error drawing on frame {current_frame}: {e}")

                try:
                    if use_ffmpeg:
                        if proc and proc.stdin:
                            proc.stdin.write(frame.tobytes())
                    else:
                        if writer is not None:
                            writer.write(frame)
                except Exception as e:
                    logger.error(f"  Error writing frame {current_frame}: {e}")
                    break

                # if current_frame >= max_frame:
                #     break
        except Exception as e:
            logger.error(f"  Unexpected error in rendering loop: {e}")
        finally:
            try:
                cap.release()
            except Exception as e:
                logger.warning(f"  Error releasing video capture: {e}")

            if use_ffmpeg and proc:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"  Error closing ffmpeg: {e}")
                    try:
                        proc.kill()
                    except:
                        pass
            elif writer:
                try:
                    writer.release()
                except Exception as e:
                    logger.warning(f"  Error releasing video writer: {e}")

        logger.info(f"  Rendered target track video: {output_path}")

    def load_track_embeddings(self, h5_file: Path) -> Optional[EmbeddingProfile]:
        """
        Load embeddings from a single track HDF5 file

        Args:
            h5_file: Path to track_{id}.h5 file

        Returns:
            EmbeddingProfile or None if load fails
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                profile = EmbeddingProfile()

                if 'embeddings' in f:
                    emb_grp = f['embeddings']

                    if 'face_feature' in emb_grp:
                        profile.face_feature = emb_grp['face_feature'][:]

                    if 'upper_feature' in emb_grp:
                        profile.upper_feature = emb_grp['upper_feature'][:]

                    if 'lower_feature' in emb_grp:
                        profile.lower_feature = emb_grp['lower_feature'][:]

                # Get metadata
                if 'metadata' in f:
                    meta = f['metadata']
                    profile.num_observations = meta.attrs.get('num_frames', 0)

                return profile

        except Exception as e:
            logger.error(f"Error loading {h5_file}: {e}")
            return None

    def build_reference_profiles(self, child_id: str):
        """
        Build reference embedding profiles from solo videos

        Args:
            child_id: Child ID to build profiles for
        """
        logger.info(f"Building reference profiles for child {child_id}...")

        # Clear any previously stored reference data for this child
        self.reference_profiles[child_id] = {}
        self.reference_track_diagnostics[child_id] = {}
        self.reference_videos[child_id] = {}

        # Filter to solo videos for this child
        child_videos = self.df[self.df['ID'] == child_id]
        solo_videos = child_videos[child_videos['#_children'] == 1]

        if len(solo_videos) == 0:
            logger.warning(f"No solo videos found for child {child_id}")
            return

        logger.info(f"Found {len(solo_videos)} solo videos for child {child_id}")

        # Infer timepoints for videos that don't have explicit timepoint
        solo_videos = solo_videos.copy()
        for idx, row in solo_videos.iterrows():
            timepoint = row.get('timepoint')
            if pd.isna(timepoint) or (isinstance(timepoint, str) and not timepoint.strip()):
                # Try to infer from age
                age_value = row.get('Age')
                inferred_timepoint, _ = self._infer_timepoint_from_age(child_id, age_value)
                if inferred_timepoint:
                    solo_videos.at[idx, 'timepoint'] = inferred_timepoint
                    logger.info(f"  Inferred timepoint '{inferred_timepoint}' from age for video: {row.get('SourceFile', 'unknown')}")

        # Group by timepoint
        timepoints = solo_videos['timepoint'].dropna().unique()
        for timepoint in timepoints:
            tp_videos = solo_videos[solo_videos['timepoint'] == timepoint]
            profile = self._build_profile_from_videos(tp_videos, timepoint)

            if profile:
                self.reference_profiles[child_id][timepoint] = profile
                logger.info(f"  {timepoint} profile ({len(tp_videos)} videos): "
                      f"face={'✓' if profile.face_feature is not None else '✗'} "
                      f"upper={'✓' if profile.upper_feature is not None else '✗'} "
                      f"lower={'✓' if profile.lower_feature is not None else '✗'}")

    def _build_profile_from_videos(self, videos_df, timepoint: str) -> Optional[EmbeddingProfile]:
        """
        Build averaged embedding profile from multiple videos

        Args:
            videos_df: DataFrame of videos
            timepoint: Timepoint label (e.g., '14_month', '36_month')

        Returns:
            Averaged EmbeddingProfile or None
        """
        face_embeddings = []
        upper_embeddings = []
        lower_embeddings = []
        source_videos = []

        for _, video_series in videos_df.iterrows():
            video_info = video_series.to_dict()
            emb_dir = self._get_embedding_path(video_info)

            if not emb_dir.exists():
                logger.warning(f"  Embeddings not found for {video_info['SourceFile']}")
                continue

            selection_result = self._select_reference_track(emb_dir, video_info, include_details=True)
            if not selection_result:
                logger.warning(f"  Could not select reference track in {emb_dir}")
                continue
            selection, selector_diagnostics = selection_result

            if not selection:
                logger.warning(f"  Could not select reference track in {emb_dir}")
                continue

            source_file_key = str(video_info.get('SourceFile'))
            if selector_diagnostics is not None:
                self.reference_track_diagnostics.setdefault(video_info['ID'], {})[source_file_key] = selector_diagnostics

            source_h5 = selection.track.meta.get('source_h5')
            h5_path = Path(source_h5) if source_h5 else None

            if not h5_path or not h5_path.exists():
                h5_candidate = emb_dir / f"track_{selection.track.id:04d}.h5"
                if h5_candidate.exists():
                    h5_path = h5_candidate
                else:
                    logger.warning(f"  Track file missing for selected track {selection.track.id}")
                    continue

            best_track = self.load_track_embeddings(h5_path)

            if best_track:
                # Only store reference video info after successful embedding load
                if video_info['ID'] not in self.reference_videos:
                    self.reference_videos[video_info['ID']] = {}
                self.reference_videos[video_info['ID']][str(video_info['SourceFile'])] = selection.track.id

                if best_track.face_feature is not None:
                    face_embeddings.append(best_track.face_feature)
                if best_track.upper_feature is not None:
                    upper_embeddings.append(best_track.upper_feature)
                if best_track.lower_feature is not None:
                    lower_embeddings.append(best_track.lower_feature)
                source_videos.append(video_info['SourceFile'])
                logger.info(
                    f"  Selected track {selection.track.id} "
                    f"(score={selection.node.score:.3f}, "
                    f"weight={selection.node.weight:.2f}) from {emb_dir}"
                )

        if not face_embeddings and not upper_embeddings and not lower_embeddings:
            logger.warning(f"  No valid embeddings found for {timepoint} profile")
            return None

        # Average embeddings
        profile = EmbeddingProfile()
        profile.source_videos = source_videos
        profile.num_observations = len(source_videos)

        if face_embeddings:
            profile.face_feature = np.mean(face_embeddings, axis=0)
            # Normalize
            norm = np.linalg.norm(profile.face_feature)
            if norm > 0:
                profile.face_feature = profile.face_feature / norm

        if upper_embeddings:
            profile.upper_feature = np.mean(upper_embeddings, axis=0)
            norm = np.linalg.norm(profile.upper_feature)
            if norm > 0:
                profile.upper_feature = profile.upper_feature / norm

        if lower_embeddings:
            profile.lower_feature = np.mean(lower_embeddings, axis=0)
            norm = np.linalg.norm(profile.lower_feature)
            if norm > 0:
                profile.lower_feature = profile.lower_feature / norm

        return profile

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if a is None or b is None:
            return 0.0

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def compute_track_similarity(self, track_profile: EmbeddingProfile,
                                 ref_profile: EmbeddingProfile,
                                 weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted similarity between track and reference profile

        Args:
            track_profile: Track embeddings
            ref_profile: Reference embeddings
            weights: Optional custom weights (default: face=0.5, upper=0.3, lower=0.2)

        Returns:
            (total_score, individual_scores)
        """
        if weights is None:
            weights = {'face': 0.5, 'upper': 0.3, 'lower': 0.2}

        scores = {}
        scores['face'] = self.cosine_similarity(track_profile.face_feature, ref_profile.face_feature)
        scores['upper'] = self.cosine_similarity(track_profile.upper_feature, ref_profile.upper_feature)
        scores['lower'] = self.cosine_similarity(track_profile.lower_feature, ref_profile.lower_feature)

        # Compute weighted total
        total_score = (
            scores['face'] * weights['face'] +
            scores['upper'] * weights['upper'] +
            scores['lower'] * weights['lower']
        )

        return total_score, scores

    def _create_match_from_track_id(self, child_id: str, video_info: Dict, track_id: int) -> Optional[TrackMatch]:
        """
        Create a TrackMatch object from a known track ID (for reference videos)

        Args:
            child_id: Child ID
            video_info: Video metadata dict
            track_id: Known track ID from single-child identification

        Returns:
            TrackMatch object with metadata loaded from track file
        """
        emb_dir = self._get_embedding_path(video_info)
        track_path = self._resolve_track_path(emb_dir, track_id)

        # Default values
        num_frames = 0
        start_frame = 0
        end_frame = 0

        # Try to load metadata from track file
        if track_path and track_path.exists():
            try:
                with h5py.File(track_path, 'r') as f:
                    if 'metadata' in f:
                        meta = f['metadata']
                        num_frames = meta.attrs.get('num_frames', 0)
                        start_frame = meta.attrs.get('start_frame', 0)
                        end_frame = meta.attrs.get('end_frame', 0)
            except Exception as e:
                logger.warning(f"  Could not load metadata for track {track_id}: {e}")

        timepoint = video_info.get('timepoint')
        video_basename = self._video_basename(video_info)

        # Create match with perfect scores since this is from single-child identification
        match = TrackMatch(
            video_id=video_basename,
            track_id=track_id,
            similarity_score=1.0,  # Perfect match - from single-child ID
            face_score=1.0,
            upper_score=1.0,
            lower_score=1.0,
            num_frames=num_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            confidence='high',
            timepoint=timepoint,
            is_reference=True
        )

        return match

    def identify_target_in_video(self, child_id: str, video_info: Dict) -> Tuple[Optional[TrackMatch], Optional[str]]:
        """
        Identify target child in a single video

        Args:
            child_id: Child ID
            video_info: Video metadata dict

        Returns:
            (Best TrackMatch, failure_reason)
        """
        # Check if this is a reference video - if so, use the already-identified track
        source_file = str(video_info.get('SourceFile'))
        if child_id in self.reference_videos and source_file in self.reference_videos[child_id]:
            track_id = self.reference_videos[child_id][source_file]
            logger.info(f"  Using reference track {track_id} (from single-child identification)")
            match = self._create_match_from_track_id(child_id, video_info, track_id)
            return match, None

        # Get reference profile for this timepoint
        timepoint = video_info.get('timepoint')
        age_value = video_info.get('Age')
        if age_value is None and 'age' in video_info:
            age_value = video_info.get('age')

        timepoint_missing = pd.isna(timepoint) or (isinstance(timepoint, str) and not timepoint.strip())
        if timepoint_missing:
            inferred_timepoint, age_months = self._infer_timepoint_from_age(child_id, age_value)
            if inferred_timepoint:
                timepoint = inferred_timepoint
                video_info['timepoint'] = timepoint
                if age_months is not None and np.isfinite(age_months):
                    logger.info(f"No timepoint specified for video; inferred '{timepoint}' from age ≈ {age_months:.1f} months")
                else:
                    logger.info(f"No timepoint specified for video; inferred '{timepoint}' from available age")
            else:
                reason = "timepoint missing in CSV and unable to infer from age"
                logger.warning(f"No timepoint specified for video and unable to infer from age")
                return None, reason

        if child_id not in self.reference_profiles:
            reason = f"no reference profile available for child {child_id}"
            logger.warning(f"No reference profile for child {child_id}")
            return None, reason

        if timepoint not in self.reference_profiles[child_id]:
            reason = f"no reference profile for timepoint '{timepoint}'"
            logger.warning(f"No reference profile for timepoint {timepoint}")
            return None, reason

        ref_profile = self.reference_profiles[child_id][timepoint]

        # Get embeddings directory
        emb_dir = self._get_embedding_path(video_info)

        if not emb_dir.exists():
            reason = f"embeddings directory not found ({emb_dir})"
            logger.warning(f"Embeddings not found: {emb_dir}")
            return None, reason

        # Load all tracks
        h5_files = list(emb_dir.glob('track_*.h5'))

        if not h5_files:
            reason = f"no track files present in {emb_dir}"
            logger.warning(f"No track files found in {emb_dir}")
            return None, reason

        # Score all tracks
        best_match = None
        best_score = -1
        best_reason = "tracks evaluated but none scored above current best threshold"

        for h5_file in h5_files:
            track_id = int(h5_file.stem.split('_')[1])
            track_profile = self.load_track_embeddings(h5_file)

            if not track_profile:
                continue

            # Compute similarity
            total_score, individual_scores = self.compute_track_similarity(
                track_profile,
                ref_profile,
                weights=self.similarity_weights
            )

            # Metadata-based boosting
            boost = 1.0

            # Boost if child is clearly visible
            if video_info.get('Child_of_interest_clear') == 'Yes':
                boost *= 1.2

            # Boost if track has many frames (likely main subject)
            # We'll need to load metadata for this
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'metadata' in f:
                        num_frames = f['metadata'].attrs.get('num_frames', 0)
                        start_frame = f['metadata'].attrs.get('start_frame', 0)
                        end_frame = f['metadata'].attrs.get('end_frame', 0)
            except:
                num_frames = track_profile.num_observations
                start_frame = 0
                end_frame = 0

            # Boost if track is long-duration
            if num_frames > 100:  # Threshold for "substantial presence"
                boost *= 1.1

            final_score = total_score * boost

            if final_score > best_score:
                best_score = final_score

                # Determine confidence
                if final_score > 0.8:
                    confidence = 'high'
                elif final_score > 0.6:
                    confidence = 'medium'
                else:
                    confidence = 'low'

                best_match = TrackMatch(
                    video_id=f"{video_info['ID']}_{video_info['Coder']}_{Path(video_info['SourceFile']).stem}",
                    track_id=track_id,
                    similarity_score=final_score,
                    face_score=individual_scores['face'],
                    upper_score=individual_scores['upper'],
                    lower_score=individual_scores['lower'],
                    num_frames=num_frames,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    confidence=confidence,
                    timepoint=timepoint,
                    is_reference=False
                )
                best_reason = None

        if best_match is None:
            reason = best_reason or "unable to identify a confident target track"
            return None, reason

        if best_match.similarity_score < self.min_score:
            reason = (f"best score {best_match.similarity_score:.3f} below minimum threshold "
                      f"{self.min_score:.3f}")
            return None, reason

        return best_match, None

    def process_child(self, child_id: str):
        """
        Process all videos for a single child

        Args:
            child_id: Child ID to process
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing child: {child_id}")
        logger.info(f"{'='*60}")

        # Build reference profiles
        self.build_reference_profiles(child_id)

        # Get all videos for this child
        child_videos = self.df[self.df['ID'] == child_id]
        logger.info(f"Processing {len(child_videos)} videos for child {child_id}...")

        matches = []
        count = 0
        for idx, video_info in child_videos.iterrows():
            video_dict = video_info.to_dict()
            video_name = video_dict['SourceFile']
            logger.info(f"\n[{count+1}/{len(child_videos)}] {video_name}")

            match, miss_reason = self.identify_target_in_video(child_id, video_dict)
            timepoint_label = video_dict.get('timepoint')

            if match:
                matches.append(match)
                logger.info(f"  ✓ Track {match.track_id}: score={match.similarity_score:.3f} "
                      f"(face={match.face_score:.3f}, upper={match.upper_score:.3f}, "
                      f"lower={match.lower_score:.3f}) [{match.confidence}]")
                self._render_target_track(video_dict, match)
            else:
                if miss_reason:
                    logger.warning(f"  ✗ No match found: {miss_reason}")
                else:
                    logger.warning(f"  ✗ No match found")

            self._store_video_result(child_id, timepoint_label, video_dict, match, miss_reason)
            self._record_match_outcome(child_id, match is not None, miss_reason)
            count += 1
        self.match_results[child_id] = matches
        self.global_metrics['children_processed'] += 1

        # Save results
        self.save_results(child_id)

    def save_results(self, child_id: str):
        """Save matching results to JSON"""
        child_dir = self.output_dir / _sanitize_for_path(child_id)
        child_dir.mkdir(parents=True, exist_ok=True)

        reference_profiles = {
            timepoint: {
                'source_videos': profile.source_videos,
                'num_observations': profile.num_observations,
                'has_face': profile.face_feature is not None,
                'has_upper': profile.upper_feature is not None,
                'has_lower': profile.lower_feature is not None
            }
            for timepoint, profile in self.reference_profiles.get(child_id, {}).items()
        }

        reference_tracks = [
            {
                'source_file': source_file,
                'track_id': track_id,
                'selector_candidates': self.reference_track_diagnostics.get(child_id, {}).get(source_file, [])
            }
            for source_file, track_id in self.reference_videos.get(child_id, {}).items()
        ]

        matches_serialized = [
            {
                'video_id': m.video_id,
                'track_id': m.track_id,
                'similarity_score': float(m.similarity_score),
                'face_score': float(m.face_score),
                'upper_score': float(m.upper_score),
                'lower_score': float(m.lower_score),
                'num_frames': int(m.num_frames) if m.num_frames is not None else None,
                'start_frame': int(m.start_frame) if m.start_frame is not None else None,
                'end_frame': int(m.end_frame) if m.end_frame is not None else None,
                'confidence': m.confidence,
                'timepoint': m.timepoint,
                'is_reference': m.is_reference,
            }
            for m in self.match_results[child_id]
        ]

        metrics = self.child_metrics[child_id]
        total_videos = metrics['total_videos']
        successes = metrics['successes']
        failures = metrics['failures']
        failure_breakdown = self._format_failure_breakdown(metrics['failure_reasons'], total_videos)
        summary = {
            'total_videos': total_videos,
            'matches_found': successes,
            'failures': failures,
            'success_rate': float(successes) / float(total_videos) if total_videos else 0.0,
            'failure_rate': float(failures) / float(total_videos) if total_videos else 0.0,
            'failure_breakdown': failure_breakdown,
            'total_matched_videos': successes,
            'high_confidence': sum(1 for m in self.match_results[child_id] if m.confidence == 'high'),
            'medium_confidence': sum(1 for m in self.match_results[child_id] if m.confidence == 'medium'),
            'low_confidence': sum(1 for m in self.match_results[child_id] if m.confidence == 'low'),
            'avg_score': float(np.mean([m.similarity_score for m in self.match_results[child_id]])) if self.match_results[child_id] else 0.0
        }

        results = {
            'child_id': child_id,
            'reference_profiles': reference_profiles,
            'reference_tracks': reference_tracks,
            'matches': matches_serialized,
            'summary': summary,
        }

        summary_path = child_dir / f"{_sanitize_for_path(child_id)}_target_identification.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)

        summary_by_timepoint: Dict[str, List[Dict[str, any]]] = defaultdict(list)
        for entry in matches_serialized:
            tp_key = entry.get('timepoint') or 'unknown'
            summary_by_timepoint[tp_key].append(entry)

        for tp_key, entries in summary_by_timepoint.items():
            tp_dir = self._get_output_subdir(self.output_dir, child_id, tp_key)
            with open(tp_dir / "matches.json", 'w') as f:
                json.dump(entries, f, indent=2, default=_json_default)

        logger.info(f"\nResults saved to: {summary_path}")
        logger.info(f"Summary: {summary['high_confidence']} high, "
              f"{summary['medium_confidence']} medium, "
              f"{summary['low_confidence']} low confidence matches")

    def save_global_summary(self):
        """Write a run-level summary of successes/failures across all children."""
        total_videos = self.global_metrics['total_videos']
        successes = self.global_metrics['successes']
        failures = self.global_metrics['failures']

        per_child = {}
        for child_id, metrics in self.child_metrics.items():
            child_total = metrics['total_videos']
            child_successes = metrics['successes']
            child_failures = metrics['failures']
            per_child[child_id] = {
                'total_videos': child_total,
                'matches_found': child_successes,
                'failures': child_failures,
                'success_rate': float(child_successes) / float(child_total) if child_total else 0.0,
                'failure_rate': float(child_failures) / float(child_total) if child_total else 0.0,
                'failure_breakdown': self._format_failure_breakdown(metrics['failure_reasons'], child_total),
            }

        run_summary = {
            'total_videos': total_videos,
            'matches_found': successes,
            'failures': failures,
            'success_rate': float(successes) / float(total_videos) if total_videos else 0.0,
            'failure_rate': float(failures) / float(total_videos) if total_videos else 0.0,
            'failure_breakdown': self._format_failure_breakdown(self.global_metrics['failure_reasons'], total_videos),
            'children_processed': self.global_metrics['children_processed'],
            'per_child': per_child,
        }

        summary_path = self.output_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(run_summary, f, indent=2, default=_json_default)

    def process_all(self):
        """Process all children"""
        # Get unique child IDs
        child_ids = self.df['ID'].unique()

        if self.filter_ids:
            child_ids = [cid for cid in child_ids if cid in self.filter_ids]

        logger.info(f"Processing {len(child_ids)} children: {', '.join(child_ids)}")

        count = 0
        for child_id in child_ids:
            logger.info(f"Processing child {count+1}/{len(child_ids)}: {child_id}")
            count += 1
            try:
                self.process_child(child_id)
            except Exception as e:
                logger.error(f"Error processing child {child_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.save_global_summary()

def main():
    parser = argparse.ArgumentParser(
        description='Identify target child across multiple videos using embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all children
  python batch_identify_target.py annotations.csv --embeddings-dir /path/to/pipeline_outputs

  # Process specific children
  python batch_identify_target.py annotations.csv --embeddings-dir /path/to/outputs --ids A2P7X9N8L7 B3Q8Y1M9K2
        """
    )

    parser.add_argument('csv_file', help='CSV file containing video annotations')
    parser.add_argument('--embeddings-dir', required=True,
                       help='Base directory containing tracks_hdf5 subdirectories (e.g., pipeline_outputs/)')
    parser.add_argument('--output-dir', default='./target_identification_results',
                       help='Output directory for results (default: ./target_identification_results)')
    parser.add_argument('--ids', nargs='+',
                       help='Filter by specific child IDs (space-separated list)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering of target-track visualizations (bbox + pose)')
    parser.add_argument('--video-dir', default='/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized',
                       help='Base directory containing source videos for path conversion')
    parser.add_argument('--rmm', action='store_true', default=True,
                       help='Use RMM dataset path conversion logic when resolving video paths')
    parser.add_argument('--face-only', action='store_true', default=True,
                       help='Only use face embeddings when scoring tracks')
    parser.add_argument('--min-score', type=float, default=0.7,
                       help='Minimum similarity score required to accept a match (default: 0.5)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.embeddings_dir):
        logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
        sys.exit(1)

    if args.video_dir and not os.path.exists(args.video_dir):
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)

    # Run identification
    identifier = TargetIdentifier(
        args.csv_file,
        args.embeddings_dir,
        args.output_dir,
        filter_ids=args.ids,
        render=args.render,
        video_base_dir=args.video_dir,
        rmm=args.rmm,
        face_only=args.face_only,
        min_score=args.min_score,
    )

    logger.info("="*60)
    logger.info("Starting target identification process")
    logger.info("="*60)

    identifier.process_all()

    logger.info("\n" + "="*60)
    logger.info("Target identification complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
