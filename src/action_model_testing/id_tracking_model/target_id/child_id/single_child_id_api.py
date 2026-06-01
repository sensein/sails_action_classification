#!/usr/bin/env python3
"""
Single Child Identification API

High-level API wrapper for child identification in single-child videos.
Provides a simple interface that takes tracking JSON and configuration parameters.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import cv2
import subprocess
import os

from .single_child_identification import (
    Track, AnnotationInfo, ChildIdentificationConfig,
    identify_single_child, ChildResult
)


def convert_tracking_json_to_tracks(tracking_data: Dict[str, Any]) -> List[Track]:
    """
    Convert tracking JSON results to Track objects.

    Parameters
    ----------
    tracking_data : dict
        The loaded tracking JSON data containing video_metadata and tracking_results

    Returns
    -------
    tracks : list of Track
        List of Track objects ready for child identification
    """
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
        sorted_frame_nums = sorted([int(f) for f in frames_data.keys()])

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


def child_result_to_dict(
    result: ChildResult,
    tracking_data: Dict[str, Any],
    config: ChildIdentificationConfig,
    video_name: str,
    processing_time: float
) -> Dict[str, Any]:
    """
    Convert ChildResult object to JSON-serializable dictionary.

    Parameters
    ----------
    result : ChildResult
        The result from identify_single_child
    tracking_data : dict
        Original tracking data for metadata
    config : ChildIdentificationConfig
        Configuration used for identification
    video_name : str
        Name of the video file
    processing_time : float
        Time taken to process in seconds

    Returns
    -------
    result_dict : dict
        JSON-serializable dictionary containing all results
    """
    diagnostics = result.diagnostics

    result_dict = {
        "video_info": {
            "filename": video_name,
            "source_video": tracking_data['video_metadata']['input_path'],
            "fps": tracking_data['video_metadata']['fps'],
            "total_frames": tracking_data['video_metadata']['total_frames'],
            "width": tracking_data['video_metadata'].get('width', 'unknown'),
            "height": tracking_data['video_metadata'].get('height', 'unknown'),
            "processing_time_seconds": round(processing_time, 2)
        },

        "child_identification": {
            "selected_track_ids": result.child_track_id_sequence,
            "confidence": round(result.confidence, 4),
            "uncertainty": result.uncertainty,
            "num_segments": len(result.segments),
            "total_duration_seconds": sum(seg.duration_seconds() for seg in result.segments),
            "segments": [
                {
                    "track_id": seg.id,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "duration_seconds": round(seg.duration_seconds(), 2),
                    "duration_frames": seg.duration_frames()
                }
                for seg in result.segments
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
                    "skeleton_prob": round(node.evidence.p_skeleton, 4) if node.evidence.p_skeleton is not None else None,
                    "rigidity_prob": round(node.evidence.p_rigidity, 4) if node.evidence.p_rigidity is not None else None
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
            "age_estimation_method": config.age_estimation_method,
            "enable_body_visibility_filter": config.enable_body_visibility_filter,
            "min_visible_keypoints": config.min_visible_keypoints,
            "min_track_frames": config.min_track_frames,
            "sampling_percentage": config.sampling_percentage,
            "sampling_max_frames": config.sampling_max_frames_per_track,
            "age_child_years_threshold": config.age_child_years_threshold,
            "enable_skeleton_ratios": config.enable_skeleton_ratios,
            "skeleton_min_confidence": config.skeleton_min_confidence,
            "min_rigidity_score": config.min_rigidity_score,
        }
    }

    return result_dict


def create_child_video(
    video_path: str,
    child_result: ChildResult,
    tracking_data: Dict[str, Any],
    output_path: Path,
    max_frames: Optional[int] = None
) -> bool:
    """Create video with bounding boxes around identified child"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try to use ffmpeg for h264 encoding, fall back to cv2 if not available
        use_ffmpeg = True
        proc = None

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Using ffmpeg h264 encoding")
        except FileNotFoundError:
            # Fallback to cv2 video writer
            use_ffmpeg = False
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print("ffmpeg not found, using cv2 video writer")

        # Create frame-to-bbox mapping for child segments
        child_frame_bboxes = {}

        # Find the original tracks that correspond to child segments
        for segment in child_result.segments:
            # Find matching track by ID in tracking_results
            track_key = str(segment.id)
            if track_key in tracking_data['tracking_results']:
                track_data = tracking_data['tracking_results'][track_key]

                # Map frame numbers to bboxes for this segment's time range
                for frame_str, frame_data in track_data['frames'].items():
                    frame_num = int(frame_str)
                    if segment.start_frame <= frame_num <= segment.end_frame:
                        child_frame_bboxes[frame_num] = frame_data['bbox']

        frame_count = 0
        processed_frames = 0

        print(f"Processing video: {total_frames} frames at {fps:.1f} fps")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Limit frames if specified
                if max_frames and processed_frames >= max_frames:
                    break

                # Draw bounding box if this frame contains the child
                if frame_count in child_frame_bboxes:
                    bbox = child_frame_bboxes[frame_count]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]

                    # Draw green bounding box for child
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Add child label
                    label = f"CHILD (conf: {child_result.confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Write frame based on encoding method
                if use_ffmpeg:
                    # Ensure frame size matches expected dimensions
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Write frame to ffmpeg stdin
                    try:
                        proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        # ffmpeg died early
                        err = proc.stderr.read().decode(errors="ignore")
                        print(f"ffmpeg exited early: {err}")
                        break
                else:
                    # Use cv2 video writer
                    out.write(frame)

                processed_frames += 1

                if processed_frames % 1000 == 0:
                    print(f"Processed {processed_frames} frames...")

        finally:
            cap.release()

            if use_ffmpeg and proc:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass

                # Wait for ffmpeg to finish
                rc = proc.wait()
                if rc != 0:
                    err = proc.stderr.read().decode(errors="ignore")
                    print(f"ffmpeg failed (code {rc}): {err}")
                    return False
            else:
                # Release cv2 video writer
                out.release()

        print(f"Video created: {output_path} ({processed_frames} frames)")
        return True

    except Exception as e:
        print(f"Error creating video: {e}")
        return False
        
def identify_child_in_video(
    tracking_json_path: str,
    video_path: str,
    video_output_path: Union[Path, str],
    age_estimation_method: str = 'siglip',
    enable_body_visibility_filter: bool = True,
    min_visible_keypoints: int = 4,
    min_track_frames: int = 10,
    sampling_percentage: float = 0.25,
    sampling_max_frames: int = 30,
    age_child_years_threshold: float = 10.0,
    enable_skeleton_ratios: bool = False,
    skeleton_min_confidence: float = 0.3,
    estimated_age_months: Optional[float] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Identify child in a single-child video from tracking results.

    This is a high-level API function that takes tracking JSON and configuration
    parameters, runs child identification, and returns results as a dictionary.

    Parameters
    ----------
    tracking_json_path : str
        Path to the tracking JSON file
    video_path : str
        Path to the video file (for extracting frames if needed)
    video_output_path : str or Path
        Path to save the output video with child bounding boxes
    age_estimation_method : str, default='siglip'
        Method for age estimation ('siglip' or 'deepface')
    enable_body_visibility_filter : bool, default=True
        Filter out detections with too few visible keypoints
    min_visible_keypoints : int, default=4
        Minimum number of visible keypoints required
    min_track_frames : int, default=10
        Minimum number of frames required for a track to be considered
    sampling_percentage : float, default=0.25
        Percentage of frames to sample from each track (0.0 to 1.0)
    sampling_max_frames : int, default=30
        Maximum number of frames to sample per track
    age_child_years_threshold : float, default=10.0
        Age threshold in years for child classification
    enable_skeleton_ratios : bool, default=False
        Enable skeleton ratio analysis for child detection
    skeleton_min_confidence : float, default=0.3
        Minimum keypoint confidence for skeleton ratio computation
    estimated_age_months : float, optional
        Estimated child age in months (if known)
    **kwargs : dict
        Additional configuration parameters to pass to ChildIdentificationConfig

    Returns
    -------
    result : dict
        Dictionary containing:
        - video_info: Video metadata and processing info
        - child_identification: Selected tracks and confidence
        - detailed_analysis: Per-track scores and evidence
        - configuration: Configuration used

    Examples
    --------
    >>> result = identify_child_in_video(
    ...     tracking_json_path='video_tracking.json',
    ...     video_path='video.mp4',
    ...     enable_skeleton_ratios=True,
    ...     skeleton_min_confidence=0.3
    ... )
    >>> print(f"Selected tracks: {result['child_identification']['selected_track_ids']}")
    >>> print(f"Confidence: {result['child_identification']['confidence']}")
    """
    # Load tracking data
    with open(tracking_json_path, 'r') as f:
        tracking_data = json.load(f)

    # Convert tracking data to Track objects
    tracks = convert_tracking_json_to_tracks(tracking_data)

    # Create annotations
    if estimated_age_months is None:
        # Default to 18 months if not provided
        estimated_age_months = 18.0

    annotations = AnnotationInfo(
        age_in_months=estimated_age_months,
        quality_flags={}
    )

    # Create configuration
    config = ChildIdentificationConfig(
        age_estimation_method=age_estimation_method,
        enable_body_visibility_filter=enable_body_visibility_filter,
        min_visible_keypoints=min_visible_keypoints,
        min_track_frames=min_track_frames,
        sampling_percentage=sampling_percentage,
        sampling_max_frames_per_track=sampling_max_frames,
        age_child_years_threshold=age_child_years_threshold,
        enable_skeleton_ratios=enable_skeleton_ratios,
        skeleton_min_confidence=skeleton_min_confidence,
        **kwargs
    )

    # Run child identification
    start_time = datetime.now()
    result = identify_single_child(tracks, annotations, config)
    processing_time = (datetime.now() - start_time).total_seconds()

    # Get video name from path
    video_name = Path(tracking_json_path).stem.replace('_tracking', '')

    # Convert result to dictionary
    result_dict = child_result_to_dict(
        result, tracking_data, config, video_name, processing_time
    )

    video_output_path = Path(video_output_path)
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    create_child_video(
        video_path, result, tracking_data,
        output_path=Path(video_output_path),
        max_frames=10000  # Limit to first 10,000 frames for performance
    )

    return result_dict
