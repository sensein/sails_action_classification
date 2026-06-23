"""
Visualize Tracking Results from Child ID Output

This script takes a child_id video or log file path and creates a visualization
video showing all tracking results (not just the identified child) from the
original tracking JSON data.

Usage:
    python visualize_tracking_from_child_id.py <child_id_video_or_log_path>
    python visualize_tracking_from_child_id.py <child_id_video_or_log_path> --max-frames 3000
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


def find_tracking_json_from_child_id_path(child_id_path: Path) -> Path | None:
    """
    Given a child_id video or log path, find the corresponding tracking JSON file.

    Expected structure:
    pipeline_outputs/
      subset_name/
        tracking/
          VIDEO_ID_CODER_BASENAME_tracking.json
        child_classifications/
          videos/
            VIDEO_ID_CODER_BASENAME_child_identified.mp4
          logs/
            VIDEO_ID_CODER_BASENAME_analysis.json
    """
    child_id_path = Path(child_id_path)

    # Extract the base filename
    if child_id_path.name.endswith('_child_identified.mp4'):
        base_name = child_id_path.stem.replace('_child_identified', '')
    elif child_id_path.name.endswith('_analysis.json'):
        base_name = child_id_path.stem.replace('_analysis', '')
    else:
        # Assume it's already the base name
        base_name = child_id_path.stem

    # Navigate up to find the tracking folder
    # From: subset_name/child_classifications/videos/file.mp4
    # To:   subset_name/tracking/file_tracking.json

    current = child_id_path.parent

    # Go up until we find a directory containing 'tracking' subdirectory
    max_depth = 5
    for _ in range(max_depth):
        tracking_dir = current / "tracking"
        if tracking_dir.exists() and tracking_dir.is_dir():
            # Found the tracking directory
            tracking_json = tracking_dir / f"{base_name}_tracking.json"
            if tracking_json.exists():
                return tracking_json
            else:
                print(f"Warning: Expected tracking file not found: {tracking_json}")
                return None

        # Go up one level
        parent = current.parent
        if parent == current:
            break
        current = parent

    print(f"Error: Could not find tracking directory from path: {child_id_path}")
    return None


def find_original_video_path(tracking_data: dict[str, Any]) -> str | None:
    """Extract the original video path from tracking JSON metadata"""
    try:
        video_path: str = tracking_data['video_metadata']['input_path']
        if os.path.exists(video_path):
            return video_path
        else:
            print(f"Warning: Original video not found at: {video_path}")
            return None
    except KeyError:
        print("Error: Could not find video path in tracking metadata")
        return None


def create_tracking_visualization(
    video_path: str,
    tracking_data: dict[str, Any],
    output_path: Path,
    max_frames: int | None = None
) -> bool:
    """
    Create a visualization video showing all tracking results.

    Args:
        video_path: Path to original video
        tracking_data: Loaded tracking JSON data
        output_path: Where to save the visualization video
        max_frames: Optional limit on number of frames to process
    """
    ffmpeg_failed = False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

        # Create frame-to-detections mapping
        frame_detections: dict[int, list[tuple[int, Any]]] = {}

        for track_id_str, track_data in tracking_data['tracking_results'].items():
            track_id = int(track_id_str)

            for frame_str, frame_data in track_data['frames'].items():
                frame_num = int(frame_str)
                bbox = frame_data['bbox']

                if frame_num not in frame_detections:
                    frame_detections[frame_num] = []
                frame_detections[frame_num].append((track_id, bbox))

        # Setup ffmpeg for h264 encoding
        use_ffmpeg = True
        proc: subprocess.Popen[bytes] | None = None
        out: cv2.VideoWriter | None = None

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(output_path)
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Using ffmpeg h264 encoding")
        except FileNotFoundError:
            # Fallback to cv2 video writer
            use_ffmpeg = False
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print("ffmpeg not found, using cv2 video writer")

        # Define colors for different track IDs (cycle through a palette)
        color_palette = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]

        def get_color_for_track(track_id: int) -> tuple[int, int, int]:
            return color_palette[track_id % len(color_palette)]

        frame_count = 0
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Limit frames if specified
                if max_frames and processed_frames >= max_frames:
                    break

                # Draw all tracking bounding boxes for this frame
                if frame_count in frame_detections:
                    detections = frame_detections[frame_count]

                    for track_id, bbox in detections:
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]

                        # Get color for this track ID
                        color = get_color_for_track(track_id)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add track ID label
                        label = f"ID: {track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 6),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 3),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add frame counter in top-left
                frame_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(frame, frame_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Write frame
                if use_ffmpeg and proc is not None and proc.stdin is not None:
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                    try:
                        proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        err = proc.stderr.read().decode(errors="ignore") if proc.stderr is not None else ""
                        print(f"ffmpeg exited early: {err}")
                        break
                elif out is not None:
                    out.write(frame)

                processed_frames += 1

                if processed_frames % 1000 == 0:
                    print(f"Processed {processed_frames} frames...")

        finally:
            cap.release()

            if use_ffmpeg and proc is not None:
                if proc.stdin is not None:
                    with contextlib.suppress(BrokenPipeError):
                        proc.stdin.close()

                rc = proc.wait()
                if rc != 0:
                    err = proc.stderr.read().decode(errors="ignore") if proc.stderr is not None else ""
                    print(f"ffmpeg failed (code {rc}): {err}")
                    ffmpeg_failed = True
            elif out is not None:
                out.release()

        if ffmpeg_failed:
            return False

        print(f"✓ Visualization created: {output_path} ({processed_frames} frames)")
        return True

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create tracking visualization from child_id output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From child_id video
  python visualize_tracking_from_child_id.py /path/to/child_classifications/videos/123_ABC_video_child_identified.mp4

  # From child_id log
  python visualize_tracking_from_child_id.py /path/to/child_classifications/logs/123_ABC_video_analysis.json

  # Limit to first 3000 frames
  python visualize_tracking_from_child_id.py /path/to/video.mp4 --max-frames 3000
        """
    )
    parser.add_argument('child_id_path', help='Path to child_id video or log file')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--output', help='Custom output path (default: auto-generated in tracking folder)')

    args = parser.parse_args()

    child_id_path = Path(args.child_id_path)

    if not child_id_path.exists():
        print(f"Error: File not found: {child_id_path}")
        sys.exit(1)

    # Find corresponding tracking JSON
    print(f"Looking for tracking JSON from: {child_id_path}")
    tracking_json_path = find_tracking_json_from_child_id_path(child_id_path)

    if not tracking_json_path:
        print("Error: Could not locate tracking JSON file")
        sys.exit(1)

    print(f"Found tracking JSON: {tracking_json_path}")

    # Load tracking data
    try:
        with open(tracking_json_path) as f:
            tracking_data = json.load(f)
    except Exception as e:
        print(f"Error loading tracking JSON: {e}")
        sys.exit(1)

    # Find original video
    video_path = find_original_video_path(tracking_data)
    if not video_path:
        print("Error: Could not locate original video")
        sys.exit(1)

    print(f"Found original video: {video_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Place in videos folder at same level as tracking folder
        # From: subset_name/tracking/file_tracking.json
        # To:   subset_name/videos/file_tracking_viz.mp4
        videos_dir = tracking_json_path.parent.parent / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        base_name = tracking_json_path.stem.replace('_tracking', '')
        output_path = videos_dir / f"{base_name}.mp4"

    print(f"Output will be saved to: {output_path}")

    # Create visualization
    success = create_tracking_visualization(
        video_path,
        tracking_data,
        output_path,
        max_frames=args.max_frames
    )

    if success:
        print(f"\n✓ Success! Visualization saved to: {output_path}")
        sys.exit(0)
    else:
        print("\n✗ Failed to create visualization")
        sys.exit(1)


if __name__ == "__main__":
    main()