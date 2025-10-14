#!/usr/bin/env python3
"""
EntitySAM Bounding Box Overlay Script
Overlays bounding boxes from EntitySAM inference results onto the original video.

Usage:
    python entitysam_bbox_overlay.py --video /path/to/video.mp4
    python entitysam_bbox_overlay.py --video /path/to/video.mp4 --video_id custom_id
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV (cv2) is required for video processing. "
        "Install it with: pip install opencv-python"
    )

import numpy as np
from tqdm import tqdm


def find_entitysam_results(video_path, video_id=None, entitysam_out_dir="/orcd/data/satra/001/users/brukew/entitysam_out"):
    """Find EntitySAM results directory and files."""
    video_path = Path(video_path)

    if video_id is None:
        video_id = video_path.stem

    results_dir = Path(entitysam_out_dir) / video_id
    pred_json = results_dir / "pred.json"

    if not results_dir.exists():
        raise FileNotFoundError(f"EntitySAM results directory not found: {results_dir}")

    if not pred_json.exists():
        raise FileNotFoundError(f"EntitySAM prediction file not found: {pred_json}")

    return results_dir, pred_json


def load_bbox_data(pred_json_path, min_area=100):
    """Load bounding box data from EntitySAM results."""
    with open(pred_json_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations'][0]['annotations']

    # Group bboxes by frame
    frame_bboxes = {}
    max_category_id = 0

    for frame_idx, frame_data in enumerate(annotations):
        frame_name = frame_data['file_name']
        segments_info = frame_data['segments_info']

        bboxes = []
        for segment in segments_info:
            area = segment['area']

            # Pre-filter by area
            if area < min_area:
                continue

            # Convert bbox to integers once
            bbox = [int(v) for v in segment['bbox']]  # [x, y, width, height]
            category_id = segment['category_id']

            # Track max category ID
            max_category_id = max(max_category_id, category_id)

            bboxes.append({
                'bbox': bbox,
                'category_id': category_id,
                'area': area,
                'id': segment['id']
            })

        frame_bboxes[frame_idx] = {
            'frame_name': frame_name,
            'bboxes': bboxes
        }

    return frame_bboxes, max_category_id


def generate_colors(num_categories):
    """Generate distinct colors for different categories."""
    np.random.seed(42)  # For consistent colors
    colors = []
    for _ in range(num_categories):
        color = tuple(int(c) for c in np.random.randint(0, 256, size=3))
        colors.append(color)
    return colors


def draw_bboxes_on_frame(frame, bboxes, colors):
    """Draw bounding boxes on a frame (modifies frame in-place)."""
    for bbox_data in bboxes:
        bbox = bbox_data['bbox']
        category_id = bbox_data['category_id']
        area = bbox_data['area']
        obj_id = bbox_data['id']

        x, y, w, h = bbox

        # Use category_id to select color (mod length to wrap around)
        color = colors[category_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label with ID and area
        label = f"ID:{obj_id} A:{area}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Draw label background
        cv2.rectangle(frame,
                     (x, y - label_size[1] - 10),
                     (x + label_size[0], y),
                     color, -1)

        # Draw label text
        cv2.putText(frame, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def create_bbox_overlay_video(video_path, frame_bboxes, output_path, max_category_id):
    """Create video with bounding box overlays using direct ffmpeg pipe."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback if frame count is unreliable
    if total_frames <= 0:
        total_frames = len(frame_bboxes)

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Generate colors for categories
    colors = generate_colors(max_category_id + 1)

    # Setup ffmpeg process with stdin pipe for direct frame writing
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
        "-an",  # No audio
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # High quality
        str(output_path)
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    try:
        frame_idx = 0
        print("Processing frames and encoding video...")

        with tqdm(total=total_frames, desc="Adding bounding boxes") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply bounding boxes if available for this frame
                if frame_idx in frame_bboxes:
                    bboxes = frame_bboxes[frame_idx]['bboxes']
                    frame = draw_bboxes_on_frame(frame, bboxes, colors)

                # Write frame directly to ffmpeg stdin (no disk I/O!)
                proc.stdin.write(frame.tobytes())

                frame_idx += 1
                pbar.update(1)

        cap.release()

        # Close stdin to signal end of input
        proc.stdin.close()

        # Wait for ffmpeg to finish
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        print(f"Video encoded successfully: {output_path}")

    except Exception as e:
        # Clean up ffmpeg process on error
        proc.kill()
        proc.wait()
        raise e
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay bounding boxes from EntitySAM results onto original video."
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Path to original video file"
    )
    parser.add_argument(
        '--video_id',
        type=str,
        default=None,
        help="Video ID for EntitySAM results lookup (default: auto-detect from filename)"
    )
    parser.add_argument(
        '--entitysam_out',
        type=str,
        default="/orcd/data/satra/001/users/brukew/entitysam_out",
        help="EntitySAM output directory (default: /orcd/data/satra/001/users/brukew/entitysam_out)"
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=100,
        help="Minimum bounding box area to display (default: 100)"
    )

    args = parser.parse_args()

    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    # Find EntitySAM results
    print(f"Looking for EntitySAM results...")
    results_dir, pred_json = find_entitysam_results(args.video, args.video_id, args.entitysam_out)
    print(f"Found results: {results_dir}")

    # Load bounding box data (with pre-filtering by min_area)
    print("Loading bounding box data...")
    frame_bboxes, max_category_id = load_bbox_data(pred_json, args.min_area)
    print(f"Loaded bboxes for {len(frame_bboxes)} frames")

    # Calculate total number of bboxes and entities
    total_bboxes = sum(len(frame_data['bboxes']) for frame_data in frame_bboxes.values())
    unique_ids = set()
    for frame_data in frame_bboxes.values():
        for bbox_data in frame_data['bboxes']:
            unique_ids.add(bbox_data['id'])

    print(f"Total bounding boxes: {total_bboxes}")
    print(f"Unique entities: {len(unique_ids)}")

    # Set output path
    output_path = results_dir / "bbox_overlay.mp4"

    # Create overlay video
    print(f"Creating overlay video: {output_path}")
    create_bbox_overlay_video(video_path, frame_bboxes, output_path, max_category_id)

    print("\n" + "=" * 60)
    print("BOUNDING BOX OVERLAY COMPLETE")
    print("=" * 60)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {len(frame_bboxes)}")
    print(f"Total bounding boxes: {total_bboxes}")
    print(f"Unique entities: {len(unique_ids)}")
    print(f"Minimum area filter: {args.min_area}")
    print("=" * 60)


if __name__ == "__main__":
    main()