"""
generate_masked_videos.py
Regenerates masked_video.mp4 from saved masks/ and bboxes/ .npy files
produced by the SAM3 pipeline.

Usage:
conda activate hf_env
python generate_masked_videos.py \
    --results_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
    --multi
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

MASK_COLOR  = (0, 255, 0)
MASK_ALPHA  = 0.35
BBOX_COLOR  = (0, 0, 255)
BBOX_THICK  = 2
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 0, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
FONT_THICK  = 2

MULTI_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 165, 0), (0, 128, 255), (128, 0, 255), (255, 255, 0),
]


def draw_overlay(frame, masks_np, bboxes, obj_ids, scores, multi_color=False):
    out = frame.copy()
    H, W = frame.shape[:2]
    n = len(obj_ids)
    for i in range(n):
        if multi_color and n > 1:
            color_mask = MULTI_COLORS[i % len(MULTI_COLORS)]
            color_bbox = color_mask
            color_lbl  = color_mask
        else:
            color_mask = MASK_COLOR
            color_bbox = BBOX_COLOR
            color_lbl  = LABEL_BG

        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i]
            if mask.shape == (H, W) and np.any(mask):
                overlay = out.copy()
                overlay[mask > 0] = color_mask
                out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)

        if i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            if x1 >= 0:
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                              color_bbox, BBOX_THICK)
                oid = obj_ids[i]
                label = f"ID:{oid}"
                if i < len(scores) and scores[i] is not None and scores[i] != "":
                    try:
                        label += f" {float(scores[i]):.2f}"
                    except (TypeError, ValueError):
                        pass
                (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
                lx1 = int(x1)
                ly1 = max(int(y1) - th - bl - 4, 0)
                lx2 = int(x1) + tw + 4
                ly2 = max(int(y1), th + bl + 4)
                cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color_lbl, -1)
                cv2.putText(out, label, (lx1 + 2, ly2 - bl - 2),
                            FONT, FONT_SCALE, LABEL_COLOR, FONT_THICK, cv2.LINE_AA)
    return out


def load_frame_data(video_out_dir):
    """Load per-frame masks, bboxes, obj_ids, scores from a video output dir."""
    results_path = os.path.join(video_out_dir, "results.json")
    if not os.path.exists(results_path):
        return None, None
    with open(results_path) as f:
        results = json.load(f)

    masks_dir  = os.path.join(video_out_dir, "masks")
    bboxes_dir = os.path.join(video_out_dir, "bboxes")

    per_frame = {}
    for frame_str, info in results.get("per_frame", {}).items():
        fi = int(frame_str)
        if info["num_objects"] == 0:
            continue
        obj_ids = info["object_ids"]
        scores  = info.get("scores", [])

        bboxes = None
        bbox_file = os.path.join(bboxes_dir, f"frame_{fi:06d}.npy")
        if os.path.exists(bbox_file):
            bboxes = np.load(bbox_file)
        else:
            bboxes = np.array(info.get("bboxes_xyxy", []), dtype=np.int32)

        masks = None
        mask_file = os.path.join(masks_dir, f"frame_{fi:06d}.npy")
        if os.path.exists(mask_file):
            masks = np.load(mask_file)

        per_frame[fi] = {
            "obj_ids": obj_ids,
            "scores":  scores,
            "bboxes":  bboxes,
            "masks":   masks,
        }
    return results, per_frame


def regenerate_masked_video(video_path, per_frame_data, output_path, multi_color=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Cannot open source video: {video_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0
    pbar = tqdm(total=total, desc="    rendering", leave=False)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in per_frame_data:
                fd = per_frame_data[frame_idx]
                frame = draw_overlay(
                    frame,
                    fd.get("masks"),
                    fd.get("bboxes", np.zeros((0, 4), dtype=np.int32)),
                    fd.get("obj_ids", []),
                    fd.get("scores", []),
                    multi_color=multi_color,
                )
            proc.stdin.write(frame.tobytes())
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        proc.stdin.close()
        proc.wait()

    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Directory containing per-video output folders")
    ap.add_argument("--multi", action="store_true",
                    help="Use multi-color overlays for multiple objects")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing masked_video.mp4 files")
    ap.add_argument("--only", type=str, default=None,
                    help="Only regenerate for this video name (stem)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    video_dirs = [d for d in results_dir.iterdir()
                  if d.is_dir() and (d / "results.json").exists()]
    video_dirs.sort()

    if args.only:
        video_dirs = [d for d in video_dirs if d.name == args.only]

    print(f"Found {len(video_dirs)} video output dirs")

    done = skipped = failed = 0
    for vd in video_dirs:
        out_mp4 = vd / "masked_video.mp4"
        if out_mp4.exists() and not args.force:
            skipped += 1
            continue

        results, per_frame = load_frame_data(str(vd))
        if results is None:
            print(f"  ⚠ {vd.name}: no results.json")
            failed += 1
            continue

        src_video = results.get("video_path")
        if not src_video or not os.path.exists(src_video):
            print(f"  ⚠ {vd.name}: source video missing ({src_video})")
            failed += 1
            continue

        print(f"  → {vd.name}  ({len(per_frame)} frames with detections)")
        ok = regenerate_masked_video(src_video, per_frame, str(out_mp4),
                                     multi_color=args.multi)
        if ok:
            done += 1
            print(f"    ✓ {out_mp4}")
        else:
            failed += 1
            print(f"    ✗ ffmpeg failed")

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()