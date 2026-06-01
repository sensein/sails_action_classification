"""
reconstruct_results_json.py
Rebuilds missing results.json for folders that have masks/ and bboxes/ but no results.json.

Usage:
python reconstruct_results_json.py \
    --results_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
    --video_root /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def find_video_path(video_name, video_root):
    """Reconstruct the source video path from the folder name."""
    # Parse sub, ses, task, run from folder name
    # e.g. sub-A4E8K1L5Y2_ses-01_task-other_run-01_desc-processed_beh
    parts = video_name.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), None)
    ses = next((p for p in parts if p.startswith("ses-")), None)
    if not sub or not ses:
        return None
    video_file = f"{video_name}.mp4"
    video_path = os.path.join(video_root, sub, ses, "beh", video_file)
    return video_path


def reconstruct_results_json(video_dir, video_root):
    video_name = video_dir.name
    masks_dir  = video_dir / "masks"
    bboxes_dir = video_dir / "bboxes"

    mask_files = sorted(masks_dir.glob("frame_*.npy"))
    if not mask_files:
        print(f"  ⚠ No mask files in {video_name}")
        return False

    video_path = find_video_path(video_name, video_root)

    per_frame = {}
    total_frames = 0
    frames_with_detections = 0

    for mf in tqdm(mask_files, desc=f"  scanning {video_name}", leave=False):
        frame_idx = int(mf.stem.split("_")[1])
        total_frames = max(total_frames, frame_idx + 1)

        masks = np.load(mf)
        n_obj = masks.shape[0]

        bboxes = []
        bf = bboxes_dir / mf.name
        if bf.exists():
            bboxes = np.load(bf).tolist()

        if n_obj == 0:
            continue

        frames_with_detections += 1
        per_frame[str(frame_idx)] = {
            "num_objects": n_obj,
            "object_ids":  list(range(1, n_obj + 1)),
            "scores":      [None] * n_obj,
            "bboxes_xyxy": bboxes,
        }

    results = {
        "video_path":               video_path,
        "video_name":               video_name,
        "prompt":                   "Human Young Child",
        "total_frames":             total_frames,
        "detection_mode":           "multi (all verified humans)",
        "backend_used":             "transformers",
        "verification":             "YOLOv8 person + aspect ratio",
        "min_confidence":           0.15,
        "max_aspect_ratio":         3.5,
        "resize_shorter_side":      192,
        "sam3_resolution":          "reconstructed",
        "original_resolution":      "reconstructed",
        "frames_processed":         total_frames,
        "frames_with_detections":   frames_with_detections,
        "frames_rejected_not_human": 0,
        "per_frame":                per_frame,
    }

    out_path = video_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  ✓ Written: {out_path}  ({frames_with_detections} frames with detections)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--video_root",  required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    # Find folders with masks/ but no results.json
    missing = [
        d for d in sorted(results_dir.iterdir())
        if d.is_dir()
        and (d / "masks").exists()
        and not (d / "results.json").exists()
    ]

    print(f"Found {len(missing)} folders missing results.json")
    ok = fail = 0
    for d in missing:
        success = reconstruct_results_json(d, args.video_root)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok}  Failed: {fail}")


if __name__ == "__main__":
    main()