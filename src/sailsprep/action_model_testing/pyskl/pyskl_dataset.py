"""
Convert ViTPose per-clip JSONs + split CSV into pyskl pickle format.

Uses ffprobe (subprocess) to read real video dimensions. Avoids cv2 which
segfaults on network-mounted videos on this cluster.

Usage:
    python json_to_pyskl.py --task loco --out /path/to/loco_pyskl.pkl
    python json_to_pyskl.py --task rmm  --out /path/to/rmm_pyskl.pkl
"""

import os
import sys
import json
import pickle
import argparse
import traceback
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
POSE_ROOT  = "/orcd/scratch/bcs/001/sensein/sails/pose_h5_outputs/vit_pose_clip_video"
SPLIT_CSV  = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

TASK_CLASSES = {
    "loco": ["Crawling", "Cruising", "Running", "Vehicle", "Walking"],
    "rmm":  ["Hands_flapping", "Jumping", "Rocking", "Spinning"],
}

CLIP_T = 30   # max temporal length per clip (ann-frames @ 15 fps)

COCO_17_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]
NUM_KP = len(COCO_17_NAMES)  # 17

DEFAULT_HW = (720, 1280)  # fallback if ffprobe fails


# ============================================================
# ffprobe helper — gets real (H, W) without cv2
# ============================================================
def ffprobe_dims(video_path, timeout=10):
    """Return (H, W) using ffprobe. Returns None on any failure."""
    if not os.path.exists(video_path):
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x",
             video_path],
            timeout=timeout, stderr=subprocess.STDOUT,
        ).decode().strip()
        # Output like "1920x1080"
        if "x" in out:
            w, h = out.split("x")
            return (int(h), int(w))
    except Exception:
        return None
    return None


# ============================================================
# HELPERS
# ============================================================
def build_video_to_split(split_csv_path):
    """
    Returns dict: basename_no_ext -> (split_name, video_path)
    """
    df = pd.read_csv(split_csv_path)
    if "video_path" not in df.columns or "split" not in df.columns:
        raise ValueError("Split CSV must have 'video_path' and 'split' columns")
    mapping = {}
    for _, row in df.iterrows():
        vp = str(row["video_path"]).strip()
        sp = str(row["split"]).strip().lower()
        basename = os.path.splitext(os.path.basename(vp))[0]
        mapping[basename] = (sp, vp)
    return mapping


def parse_video_key_from_filename(fname):
    """
    Filename pattern: <basename>_<a>_<b>_<c>_<d>_clip<N>.json
    where basename may itself contain underscores.
    Strip trailing 'clipN' then up to 4 trailing integer tokens.
    """
    parts = fname.replace(".json", "").split("_")
    if parts and parts[-1].startswith("clip"):
        parts = parts[:-1]
    for _ in range(4):
        if parts and parts[-1].lstrip("-").isdigit():
            parts.pop()
        else:
            break
    return "_".join(parts)


def resolve_split_for_clip(data, fname, video_split):
    """
    Returns (split_name, video_path, matched_key) or (None, None, tried_key).
    """
    video_key = (data.get("video") or "").strip()
    if video_key and video_key in video_split:
        sp, vp = video_split[video_key]
        return sp, vp, video_key

    fn_key = parse_video_key_from_filename(fname)
    if fn_key in video_split:
        sp, vp = video_split[fn_key]
        return sp, vp, fn_key

    # startswith fallback
    if video_key:
        for k, (sp, vp) in video_split.items():
            if k.startswith(video_key) or video_key.startswith(k):
                return sp, vp, k
    if fn_key:
        for k, (sp, vp) in video_split.items():
            if k.startswith(fn_key) or fn_key.startswith(k):
                return sp, vp, k

    return None, None, (fn_key or video_key)


def json_to_arrays(json_data, clip_t=CLIP_T):
    """
    Return (keypoint (1,T,17,2) float32, score (1,T,17) float32, total_frames int).
    """
    frames_dict = json_data.get("frames", {})
    if not frames_dict:
        return None, None, 0

    try:
        ann_frames_present = sorted(int(k) for k in frames_dict.keys())
    except Exception:
        return None, None, 0
    if not ann_frames_present:
        return None, None, 0

    clip_start = ann_frames_present[0]

    kp    = np.zeros((1, clip_t, NUM_KP, 2), dtype=np.float32)
    score = np.zeros((1, clip_t, NUM_KP),    dtype=np.float32)

    for af_str, kmap in frames_dict.items():
        try:
            af_int = int(af_str)
        except Exception:
            continue
        t_idx = af_int - clip_start
        if t_idx < 0 or t_idx >= clip_t:
            continue
        if not isinstance(kmap, dict):
            continue
        for kp_idx, kp_name in enumerate(COCO_17_NAMES):
            v = kmap.get(kp_name)
            if v is None:
                continue
            try:
                kp[0, t_idx, kp_idx, 0] = float(v.get("x", 0.0))
                kp[0, t_idx, kp_idx, 1] = float(v.get("y", 0.0))
                score[0, t_idx, kp_idx] = float(v.get("confidence", 0.0))
            except Exception:
                continue

    total_frames = int(np.any(score > 0, axis=(0, 2)).sum())
    return kp, score, total_frames


# ============================================================
# MAIN CONVERSION
# ============================================================
def convert_task(task, out_path, pose_root, split_csv, min_frames=5,
                 print_every=500):
    classes = TASK_CLASSES[task]
    label_map = {name: i for i, name in enumerate(classes)}
    print(f"Task      : {task}")
    print(f"Classes   : {label_map}")
    print(f"Pose root : {pose_root}/{task}")
    print(f"Output    : {out_path}")
    print(f"img_shape : from ffprobe (per video, cached); fallback {DEFAULT_HW}\n")

    video_split = build_video_to_split(split_csv)
    print(f"Split CSV : {len(video_split)} videos indexed\n")

    dim_cache = {}    # video_path -> (H, W)

    anns = []
    split_buckets = {"train": [], "val": [], "test": []}

    n_seen = n_ok = n_nosplit = n_empty = n_jsonerr = n_fallback_dims = 0
    per_class_per_split = defaultdict(lambda: defaultdict(int))
    missing_videos = set()

    task_root = os.path.join(pose_root, task)

    for class_name in classes:
        class_dir = os.path.join(task_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARN: missing class dir {class_dir}")
            continue

        json_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".json"))
        n_class = len(json_files)
        print(f"  [{task}/{class_name}] {n_class} JSONs ...")

        for idx, fname in enumerate(json_files):
            n_seen += 1
            if (idx + 1) % print_every == 0:
                print(f"    ... {idx+1}/{n_class}  (dim_cache={len(dim_cache)})")

            jpath = os.path.join(class_dir, fname)
            try:
                with open(jpath) as f:
                    data = json.load(f)
            except Exception as e:
                n_jsonerr += 1
                if n_jsonerr <= 5:
                    print(f"    JSON read error ({e}): {fname}")
                continue

            split_name, video_path, matched_key = resolve_split_for_clip(
                data, fname, video_split)
            if split_name is None or split_name not in split_buckets:
                n_nosplit += 1
                missing_videos.add(matched_key)
                continue

            kp, score, total_frames = json_to_arrays(data)
            if kp is None or total_frames < min_frames:
                n_empty += 1
                continue

            # Look up real video dims via ffprobe, cached per video
            if video_path not in dim_cache:
                dims = ffprobe_dims(video_path)
                if dims is None:
                    dims = DEFAULT_HW
                    n_fallback_dims += 1
                dim_cache[video_path] = dims
            hw = dim_cache[video_path]

            frame_dir = f"{task}_{class_name}_{os.path.splitext(fname)[0]}"
            ann = {
                "frame_dir":      frame_dir,
                "label":          label_map[class_name],
                "img_shape":      hw,
                "original_shape": hw,
                "total_frames":   total_frames,
                "keypoint":       kp,
                "keypoint_score": score,
            }
            anns.append(ann)
            split_buckets[split_name].append(frame_dir)
            per_class_per_split[split_name][class_name] += 1
            n_ok += 1

        print(f"    done {class_name}: {n_class} processed")

    # ---- summary ----
    print(f"\n{'='*60}\nCONVERSION SUMMARY  [{task}]\n{'='*60}")
    print(f"  seen          : {n_seen}")
    print(f"  saved         : {n_ok}")
    print(f"  no split match: {n_nosplit}  ({len(missing_videos)} unique videos)")
    print(f"  empty clips   : {n_empty}")
    print(f"  json errors   : {n_jsonerr}")
    print(f"  videos probed : {len(dim_cache)}  (fallback used: {n_fallback_dims})")

    if dim_cache:
        sample_dims = list(dim_cache.values())[:5]
        print(f"  sample dims   : {sample_dims}")

    for sp in ("train", "val", "test"):
        n = len(split_buckets[sp])
        print(f"\n  {sp}: {n} clips")
        for c in classes:
            print(f"    {c:25s}: {per_class_per_split[sp].get(c, 0)}")

    if missing_videos:
        print(f"\n  Sample missing videos (first 10):")
        for v in list(missing_videos)[:10]:
            print(f"    {v}")

    if n_ok == 0:
        print("\nERROR: no annotations built, refusing to write empty pickle")
        sys.exit(1)

    # ---- write pickle ----
    out = {"split": split_buckets, "annotations": anns}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB, {len(anns)} annotations)")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CLASSES.keys()))
    p.add_argument("--out",  required=True, help="Output pickle path")
    p.add_argument("--pose_root", default=POSE_ROOT)
    p.add_argument("--split_csv", default=SPLIT_CSV)
    p.add_argument("--min_frames", type=int, default=5,
                   help="Minimum non-empty frames required per clip")
    args = p.parse_args()

    try:
        convert_task(args.task, args.out, args.pose_root, args.split_csv,
                     min_frames=args.min_frames)
    except Exception:
        traceback.print_exc()
        sys.exit(1)