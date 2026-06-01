"""
Extract per-clip features using BOTH I3D (R3D-18) and R2Plus1D backbones.
Mirrors the V-JEPA extraction pipeline — same clipping, same H5 bbox crop,
same FPS conversion, same output layout.

Key FPS logic:
  - Annotations are at 15 FPS
  - Video is at 30 FPS
  - video_frame = ann_frame * 2
  - Clipping rules in annotation-frame space (15 FPS):
      < 15 ann-frames  (<1 s)       -> SKIP
      15-44 ann-frames (1-2.99 s)   -> 1 clip
      45-59 ann-frames (3-3.99 s)   -> 2 clips (first 30 ann-frames + remainder)
      >= 60 ann-frames (>=4 s)      -> 30-ann-frame chunks; last kept if >= 15 ann-frames

Output layout:
  <out_root>/i3d/<task>/<ClassName>/<basename>_<ann_start>_<ann_end>_clip<N>.npy
  <out_root>/r2plus1d/<task>/<ClassName>/<basename>_<ann_start>_<ann_end>_clip<N>.npy
  Shape: (512, T) for both backbones

Tasks:
  --task rmm   -> label col 'Repetitive_Motor_Movements'
  --task loco  -> label col 'Locomotion'

Backbone:
  --backbone i3d       -> run only I3D (R3D-18)
  --backbone r2plus1d  -> run only R2Plus1D
  (omit)               -> run both
"""

from __future__ import annotations

import argparse
import math
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torchvision.transforms as T


# ============================================================
# Constants
# ============================================================
ANN_FPS         = 15.0
VID_FPS         = 30.0
ANN_TO_VID      = int(VID_FPS / ANN_FPS)   # = 2

MIN_ANN_FRAMES  = 15
CLIP_ANN_FRAMES = 30
CROP_SIZE       = 224
CLIP_LEN        = 16
FEATURE_DIM     = 512

LABEL_COLUMNS = {
    "rmm":  "Repetitive_Motor_Movements",
    "loco": "Locomotion",
}

BACKBONES = ["i3d", "r2plus1d"]


# ============================================================
# Build backbones
# ============================================================
def build_backbones(gpu: int, active_backbones: list[str] = BACKBONES):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    models = {}
    for name in active_backbones:
        if name == "i3d":
            m = video_models.r3d_18(pretrained=True)
        else:
            m = video_models.r2plus1d_18(pretrained=True)
        m.fc = nn.Identity()
        m.eval()
        m.to(device)
        models[name] = m
        print(f"  Loaded backbone: {name}")
    return models, device


# ============================================================
# Clipping logic — annotation-frame space
# ============================================================
def chunk_run(start_ann: int, end_ann: int) -> list[tuple[int, int]]:
    total = end_ann - start_ann + 1
    if total < MIN_ANN_FRAMES:
        return []
    if total < CLIP_ANN_FRAMES * 2:
        if total < 45:
            return [(start_ann, end_ann)]
        else:
            split_pt = start_ann + CLIP_ANN_FRAMES
            return [(start_ann, split_pt - 1), (split_pt, end_ann)]
    clips = []
    s = start_ann
    while s <= end_ann:
        e = min(s + CLIP_ANN_FRAMES - 1, end_ann)
        if (e - s + 1) >= MIN_ANN_FRAMES:
            clips.append((s, e))
        s += CLIP_ANN_FRAMES
    return clips


# ============================================================
# Find contiguous same-label runs
# ============================================================
def find_action_runs(ann: pd.DataFrame, label_col: str) -> list[tuple[int, int, str]]:
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()
    runs = []
    i, n = 0, len(df)
    while i < n:
        lab = labels[i].strip()
        if lab in ("N/A", "", "nan", "NaN"):
            i += 1
            continue
        j = i
        while (j + 1 < n
               and labels[j + 1].strip() == lab
               and frames[j + 1] == frames[j] + 1):
            j += 1
        run_len = frames[j] - frames[i] + 1
        if run_len >= MIN_ANN_FRAMES:
            runs.append((frames[i], frames[j], lab))
        i = j + 1
    return runs


# ============================================================
# H5 bbox map
# ============================================================
def load_bbox_map(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def get_bbox_for_ann_frame(ann_frame: int, bbox_map: dict, bbox_keys: np.ndarray):
    if ann_frame in bbox_map:
        return bbox_map[ann_frame]
    nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - ann_frame))])
    return bbox_map[nearest]


def crop_frame(frame_rgb: np.ndarray, bbox: tuple, out_size: int = CROP_SIZE) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size))


# ============================================================
# Read + crop clip frames (done ONCE, shared by both backbones)
# ============================================================
def read_clip_frames(
    video_path: str,
    bbox_map: dict,
    bbox_keys: np.ndarray,
    clip_ann_start: int,
    clip_ann_end: int,
    crop_size: int = CROP_SIZE,
) -> np.ndarray | None:
    """
    Returns (T, H, W, 3) uint8 RGB.
    video_frame = ann_frame * 2
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return None
    out = []
    for af in range(clip_ann_start, clip_ann_end + 1):
        vf = af * ANN_TO_VID
        cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ret, frame = cap.read()
        if not ret:
            out.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bb = get_bbox_for_ann_frame(af, bbox_map, bbox_keys)
        out.append(crop_frame(frame_rgb, bb, crop_size))
    cap.release()
    if not out:
        return None
    return np.stack(out, axis=0)   # (T, H, W, 3)


# ============================================================
# Feature extraction — sliding window, one vector per ann-frame
# ============================================================
def preprocess_clip_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803,  0.22145,  0.216989]),
    ])
    processed = [transform(f) for f in frames]
    return torch.stack(processed, dim=1)   # (3, T, H, W)


def extract_features(
    model: nn.Module,
    frames: np.ndarray,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Centered sliding window: one 512-d vector per ann-frame.
    frames : (T, H, W, 3) uint8 RGB
    Returns: (512, T) float32
    """
    num_frames = frames.shape[0]
    if num_frames == 0:
        return np.zeros((FEATURE_DIM, 0), dtype=np.float32)

    half       = CLIP_LEN // 2
    frame_list = [frames[i] for i in range(num_frames)]
    all_feats  = []
    clip_batch = []

    for i in range(num_frames):
        start = max(0, i - half)
        end   = start + CLIP_LEN
        if end > num_frames:
            end   = num_frames
            start = max(0, end - CLIP_LEN)
        clip_frames = frame_list[start:end]
        while len(clip_frames) < CLIP_LEN:
            clip_frames.append(clip_frames[-1])
        clip_batch.append(preprocess_clip_tensor(clip_frames))

        if len(clip_batch) == batch_size or i == num_frames - 1:
            batch = torch.stack(clip_batch, dim=0).to(device)
            with torch.no_grad():
                feats = model(batch)
            all_feats.append(feats.cpu().numpy())
            clip_batch = []

    return np.concatenate(all_feats, axis=0).T   # (512, T)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Extract I3D + R2Plus1D features using annotation clips.")
    p.add_argument("--splits_csv",  required=True)
    p.add_argument("--out_root",    required=True,
                   help="Root output dir. Saves to <out_root>/i3d/<task>/ and <out_root>/r2plus1d/<task>/")
    p.add_argument("--task",        required=True, choices=list(LABEL_COLUMNS.keys()),
                   help="'rmm' or 'loco'")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--task_id",     type=int, default=None)
    p.add_argument("--num_tasks",   type=int, default=5)
    p.add_argument("--backbone",    type=str, default=None, choices=BACKBONES,
                   help="Run only this backbone: 'i3d' or 'r2plus1d'. Omit to run both.")
    args = p.parse_args()

    label_col = LABEL_COLUMNS[args.task]

    # Which backbones to run
    active_backbones = [args.backbone] if args.backbone else BACKBONES

    # Output dirs per active backbone
    out_dirs = {
        name: os.path.join(args.out_root, name, args.task)
        for name in active_backbones
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    log_dir = os.path.join(args.out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Load backbones
    print("Loading backbones...")
    models, device = build_backbones(args.gpu, active_backbones)

    print(f"\nTask      : {args.task}")
    print(f"Backbone  : {args.backbone if args.backbone else 'both'}")
    print(f"Label col : {label_col}")
    print(f"Ann FPS   : {ANN_FPS}  |  Video FPS: {VID_FPS}  |  Multiplier: {ANN_TO_VID}x")
    print(f"Clip size : {CLIP_ANN_FRAMES} ann-frames = {CLIP_ANN_FRAMES * ANN_TO_VID} video frames")
    for name, d in out_dirs.items():
        print(f"Output [{name}]: {d}")
    print()

    # Load CSV
    df = pd.read_csv(args.splits_csv)
    for c in ["video_path", "label_path", "interpolated_anno_h5"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")
    rows = list(df.itertuples(index=False))

    # SLURM array sharding
    if args.task_id is not None:
        chunk = math.ceil(len(rows) / args.num_tasks)
        s = args.task_id * chunk
        e = min(s + chunk, len(rows))
        rows = rows[s:e]
        print(f"SLURM task {args.task_id}/{args.num_tasks - 1}: rows [{s}:{e}] ({len(rows)} videos)")

    success = skipped = failed = 0

    for i, r in enumerate(rows):
        vp       = str(r.video_path).strip()
        lp       = str(r.label_path).strip()
        hp       = str(r.interpolated_anno_h5).strip()
        basename = os.path.splitext(os.path.basename(vp))[0]

        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            print(f"[{i+1}/{len(rows)}] SKIP (missing file): {basename}")
            skipped += 1
            continue

        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as ex:
            print(f"[{i+1}/{len(rows)}] SKIP (CSV error) {basename}: {ex}")
            skipped += 1
            continue

        if label_col not in ann.columns:
            print(f"[{i+1}/{len(rows)}] SKIP (no '{label_col}') {basename}")
            skipped += 1
            continue

        try:
            bbox_map  = load_bbox_map(hp)
            bbox_keys = np.array(sorted(bbox_map.keys()))
        except Exception as ex:
            print(f"[{i+1}/{len(rows)}] SKIP (H5 error) {basename}: {ex}")
            skipped += 1
            continue

        runs = find_action_runs(ann, label_col)
        print(f"\n[{i+1}/{len(rows)}] {basename}  |  runs={len(runs)}")

        for (run_sf, run_ef, lab) in runs:
            clips = chunk_run(run_sf, run_ef)

            for clip_idx, (cs, ce) in enumerate(clips):
                fname = f"{basename}_{cs}_{ce}_clip{clip_idx}.npy"

                # Check if all active backbones already done
                all_exist = all(
                    os.path.exists(os.path.join(out_dirs[name], lab.replace("/", "_"), fname))
                    for name in active_backbones
                )
                if all_exist and not args.overwrite:
                    print(f"  EXISTS (all)  {lab}/{fname}")
                    skipped += len(active_backbones)
                    continue

                # Read frames ONCE — shared by all active backbones
                frames = read_clip_frames(vp, bbox_map, bbox_keys, cs, ce)
                if frames is None or frames.shape[0] == 0:
                    print(f"  FAIL read  ann[{cs}-{ce}]  {lab}")
                    failed += len(active_backbones)
                    continue

                ann_len = ce - cs + 1

                # Run each active backbone on the same frames
                for backbone_name, model in models.items():
                    class_dir = os.path.join(out_dirs[backbone_name], lab.replace("/", "_"))
                    os.makedirs(class_dir, exist_ok=True)
                    out_path = os.path.join(class_dir, fname)

                    if os.path.exists(out_path) and not args.overwrite:
                        print(f"  EXISTS [{backbone_name}]  {lab}/{fname}")
                        skipped += 1
                        continue

                    try:
                        feats = extract_features(model, frames,
                                                 batch_size=args.batch_size,
                                                 device=device)
                    except Exception as ex:
                        print(f"  FAIL extract [{backbone_name}]  ann[{cs}-{ce}]  {lab}: {ex}")
                        failed += 1
                        continue

                    np.save(out_path, feats)
                    print(f"  SAVED [{backbone_name}]  {lab}/{fname}"
                          f"  ann_frames={ann_len}"
                          f"  vid_frames=[{cs*ANN_TO_VID}-{ce*ANN_TO_VID}]"
                          f"  shape={feats.shape}")
                    success += 1

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  success  = {success}")
    print(f"  skipped  = {skipped}")
    print(f"  failed   = {failed}")
    for name, d in out_dirs.items():
        print(f"  [{name}] -> {d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()