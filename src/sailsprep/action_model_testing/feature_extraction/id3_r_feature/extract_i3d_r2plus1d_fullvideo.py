"""
Extract per-frame features using BOTH I3D (R3D-18) and R2Plus1D backbones
for FULL VIDEOS — mirrors the V-JEPA extraction pipeline exactly.

Same inputs:
  - splits_csv with columns: video_path, interpolated_full_h5
  - H5 bbox crop (same convention as V-JEPA)
  - target_fps (default 15, matching annotations)

Same output layout as V-JEPA:
  <output_dir>/i3d/<basename>.npy        shape: (512, T)
  <output_dir>/r2plus1d/<basename>.npy   shape: (512, T)

Key design choices (matching V-JEPA):
  - Decode at target_fps (15 fps) using OpenCV
  - Per-frame bbox crop from H5 (nearest-frame fallback)
  - Centered sliding window of CLIP_LEN=16 frames → one 512-d vector per frame
  - Output shape (512, T) where T = number of decoded frames
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
ANN_FPS       = 15.0
CROP_SIZE     = 224
CLIP_LEN      = 16        # sliding-window width (frames)
FEATURE_DIM   = 512
BACKBONES     = ["i3d", "r2plus1d"]


# ============================================================
# Build both backbones
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
# H5 bbox loading  (identical to V-JEPA script)
# ============================================================
def load_bbox_map(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def crop_frame_with_bbox(frame: np.ndarray, bbox, out_size: int = CROP_SIZE) -> np.ndarray:
    """Crop frame to bbox and resize. frame is HWC uint8 RGB."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size))


# ============================================================
# Read full video with bbox cropping  (mirrors V-JEPA)
# ============================================================
def read_video_cropped(
    video_path: str,
    h5_path: str,
    target_fps: float = ANN_FPS,
    crop_size: int = CROP_SIZE,
) -> np.ndarray | None:
    """
    Decode video at target_fps, crop each frame using the H5 bbox map.
    Returns (T, crop_size, crop_size, 3) uint8 RGB, or None on failure.
    Uses OpenCV (consistent with the clip-based I3D script).
    """
    if not os.path.exists(video_path):
        print(f"  video not found: {video_path}"); return None
    if not os.path.exists(h5_path):
        print(f"  h5 not found: {h5_path}"); return None

    try:
        bbox_map = load_bbox_map(h5_path)
        if not bbox_map:
            print(f"  empty bbox map: {h5_path}"); return None
        bbox_keys = np.array(sorted(bbox_map.keys()))
    except Exception as e:
        print(f"  H5 load error {h5_path}: {e}"); return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  cannot open video: {video_path}"); return None

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps
    num_target   = int(duration * target_fps)

    if num_target == 0:
        print(f"  video too short: {video_path}"); cap.release(); return None

    # Evenly spaced native-frame indices to match target_fps
    native_indices = np.linspace(0, total_frames - 1, num_target, dtype=int)

    cropped = []
    for out_idx, vf in enumerate(native_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf))
        ret, frame = cap.read()
        if not ret:
            # Pad with zeros on read failure
            cropped.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Annotation-frame index = position in the target-fps sequence
        af = out_idx
        if af in bbox_map:
            bb = bbox_map[af]
        else:
            nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
            bb = bbox_map[nearest]

        cropped.append(crop_frame_with_bbox(frame_rgb, bb, out_size=crop_size))

    cap.release()
    if not cropped:
        return None
    return np.stack(cropped, axis=0)   # (T, H, W, 3)


# ============================================================
# Feature extraction — sliding window, one vector per frame
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
    Centered sliding window: one 512-d vector per frame.
    frames  : (T, H, W, 3) uint8 RGB
    Returns : (512, T) float32
    """
    num_frames = frames.shape[0]
    if num_frames == 0:
        return np.zeros((FEATURE_DIM, 0), dtype=np.float32)

    half       = CLIP_LEN // 2
    frame_list = [frames[i] for i in range(num_frames)]
    all_feats  = []
    clip_batch = []

    for i in range(num_frames):
        # Centered window clamped to valid range
        start = max(0, i - half)
        end   = start + CLIP_LEN
        if end > num_frames:
            end   = num_frames
            start = max(0, end - CLIP_LEN)

        clip_frames = frame_list[start:end]
        # Pad by repeating last frame if clip is shorter than CLIP_LEN
        while len(clip_frames) < CLIP_LEN:
            clip_frames.append(clip_frames[-1])

        clip_batch.append(preprocess_clip_tensor(clip_frames))

        if len(clip_batch) == batch_size or i == num_frames - 1:
            batch = torch.stack(clip_batch, dim=0).to(device)   # (B, 3, 16, H, W)
            with torch.no_grad():
                feats = model(batch)                             # (B, 512)
            all_feats.append(feats.cpu().numpy())
            clip_batch = []

    return np.concatenate(all_feats, axis=0).T   # (512, T)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Extract full-video I3D + R2Plus1D features (mirrors V-JEPA pipeline)."
    )
    p.add_argument("--splits_csv",  required=True,
                   help="CSV with columns: video_path, interpolated_full_h5")
    p.add_argument("--output_dir",  required=True,
                   help="Root output dir. Saves to <output_dir>/i3d/ and <output_dir>/r2plus1d/")
    p.add_argument("--target_fps",  type=float, default=ANN_FPS,
                   help="FPS to decode video at (default: 15, matching annotations)")
    p.add_argument("--crop_size",   type=int,   default=CROP_SIZE)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--gpu",         type=int,   default=0)
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--task_id",     type=int,   default=None)
    p.add_argument("--num_tasks",   type=int,   default=1)
    p.add_argument("--backbone",    type=str,   default=None, choices=BACKBONES,
                   help="Run only this backbone. Omit to run both.")
    args = p.parse_args()

    # Which backbones to run
    active_backbones = [args.backbone] if args.backbone else BACKBONES

    # Output dirs — one per backbone, same structure as V-JEPA
    out_dirs = {name: os.path.join(args.output_dir, name) for name in active_backbones}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Loading backbones...")
    models, device = build_backbones(args.gpu, active_backbones)

    print(f"\nTarget FPS : {args.target_fps}")
    print(f"Crop size  : {args.crop_size}")
    print(f"Clip len   : {CLIP_LEN} frames (sliding window)")
    for name, d in out_dirs.items():
        print(f"Output [{name}]: {d}")
    print()

    df = pd.read_csv(args.splits_csv)
    for col in ["video_path", "interpolated_full_h5"]:
        if col not in df.columns:
            raise ValueError(f"Split CSV missing column: '{col}'")

    rows = list(df.itertuples(index=False))

    # SLURM array sharding (identical pattern to V-JEPA script)
    if args.task_id is not None:
        chunk = math.ceil(len(rows) / args.num_tasks)
        s = args.task_id * chunk
        e = min(s + chunk, len(rows))
        rows = rows[s:e]
        print(f"SLURM task {args.task_id}/{args.num_tasks - 1}: "
              f"rows [{s}:{e}] ({len(rows)} videos)")
    else:
        print(f"Processing {len(rows)} videos")

    success = skipped = failed = 0

    for i, r in enumerate(rows):
        vp       = str(r.video_path).strip()
        hp       = str(r.interpolated_full_h5).strip()
        basename = os.path.splitext(os.path.basename(vp))[0]

        # Check if BOTH backbone outputs already exist
        both_exist = all(
            os.path.exists(os.path.join(out_dirs[name], f"{basename}.npy"))
            for name in out_dirs
        )
        if both_exist and not args.overwrite:
            print(f"[{i+1}/{len(rows)}] EXISTS (both)  {basename}")
            skipped += 2
            continue

        print(f"\n[{i+1}/{len(rows)}] {basename}", flush=True)

        # Read full video — ONCE, shared by both backbones
        frames = read_video_cropped(vp, hp, args.target_fps, args.crop_size)
        if frames is None:
            print(f"  FAIL read: {basename}")
            failed += 2
            continue

        print(f"  frames (cropped): {frames.shape}")   # (T, H, W, 3)

        # Run both backbones on the same frames
        for backbone_name, model in models.items():
            out_path = os.path.join(out_dirs[backbone_name], f"{basename}.npy")

            if os.path.exists(out_path) and not args.overwrite:
                print(f"  EXISTS [{backbone_name}]  {basename}.npy")
                skipped += 1
                continue

            try:
                feats = extract_features(
                    model, frames,
                    batch_size=args.batch_size,
                    device=device,
                )
            except Exception as ex:
                print(f"  FAIL extract [{backbone_name}]  {basename}: {ex}")
                failed += 1
                continue

            np.save(out_path, feats)
            print(f"  SAVED [{backbone_name}]  {basename}.npy  shape={feats.shape}")
            success += 1

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  success = {success}")
    print(f"  skipped = {skipped}")
    print(f"  failed  = {failed}")
    for name, d in out_dirs.items():
        print(f"  [{name}] -> {d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()