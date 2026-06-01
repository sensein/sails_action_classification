"""
Extract V-JEPA 2 features per action run (no clip chunking).

Key FPS logic:
  - Annotations are at 15 FPS  (frame numbers in annotation CSV)
  - Video is at 30 FPS
  - To convert annotation frame  ->  video frame:  video_frame = ann_frame * 2
  - Each contiguous run is saved as ONE clip regardless of length.
  - Only rule: < 15 ann-frames -> SKIP (everything else -> 1 clip, as-is)
  - Non-consecutive annotation frames within a run start a new clip.

Output layout:
  <out_root>/<task>/<ClassName>/<basename>_<ann_start>_<ann_end>_clip<N>.npy
  Shape: (D=1536, T) where T = number of annotation frames in the clip

Tasks:
  --task rmm   -> label col 'Repetitive_Motor_Movements'
  --task loco  -> label col 'Locomotion'
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
from transformers import AutoModel, AutoVideoProcessor


# ============================================================
# Constants
# ============================================================
MODEL_NAME      = "facebook/vjepa2-vitg-fpc64-256"
EMBED_DIM       = 1536
TUBELET_SIZE    = 2
FRAMES_PER_CLIP = 64          # V-JEPA 2 context window (video frames)
ANN_FPS         = 15.0        # annotation frame rate
VID_FPS         = 30.0        # video frame rate
ANN_TO_VID      = int(VID_FPS / ANN_FPS)   # = 2  (multiply ann frame by this)

# Only rule: skip runs shorter than this
MIN_ANN_FRAMES  = 15

CROP_SIZE       = 256

LABEL_COLUMNS = {
    "rmm":  "Repetitive_Motor_Movements",
    "loco": "Locomotion",
}


# ============================================================
# Find contiguous same-label runs in annotation CSV
# ============================================================
def find_action_runs(ann: pd.DataFrame, label_col: str) -> list[tuple[int, int, str]]:
    """
    Returns list of (start_ann_frame, end_ann_frame, label_str)
    for consecutive same-label blocks. NA / empty / nan -> ignored.
    Non-consecutive frames (gaps) split into separate runs.
    Only runs >= MIN_ANN_FRAMES are returned.
    """
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
    """
    Load per-annotation-frame bounding boxes from H5 file.
    Returns {ann_frame_int: (x1, y1, x2, y2)}.
    """
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def get_bbox_for_ann_frame(ann_frame: int, bbox_map: dict, bbox_keys: np.ndarray):
    """Return bbox for ann_frame, falling back to nearest available frame."""
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
# Read + crop a single clip from the video
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
    Read video frames for annotation range [clip_ann_start, clip_ann_end].
    Video frame index = ann_frame * ANN_TO_VID  (i.e. * 2 for 30fps video).

    Returns (T, H, W, 3) uint8 RGB, where T = number of ann frames in clip.
    Returns None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return None

    ann_frames = list(range(clip_ann_start, clip_ann_end + 1))
    out = []
    for af in ann_frames:
        # --- KEY FPS CONVERSION: ann frame -> video frame ---
        vf = af * ANN_TO_VID   # multiply by 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ret, frame = cap.read()
        if not ret:
            # Pad with zeros if frame unreadable
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
# V-JEPA 2 feature extraction
# ============================================================
def extract_features(
    model,
    processor,
    frames: np.ndarray,
    batch_clips: int = 2,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    frames: (T, H, W, 3) uint8 RGB
    Returns: (D, T) float32 features — one D-dim vector per ann frame.
    """
    t_total = frames.shape[0]
    patches_per_clip = FRAMES_PER_CLIP // TUBELET_SIZE

    # Pad to multiple of FRAMES_PER_CLIP
    pad_total = math.ceil(t_total / FRAMES_PER_CLIP) * FRAMES_PER_CLIP
    if pad_total > t_total:
        pad = np.repeat(frames[-1:], pad_total - t_total, axis=0)
        frames_padded = np.concatenate([frames, pad], axis=0)
    else:
        frames_padded = frames

    num_clips = pad_total // FRAMES_PER_CLIP
    clips = frames_padded.reshape(num_clips, FRAMES_PER_CLIP, *frames.shape[1:])

    all_feats = []
    for s in range(0, num_clips, batch_clips):
        e = min(s + batch_clips, num_clips)
        batch = clips[s:e]
        clip_list = [batch[i] for i in range(batch.shape[0])]

        inputs = processor(clip_list, return_tensors="pt")
        inputs = {
            k: (v.to(device, dtype=torch.float16) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)

        tokens = outputs.last_hidden_state.float()       # (B, N_patches, D)
        b, n_patches, d = tokens.shape
        spatial_patches = n_patches // patches_per_clip
        tokens = tokens.reshape(b, patches_per_clip, spatial_patches, d)
        pooled = tokens.mean(dim=2)                      # (B, Tp, D)
        all_feats.append(pooled.cpu().numpy())

    patch_feats = np.concatenate(all_feats, axis=0)     # (num_clips, Tp, D)
    per_frame   = np.repeat(patch_feats, TUBELET_SIZE, axis=1)  # upsample to video frames
    per_frame   = per_frame.reshape(-1, per_frame.shape[-1])[:t_total]
    return per_frame.T   # (D, T)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Extract V-JEPA 2 features using annotation runs (no chunking).")
    p.add_argument("--splits_csv",  required=True,
                   help="CSV with columns: video_path, label_path, interpolated_anno_h5, split")
    p.add_argument("--out_root",    required=True,
                   help="Root output dir, e.g. /scratch/.../clips")
    p.add_argument("--task",        required=True, choices=list(LABEL_COLUMNS.keys()),
                   help="'rmm' or 'loco'")
    p.add_argument("--model_name",  default=MODEL_NAME)
    p.add_argument("--batch_clips", type=int, default=2)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--overwrite",   action="store_true")
    # SLURM array support
    p.add_argument("--task_id",     type=int, default=None,
                   help="SLURM array task ID (0-indexed)")
    p.add_argument("--num_tasks",   type=int, default=1,
                   help="Total number of SLURM array tasks")
    args = p.parse_args()

    label_col = LABEL_COLUMNS[args.task]
    out_base  = os.path.join(args.out_root, args.task)
    os.makedirs(out_base, exist_ok=True)

    # ── Load model ────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading V-JEPA 2: {args.model_name} ...")
    processor = AutoVideoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{args.gpu}",
        attn_implementation="sdpa",
    )
    model.eval()

    print(f"\nTask      : {args.task}")
    print(f"Label col : {label_col}")
    print(f"Ann FPS   : {ANN_FPS}  |  Video FPS: {VID_FPS}  |  Multiplier: {ANN_TO_VID}x")
    print(f"Min run   : {MIN_ANN_FRAMES} ann-frames (shorter runs skipped)")
    print(f"Chunking  : NONE — each run saved as one clip")
    print(f"Output    : {out_base}\n")

    # ── Load CSV ──────────────────────────────────────────────
    df = pd.read_csv(args.splits_csv)
    for c in ["video_path", "label_path", "interpolated_anno_h5"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in splits CSV: '{c}'")
    rows = list(df.itertuples(index=False))

    # SLURM array sharding
    if args.task_id is not None:
        chunk = math.ceil(len(rows) / args.num_tasks)
        s = args.task_id * chunk
        e = min(s + chunk, len(rows))
        rows = rows[s:e]
        print(f"SLURM task {args.task_id}/{args.num_tasks - 1}: "
              f"processing rows [{s}:{e}]  ({len(rows)} videos)")

    success = skipped = failed = 0

    for i, r in enumerate(rows):
        vp = str(r.video_path).strip()
        lp = str(r.label_path).strip()
        hp = str(r.interpolated_anno_h5).strip()
        basename = os.path.splitext(os.path.basename(vp))[0]

        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            print(f"[{i+1}/{len(rows)}] SKIP (missing file): {basename}")
            skipped += 1
            continue

        # Load annotation CSV
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

        # Load bbox map once per video
        try:
            bbox_map  = load_bbox_map(hp)
            bbox_keys = np.array(sorted(bbox_map.keys()))
        except Exception as ex:
            print(f"[{i+1}/{len(rows)}] SKIP (H5 error) {basename}: {ex}")
            skipped += 1
            continue

        # Find contiguous label runs (each run -> one clip, no chunking)
        runs = find_action_runs(ann, label_col)
        print(f"\n[{i+1}/{len(rows)}] {basename}  |  runs={len(runs)}")

        for clip_idx, (run_sf, run_ef, lab) in enumerate(runs):
            # Each run is saved as a single clip — no chunking
            cs, ce = run_sf, run_ef

            class_dir = os.path.join(out_base, lab.replace("/", "_"))
            os.makedirs(class_dir, exist_ok=True)
            out_path = os.path.join(
                class_dir,
                f"{basename}_{cs}_{ce}_clip{clip_idx}.npy"
            )

            if os.path.exists(out_path) and not args.overwrite:
                print(f"  EXISTS  {lab}/{os.path.basename(out_path)}")
                skipped += 1
                continue

            # Read video frames for this run
            ann_len = ce - cs + 1
            frames = read_clip_frames(vp, bbox_map, bbox_keys, cs, ce)
            if frames is None or frames.shape[0] == 0:
                print(f"  FAIL read  ann[{cs}-{ce}]  {lab}")
                failed += 1
                continue

            # Extract V-JEPA 2 features
            try:
                feats = extract_features(
                    model, processor, frames,
                    batch_clips=args.batch_clips,
                    device=device,
                )
            except Exception as ex:
                print(f"  FAIL extract  ann[{cs}-{ce}]  {lab}: {ex}")
                failed += 1
                continue

            np.save(out_path, feats)
            print(f"  SAVED  {lab}/{os.path.basename(out_path)}"
                  f"  ann_frames={ann_len}"
                  f"  vid_frames=[{cs*ANN_TO_VID}-{ce*ANN_TO_VID}]"
                  f"  shape={feats.shape}")
            success += 1

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  success  = {success}")
    print(f"  skipped  = {skipped}")
    print(f"  failed   = {failed}")
    print(f"  output   = {out_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()