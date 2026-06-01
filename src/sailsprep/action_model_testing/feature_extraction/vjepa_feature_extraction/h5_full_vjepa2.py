"""Extract per-frame V-JEPA 2 features from full videos with H5 bbox cropping.

For each video:
  1. Decode frames at target_fps (15fps by default, matching annotations).
  2. For each frame, look up the interpolated bbox from the H5 file.
  3. Crop the frame to the subject region, then pass to V-JEPA 2.
  4. Save per-frame features as (D, T) .npy where D=1536 for ViT-g.
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
from decord import VideoReader, cpu as decord_cpu
from transformers import AutoModel, AutoVideoProcessor


# ============================================================
# Constants
# ============================================================
MODEL_NAME      = "facebook/vjepa2-vitg-fpc64-256"
EMBED_DIM       = 1408        # ViT-g
TUBELET_SIZE    = 2
FRAMES_PER_CLIP = 64
ANN_FPS         = 15.0


# ============================================================
# H5 bbox loading (same convention as SlowFast pipeline)
# ============================================================
def load_bbox_map(h5_path: str) -> dict:
    """Return {ann_frame_idx: (x1,y1,x2,y2)} from interpolated H5."""
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def crop_frame_with_bbox(frame: np.ndarray, bbox, out_size: int = 256) -> np.ndarray:
    """Crop frame to bbox and resize to (out_size, out_size). frame is HWC uint8 RGB."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size))
    return crop


# ============================================================
# Video reading with bbox cropping
# ============================================================
def read_video_cropped(video_path: str, h5_path: str,
                       target_fps: float = ANN_FPS,
                       crop_size: int = 256) -> np.ndarray | None:
    """
    Decode video at target_fps, crop each frame using the H5 bbox map.

    Returns (T, crop_size, crop_size, 3) uint8 RGB, or None on failure.
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

        vr = VideoReader(video_path, ctx=decord_cpu(0))
        native_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / native_fps
        num_target = int(duration * target_fps)
        if num_target == 0:
            print(f"  video too short: {video_path}"); return None

        # Evenly spaced native-frame indices to hit target_fps.
        indices = np.linspace(0, total_frames - 1, num_target, dtype=int)

        # Batch decode (RGB uint8) then crop per frame.
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        cropped = np.empty((num_target, crop_size, crop_size, 3), dtype=np.uint8)
        for i in range(num_target):
            af = i  # annotation-frame index (since we sampled at ann fps)
            if af in bbox_map:
                bb = bbox_map[af]
            else:
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                bb = bbox_map[nearest]
            cropped[i] = crop_frame_with_bbox(frames[i], bb, out_size=crop_size)

        return cropped
    except Exception as e:
        print(f"  decode/crop error {video_path}: {e}")
        return None


# ============================================================
# Feature extraction
# ============================================================
def extract_features_single_video(model, processor, frames: np.ndarray,
                                   batch_clips: int = 2, device=torch.device("cpu")) -> np.ndarray:
    """
    Split frames into 64-frame clips, run V-JEPA 2, spatial-pool, and map
    temporal patches back to per-frame features. Returns (D, T).
    """
    t_total = frames.shape[0]
    patches_per_clip = FRAMES_PER_CLIP // TUBELET_SIZE  # 32

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
    for start in range(0, num_clips, batch_clips):
        end = min(start + batch_clips, num_clips)
        batch = clips[start:end]  # (B, 64, H, W, 3)
        clip_list = [batch[i] for i in range(batch.shape[0])]

        inputs = processor(clip_list, return_tensors="pt")
        inputs = {k: (v.to(device, dtype=torch.float16) if v.is_floating_point()
                      else v.to(device)) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        tokens = outputs.last_hidden_state.float()         # (B, N, D)
        b, n_patches, d = tokens.shape
        spatial_patches = n_patches // patches_per_clip
        tokens = tokens.reshape(b, patches_per_clip, spatial_patches, d)
        pooled = tokens.mean(dim=2)                        # (B, Tp, D)
        all_feats.append(pooled.cpu().numpy())

    patch_feats = np.concatenate(all_feats, axis=0)        # (num_clips, Tp, D)
    per_frame = np.repeat(patch_feats, TUBELET_SIZE, axis=1)
    per_frame = per_frame.reshape(-1, per_frame.shape[-1])
    per_frame = per_frame[:t_total]                        # (T, D)
    return per_frame.T                                     # (D, T)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--target_fps", type=float, default=ANN_FPS)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_clips", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading V-JEPA 2: {args.model_name}")
    processor = AutoVideoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{args.gpu}",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}  |  Device: {device}")

    df = pd.read_csv(args.splits_csv)
    for col in ["video_path", "interpolated_full_h5"]:
        if col not in df.columns:
            raise ValueError(f"Split CSV missing column: {col}")

    rows = list(df.itertuples(index=False))

    if args.task_id is not None:
        chunk = math.ceil(len(rows) / args.num_tasks)
        s = args.task_id * chunk
        e = min(s + chunk, len(rows))
        rows = rows[s:e]
        print(f"\nSLURM task {args.task_id}/{args.num_tasks - 1}: "
              f"processing {len(rows)} videos [{s}:{e}]")
    else:
        print(f"\nProcessing {len(rows)} videos")

    success = skipped = failed = 0
    for i, r in enumerate(rows):
        vp = str(r.video_path).strip()
        hp = str(r.interpolated_full_h5).strip()
        basename = os.path.splitext(os.path.basename(vp))[0]
        out = os.path.join(args.output_dir, f"{basename}.npy")

        if os.path.exists(out) and not args.overwrite:
            skipped += 1; continue

        print(f"\n[{i+1}/{len(rows)}] {basename}", flush=True)

        frames = read_video_cropped(vp, hp, args.target_fps, args.crop_size)
        if frames is None:
            failed += 1; continue
        print(f"  frames (cropped): {frames.shape}")

        try:
            feats = extract_features_single_video(
                model, processor, frames,
                batch_clips=args.batch_clips, device=device,
            )
        except Exception as e:
            print(f"  feature error: {e}")
            failed += 1; continue

        print(f"  features: {feats.shape}")  # (1536, T)
        np.save(out, feats)
        success += 1

    tag = f" (task {args.task_id})" if args.task_id is not None else ""
    print(f"\n{'='*50}\nDone{tag}.")
    print(f"  success: {success}\n  skipped: {skipped}\n  failed : {failed}")
    print(f"  output : {args.output_dir}\n  dim    : {EMBED_DIM}\n{'='*50}")


if __name__ == "__main__":
    main()