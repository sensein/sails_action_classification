"""
sam3_targeted_noyolo.py
========================
Runs SAM3 on a specific list of 7 video folders ONLY.
- No YOLO verification
- No aspect ratio filtering
- Raw SAM3 detections → masks .npy, bboxes .npy, detections.csv, results.json, masked_video.mp4
- Overwrites existing outputs in those folders
- Does NOT touch any other folder in the output dir

The video folder name = Path(video_path).stem, matching the output dir convention
of the original SAM3 pipeline. You must supply the video paths in the TARGET_VIDEOS
dict below (folder_name → full video path).

Usage:
    python sam3_targeted_noyolo.py \
        --output_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
        --video_root /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed \
        --resize 192
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# TARGET: folder_name → relative path hint for auto-discovery
# Fill in the full video_path for each, or use --video_root + auto-discovery
# ─────────────────────────────────────────────────────────────────────────────
TARGET_FOLDER_NAMES = [
    "sub-N3L7A1I2B9_ses-01_task-toyplay_run-02_desc-processed_beh",
    "VAMZwnfAHyY_3_10",
    "VxK45NHvHTg_25_28",
    "ZHJr17Q4384_2_35",
    "v_Spinning_25_b_01_Spinning_0025s-0028s",
    "-YJhyNoHuUw_0_24",
    "Vwlc3fLmipY_0_4",
]

TEXT_PROMPT      = "Human Young Child"
DTYPE            = torch.bfloat16
EMPTY_CACHE_EVERY = 50

# Visual overlay constants
MASK_COLOR  = (0, 255, 0)
MASK_ALPHA  = 0.35
BBOX_COLOR  = (0, 0, 255)
BBOX_THICK  = 2
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 0, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
FONT_THICK  = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def light_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def masks_to_bboxes(masks_np):
    """Convert binary masks (N, H, W) → bboxes (N, 4) in xyxy."""
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis]
    N = masks_np.shape[0]
    bboxes = np.full((N, 4), -1, dtype=np.int32)
    for i, mask in enumerate(masks_np):
        ys, xs = np.where(mask > 0)
        if len(xs):
            bboxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return bboxes


def resize_frame(frame, shorter_side):
    h, w = frame.shape[:2]
    if min(h, w) <= shorter_side:
        return frame
    scale  = shorter_side / min(h, w)
    new_w  = int(round(w * scale)); new_w += new_w % 2
    new_h  = int(round(h * scale)); new_h += new_h % 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_overlay(frame, masks_np, bboxes, obj_ids, scores):
    out = frame.copy()
    H, W = frame.shape[:2]
    for i, oid in enumerate(obj_ids):
        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i]
            if mask.shape == (H, W) and np.any(mask):
                overlay = out.copy()
                overlay[mask > 0] = MASK_COLOR
                out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)
        if i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            if x1 >= 0:
                cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICK)
                sc    = scores[i] if i < len(scores) else None
                label = f"ID:{oid}" + (f" {float(sc):.2f}" if sc is not None else "")
                (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
                lx1, ly1 = x1, max(y1 - th - bl - 4, 0)
                lx2, ly2 = x1 + tw + 4, max(y1, th + bl + 4)
                cv2.rectangle(out, (lx1, ly1), (lx2, ly2), LABEL_BG, -1)
                cv2.putText(out, label, (lx1 + 2, ly2 - bl - 2),
                            FONT, FONT_SCALE, LABEL_COLOR, FONT_THICK, cv2.LINE_AA)
    return out


def write_masked_video(video_path, per_frame_data, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Cannot open video for overlay: {video_path}")
        return
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(output_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in per_frame_data:
                fd    = per_frame_data[frame_idx]
                frame = draw_overlay(
                    frame,
                    fd.get("masks",   np.zeros((0, height, width), dtype=np.uint8)),
                    fd.get("bboxes",  np.zeros((0, 4), dtype=np.int32)),
                    fd.get("obj_ids", []),
                    fd.get("scores",  []),
                )
            proc.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode == 0:
        print(f"    ✓ Masked video → {output_path}")
    else:
        print(f"    ⚠ ffmpeg returned code {proc.returncode}")


def write_detections_csv(csv_rows, path):
    fieldnames = ["frame_idx", "obj_id", "x1", "y1", "x2", "y2", "score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"    ✓ Detections CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video path discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_video_path(folder_name, video_root, output_dir):
    """
    Try to locate the source .mp4 for a given folder name.

    Priority:
      1. BIDS-style:  <video_root>/<sub>/<ses>/beh/<folder_name>.mp4
      2. Flat search: <video_root>/**/<folder_name>.mp4   (up to 4 levels deep)
      3. Already in output dir (some pipelines copy the source): <output_dir>/<folder_name>/<folder_name>.mp4
    """
    # 1. BIDS
    parts = folder_name.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), None)
    ses = next((p for p in parts if p.startswith("ses-")), None)
    if sub and ses and video_root:
        bids_path = os.path.join(video_root, sub, ses, "beh", f"{folder_name}.mp4")
        if os.path.exists(bids_path):
            return bids_path

    # 2. Flat search under video_root (up to 4 levels)
    if video_root:
        for depth in range(1, 5):
            pattern = os.path.join(video_root, *["*"] * depth, f"{folder_name}.mp4")
            import glob
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

    # 3. Fallback: check if the video sits inside its own output folder
    local = os.path.join(output_dir, folder_name, f"{folder_name}.mp4")
    if os.path.exists(local):
        return local

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core processing — Native SAM3 backend, no YOLO, no aspect filter
# ─────────────────────────────────────────────────────────────────────────────

def process_video_native(predictor, video_path, output_dir,
                         resize_shorter_side, log_every, empty_cache_every):
    video_name    = Path(video_path).stem
    video_out_dir = os.path.join(output_dir, video_name)
    masks_dir     = os.path.join(video_out_dir, "masks")
    bboxes_dir    = os.path.join(video_out_dir, "bboxes")
    os.makedirs(masks_dir,  exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)

    # ── frame reader ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if resize_shorter_side and min(orig_h, orig_w) > resize_shorter_side:
        scale = resize_shorter_side / min(orig_h, orig_w)
        sam_w = int(round(orig_w * scale)); sam_w += sam_w % 2
        sam_h = int(round(orig_h * scale)); sam_h += sam_h % 2
    else:
        scale = 1.0
        sam_w, sam_h = orig_w, orig_h

    print(f"    Resolution  : {orig_w}x{orig_h} → SAM3 {sam_w}x{sam_h} (scale={scale:.3f})")
    print(f"    Total frames: {total}")

    def get_frame(idx):
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap2.read()
        cap2.release()
        if ret and scale < 1.0:
            frm = cv2.resize(frm, (sam_w, sam_h), interpolation=cv2.INTER_AREA)
        return frm if ret else None

    # ── SAM3 session ──────────────────────────────────────────────────────────
    resp       = predictor.handle_request(dict(type="start_session", resource_path=video_path))
    session_id = resp["session_id"]
    predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                  frame_index=0, text=TEXT_PROMPT))

    results_summary = {
        "video_path":                video_path,
        "video_name":                video_name,
        "prompt":                    TEXT_PROMPT,
        "total_frames":              total,
        "detection_mode":            "raw SAM3 (no YOLO, no aspect filter)",
        "backend_used":              "native",
        "verification":              "none",
        "min_confidence":            0.0,
        "max_aspect_ratio":          -1,
        "resize_shorter_side":       resize_shorter_side,
        "sam3_resolution":           f"{sam_w}x{sam_h}",
        "original_resolution":       f"{orig_w}x{orig_h}",
        "frames_processed":          0,
        "frames_with_detections":    0,
        "frames_rejected_not_human": 0,
        "per_frame":                 {},
    }
    per_frame_data = {}
    csv_rows       = []

    def to_list(x):
        if isinstance(x, list):  return x
        if hasattr(x, "tolist"): return x.tolist()
        return list(x) if x else []

    propagated = 0
    try:
        with torch.inference_mode():
            ctx = (torch.autocast("cuda", dtype=DTYPE) if torch.cuda.is_available()
                   else torch.autocast("cpu",  dtype=DTYPE))
            with ctx:
                for response in predictor.handle_stream_request(
                    request=dict(type="propagate_in_video", session_id=session_id)
                ):
                    frame_idx = response["frame_index"]
                    outputs   = response["outputs"]

                    obj_ids = to_list(outputs.get("out_obj_ids", []))
                    scores  = to_list(outputs.get("out_probs",   []))
                    masks   = outputs.get("out_binary_masks", None)

                    # ── masks → numpy (H, W each) ──────────────────────────
                    if masks is not None and len(obj_ids) > 0:
                        if isinstance(masks, torch.Tensor):
                            masks_np = masks.detach().cpu().numpy().astype(np.uint8)
                            del masks
                        else:
                            masks_np = np.array(masks, dtype=np.uint8)
                        if masks_np.ndim == 2:
                            masks_np = masks_np[np.newaxis]
                        while masks_np.ndim > 3:
                            masks_np = masks_np[:, 0]
                        # upsample to SAM resolution if needed
                        if masks_np.shape[-2:] != (sam_h, sam_w):
                            masks_np = np.stack([
                                cv2.resize(m, (sam_w, sam_h),
                                           interpolation=cv2.INTER_NEAREST)
                                for m in masks_np
                            ], axis=0)
                    else:
                        masks_np = None

                    # ── scale back to original resolution ─────────────────
                    if scale < 1.0 and masks_np is not None:
                        masks_orig = np.stack([
                            cv2.resize(m, (orig_w, orig_h),
                                       interpolation=cv2.INTER_NEAREST)
                            for m in masks_np
                        ], axis=0)
                    else:
                        masks_orig = masks_np

                    # ── bboxes from masks ──────────────────────────────────
                    n_obj = len(obj_ids)
                    if masks_orig is not None and n_obj > 0 and masks_orig.sum() > 0:
                        bboxes_orig = masks_to_bboxes(masks_orig)
                    else:
                        bboxes_orig = np.full((n_obj, 4), -1, dtype=np.int32)

                    # ── save mask / bbox npy ───────────────────────────────
                    if masks_orig is not None and n_obj > 0 and masks_orig.sum() > 0:
                        np.save(os.path.join(masks_dir,  f"frame_{frame_idx:06d}.npy"),
                                masks_orig)
                    if n_obj > 0:
                        np.save(os.path.join(bboxes_dir, f"frame_{frame_idx:06d}.npy"),
                                bboxes_orig)

                    # ── csv rows ───────────────────────────────────────────
                    for k, oid in enumerate(obj_ids):
                        x1, y1, x2, y2 = (bboxes_orig[k].tolist()
                                           if k < len(bboxes_orig)
                                           else [-1, -1, -1, -1])
                        sc = scores[k] if k < len(scores) else None
                        csv_rows.append({
                            "frame_idx": frame_idx, "obj_id": oid,
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "score": round(float(sc), 4) if sc is not None else "",
                        })

                    # ── results.json per_frame ─────────────────────────────
                    results_summary["per_frame"][str(frame_idx)] = {
                        "num_objects": n_obj,
                        "object_ids":  list(obj_ids),
                        "scores":      [round(float(s), 4) for s in scores],
                        "bboxes_xyxy": bboxes_orig.tolist() if n_obj > 0 else [],
                    }
                    results_summary["frames_processed"] += 1
                    if n_obj > 0:
                        results_summary["frames_with_detections"] += 1

                    # ── overlay data for masked video ──────────────────────
                    if n_obj > 0:
                        per_frame_data[frame_idx] = {
                            "masks":   masks_orig if (masks_orig is not None
                                                      and masks_orig.sum() > 0)
                                       else np.zeros((0, orig_h, orig_w), dtype=np.uint8),
                            "bboxes":  bboxes_orig,
                            "obj_ids": list(obj_ids),
                            "scores":  list(scores),
                        }

                    propagated += 1
                    if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                        light_cache()

                    if results_summary["frames_processed"] % log_every == 0:
                        print(f"    Frame {frame_idx:6d}: {n_obj} objects"
                              + (f"  scores={[f'{s:.3f}' for s in scores]}"
                                 if scores else ""))
    finally:
        try:
            predictor.handle_request(dict(type="close_session", session_id=session_id))
        except Exception as e:
            print(f"    ⚠ close_session error: {e}")

    cleanup_gpu()

    # ── write outputs ──────────────────────────────────────────────────────────
    write_masked_video(video_path, per_frame_data,
                       os.path.join(video_out_dir, "masked_video.mp4"))
    write_detections_csv(csv_rows, os.path.join(video_out_dir, "detections.csv"))
    with open(os.path.join(video_out_dir, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"    ✓ results.json → {video_out_dir}/results.json")

    del per_frame_data
    cleanup_gpu()
    return results_summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",  required=True,
                    help="Root SAM3 output directory (same as original pipeline)")
    ap.add_argument("--video_root",  default=None,
                    help="Root of the BIDS / video source tree for path discovery")
    ap.add_argument("--resize",      type=int, default=192,
                    help="Resize shorter side before SAM3 (default 192, None = no resize)")
    ap.add_argument("--log_every",   type=int, default=50)
    ap.add_argument("--empty_cache_every", type=int, default=EMPTY_CACHE_EVERY)
    ap.add_argument("--model_id",    default="facebook/sam3")
    # Allow overriding target list via CLI (space-separated folder names)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Override TARGET_FOLDER_NAMES list")
    return ap.parse_args()


def main():
    args   = parse_args()
    targets = args.targets if args.targets is not None else TARGET_FOLDER_NAMES

    print(f"\n{'='*60}")
    print(f"SAM3 Targeted Run — NO YOLO — NO aspect filter")
    print(f"Output dir : {args.output_dir}")
    print(f"Video root : {args.video_root}")
    print(f"Resize     : {args.resize}")
    print(f"Targets    : {len(targets)} folders")
    print(f"{'='*60}\n")

    # ── resolve video paths ────────────────────────────────────────────────────
    resolved = {}   # folder_name → video_path
    missing  = []
    for name in targets:
        vp = find_video_path(name, args.video_root, args.output_dir)
        if vp:
            resolved[name] = vp
            print(f"  ✓ Found  : {name}")
            print(f"           → {vp}")
        else:
            missing.append(name)
            print(f"  ✗ NOT FOUND: {name}")

    if missing:
        print(f"\n⚠ {len(missing)} video(s) could not be located — they will be skipped.")
        print("  Add explicit paths or check --video_root.\n")

    if not resolved:
        print("No videos to process. Exiting.")
        return

    # ── load SAM3 native predictor ─────────────────────────────────────────────
    print("\nLoading SAM3 native predictor...")
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()
    print("Predictor ready.\n")

    ok = fail = 0
    for i, (folder_name, video_path) in enumerate(resolved.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(resolved)}] {folder_name}")
        print(f"{'='*60}")

        video_out_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(video_out_dir, exist_ok=True)

        try:
            t0     = time.time()
            result = process_video_native(
                predictor=predictor,
                video_path=video_path,
                output_dir=args.output_dir,
                resize_shorter_side=args.resize,
                log_every=args.log_every,
                empty_cache_every=args.empty_cache_every,
            )
            elapsed = time.time() - t0
            print(f"  ✓ Done in {elapsed:.1f}s | "
                  f"{result['frames_with_detections']}/{result['frames_processed']} "
                  f"frames with detections")
            ok += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            fail += 1
            cleanup_gpu()

    print(f"\n{'='*60}")
    print(f"DONE — {ok} succeeded, {fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()