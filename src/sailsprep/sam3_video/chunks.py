"""
sam3_masks_only.py
==================
Stripped-down SAM3 inference — outputs ONLY:
  • masks/frame_XXXXXX.npy   — binary masks (N, H, W) at ORIGINAL resolution
  • bboxes/frame_XXXXXX.npy  — bboxes (N, 4) xyxy at ORIGINAL resolution
  • results.json              — frame-level metadata (no masked video, no CSV)

Removed vs. the full pipeline:
  ✗ YOLO person verification  (all SAM3 detections kept as-is)
  ✗ Aspect ratio filter        (all detections kept)
  ✗ max_frames limit           (always runs the full video)
  ✗ masked_video.mp4           (skipped entirely — saves time + RAM)
  ✗ detections.csv             (skipped — can be rebuilt from npy + json later)

Use reconstruct_detections_csv.py + reconstruct_results_json.py afterwards
to generate those files from the npy outputs if needed.

Runs the HYBRID router:
  • Short videos  (<hybrid_threshold frames) → NATIVE  backend (GPU, fast)
  • Long  videos  (≥hybrid_threshold frames) → TRANSFORMERS backend (CPU offload)

Target videos are hardcoded in TARGET_FOLDER_NAMES. The script auto-discovers
video paths via BIDS layout under --video_root, then falls back to a glob search.

Usage:
    python sam3_masks_only.py \\
        --output_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \\
        --video_root /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed \\
        --resize 192 \\
        --hybrid_threshold 1500
"""

import argparse
import gc
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# TARGET FOLDERS  (output folder name = video stem)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_FOLDER_NAMES = [
    "sub-A4E8K1L5Y2_ses-01_task-other_run-01_desc-processed_beh",
    "sub-A4E8K1L5Y2_ses-01_task-other_run-02_desc-processed_beh",
    "sub-A4E8K1L5Y2_ses-01_task-other_run-03_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-generalsocialcommunicationinteraction_run-01_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-10_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-11_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-12_desc-processed_beh",
    "sub-L0B0Q5O3Q3_ses-02_task-toyplay_run-08_desc-processed_beh",
    "sub-N3L7A1I2B9_ses-01_task-generalsocialcommunicationinteraction_run-05_desc-processed_beh",
    "sub-N3L7A1I2B9_ses-01_task-generalsocialcommunicationinteraction_run-34_desc-processed_beh",
    "sub-O7X6W5O8E0_ses-02_task-toyplay_run-02_desc-processed_beh",
]

TEXT_PROMPT       = "Human Young Child"
DTYPE             = torch.bfloat16
EMPTY_CACHE_EVERY = 50

# Chunked processing: split long videos into segments of this many frames.
# Each chunk is a fresh SAM3 session prompted on its own first frame.
# Only used when total_frames >= chunk_threshold.
# Set to 0 or None to disable chunking (process whole video at once).
DEFAULT_CHUNK_SIZE      = 1000   # frames per chunk
DEFAULT_CHUNK_THRESHOLD = 2000  # only chunk videos this long or longer


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def light_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def masks_to_bboxes(masks_np):
    """Binary masks (N,H,W) → xyxy bboxes (N,4). Returns -1 for empty masks."""
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis]
    N = masks_np.shape[0]
    bboxes = np.full((N, 4), -1, dtype=np.int32)
    for i, mask in enumerate(masks_np):
        ys, xs = np.where(mask > 0)
        if len(xs):
            bboxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# Video path discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_video_path(folder_name, video_root, output_dir):
    """
    Locate source .mp4 for a given folder/stem name.
    Priority:
      1. BIDS layout: <video_root>/<sub>/<ses>/beh/<folder_name>.mp4
      2. Recursive glob: <video_root>/**/<folder_name>.mp4  (up to 5 levels)
      3. Sibling copy:   <output_dir>/<folder_name>/<folder_name>.mp4
    """
    # 1. BIDS
    parts = folder_name.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), None)
    ses = next((p for p in parts if p.startswith("ses-")), None)
    if sub and ses and video_root:
        bids = os.path.join(video_root, sub, ses, "beh", f"{folder_name}.mp4")
        if os.path.exists(bids):
            return bids

    # 2. Glob search
    if video_root:
        for depth in range(1, 6):
            pattern = os.path.join(video_root, *["*"] * depth, f"{folder_name}.mp4")
            hits = glob.glob(pattern)
            if hits:
                return hits[0]

    # 3. Output-dir sibling
    local = os.path.join(output_dir, folder_name, f"{folder_name}.mp4")
    if os.path.exists(local):
        return local

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame reader with optional resize  (keeps original res for saving)
# ─────────────────────────────────────────────────────────────────────────────

class LazyFrameReader:
    def __init__(self, video_path, resize_shorter_side=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self._last  = -1

        if resize_shorter_side and min(self.orig_h, self.orig_w) > resize_shorter_side:
            scale = resize_shorter_side / min(self.orig_h, self.orig_w)
            rw = int(round(self.orig_w * scale)); rw += rw % 2
            rh = int(round(self.orig_h * scale)); rh += rh % 2
            self.sam_w = rw; self.sam_h = rh
            self.scale = rw / self.orig_w
        else:
            self.sam_w = self.orig_w; self.sam_h = self.orig_h
            self.scale = 1.0

    def get_frame(self, idx):
        if idx < 0 or idx >= self.total:
            return None
        if idx != self._last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self._last = idx
            if self.scale < 1.0:
                frame = cv2.resize(frame, (self.sam_w, self.sam_h),
                                   interpolation=cv2.INTER_AREA)
            return frame
        return None

    def upscale_masks(self, masks_np):
        """Scale masks from SAM resolution back to original resolution."""
        if self.scale == 1.0:
            return masks_np
        return np.stack([
            cv2.resize(m, (self.orig_w, self.orig_h),
                       interpolation=cv2.INTER_NEAREST)
            for m in masks_np
        ], axis=0)

    def upscale_bboxes(self, bboxes):
        """Scale bboxes from SAM resolution back to original resolution."""
        if self.scale == 1.0:
            return bboxes
        inv = 1.0 / self.scale
        out = bboxes.astype(np.float32) * inv
        out[:, 0] = np.clip(out[:, 0], 0, self.orig_w - 1)
        out[:, 1] = np.clip(out[:, 1], 0, self.orig_h - 1)
        out[:, 2] = np.clip(out[:, 2], 0, self.orig_w - 1)
        out[:, 3] = np.clip(out[:, 3], 0, self.orig_h - 1)
        return out.astype(np.int32)

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Core save — masks + bboxes npy only
# ─────────────────────────────────────────────────────────────────────────────

def save_frame(frame_idx, masks_np_sam, obj_ids, scores,
               masks_dir, bboxes_dir, reader, results_summary):
    """
    Scale masks/bboxes to original resolution and save .npy files.
    Updates results_summary in place.
    """
    n = len(obj_ids)

    if masks_np_sam is not None and n > 0 and masks_np_sam.sum() > 0:
        masks_orig = reader.upscale_masks(masks_np_sam)
    else:
        masks_orig = None

    # Bboxes derived from upscaled masks (most accurate source)
    if masks_orig is not None and masks_orig.sum() > 0:
        bboxes_orig = masks_to_bboxes(masks_orig)
        # If upscaling lost a mask, fall back from SAM-res bbox
        if reader.scale < 1.0:
            sam_bboxes = masks_to_bboxes(masks_np_sam)
            fallback_mask = bboxes_orig[:, 0] < 0
            if fallback_mask.any():
                bboxes_orig[fallback_mask] = reader.upscale_bboxes(
                    sam_bboxes[fallback_mask])
    else:
        bboxes_orig = np.full((n, 4), -1, dtype=np.int32)

    # ── write npy ──────────────────────────────────────────────────────────────
    if masks_orig is not None and n > 0 and masks_orig.sum() > 0:
        np.save(os.path.join(masks_dir,  f"frame_{frame_idx:06d}.npy"), masks_orig)
    if n > 0:
        np.save(os.path.join(bboxes_dir, f"frame_{frame_idx:06d}.npy"), bboxes_orig)

    # ── update summary ─────────────────────────────────────────────────────────
    results_summary["per_frame"][str(frame_idx)] = {
        "num_objects": n,
        "object_ids":  list(obj_ids),
        "scores":      [round(float(s), 4) for s in scores] if scores else [],
        "bboxes_xyxy": bboxes_orig.tolist() if n > 0 else [],
    }
    results_summary["frames_processed"] += 1
    if n > 0:
        results_summary["frames_with_detections"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# NATIVE backend  (fast GPU path)
# ─────────────────────────────────────────────────────────────────────────────

def run_native_chunk(predictor, video_path, reader,
                     start_frame, end_frame,
                     log_every, empty_cache_every,
                     results_summary, masks_dir, bboxes_dir):
    """
    Run one chunk [start_frame, end_frame) as a fresh SAM3 session.
    Prompts on the first frame of the chunk, propagates to end_frame.
    """
    n_frames_in_chunk = end_frame - start_frame
    print(f"    ── Chunk frames [{start_frame}, {end_frame})  "
          f"({n_frames_in_chunk} frames) ──")

    def to_list(x):
        if isinstance(x, list):  return x
        if hasattr(x, "tolist"): return x.tolist()
        return list(x) if x else []

    resp       = predictor.handle_request(dict(type="start_session",
                                               resource_path=video_path))
    session_id = resp["session_id"]

    # Prompt on the first frame of this chunk
    predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                  frame_index=start_frame, text=TEXT_PROMPT))

    propagated = 0
    try:
        with torch.inference_mode():
            ctx = (torch.autocast("cuda", dtype=DTYPE) if torch.cuda.is_available()
                   else torch.autocast("cpu", dtype=DTYPE))
            with ctx:
                for response in predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_idx=start_frame,
                        max_frame_num_to_track=n_frames_in_chunk,
                    )
                ):
                    frame_idx = response["frame_index"]
                    outputs   = response["outputs"]

                    obj_ids = to_list(outputs.get("out_obj_ids", []))
                    scores  = to_list(outputs.get("out_probs",   []))
                    masks   = outputs.get("out_binary_masks", None)

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
                        if masks_np.shape[-2:] != (reader.sam_h, reader.sam_w):
                            masks_np = np.stack([
                                cv2.resize(m, (reader.sam_w, reader.sam_h),
                                           interpolation=cv2.INTER_NEAREST)
                                for m in masks_np
                            ], axis=0)
                    else:
                        masks_np = None

                    save_frame(frame_idx, masks_np, obj_ids, scores,
                               masks_dir, bboxes_dir, reader, results_summary)

                    propagated += 1
                    if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                        light_cache()

                    if results_summary["frames_processed"] % log_every == 0:
                        n = len(obj_ids)
                        sc_str = (f"  scores={[f'{s:.3f}' for s in scores]}"
                                  if scores else "")
                        print(f"    Frame {frame_idx:6d}: {n} obj{sc_str}")
    finally:
        try:
            predictor.handle_request(dict(type="close_session",
                                          session_id=session_id))
        except Exception as e:
            print(f"    ⚠ close_session (chunk): {e}")

    cleanup_gpu()


def run_native(predictor, video_path, output_dir, reader,
               log_every, empty_cache_every, results_summary,
               masks_dir, bboxes_dir,
               chunk_size=0, chunk_threshold=0):

    total = reader.total
    use_chunks = (chunk_size and chunk_size > 0
                  and chunk_threshold and chunk_size > 0
                  and total >= chunk_threshold)

    if use_chunks:
        # ── Chunked mode: fresh session per chunk ──────────────────────────
        chunk_starts = list(range(0, total, chunk_size))
        print(f"    Chunked mode: {len(chunk_starts)} chunks of ≤{chunk_size} frames")
        for ci, start in enumerate(chunk_starts):
            end = min(start + chunk_size, total)
            print(f"    Chunk [{ci+1}/{len(chunk_starts)}]", end="  ")
            run_native_chunk(
                predictor=predictor,
                video_path=video_path,
                reader=reader,
                start_frame=start,
                end_frame=end,
                log_every=log_every,
                empty_cache_every=empty_cache_every,
                results_summary=results_summary,
                masks_dir=masks_dir,
                bboxes_dir=bboxes_dir,
            )
            cleanup_gpu()   # full cleanup between chunks
    else:
        # ── Single-session mode (short videos) ────────────────────────────
        def to_list(x):
            if isinstance(x, list):  return x
            if hasattr(x, "tolist"): return x.tolist()
            return list(x) if x else []

        resp       = predictor.handle_request(dict(type="start_session",
                                                   resource_path=video_path))
        session_id = resp["session_id"]
        predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                      frame_index=0, text=TEXT_PROMPT))

        propagated = 0
        try:
            with torch.inference_mode():
                ctx = (torch.autocast("cuda", dtype=DTYPE) if torch.cuda.is_available()
                       else torch.autocast("cpu", dtype=DTYPE))
                with ctx:
                    for response in predictor.handle_stream_request(
                        request=dict(type="propagate_in_video", session_id=session_id)
                    ):
                        frame_idx = response["frame_index"]
                        outputs   = response["outputs"]

                        obj_ids = to_list(outputs.get("out_obj_ids", []))
                        scores  = to_list(outputs.get("out_probs",   []))
                        masks   = outputs.get("out_binary_masks", None)

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
                            if masks_np.shape[-2:] != (reader.sam_h, reader.sam_w):
                                masks_np = np.stack([
                                    cv2.resize(m, (reader.sam_w, reader.sam_h),
                                               interpolation=cv2.INTER_NEAREST)
                                    for m in masks_np
                                ], axis=0)
                        else:
                            masks_np = None

                        save_frame(frame_idx, masks_np, obj_ids, scores,
                                   masks_dir, bboxes_dir, reader, results_summary)

                        propagated += 1
                        if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                            light_cache()

                        if results_summary["frames_processed"] % log_every == 0:
                            n = len(obj_ids)
                            sc_str = (f"  scores={[f'{s:.3f}' for s in scores]}"
                                      if scores else "")
                            print(f"    Frame {frame_idx:6d}: {n} obj{sc_str}")
        finally:
            try:
                predictor.handle_request(dict(type="close_session",
                                              session_id=session_id))
            except Exception as e:
                print(f"    ⚠ close_session: {e}")

        cleanup_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMERS backend  (CPU-offload, for long videos)
# ─────────────────────────────────────────────────────────────────────────────

def run_transformers(model, processor, device,
                     video_path, reader,
                     log_every, empty_cache_every, results_summary,
                     masks_dir, bboxes_dir,
                     chunk_size=0, chunk_threshold=0):

    img_w, img_h = reader.sam_w, reader.sam_h
    total        = reader.total

    use_chunks = (chunk_size and chunk_size > 0
                  and chunk_threshold and chunk_threshold > 0
                  and total >= chunk_threshold)

    chunk_starts = list(range(0, total, chunk_size)) if use_chunks else [0]
    if use_chunks:
        print(f"    Chunked mode (transformers): "
              f"{len(chunk_starts)} chunks of ≤{chunk_size} frames")

    oom_frame = None

    for ci, start in enumerate(chunk_starts):
        end = min(start + chunk_size, total) if use_chunks else total
        n_chunk = end - start
        if use_chunks:
            print(f"    Chunk [{ci+1}/{len(chunk_starts)}]  "
                  f"frames [{start}, {end})  ({n_chunk} frames)")

        # Load only this chunk's frames into RAM
        print(f"    Loading {n_chunk} frames via OpenCV ...")
        cap    = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(n_chunk):
            ret, frm = cap.read()
            if not ret:
                break
            if reader.scale < 1.0:
                frm = cv2.resize(frm, (img_w, img_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"    Loaded {len(frames)} frames.")

        sess = processor.init_video_session(
            video=frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=DTYPE,
        )
        sess = processor.add_text_prompt(inference_session=sess, text=TEXT_PROMPT)

        propagated = 0
        try:
            with torch.inference_mode():
                ctx = (torch.autocast("cuda", dtype=DTYPE) if torch.cuda.is_available()
                       else torch.autocast("cpu", dtype=DTYPE))
                with ctx:
                    for model_out in model.propagate_in_video_iterator(
                        inference_session=sess,
                        max_frame_num_to_track=len(frames),
                    ):
                        # model_out.frame_idx is 0-based within the chunk
                        # → translate back to global frame index
                        local_idx  = model_out.frame_idx
                        frame_idx  = start + local_idx
                        proc       = processor.postprocess_outputs(sess, model_out)

                        obj_ids = (proc["object_ids"].tolist()
                                   if len(proc["object_ids"]) > 0 else [])
                        scores  = (proc["scores"].tolist()
                                   if len(proc["scores"])     > 0 else [])

                        masks_t = proc["masks"]
                        if len(obj_ids) > 0:
                            masks_np = masks_t.detach().cpu().numpy().astype(np.uint8)
                        else:
                            masks_np = None
                        del masks_t
                        del proc["boxes"]

                        save_frame(frame_idx, masks_np, obj_ids, scores,
                                   masks_dir, bboxes_dir, reader, results_summary)

                        propagated += 1
                        if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                            light_cache()

                        if results_summary["frames_processed"] % log_every == 0:
                            n = len(obj_ids)
                            sc_str = (f"  scores={[f'{s:.3f}' for s in scores]}"
                                      if scores else "")
                            print(f"    Frame {frame_idx:6d}: {n} obj{sc_str}")

        except torch.cuda.OutOfMemoryError:
            oom_frame = results_summary["frames_processed"]
            print(f"    ⚠ OOM at frame ~{oom_frame} in chunk [{start},{end}) "
                  f"— partial results saved")
            results_summary["oom_at_frame"] = oom_frame
            torch.cuda.empty_cache()
            try: del sess
            except Exception: pass
            try: del frames
            except Exception: pass
            gc.collect()
            cleanup_gpu()
            return oom_frame   # abort remaining chunks
        finally:
            try: del sess
            except Exception: pass
            try: del frames
            except Exception: pass
            gc.collect()
            cleanup_gpu()

    return None   # success


# ─────────────────────────────────────────────────────────────────────────────
# Process one video end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path, output_dir, router_state,
                  resize_shorter_side, hybrid_threshold,
                  log_every, empty_cache_every,
                  chunk_size=DEFAULT_CHUNK_SIZE,
                  chunk_threshold=DEFAULT_CHUNK_THRESHOLD):

    video_name    = Path(video_path).stem
    video_out_dir = os.path.join(output_dir, video_name)
    masks_dir     = os.path.join(video_out_dir, "masks")
    bboxes_dir    = os.path.join(video_out_dir, "bboxes")
    os.makedirs(masks_dir,  exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)

    reader = LazyFrameReader(video_path, resize_shorter_side=resize_shorter_side)
    total  = reader.total

    print(f"    Resolution  : {reader.orig_w}x{reader.orig_h}"
          f" → SAM {reader.sam_w}x{reader.sam_h}  (scale={reader.scale:.3f})")
    print(f"    Total frames: {total}")

    results_summary = {
        "video_path":                video_path,
        "video_name":                video_name,
        "prompt":                    TEXT_PROMPT,
        "total_frames":              total,
        "detection_mode":            "raw SAM3 — no YOLO, no aspect filter, no frame limit",
        "backend_used":              None,          # filled below
        "verification":              "none",
        "min_confidence":            0.0,
        "max_aspect_ratio":          -1,
        "resize_shorter_side":       resize_shorter_side,
        "sam3_resolution":           f"{reader.sam_w}x{reader.sam_h}",
        "original_resolution":       f"{reader.orig_w}x{reader.orig_h}",
        "frames_processed":          0,
        "frames_with_detections":    0,
        "frames_rejected_not_human": 0,
        "per_frame":                 {},
    }

    if total >= hybrid_threshold:
        # ── TRANSFORMERS backend ──────────────────────────────────────────────
        backend_name = "transformers"
        results_summary["backend_used"] = backend_name
        print(f"    Backend     : TRANSFORMERS (CPU offload)  [{total} ≥ {hybrid_threshold}]")

        if router_state["transformers_model"] is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            from accelerate import Accelerator
            device = Accelerator().device
            print(f"    Loading transformers model on {device} ...")
            router_state["transformers_model"]     = Sam3VideoModel.from_pretrained(
                router_state["model_id"]).to(device, dtype=DTYPE)
            router_state["transformers_processor"] = Sam3VideoProcessor.from_pretrained(
                router_state["model_id"])
            router_state["transformers_device"]    = device
            print(f"    Transformers model ready.")

        oom = run_transformers(
            model=router_state["transformers_model"],
            processor=router_state["transformers_processor"],
            device=router_state["transformers_device"],
            video_path=video_path,
            reader=reader,
            log_every=log_every,
            empty_cache_every=empty_cache_every,
            results_summary=results_summary,
            masks_dir=masks_dir,
            bboxes_dir=bboxes_dir,
            chunk_size=chunk_size,
            chunk_threshold=chunk_threshold,
        )
        if oom is not None:
            # Don't write results.json so the video is retried next run
            reader.close()
            return results_summary, False

    else:
        # ── NATIVE backend ────────────────────────────────────────────────────
        backend_name = "native"
        results_summary["backend_used"] = backend_name
        print(f"    Backend     : NATIVE (GPU)  [{total} < {hybrid_threshold}]")

        if router_state["native_predictor"] is None:
            from sam3.model_builder import build_sam3_video_predictor
            print(f"    Loading native SAM3 predictor ...")
            router_state["native_predictor"] = build_sam3_video_predictor()
            print(f"    Native predictor ready.")

        run_native(
            predictor=router_state["native_predictor"],
            video_path=video_path,
            output_dir=output_dir,
            reader=reader,
            log_every=log_every,
            empty_cache_every=empty_cache_every,
            results_summary=results_summary,
            masks_dir=masks_dir,
            bboxes_dir=bboxes_dir,
            chunk_size=chunk_size,
            chunk_threshold=chunk_threshold,
        )

    reader.close()

    # ── write results.json (no masked_video, no detections.csv) ──────────────
    json_path = os.path.join(video_out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"    ✓ results.json → {json_path}")

    cleanup_gpu()
    return results_summary, True


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",  required=True,
                    help="Root SAM3 output directory")
    ap.add_argument("--video_root",  default=None,
                    help="Root of BIDS / video source tree for path discovery")
    ap.add_argument("--model_id",    default="facebook/sam3")
    ap.add_argument("--resize",      type=int, default=192,
                    help="Resize shorter side before SAM3 (default 192)")
    ap.add_argument("--hybrid_threshold", type=int, default=1500,
                    help="Frames ≥ this → transformers backend (default 1500)")
    ap.add_argument("--log_every",   type=int, default=50)
    ap.add_argument("--empty_cache_every", type=int, default=EMPTY_CACHE_EVERY)
    ap.add_argument("--overwrite",   action="store_true", default=False,
                    help="Re-run even if results.json already exists")
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help=f"Frames per chunk for long videos (default {DEFAULT_CHUNK_SIZE}). "
                         f"Set to 0 to disable chunking.")
    ap.add_argument("--chunk_threshold", type=int, default=DEFAULT_CHUNK_THRESHOLD,
                    help=f"Only chunk videos with >= this many frames "
                         f"(default {DEFAULT_CHUNK_THRESHOLD}).")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Override TARGET_FOLDER_NAMES (space-separated folder names)")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    targets = args.targets if args.targets is not None else TARGET_FOLDER_NAMES

    print(f"\n{'='*65}")
    print(f"  SAM3 MASKS-ONLY  —  no YOLO  —  no aspect filter  —  no frame cap")
    print(f"{'='*65}")
    print(f"  Output dir        : {args.output_dir}")
    print(f"  Video root        : {args.video_root}")
    print(f"  Resize            : {args.resize}")
    print(f"  Hybrid threshold  : {args.hybrid_threshold} frames")
    print(f"  Chunk size        : {args.chunk_size} frames  "
          f"(applied to videos ≥ {args.chunk_threshold} frames)")
    print(f"  Targets           : {len(targets)} folders")
    print(f"  Outputs per video : masks/*.npy  bboxes/*.npy  results.json")
    print(f"  Skipped outputs   : masked_video.mp4  detections.csv")
    print(f"{'='*65}\n")

    # ── resolve video paths ───────────────────────────────────────────────────
    resolved  = {}
    not_found = []
    for name in targets:
        vp = find_video_path(name, args.video_root, args.output_dir)
        if vp:
            resolved[name] = vp
            print(f"  ✓ {name}")
            print(f"      → {vp}")
        else:
            not_found.append(name)
            print(f"  ✗ NOT FOUND: {name}")

    if not_found:
        print(f"\n  ⚠ {len(not_found)} path(s) unresolved — will be skipped.\n")
    if not resolved:
        print("Nothing to process. Exiting.")
        return

    # ── router state (lazy-loaded backends) ───────────────────────────────────
    router_state = {
        "model_id":               args.model_id,
        "native_predictor":       None,
        "transformers_model":     None,
        "transformers_processor": None,
        "transformers_device":    None,
    }

    ok = fail = skip = 0

    for i, (folder_name, video_path) in enumerate(resolved.items(), 1):
        print(f"\n{'='*65}")
        print(f"  [{i}/{len(resolved)}]  {folder_name}")
        print(f"{'='*65}")

        video_out_dir = os.path.join(args.output_dir, folder_name)
        done_marker   = os.path.join(video_out_dir, "results.json")

        if os.path.exists(done_marker) and not args.overwrite:
            print(f"  ⏭ Already done (results.json exists). Use --overwrite to redo.")
            skip += 1
            continue

        # ── clean previous outputs for THIS folder only ───────────────────────
        # Deletes results.json, detections.csv, masked_video.mp4
        # Also wipes masks/ and bboxes/ contents so stale frames don't linger
        CLEAN_FILES = ["results.json", "detections.csv", "masked_video.mp4"]
        CLEAN_DIRS  = ["masks", "bboxes"]
        for fname in CLEAN_FILES:
            fp = os.path.join(video_out_dir, fname)
            if os.path.exists(fp):
                os.remove(fp)
                print(f"    🗑  Removed: {fp}")
        for dname in CLEAN_DIRS:
            dp = os.path.join(video_out_dir, dname)
            if os.path.isdir(dp):
                npy_files = [f for f in os.listdir(dp) if f.endswith(".npy")]
                for nf in npy_files:
                    os.remove(os.path.join(dp, nf))
                print(f"    🗑  Cleared {len(npy_files)} .npy files from: {dp}")

        try:
            t0 = time.time()
            _, success = process_video(
                video_path=video_path,
                output_dir=args.output_dir,
                router_state=router_state,
                resize_shorter_side=args.resize,
                hybrid_threshold=args.hybrid_threshold,
                log_every=args.log_every,
                empty_cache_every=args.empty_cache_every,
                chunk_size=args.chunk_size,
                chunk_threshold=args.chunk_threshold,
            )
            elapsed = time.time() - t0

            if success:
                ok += 1
                print(f"  ✓ Done in {elapsed:.1f}s")
            else:
                fail += 1
                print(f"  ✗ OOM — partial npy files saved, results.json NOT written")
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            fail += 1
            cleanup_gpu()

    print(f"\n{'='*65}")
    print(f"  DONE  —  {ok} succeeded  |  {fail} failed  |  {skip} skipped")
    print(f"{'='*65}")
    print(f"\n  To generate detections.csv from the saved npy/json files, run:")
    print(f"    python reconstruct_detections_csv.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()