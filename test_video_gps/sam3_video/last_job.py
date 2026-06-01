"""
SAM3.1 Video Inference Script — HYBRID v2
==========================================
Hybrid backend routing to defeat OOM on long videos while preserving
tracking IDs and accuracy:

  • Videos < hybrid_threshold frames  → NATIVE backend (fast, GPU)
  • Videos ≥ hybrid_threshold frames  → TRANSFORMERS backend (CPU offload)

Both backends:
  ✓ Use the same SAM3 weights (identical accuracy)
  ✓ Run a SINGLE continuous session per video (tracking IDs intact)
  ✓ Never chunk

Plus all v1 OOM fixes:
  ✓ Adaptive resize per video length (constant within a video)
  ✓ torch.inference_mode() + bf16 autocast
  ✓ Periodic empty_cache() inside propagation loop (defrag only)
  ✓ Immediate del of GPU tensors after CPU copy
  ✓ Loud close_session errors
  ✓ Periodic backend rebuild BETWEEN videos only

Changes from v2:
  ✓ Removed BIDS filtering — ALL videos in CSV are processed
  ✓ Job array support via --array_index / --array_total
"""

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TEXT_PROMPT = "Human Young Child"
DTYPE = torch.bfloat16

MIN_CONFIDENCE = 0.15

YOLO_MODEL_NAME   = "yolov8n.pt"
YOLO_PERSON_CLASS = 0
YOLO_PERSON_CONF  = 0.3
YOLO_BBOX_PAD     = 20

MAX_ASPECT_RATIO = 3.5

MODEL_ID_BEST     = "facebook/sam3"
MODEL_ID_FALLBACK = "facebook/sam3"

EMPTY_CACHE_EVERY = 50
REBUILD_BACKEND_EVERY = 25

# Hybrid routing threshold (frames). Videos with >= this many frames
# go to the transformers backend with CPU offload.
HYBRID_THRESHOLD = 1500

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


# ─────────────────────────────────────────────
# Resize helpers
# ─────────────────────────────────────────────

def resize_frame(frame: np.ndarray, shorter_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) <= shorter_side:
        return frame
    scale = shorter_side / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pick_adaptive_resize(total_frames, resize_default, resize_long, resize_xlong,
                         long_threshold, xlong_threshold):
    if total_frames >= xlong_threshold:
        return resize_xlong
    if total_frames >= long_threshold:
        return resize_long
    return resize_default


# ─────────────────────────────────────────────
# Lazy frame reader
# ─────────────────────────────────────────────

class LazyFrameReader:
    def __init__(self, BidsProcessed, resize_shorter_side=None):
        self.BidsProcessed = BidsProcessed
        self.resize_shorter_side = resize_shorter_side
        self.cap = cv2.VideoCapture(BidsProcessed)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {BidsProcessed}")

        self.orig_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total       = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps         = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self._last_idx   = -1

        if (resize_shorter_side is not None
                and min(self.orig_height, self.orig_width) > resize_shorter_side):
            scale = resize_shorter_side / min(self.orig_height, self.orig_width)
            rw = int(round(self.orig_width  * scale))
            rh = int(round(self.orig_height * scale))
            self.width  = rw if rw % 2 == 0 else rw + 1
            self.height = rh if rh % 2 == 0 else rh + 1
            self.scale  = self.width / self.orig_width
        else:
            self.width  = self.orig_width
            self.height = self.orig_height
            self.scale  = 1.0

        if resize_shorter_side is not None:
            print(f"    Resize: {self.orig_width}x{self.orig_height} → "
                  f"{self.width}x{self.height} "
                  f"(shorter_side={resize_shorter_side}, scale={self.scale:.3f})")
            est = (self.orig_width * self.orig_height) / (self.width * self.height)
            print(f"    Estimated VRAM reduction: ~{est:.1f}x")

    def get_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.total:
            return None
        if frame_idx != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self._last_idx = frame_idx
            if self.resize_shorter_side is not None and self.scale < 1.0:
                frame = cv2.resize(frame, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)
            return frame
        return None

    def scale_bbox_to_original(self, bbox_xyxy):
        if self.scale == 1.0 or bbox_xyxy is None or len(bbox_xyxy) == 0:
            return bbox_xyxy
        inv = 1.0 / self.scale
        scaled = bbox_xyxy.astype(np.float32) * inv
        scaled[:, 0] = np.clip(scaled[:, 0], 0, self.orig_width  - 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, self.orig_height - 1)
        scaled[:, 2] = np.clip(scaled[:, 2], 0, self.orig_width  - 1)
        scaled[:, 3] = np.clip(scaled[:, 3], 0, self.orig_height - 1)
        return scaled.astype(np.int32)

    def scale_mask_to_original(self, mask):
        if self.scale == 1.0:
            return mask
        return cv2.resize(mask, (self.orig_width, self.orig_height),
                          interpolation=cv2.INTER_NEAREST)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


def peek_total_frames(BidsProcessed):
    cap = cv2.VideoCapture(BidsProcessed)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ─────────────────────────────────────────────
# GPU cleanup
# ─────────────────────────────────────────────

def cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def light_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# YOLO verifier
# ─────────────────────────────────────────────

class YOLOPersonVerifier:
    def __init__(self, model_name=YOLO_MODEL_NAME, person_conf=YOLO_PERSON_CONF):
        from ultralytics import YOLO
        print(f"Loading YOLO person verifier: {model_name}")
        self.model = YOLO(model_name)
        self.person_conf = person_conf
        print("YOLO person verifier ready.")

    def contains_person(self, frame_bgr, bbox_xyxy, pad=YOLO_BBOX_PAD):
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        if x1 < 0 or x2 <= x1 or y2 <= y1:
            return False
        cx1 = max(0, int(x1) - pad); cy1 = max(0, int(y1) - pad)
        cx2 = min(W, int(x2) + pad); cy2 = min(H, int(y2) + pad)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False
        results = self.model(crop, conf=self.person_conf, verbose=False)
        if len(results) == 0:
            return False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy().astype(int)
                if YOLO_PERSON_CLASS in classes:
                    return True
        return False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def convert_xywh_rel_to_xyxy_abs(boxes_xywh, img_width, img_height):
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    bboxes = []
    for box in boxes_xywh:
        if hasattr(box, "tolist"):
            box = box.tolist()
        cx, cy, w, h = box
        x1 = max(0, int((cx - w / 2) * img_width))
        y1 = max(0, int((cy - h / 2) * img_height))
        x2 = min(img_width - 1,  int((cx + w / 2) * img_width))
        y2 = min(img_height - 1, int((cy + h / 2) * img_height))
        bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes, dtype=np.int32)


def masks_to_bboxes(masks_np):
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis]
    N = masks_np.shape[0]
    bboxes = np.full((N, 4), -1, dtype=np.int32)
    for i, mask in enumerate(masks_np):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        bboxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return bboxes


def check_aspect_ratio(bbox_xyxy, max_ratio=MAX_ASPECT_RATIO):
    if max_ratio <= 0:
        return True
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if h <= 0:
        return False
    return (w / h) <= max_ratio


def _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height):
    bbox = None
    if masks_np is not None and masks_np.ndim >= 2:
        mask_bboxes = masks_to_bboxes(masks_np)
        if idx < len(mask_bboxes):
            b = mask_bboxes[idx]
            if b[0] >= 0:
                bbox = b
    if bbox is None and boxes_xywh is not None and len(boxes_xywh) > 0:
        model_bboxes = convert_xywh_rel_to_xyxy_abs(boxes_xywh, img_width, img_height)
        if idx < len(model_bboxes):
            bbox = model_bboxes[idx]
    return bbox


def pick_top1_human_verified(obj_ids, scores, masks_np, boxes_xywh,
                              frame_bgr, yolo_verifier,
                              img_width, img_height,
                              min_confidence=MIN_CONFIDENCE,
                              max_aspect_ratio=MAX_ASPECT_RATIO):
    if not obj_ids or len(obj_ids) == 0:
        return [], [], None, []

    scores_arr = np.array(scores, dtype=np.float32)
    sorted_indices = np.argsort(-scores_arr)

    for idx in sorted_indices:
        idx = int(idx)
        score = float(scores_arr[idx])
        if score < min_confidence:
            continue
        bbox = _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height)
        if bbox is None:
            continue
        if not check_aspect_ratio(bbox, max_aspect_ratio):
            print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — too wide")
            continue
        if yolo_verifier is not None and frame_bgr is not None:
            if not yolo_verifier.contains_person(frame_bgr, bbox):
                print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — YOLO no person")
                continue

        best_obj_ids = [obj_ids[idx]]
        best_scores  = [score]
        best_masks = masks_np[idx:idx+1] if (masks_np is not None and idx < len(masks_np)) else None
        if boxes_xywh is not None and len(boxes_xywh) > 0:
            best_boxes = boxes_xywh[idx:idx+1] if hasattr(boxes_xywh, '__getitem__') else [boxes_xywh[idx]]
        else:
            best_boxes = []
        return best_obj_ids, best_scores, best_masks, best_boxes

    return [], [], None, []


def pick_all_humans_verified(obj_ids, scores, masks_np, boxes_xywh,
                              frame_bgr, yolo_verifier,
                              img_width, img_height,
                              min_confidence=MIN_CONFIDENCE,
                              max_aspect_ratio=MAX_ASPECT_RATIO):
    if not obj_ids or len(obj_ids) == 0:
        return [], [], None, []

    scores_arr = np.array(scores, dtype=np.float32)
    passed_ids, passed_scores = [], []
    passed_mask_indices, passed_box_indices = [], []

    for idx in range(len(obj_ids)):
        score = float(scores_arr[idx])
        if score < min_confidence:
            continue
        bbox = _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height)
        if bbox is None:
            continue
        if not check_aspect_ratio(bbox, max_aspect_ratio):
            print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — too wide")
            continue
        if yolo_verifier is not None and frame_bgr is not None:
            if not yolo_verifier.contains_person(frame_bgr, bbox):
                print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — YOLO no person")
                continue
        passed_ids.append(obj_ids[idx])
        passed_scores.append(score)
        passed_mask_indices.append(idx)
        passed_box_indices.append(idx)

    if len(passed_ids) == 0:
        return [], [], None, []

    if masks_np is not None and len(masks_np) > 0:
        valid = [i for i in passed_mask_indices if i < len(masks_np)]
        result_masks = masks_np[valid] if valid else None
    else:
        result_masks = None

    if boxes_xywh is not None and len(boxes_xywh) > 0:
        if hasattr(boxes_xywh, '__getitem__'):
            result_boxes = np.array([boxes_xywh[i] for i in passed_box_indices
                                     if i < len(boxes_xywh)])
        else:
            result_boxes = []
    else:
        result_boxes = []

    return passed_ids, passed_scores, result_masks, result_boxes


def draw_overlay(frame, masks_np, bboxes, obj_ids, scores, multi_color=False):
    out = frame.copy()
    H, W = frame.shape[:2]
    for i in range(len(obj_ids)):
        if multi_color and len(obj_ids) > 1:
            color_mask = MULTI_COLORS[i % len(MULTI_COLORS)]
            color_bbox = color_mask
            color_label_bg = color_mask
        else:
            color_mask = MASK_COLOR
            color_bbox = BBOX_COLOR
            color_label_bg = LABEL_BG
        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i]
            if mask.shape == (H, W) and np.any(mask):
                overlay = out.copy()
                overlay[mask > 0] = color_mask
                out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)
        if i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            if x1 >= 0:
                cv2.rectangle(out, (x1, y1), (x2, y2), color_bbox, BBOX_THICK)
                oid = obj_ids[i]
                score = scores[i] if i < len(scores) else None
                label = f"ID:{oid}"
                if score is not None:
                    try:
                        label += f" {float(score):.2f}"
                    except (TypeError, ValueError):
                        pass
                (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
                lx1 = x1
                ly1 = max(y1 - th - baseline - 4, 0)
                lx2 = x1 + tw + 4
                ly2 = max(y1, th + baseline + 4)
                cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color_label_bg, -1)
                cv2.putText(out, label, (lx1 + 2, ly2 - baseline - 2),
                            FONT, FONT_SCALE, LABEL_COLOR, FONT_THICK, cv2.LINE_AA)
    return out


def write_masked_video(BidsProcessed, per_frame_data, output_path, multi_color=False):
    cap = cv2.VideoCapture(BidsProcessed)
    if not cap.isOpened():
        print(f"    ⚠ Could not open video for masked output: {BidsProcessed}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in per_frame_data:
                fd = per_frame_data[frame_idx]
                frame = draw_overlay(
                    frame,
                    fd.get("masks",   np.zeros((0, height, width), dtype=np.uint8)),
                    fd.get("bboxes",  np.zeros((0, 4),             dtype=np.int32)),
                    fd.get("obj_ids", []),
                    fd.get("scores",  []),
                    multi_color=multi_color,
                )
            proc.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    if proc.returncode == 0:
        print(f"    ✓ Masked video saved → {output_path}")
    else:
        print(f"    ⚠ ffmpeg exited with code {proc.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3.1 video segmentation — HYBRID native/transformers")
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sam3_results")
    parser.add_argument("--model_id",   type=str, default=MODEL_ID_BEST)
    parser.add_argument("--prompt",     type=str, default=TEXT_PROMPT)
    parser.add_argument("--min_confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--yolo_model", type=str, default=YOLO_MODEL_NAME)
    parser.add_argument("--yolo_person_conf", type=float, default=YOLO_PERSON_CONF)
    parser.add_argument("--max_aspect_ratio", type=float, default=MAX_ASPECT_RATIO)
    parser.add_argument("--no_yolo", action="store_true", default=False)
    parser.add_argument("--multi", action="store_true", default=False)

    # Adaptive resize tiers
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--resize_long", type=int, default=None)
    parser.add_argument("--resize_xlong", type=int, default=None)
    parser.add_argument("--long_threshold",  type=int, default=1500)
    parser.add_argument("--xlong_threshold", type=int, default=3000)

    # Hybrid routing
    parser.add_argument("--hybrid_threshold", type=int, default=HYBRID_THRESHOLD,
                        help="Videos with >= this many frames use the transformers "
                             "backend with CPU offload. Shorter videos use native.")
    parser.add_argument("--force_backend", type=str, default="hybrid",
                        choices=["hybrid", "native", "transformers"],
                        help="Force a single backend instead of routing.")

    parser.add_argument("--empty_cache_every", type=int, default=EMPTY_CACHE_EVERY)
    parser.add_argument("--rebuild_backend_every", type=int, default=REBUILD_BACKEND_EVERY)

    parser.add_argument("--save_masks", action="store_true", default=True)
    parser.add_argument("--save_bboxes", action="store_true", default=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--log_every",  type=int, default=10)

    # Job array support
    parser.add_argument("--array_index", type=int, default=0,
                        help="0-based index of this job in the array (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--array_total", type=int, default=1,
                        help="Total number of jobs in the array")
    parser.add_argument("--skip_masked_video_over", type=int, default=3000,
                    help="Skip masked video output for videos longer than this many frames")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Result saver (scales back to original resolution)
# ─────────────────────────────────────────────

def save_frame_results(frame_idx, masks_np, obj_ids, scores,
                       boxes_from_model, masks_dir, bboxes_dir,
                       save_masks, save_bboxes, results_summary,
                       per_frame_data, csv_rows,
                       img_width, img_height,
                       frame_reader=None):
    n_objects = len(obj_ids)

    if masks_np is not None and n_objects > 0 and masks_np.sum() > 0:
        computed_bboxes = masks_to_bboxes(masks_np)
    else:
        computed_bboxes = np.full((n_objects, 4), -1, dtype=np.int32)

    all_invalid = np.all(computed_bboxes == -1) if len(computed_bboxes) > 0 else True
    if all_invalid and boxes_from_model is not None and len(boxes_from_model) > 0:
        final_bboxes = convert_xywh_rel_to_xyxy_abs(boxes_from_model, img_width, img_height)
    else:
        final_bboxes = computed_bboxes

    orig_width  = frame_reader.orig_width  if frame_reader is not None else img_width
    orig_height = frame_reader.orig_height if frame_reader is not None else img_height

    if frame_reader is not None and frame_reader.scale < 1.0:
        valid_mask = final_bboxes[:, 0] >= 0
        if valid_mask.any():
            final_bboxes_orig = final_bboxes.copy()
            final_bboxes_orig[valid_mask] = frame_reader.scale_bbox_to_original(
                final_bboxes[valid_mask])
        else:
            final_bboxes_orig = final_bboxes
        if masks_np is not None and n_objects > 0 and masks_np.sum() > 0:
            masks_np_orig = np.stack([
                frame_reader.scale_mask_to_original(m) for m in masks_np
            ], axis=0)
        else:
            masks_np_orig = masks_np
    else:
        final_bboxes_orig = final_bboxes
        masks_np_orig     = masks_np

    if save_masks and masks_dir and masks_np_orig is not None and n_objects > 0 and masks_np_orig.sum() > 0:
        np.save(os.path.join(masks_dir, f"frame_{frame_idx:06d}.npy"), masks_np_orig)

    if save_bboxes and bboxes_dir and n_objects > 0:
        np.save(os.path.join(bboxes_dir, f"frame_{frame_idx:06d}.npy"), final_bboxes_orig)

    for k, oid in enumerate(obj_ids):
        x1, y1, x2, y2 = final_bboxes_orig[k] if k < len(final_bboxes_orig) else (-1, -1, -1, -1)
        sc = scores[k] if k < len(scores) else None
        csv_rows.append({
            "frame_idx": frame_idx, "obj_id": oid,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "score": round(float(sc), 4) if sc is not None else "",
        })

    boxes_model_list = (boxes_from_model.tolist() if hasattr(boxes_from_model, 'tolist')
                        else (list(boxes_from_model) if boxes_from_model is not None else []))
    frame_info = {
        "num_objects":  n_objects,
        "object_ids":   list(obj_ids),
        "scores":       [round(float(s), 4) for s in scores] if scores else [],
        "boxes_model":  boxes_model_list,
        "bboxes_xyxy":  final_bboxes_orig.tolist(),
    }
    results_summary["per_frame"][str(frame_idx)] = frame_info
    if n_objects > 0:
        results_summary["frames_with_detections"] += 1

    if n_objects > 0:
        per_frame_data[frame_idx] = {
            "masks":   masks_np_orig if (masks_np_orig is not None and masks_np_orig.sum() > 0)
                       else np.zeros((0, orig_height, orig_width), dtype=np.uint8),
            "bboxes":  final_bboxes_orig,
            "obj_ids": list(obj_ids),
            "scores":  list(scores),
        }

    results_summary["frames_processed"] += 1


def write_detections_csv(csv_rows, path):
    fieldnames = ["frame_idx", "obj_id", "x1", "y1", "x2", "y2", "score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"    ✓ Detections CSV saved → {path}")


# ═════════════════════════════════════════════
# BACKEND 1: Native sam3 package (FAST, GPU)
# ═════════════════════════════════════════════
class NativeBackend:
    name = "native"

    def __init__(self, model_id):
        from sam3.model_builder import build_sam3_video_predictor
        print(f"[NATIVE] Loading SAM3 video predictor (model: {model_id})")
        self.predictor = build_sam3_video_predictor()
        print("[NATIVE] Predictor loaded.")

    def _process_session(self, BidsProcessed, prompt, frame_reader,
                         yolo_verifier, picker_fn,
                         min_confidence, max_aspect_ratio,
                         results_summary, per_frame_data, csv_rows,
                         masks_dir, bboxes_dir, save_masks, save_bboxes,
                         log_every, max_frames_to_track,
                         empty_cache_every):
        img_width  = frame_reader.width
        img_height = frame_reader.height

        response   = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=BidsProcessed))
        session_id = response["session_id"]

        self.predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id,
                         frame_index=0, text=prompt))

        propagate_request = dict(type="propagate_in_video", session_id=session_id)
        if max_frames_to_track is not None:
            propagate_request["max_frame_num_to_track"] = max_frames_to_track

        def to_list(x):
            if isinstance(x, list):  return x
            if hasattr(x, "tolist"): return x.tolist()
            return list(x) if x else []

        propagated = 0
        try:
            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast("cuda", dtype=DTYPE)
                    if torch.cuda.is_available()
                    else torch.autocast("cpu", dtype=DTYPE)
                )
                with autocast_ctx:
                    for response in self.predictor.handle_stream_request(request=propagate_request):
                        frame_idx = response["frame_index"]
                        outputs   = response["outputs"]

                        obj_ids_l = to_list(outputs.get("out_obj_ids", []))
                        scores_l  = to_list(outputs.get("out_probs",   []))
                        boxes     = outputs.get("out_boxes_xywh",    None)
                        masks     = outputs.get("out_binary_masks",  None)

                        if masks is not None and len(obj_ids_l) > 0:
                            if isinstance(masks, torch.Tensor):
                                masks_np = masks.detach().to("cpu").numpy().astype(np.uint8)
                                del masks
                            elif isinstance(masks, np.ndarray):
                                masks_np = masks.astype(np.uint8)
                            else:
                                masks_np = np.array(masks, dtype=np.uint8)
                            if masks_np.ndim == 2:
                                masks_np = masks_np[np.newaxis]
                            while masks_np.ndim > 3:
                                masks_np = masks_np[:, 0]
                            if masks_np.shape[-2:] != (img_height, img_width):
                                upsampled = [cv2.resize(m, (img_width, img_height),
                                                        interpolation=cv2.INTER_NEAREST)
                                             for m in masks_np]
                                masks_np = np.array(upsampled, dtype=np.uint8)
                        else:
                            masks_np = None

                        if isinstance(boxes, torch.Tensor):
                            boxes_cpu = boxes.detach().to("cpu").numpy()
                            del boxes
                            boxes = boxes_cpu

                        frame_bgr = frame_reader.get_frame(frame_idx)
                        had_candidates = len(obj_ids_l) > 0

                        obj_ids_f, scores_f, masks_f, boxes_f = picker_fn(
                            obj_ids_l, scores_l, masks_np, boxes,
                            frame_bgr=frame_bgr,
                            yolo_verifier=yolo_verifier,
                            img_width=img_width,
                            img_height=img_height,
                            min_confidence=min_confidence,
                            max_aspect_ratio=max_aspect_ratio,
                        )
                        del frame_bgr

                        if had_candidates and len(obj_ids_f) == 0:
                            results_summary["frames_rejected_not_human"] += 1

                        save_frame_results(
                            frame_idx=frame_idx, masks_np=masks_f,
                            obj_ids=obj_ids_f, scores=scores_f,
                            boxes_from_model=boxes_f,
                            masks_dir=masks_dir, bboxes_dir=bboxes_dir,
                            save_masks=save_masks, save_bboxes=save_bboxes,
                            results_summary=results_summary,
                            per_frame_data=per_frame_data, csv_rows=csv_rows,
                            img_width=img_width, img_height=img_height,
                            frame_reader=frame_reader,
                        )

                        propagated += 1
                        if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                            light_empty_cache()

                        if results_summary["frames_processed"] % log_every == 0:
                            status = f"    Frame {frame_idx}: {len(obj_ids_f)} objects"
                            if scores_f:
                                status += f" (scores={[f'{s:.3f}' for s in scores_f]})"
                            print(status)
        finally:
            try:
                self.predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id))
            except Exception as e:
                print(f"    ⚠ close_session failed: {e}")

        cleanup_gpu_memory()

    def process_video(self, BidsProcessed, prompt, output_dir,
                      yolo_verifier=None,
                      min_confidence=MIN_CONFIDENCE,
                      max_aspect_ratio=MAX_ASPECT_RATIO,
                      multi_mode=False,
                      resize_shorter_side=None,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10,
                      empty_cache_every=EMPTY_CACHE_EVERY,
                      skip_masked_video_over=3000):

        video_name    = Path(BidsProcessed).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        frame_reader = LazyFrameReader(BidsProcessed, resize_shorter_side=resize_shorter_side)
        total_frames = frame_reader.total

        print(f"    Backend     : NATIVE (GPU)")
        print(f"    Video       : {total_frames} frames")
        print(f"    Original res: {frame_reader.orig_width}x{frame_reader.orig_height}")
        print(f"    SAM3 res    : {frame_reader.width}x{frame_reader.height}"
              + (" (resized)" if frame_reader.scale < 1.0 else " (original)"))

        detection_mode = "multi (all verified humans)" if multi_mode else "top-1 human only"
        results_summary = {
            "BidsProcessed":    BidsProcessed,
            "video_name":    video_name,
            "prompt":        prompt,
            "total_frames":  total_frames,
            "detection_mode": detection_mode,
            "backend_used":  "native",
            "verification":  "YOLOv8 person + aspect ratio",
            "min_confidence": min_confidence,
            "max_aspect_ratio": max_aspect_ratio,
            "resize_shorter_side": resize_shorter_side,
            "sam3_resolution": f"{frame_reader.width}x{frame_reader.height}",
            "original_resolution": f"{frame_reader.orig_width}x{frame_reader.orig_height}",
            "frames_processed":           0,
            "frames_with_detections":     0,
            "frames_rejected_not_human":  0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []
        picker_fn = pick_all_humans_verified if multi_mode else pick_top1_human_verified

        self._process_session(
            BidsProcessed=BidsProcessed, prompt=prompt,
            frame_reader=frame_reader,
            yolo_verifier=yolo_verifier, picker_fn=picker_fn,
            min_confidence=min_confidence, max_aspect_ratio=max_aspect_ratio,
            results_summary=results_summary,
            per_frame_data=per_frame_data, csv_rows=csv_rows,
            masks_dir=masks_dir, bboxes_dir=bboxes_dir,
            save_masks=save_masks, save_bboxes=save_bboxes,
            log_every=log_every, max_frames_to_track=max_frames,
            empty_cache_every=empty_cache_every,
        )

        frame_reader.close()

        if frame_reader.total <= skip_masked_video_over:
            write_masked_video(BidsProcessed, per_frame_data,
                               os.path.join(video_out_dir, "masked_video.mp4"),
                               multi_color=multi_mode)
        else:
            print(f"    ⏭ Skipping masked video render ({frame_reader.total} > {skip_masked_video_over} frames)")
        write_detections_csv(csv_rows, os.path.join(video_out_dir, "detections.csv"))

        with open(os.path.join(video_out_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)

        del per_frame_data
        cleanup_gpu_memory()
        return results_summary


# ═════════════════════════════════════════════
# BACKEND 2: HuggingFace Transformers (CPU OFFLOAD, slower but no OOM)
# ═════════════════════════════════════════════
class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id):
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        from accelerate import Accelerator

        self.device = Accelerator().device
        print(f"[TRANSFORMERS] Loading model: {model_id} on {self.device}")
        self.model = Sam3VideoModel.from_pretrained(model_id).to(self.device, dtype=DTYPE)
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        print("[TRANSFORMERS] Model loaded (CPU offload mode for inference state).")

    def process_video(self, BidsProcessed, prompt, output_dir,
                      yolo_verifier=None,
                      min_confidence=MIN_CONFIDENCE,
                      max_aspect_ratio=MAX_ASPECT_RATIO,
                      multi_mode=False,
                      resize_shorter_side=None,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10,
                      empty_cache_every=EMPTY_CACHE_EVERY,
                      skip_masked_video_over=3000):

        video_name    = Path(BidsProcessed).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        frame_reader = LazyFrameReader(BidsProcessed, resize_shorter_side=resize_shorter_side)
        img_width    = frame_reader.width
        img_height   = frame_reader.height

        print(f"    Backend     : TRANSFORMERS (CPU offload)")
        print(f"    Video       : {frame_reader.total} frames")
        print(f"    Original res: {frame_reader.orig_width}x{frame_reader.orig_height}")
        print(f"    SAM3 res    : {img_width}x{img_height}"
              + (" (resized)" if frame_reader.scale < 1.0 else " (original)"))

        # Load frames with OpenCV — no pyav dependency
        print(f"    Loading {frame_reader.total} frames via OpenCV...")
        cap = cv2.VideoCapture(BidsProcessed)
        video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_shorter_side is not None and frame_reader.scale < 1.0:
                frame = cv2.resize(frame, (img_width, img_height),
                                   interpolation=cv2.INTER_AREA)
            # Convert BGR → RGB for transformers
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        total_frames = len(video_frames)
        print(f"    Loaded {total_frames} frames.")

        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=DTYPE,
        )
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session, text=prompt)

        track_limit = max_frames if max_frames else total_frames

        detection_mode = "multi (all verified humans)" if multi_mode else "top-1 human only"
        results_summary = {
            "BidsProcessed":    BidsProcessed,
            "video_name":    video_name,
            "prompt":        prompt,
            "total_frames":  total_frames,
            "detection_mode": detection_mode,
            "backend_used":  "transformers",
            "verification":  "YOLOv8 person + aspect ratio",
            "min_confidence": min_confidence,
            "max_aspect_ratio": max_aspect_ratio,
            "resize_shorter_side": resize_shorter_side,
            "sam3_resolution": f"{img_width}x{img_height}",
            "original_resolution": f"{frame_reader.orig_width}x{frame_reader.orig_height}",
            "frames_processed":          0,
            "frames_with_detections":    0,
            "frames_rejected_not_human": 0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []
        picker_fn = pick_all_humans_verified if multi_mode else pick_top1_human_verified

        propagated = 0
        oom_at = None
        try:
            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast("cuda", dtype=DTYPE)
                    if torch.cuda.is_available()
                    else torch.autocast("cpu", dtype=DTYPE)
                )
                with autocast_ctx:
                    for model_outputs in self.model.propagate_in_video_iterator(
                        inference_session=inference_session,
                        max_frame_num_to_track=track_limit,
                    ):
                        frame_idx = model_outputs.frame_idx
                        processed = self.processor.postprocess_outputs(inference_session, model_outputs)

                        obj_ids  = processed["object_ids"].tolist() if len(processed["object_ids"]) > 0 else []
                        scores   = processed["scores"].tolist()     if len(processed["scores"])     > 0 else []
                        boxes_t  = processed["boxes"]
                        boxes    = boxes_t.detach().cpu().numpy() if len(boxes_t) > 0 else np.zeros((0, 4))
                        del boxes_t
                        masks_t  = processed["masks"]
                        if len(obj_ids) > 0:
                            masks_np = masks_t.detach().cpu().numpy().astype(np.uint8)
                        else:
                            masks_np = None
                        del masks_t

                        frame_bgr      = frame_reader.get_frame(frame_idx)
                        had_candidates = len(obj_ids) > 0

                        obj_ids, scores, masks_np, boxes = picker_fn(
                            obj_ids, scores, masks_np, boxes,
                            frame_bgr=frame_bgr,
                            yolo_verifier=yolo_verifier,
                            img_width=img_width, img_height=img_height,
                            min_confidence=min_confidence,
                            max_aspect_ratio=max_aspect_ratio,
                        )
                        del frame_bgr

                        if had_candidates and len(obj_ids) == 0:
                            results_summary["frames_rejected_not_human"] += 1

                        save_frame_results(
                            frame_idx=frame_idx, masks_np=masks_np,
                            obj_ids=obj_ids, scores=scores,
                            boxes_from_model=boxes,
                            masks_dir=masks_dir, bboxes_dir=bboxes_dir,
                            save_masks=save_masks, save_bboxes=save_bboxes,
                            results_summary=results_summary,
                            per_frame_data=per_frame_data, csv_rows=csv_rows,
                            img_width=img_width, img_height=img_height,
                            frame_reader=frame_reader,
                        )

                        propagated += 1
                        if empty_cache_every > 0 and propagated % empty_cache_every == 0:
                            light_empty_cache()

                        if results_summary["frames_processed"] % log_every == 0:
                            status = f"    Frame {frame_idx}: {len(obj_ids)} objects"
                            if scores:
                                status += f" (scores={[f'{s:.3f}' for s in scores]})"
                            print(status)
        except torch.cuda.OutOfMemoryError as e:
            oom_at = results_summary["frames_processed"]
            print(f"    ⚠ OOM at frame ~{oom_at} — saving partial results")
            results_summary["oom_at_frame"] = oom_at
            torch.cuda.empty_cache()
        finally:
            try: del inference_session
            except Exception: pass
            try: del video_frames
            except Exception: pass
            gc.collect()
            cleanup_gpu_memory()
        frame_reader.close()
        if oom_at is not None:
            print(f"    ⚠ Not saving results.json — video will be retried next run")
            del per_frame_data
            cleanup_gpu_memory()
            return results_summary

        if total_frames <= skip_masked_video_over:
            write_masked_video(BidsProcessed, per_frame_data,
                               os.path.join(video_out_dir, "masked_video.mp4"),
                               multi_color=multi_mode)
        else:
            print(f"    ⏭ Skipping masked video ({total_frames} > {skip_masked_video_over} frames)")
        write_detections_csv(csv_rows, os.path.join(video_out_dir, "detections.csv"))

        with open(os.path.join(video_out_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)

        del per_frame_data
        cleanup_gpu_memory()
        return results_summary


# ═════════════════════════════════════════════
# Hybrid Router
# ═════════════════════════════════════════════

class HybridBackendRouter:
    def __init__(self, model_id, hybrid_threshold, force_backend="hybrid"):
        self.model_id         = model_id
        self.hybrid_threshold = hybrid_threshold
        self.force_backend    = force_backend
        self.native       = None
        self.transformers = None
        # Lazy load — backends instantiated on first use, not at startup
    
    def _ensure_native(self):
        if self.native is None:
            self.native = NativeBackend(self.model_id)
    
    def _ensure_transformers(self):
        if self.transformers is None:
            self.transformers = TransformersBackend(self.model_id)

    def pick(self, total_frames):
        if self.force_backend == "native":
            self._ensure_native()
            return self.native, "native"
        if self.force_backend == "transformers":
            self._ensure_transformers()
            return self.transformers, "transformers"
        if total_frames >= self.hybrid_threshold:
            self._ensure_transformers()
            return self.transformers, "transformers"
        self._ensure_native()
        return self.native, "native"
    
    def rebuild(self):
        if self.native is not None:
            try: del self.native
            except Exception: pass
            self.native = None
        if self.transformers is not None:
            try: del self.transformers
            except Exception: pass
            self.transformers = None
        cleanup_gpu_memory()
        # Both backends reload on next pick() call


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
def main():
    args = parse_args()

    resize_default = args.resize
    resize_long    = args.resize_long  if args.resize_long  is not None else resize_default
    resize_xlong   = args.resize_xlong if args.resize_xlong is not None else resize_long

    df = pd.read_csv(args.csv)
    assert "BidsProcessed" in df.columns, (
        f"CSV must have a 'BidsProcessed' column. Found: {list(df.columns)}"
    )

    # ALL videos — no filtering whatsoever
    BidsProcesseds = df["BidsProcessed"].dropna().tolist()
    total_in_csv = len(BidsProcesseds)
    print(f"Found {total_in_csv} videos in {args.csv}")

    # Job array slicing — each job handles its own chunk (round-robin)
    if args.array_total > 1:
        BidsProcesseds = [v for i, v in enumerate(BidsProcesseds)
                       if i % args.array_total == args.array_index]
        print(f"  Job array [{args.array_index + 1}/{args.array_total}]: "
              f"this job will process {len(BidsProcesseds)} of {total_in_csv} videos "
              f"(every {args.array_total}th video starting at index {args.array_index})")

    detection_mode = "ALL verified humans" if args.multi else "top-1 human only"
    print(f"\nSettings:")
    print(f"  Videos to process      : {len(BidsProcesseds)}")
    print(f"  Model                  : {args.model_id}")
    print(f"  Backend mode           : {args.force_backend}")
    print(f"  Hybrid threshold       : {args.hybrid_threshold} frames")
    print(f"  Prompt                 : '{args.prompt}'")
    print(f"  Detection mode         : {detection_mode}")
    print(f"  Min confidence         : {args.min_confidence}")
    print(f"  Max aspect ratio       : {args.max_aspect_ratio}")
    print(f"  YOLO person verify     : {'DISABLED' if args.no_yolo else args.yolo_model}")
    print(f"  Adaptive resize        :")
    print(f"    short  (<{args.long_threshold} frames)  : {resize_default}")
    print(f"    long   ({args.long_threshold}-{args.xlong_threshold-1} frames) : {resize_long}")
    print(f"    xlong  (>={args.xlong_threshold} frames): {resize_xlong}")
    print(f"  empty_cache_every      : {args.empty_cache_every} frames")
    print(f"  rebuild_backend_every  : {args.rebuild_backend_every} videos")
    print(f"  Output dir             : {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    yolo_verifier = None
    if not args.no_yolo:
        yolo_verifier = YOLOPersonVerifier(
            model_name=args.yolo_model,
            person_conf=args.yolo_person_conf,
        )

    model_id = args.model_id
    try:
        router = HybridBackendRouter(model_id, args.hybrid_threshold, args.force_backend)
    except Exception as e:
        if model_id == MODEL_ID_BEST:
            print(f"\n⚠ Could not load {MODEL_ID_BEST}: {e}")
            print(f"  Falling back to {MODEL_ID_FALLBACK}...")
            model_id = MODEL_ID_FALLBACK
            router = HybridBackendRouter(model_id, args.hybrid_threshold, args.force_backend)
        else:
            raise

    all_results = []
    failed      = []
    videos_processed_since_rebuild = 0
    routing_stats = {"native": 0, "transformers": 0}

    for i, BidsProcessed in enumerate(BidsProcesseds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(BidsProcesseds)}] {BidsProcessed}")
        print(f"{'='*60}")

        if not os.path.exists(BidsProcessed):
            print(f"  ⚠ File not found, skipping.")
            failed.append({"BidsProcessed": BidsProcessed, "error": "File not found"})
            continue

        video_name    = Path(BidsProcessed).stem
        video_out_dir = os.path.join(args.output_dir, video_name)
        done_marker   = os.path.join(video_out_dir, "results.json")
        if os.path.exists(done_marker):
            print(f"  ⏭ Already processed, skipping.")
            continue

        try:
            n_frames = peek_total_frames(BidsProcessed)
        except Exception:
            n_frames = 0

        chosen_resize = pick_adaptive_resize(
            total_frames=n_frames,
            resize_default=resize_default,
            resize_long=resize_long,
            resize_xlong=resize_xlong,
            long_threshold=args.long_threshold,
            xlong_threshold=args.xlong_threshold,
        )

        backend, backend_name = router.pick(n_frames)
        routing_stats[backend_name] += 1

        print(f"  Video length : {n_frames} frames")
        print(f"  → Backend    : {backend_name.upper()}"
              + (" (CPU offload)" if backend_name == "transformers" else " (GPU)"))
        print(f"  → Resize     : {chosen_resize}")

        try:
            t0     = time.time()
            result = backend.process_video(
                BidsProcessed=BidsProcessed,
                prompt=args.prompt,
                output_dir=args.output_dir,
                yolo_verifier=yolo_verifier,
                min_confidence=args.min_confidence,
                max_aspect_ratio=args.max_aspect_ratio,
                multi_mode=args.multi,
                resize_shorter_side=chosen_resize,
                max_frames=args.max_frames,
                save_masks=args.save_masks,
                save_bboxes=args.save_bboxes,
                log_every=args.log_every,
                empty_cache_every=args.empty_cache_every,
                skip_masked_video_over=args.skip_masked_video_over
            )
            elapsed = time.time() - t0
            result["processing_time_seconds"] = round(elapsed, 2)
            result["model_used"]              = model_id
            all_results.append(result)
            rejected = result.get("frames_rejected_not_human", 0)
            print(f"  ✓ Done in {elapsed:.1f}s | "
                  f"{result['frames_with_detections']}/{result['frames_processed']} "
                  f"frames with detections | "
                  f"{rejected} rejected (not human)")

        except torch.cuda.OutOfMemoryError as e:
            print(f"  ✗ CUDA OOM: {e}")
            failed.append({"BidsProcessed": BidsProcessed, "error": "CUDA OOM",
                           "backend": backend_name, "frames": n_frames})
            router.rebuild()
            videos_processed_since_rebuild = 0
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append({"BidsProcessed": BidsProcessed, "error": str(e),
                           "backend": backend_name, "frames": n_frames})

        cleanup_gpu_memory()
        videos_processed_since_rebuild += 1

        if (args.rebuild_backend_every > 0 and
                videos_processed_since_rebuild >= args.rebuild_backend_every):
            print(f"\n  ↻ Rebuilding backends after {videos_processed_since_rebuild} videos...")
            router.rebuild()
            videos_processed_since_rebuild = 0
            print(f"  ↻ Backends rebuilt.\n")

    global_summary = {
        "prompt":           args.prompt,
        "detection_mode":   detection_mode,
        "verification":     "YOLOv8 person + aspect ratio",
        "yolo_model":       args.yolo_model if not args.no_yolo else "disabled",
        "min_confidence":   args.min_confidence,
        "max_aspect_ratio": args.max_aspect_ratio,
        "resize_default":   resize_default,
        "resize_long":      resize_long,
        "resize_xlong":     resize_xlong,
        "long_threshold":   args.long_threshold,
        "xlong_threshold":  args.xlong_threshold,
        "hybrid_threshold": args.hybrid_threshold,
        "force_backend":    args.force_backend,
        "empty_cache_every":      args.empty_cache_every,
        "rebuild_backend_every":  args.rebuild_backend_every,
        "model_id":         model_id,
        "multi_mode":       args.multi,
        "array_index":      args.array_index,
        "array_total":      args.array_total,
        "total_videos_in_csv": total_in_csv,
        "this_job_attempted":  len(BidsProcesseds),
        "successful":  len(all_results),
        "failed":      len(failed),
        "routing_stats": routing_stats,
        "failed_videos": failed,
        "per_video_summary": [
            {
                "BidsProcessed":                r["BidsProcessed"],
                "total_frames":              r.get("total_frames"),
                "backend_used":              r.get("backend_used"),
                "frames_processed":          r["frames_processed"],
                "frames_with_detections":    r["frames_with_detections"],
                "frames_rejected_not_human": r.get("frames_rejected_not_human", 0),
                "processing_time_seconds":   r.get("processing_time_seconds"),
                "sam3_resolution":           r.get("sam3_resolution"),
                "original_resolution":       r.get("original_resolution"),
            }
            for r in all_results
        ],
    }

    # Each array job writes its own summary so they don't collide
    summary_name = (f"global_summary_job{args.array_index}.json"
                    if args.array_total > 1 else "global_summary.json")
    with open(os.path.join(args.output_dir, summary_name), "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL DONE — Job {args.array_index + 1}/{args.array_total}")
    print(f"{'='*60}")
    print(f"  Model         : {model_id}")
    print(f"  Routing stats : {routing_stats}")
    print(f"  Processed     : {len(all_results)}/{len(BidsProcesseds)} successful")
    print(f"  Failed        : {len(failed)}")
    print(f"  Results       : {args.output_dir}")


if __name__ == "__main__":
    main()