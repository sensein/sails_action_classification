"""
SAM3.1 Video Inference Script — Human-Only Toddler Detection
=============================================================
Runs SAM3.1 on BIDS-formatted videos listed in a CSV file.
Detects and tracks human children per frame.

MODES:
  --top1       (default) Keep only the single highest-confidence verified human.
  --multi      Keep ALL detections that pass verification (multiple children).

STRATEGY (no negative prompts — single SAM3 pass + YOLO post-filter):
  1. SAM3 detects with prompt "human toddler" (single pass).
  2. For each candidate detection, crop the bbox region from the frame.
  3. Run YOLOv8-nano (person class=0) on the cropped region.
  4. If YOLO confirms a "person" is present → KEEP the detection.
     If YOLO does NOT find a person → REJECT (it's a cat/toy/etc.).
  5. Also check aspect ratio: human toddlers are taller than wide.

This approach:
  ✓ Keeps toddlers playing with dogs (YOLO sees person + dog separately)
  ✓ Rejects cats/toys falsely detected as toddlers (YOLO sees no person)
  ✓ Only ONE SAM3 pass (no slowdown from negative prompts)

Requirements:
    pip install torch transformers accelerate pandas tqdm opencv-python
    pip install ultralytics   # for YOLOv8 person verification
    pip install git+https://github.com/facebookresearch/sam3.git

Usage:
    python run_sam3_video.py --csv /path/to/videos.csv --output_dir ./sam3_results --backend native
    python run_sam3_video.py --csv /path/to/videos.csv --output_dir ./sam3_results --backend native --multi

Output per video  (<output_dir>/<video_name>/):
    masked_video.mp4   — overlay video with mask + bbox + object ID
    masks/             — per-frame .npy mask arrays  (N, H, W)  uint8
    bboxes/            — per-frame .npy bbox arrays  (N, 4)     int32  [x1,y1,x2,y2]
    detections.csv     — one row per detection: frame_idx, obj_id, x1,y1,x2,y2, score
    results.json       — full per-frame summary
"""

import argparse
import csv
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
TEXT_PROMPT = "human child"
DTYPE = torch.bfloat16

# Minimum SAM3 confidence score to even consider a detection
MIN_CONFIDENCE = 0.15

# YOLO person verification settings
YOLO_MODEL_NAME  = "yolov8n.pt"   # nano = fastest, good enough for person/not-person
YOLO_PERSON_CLASS = 0             # COCO class 0 = "person"
YOLO_PERSON_CONF  = 0.3          # min YOLO confidence to confirm "person"
YOLO_BBOX_PAD     = 20           # pixels to pad around SAM3 bbox before cropping for YOLO

# Aspect ratio filter: reject detections that are extremely wide and short
# (cats lying down tend to be very wide; toddlers are more upright or square)
# Set to 0 to disable. A value of 3.0 means reject if width > 3x height.
MAX_ASPECT_RATIO = 3.5

# Subject-ID prefixes to SKIP in BIDS datasets (case-insensitive)
BIDS_SKIP_PREFIXES = ("a", "b", "c", "d", "e")

# Model IDs
MODEL_ID_BEST     = "facebook/sam3.1"
MODEL_ID_FALLBACK = "facebook/sam3"

# Visual style for overlay
MASK_COLOR  = (0, 255, 0)    # green  (B, G, R)
MASK_ALPHA  = 0.35
BBOX_COLOR  = (0, 0, 255)    # red    (B, G, R)
BBOX_THICK  = 2
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 0, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
FONT_THICK  = 2

# Multiple distinct colors for multi-child mode (B, G, R)
MULTI_COLORS = [
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 165, 0),   # orange
    (0, 128, 255),   # orange-red
    (128, 0, 255),   # purple
    (255, 255, 0),   # cyan
]


# ─────────────────────────────────────────────
# YOLO Person Verifier
# ─────────────────────────────────────────────

class YOLOPersonVerifier:
    """
    Lightweight YOLO-based check: given a cropped image region,
    does it contain a human person?
    """
    def __init__(self, model_name=YOLO_MODEL_NAME, person_conf=YOLO_PERSON_CONF):
        from ultralytics import YOLO
        print(f"Loading YOLO person verifier: {model_name}")
        self.model = YOLO(model_name)
        self.person_conf = person_conf
        print("YOLO person verifier ready.")

    def contains_person(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                        pad: int = YOLO_BBOX_PAD) -> bool:
        """
        Crop the bbox region (with padding) from the frame and run YOLO.
        Returns True if YOLO detects a "person" (class 0) in the crop.

        Args:
            frame_bgr: full frame (H, W, 3) in BGR
            bbox_xyxy:  [x1, y1, x2, y2] absolute coords
            pad: pixels to pad around the bbox before cropping
        """
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        # Skip invalid bboxes
        if x1 < 0 or x2 <= x1 or y2 <= y1:
            return False

        # Pad and clamp
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(W, int(x2) + pad)
        cy2 = min(H, int(y2) + pad)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False

        # Run YOLO on the crop (silent, no saving)
        results = self.model(crop, conf=self.person_conf, verbose=False)

        if len(results) == 0:
            return False

        # Check if any detection is class 0 (person)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy().astype(int)
                if YOLO_PERSON_CLASS in classes:
                    return True

        return False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def is_bids_video(video_path: str) -> bool:
    fname = Path(video_path).name
    return bool(re.search(r"sub-", fname, re.IGNORECASE))


def should_skip_bids_subject(video_path: str) -> bool:
    if not is_bids_video(video_path):
        return False
    fname = Path(video_path).name
    m = re.search(r"sub-([^_/\\]+)", fname, re.IGNORECASE)
    if not m:
        return False
    subject_id = m.group(1)
    return subject_id[0].lower() in BIDS_SKIP_PREFIXES


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


def masks_to_bboxes(masks_np: np.ndarray) -> np.ndarray:
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


def check_aspect_ratio(bbox_xyxy, max_ratio=MAX_ASPECT_RATIO) -> bool:
    """
    Returns True if the bbox aspect ratio is acceptable (not too wide/flat).
    """
    if max_ratio <= 0:
        return True  # disabled

    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if h <= 0:
        return False
    ratio = w / h
    return ratio <= max_ratio


def _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height):
    """Helper: get absolute bbox for detection at index idx."""
    bbox = None

    # Try from mask first (more accurate)
    if masks_np is not None and masks_np.ndim >= 2:
        mask_bboxes = masks_to_bboxes(masks_np)
        if idx < len(mask_bboxes):
            b = mask_bboxes[idx]
            if b[0] >= 0:
                bbox = b

    # Fallback to model boxes
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
    """
    Pick the highest-confidence SAM3 detection that:
      1. Meets minimum confidence
      2. Passes aspect ratio check (not extremely wide/flat)
      3. Is confirmed as containing a "person" by YOLO

    Falls through candidates in descending confidence order.
    Returns (obj_ids, scores, masks_np, boxes_xywh) with length 0 or 1.
    """
    if not obj_ids or len(obj_ids) == 0:
        return [], [], None, []

    scores_arr = np.array(scores, dtype=np.float32)
    sorted_indices = np.argsort(-scores_arr)

    for idx in sorted_indices:
        idx = int(idx)
        score = float(scores_arr[idx])

        # ── Check 1: Minimum confidence ──
        if score < min_confidence:
            continue

        # ── Get bbox for this detection ──
        bbox = _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height)
        if bbox is None:
            continue

        # ── Check 2: Aspect ratio ──
        if not check_aspect_ratio(bbox, max_aspect_ratio):
            print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — "
                  f"too wide (aspect ratio check failed)")
            continue

        # ── Check 3: YOLO person verification ──
        if yolo_verifier is not None and frame_bgr is not None:
            if not yolo_verifier.contains_person(frame_bgr, bbox):
                print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — "
                      f"YOLO found no person in bbox")
                continue

        # ── PASSED all checks ──
        best_obj_ids = [obj_ids[idx]]
        best_scores  = [score]

        if masks_np is not None and idx < len(masks_np):
            best_masks = masks_np[idx:idx+1]
        else:
            best_masks = None

        if boxes_xywh is not None and len(boxes_xywh) > 0:
            if hasattr(boxes_xywh, '__getitem__'):
                best_boxes = boxes_xywh[idx:idx+1]
            else:
                best_boxes = [boxes_xywh[idx]]
        else:
            best_boxes = []

        return best_obj_ids, best_scores, best_masks, best_boxes

    # Nothing passed
    return [], [], None, []


def pick_all_humans_verified(obj_ids, scores, masks_np, boxes_xywh,
                              frame_bgr, yolo_verifier,
                              img_width, img_height,
                              min_confidence=MIN_CONFIDENCE,
                              max_aspect_ratio=MAX_ASPECT_RATIO):
    """
    Return ALL SAM3 detections that pass verification (not just top-1).
    Each must:
      1. Meet minimum confidence
      2. Pass aspect ratio check
      3. Be confirmed as containing a "person" by YOLO

    Returns (obj_ids, scores, masks_np, boxes_xywh) — may contain 0, 1, or many.
    """
    if not obj_ids or len(obj_ids) == 0:
        return [], [], None, []

    scores_arr = np.array(scores, dtype=np.float32)

    passed_ids    = []
    passed_scores = []
    passed_mask_indices = []
    passed_box_indices  = []

    for idx in range(len(obj_ids)):
        score = float(scores_arr[idx])

        # ── Check 1: Minimum confidence ──
        if score < min_confidence:
            continue

        # ── Get bbox for this detection ──
        bbox = _get_bbox_for_detection(idx, masks_np, boxes_xywh, img_width, img_height)
        if bbox is None:
            continue

        # ── Check 2: Aspect ratio ──
        if not check_aspect_ratio(bbox, max_aspect_ratio):
            print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — "
                  f"too wide (aspect ratio check failed)")
            continue

        # ── Check 3: YOLO person verification ──
        if yolo_verifier is not None and frame_bgr is not None:
            if not yolo_verifier.contains_person(frame_bgr, bbox):
                print(f"      ⚠ Rejected ID={obj_ids[idx]} score={score:.3f} — "
                      f"YOLO found no person in bbox")
                continue

        # ── PASSED all checks ──
        passed_ids.append(obj_ids[idx])
        passed_scores.append(score)
        passed_mask_indices.append(idx)
        passed_box_indices.append(idx)

    if len(passed_ids) == 0:
        return [], [], None, []

    # Assemble masks
    if masks_np is not None and len(masks_np) > 0:
        valid_mask_indices = [i for i in passed_mask_indices if i < len(masks_np)]
        if valid_mask_indices:
            result_masks = masks_np[valid_mask_indices]
        else:
            result_masks = None
    else:
        result_masks = None

    # Assemble boxes
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
    """
    Draw mask overlay, bounding box, and label (ID + score) on the frame.

    Args:
        frame:       (H, W, 3) BGR image
        masks_np:    (N, H, W) binary masks
        bboxes:      (N, 4) [x1,y1,x2,y2]
        obj_ids:     list of object IDs
        scores:      list of confidence scores
        multi_color: if True, use distinct colors per detection
    """
    out = frame.copy()
    H, W = frame.shape[:2]

    for i in range(len(obj_ids)):
        # Pick color for this detection
        if multi_color and len(obj_ids) > 1:
            color_mask = MULTI_COLORS[i % len(MULTI_COLORS)]
            color_bbox = color_mask
            color_label_bg = color_mask
        else:
            color_mask = MASK_COLOR
            color_bbox = BBOX_COLOR
            color_label_bg = LABEL_BG

        # Draw mask
        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i]
            if mask.shape == (H, W) and np.any(mask):
                overlay = out.copy()
                overlay[mask > 0] = color_mask
                out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)

        # Draw bbox and label
        if i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            if x1 >= 0:
                cv2.rectangle(out, (x1, y1), (x2, y2), color_bbox, BBOX_THICK)

                oid   = obj_ids[i]
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


def write_masked_video(video_path, per_frame_data, output_path, multi_color=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Could not open video for masked output: {video_path}")
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
        description="Run SAM3.1 video segmentation (human-only, YOLO-verified)")
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sam3_results")
    parser.add_argument("--model_id",   type=str, default=MODEL_ID_BEST)
    parser.add_argument("--prompt",     type=str, default=TEXT_PROMPT)
    parser.add_argument("--min_confidence", type=float, default=MIN_CONFIDENCE,
                        help="Min SAM3 confidence to consider a detection")
    parser.add_argument("--yolo_model", type=str, default=YOLO_MODEL_NAME,
                        help="YOLO model for person verification (default: yolov8n.pt)")
    parser.add_argument("--yolo_person_conf", type=float, default=YOLO_PERSON_CONF,
                        help="Min YOLO confidence to confirm 'person'")
    parser.add_argument("--max_aspect_ratio", type=float, default=MAX_ASPECT_RATIO,
                        help="Reject bboxes wider than this ratio (0=disable)")
    parser.add_argument("--no_yolo", action="store_true", default=False,
                        help="Disable YOLO verification (use only aspect ratio)")
    parser.add_argument("--multi", action="store_true", default=False,
                        help="Detect ALL verified humans per frame (not just top-1)")
    parser.add_argument("--backend",    type=str, choices=["native", "transformers"],
                        default="native")
    parser.add_argument("--save_masks", action="store_true", default=True)
    parser.add_argument("--save_bboxes", action="store_true", default=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--log_every",  type=int, default=10)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Shared result-saver
# ─────────────────────────────────────────────

def save_frame_results(frame_idx, masks_np, obj_ids, scores,
                       boxes_from_model, masks_dir, bboxes_dir,
                       save_masks, save_bboxes, results_summary,
                       per_frame_data, csv_rows,
                       img_width, img_height):
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

    if save_masks and masks_dir and masks_np is not None and n_objects > 0 and masks_np.sum() > 0:
        np.save(os.path.join(masks_dir, f"frame_{frame_idx:06d}.npy"), masks_np)

    if save_bboxes and bboxes_dir and n_objects > 0:
        np.save(os.path.join(bboxes_dir, f"frame_{frame_idx:06d}.npy"), final_bboxes)

    for k, oid in enumerate(obj_ids):
        x1, y1, x2, y2 = final_bboxes[k] if k < len(final_bboxes) else (-1, -1, -1, -1)
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
        "bboxes_xyxy":  final_bboxes.tolist(),
    }
    results_summary["per_frame"][str(frame_idx)] = frame_info
    if n_objects > 0:
        results_summary["frames_with_detections"] += 1

    if n_objects > 0:
        per_frame_data[frame_idx] = {
            "masks": masks_np if (masks_np is not None and masks_np.sum() > 0)
                     else np.zeros((0, img_height, img_width), dtype=np.uint8),
            "bboxes":  final_bboxes,
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
# BACKEND 1: Native sam3 package
# ═════════════════════════════════════════════
class NativeBackend:
    def __init__(self, model_id):
        from sam3.model_builder import build_sam3_video_predictor
        print(f"Loading native SAM3 video predictor (model: {model_id})")
        self.predictor = build_sam3_video_predictor()
        print("Native predictor loaded.")

    def process_video(self, video_path, prompt, output_dir,
                      yolo_verifier=None,
                      min_confidence=MIN_CONFIDENCE,
                      max_aspect_ratio=MAX_ASPECT_RATIO,
                      multi_mode=False,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10):

        video_name    = Path(video_path).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        # ── Open video for frame access (for YOLO verification) ──
        cap = cv2.VideoCapture(video_path)
        img_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read ALL frames into memory for random access
        print(f"    Reading video frames into memory...")
        video_frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_bgr.append(frame)
        cap.release()
        print(f"    Loaded {len(video_frames_bgr)} frames ({img_width}x{img_height})")

        # ── Start SAM3 session ──
        response   = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path))
        session_id = response["session_id"]

        self.predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id,
                         frame_index=0, text=prompt))

        propagate_request = dict(type="propagate_in_video", session_id=session_id)
        if max_frames is not None:
            propagate_request["max_frame_num_to_track"] = max_frames

        detection_mode = "multi (all verified humans)" if multi_mode else "top-1 human only"
        results_summary = {
            "video_path": video_path, "video_name": video_name,
            "prompt": prompt,
            "detection_mode": detection_mode,
            "verification": "YOLOv8 person + aspect ratio",
            "min_confidence": min_confidence,
            "max_aspect_ratio": max_aspect_ratio,
            "frames_processed": 0, "frames_with_detections": 0,
            "frames_rejected_not_human": 0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []

        # Choose the picker function
        picker_fn = pick_all_humans_verified if multi_mode else pick_top1_human_verified

        def to_list(x):
            if isinstance(x, list):   return x
            if hasattr(x, "tolist"):  return x.tolist()
            return list(x) if x else []

        for response in self.predictor.handle_stream_request(request=propagate_request):
            frame_idx = response["frame_index"]
            outputs   = response["outputs"]

            obj_ids_l = to_list(outputs.get("out_obj_ids", []))
            scores_l  = to_list(outputs.get("out_probs", []))
            boxes     = outputs.get("out_boxes_xywh", None)
            masks     = outputs.get("out_binary_masks", None)

            # Process masks
            if masks is not None and len(obj_ids_l) > 0:
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.cpu().numpy().astype(np.uint8)
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

            # ── Get the raw frame for YOLO verification ──
            frame_bgr = None
            if frame_idx < len(video_frames_bgr):
                frame_bgr = video_frames_bgr[frame_idx]

            # ── PICK VERIFIED DETECTIONS ──
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
            )

            if results_summary["frames_processed"] % log_every == 0:
                status = f"    Frame {frame_idx}: {len(obj_ids_f)} objects"
                if scores_f:
                    status += f" (scores={[f'{s:.3f}' for s in scores_f]})"
                print(status)

        # Close SAM3 session
        try:
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=session_id))
        except Exception:
            pass

        # Free video frames from memory
        del video_frames_bgr

        # Write outputs
        masked_video_path = os.path.join(video_out_dir, "masked_video.mp4")
        write_masked_video(video_path, per_frame_data, masked_video_path,
                           multi_color=multi_mode)

        csv_path = os.path.join(video_out_dir, "detections.csv")
        write_detections_csv(csv_rows, csv_path)

        summary_path = os.path.join(video_out_dir, "results.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        return results_summary


# ═════════════════════════════════════════════
# BACKEND 2: HuggingFace Transformers
# ═════════════════════════════════════════════
class TransformersBackend:
    def __init__(self, model_id):
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        from accelerate import Accelerator

        self.device = Accelerator().device
        print(f"Loading transformers model: {model_id} on {self.device}")
        self.model = Sam3VideoModel.from_pretrained(model_id).to(self.device, dtype=DTYPE)
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        print("Transformers model loaded.")

    def process_video(self, video_path, prompt, output_dir,
                      yolo_verifier=None,
                      min_confidence=MIN_CONFIDENCE,
                      max_aspect_ratio=MAX_ASPECT_RATIO,
                      multi_mode=False,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10):
        from transformers.video_utils import load_video

        video_name    = Path(video_path).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        # Read frames for YOLO verification
        cap = cv2.VideoCapture(video_path)
        img_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    Reading video frames for YOLO verification...")
        video_frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_bgr.append(frame)
        cap.release()
        print(f"    Loaded {len(video_frames_bgr)} frames")

        video_frames, _ = load_video(video_path)
        total_frames    = len(video_frames)

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
            "video_path": video_path, "video_name": video_name,
            "prompt": prompt, "total_frames": total_frames,
            "detection_mode": detection_mode,
            "verification": "YOLOv8 person + aspect ratio",
            "min_confidence": min_confidence,
            "max_aspect_ratio": max_aspect_ratio,
            "frames_processed": 0, "frames_with_detections": 0,
            "frames_rejected_not_human": 0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []

        picker_fn = pick_all_humans_verified if multi_mode else pick_top1_human_verified

        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=track_limit,
        ):
            frame_idx = model_outputs.frame_idx
            processed = self.processor.postprocess_outputs(inference_session, model_outputs)

            obj_ids = processed["object_ids"].tolist() if len(processed["object_ids"]) > 0 else []
            scores  = processed["scores"].tolist()     if len(processed["scores"])     > 0 else []
            boxes   = processed["boxes"].cpu().numpy()  if len(processed["boxes"])      > 0 else np.zeros((0, 4))
            masks   = processed["masks"]
            masks_np = masks.cpu().numpy().astype(np.uint8) if len(obj_ids) > 0 else None

            frame_bgr = video_frames_bgr[frame_idx] if frame_idx < len(video_frames_bgr) else None
            had_candidates = len(obj_ids) > 0

            obj_ids, scores, masks_np, boxes = picker_fn(
                obj_ids, scores, masks_np, boxes,
                frame_bgr=frame_bgr,
                yolo_verifier=yolo_verifier,
                img_width=img_width, img_height=img_height,
                min_confidence=min_confidence,
                max_aspect_ratio=max_aspect_ratio,
            )

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
            )

            if results_summary["frames_processed"] % log_every == 0:
                status = f"    Frame {frame_idx}: {len(obj_ids)} objects"
                if scores:
                    status += f" (scores={[f'{s:.3f}' for s in scores]})"
                print(status)

        del video_frames_bgr

        write_masked_video(video_path, per_frame_data,
                           os.path.join(video_out_dir, "masked_video.mp4"),
                           multi_color=multi_mode)
        write_detections_csv(csv_rows, os.path.join(video_out_dir, "detections.csv"))

        with open(os.path.join(video_out_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)

        return results_summary


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    assert "video_path" in df.columns, (
        f"CSV must have a 'video_path' column. Found: {list(df.columns)}"
    )
    video_paths_all = df["video_path"].dropna().tolist()
    print(f"Found {len(video_paths_all)} videos in {args.csv}")

    # ── Filter: BIDS only, skip A-E subjects ──
    video_paths, skipped_bids, skipped_nonbids = [], [], []
    for vp in video_paths_all:
        if not is_bids_video(vp):
            skipped_nonbids.append(vp)
        elif should_skip_bids_subject(vp):
            skipped_bids.append(vp)
        else:
            video_paths.append(vp)

    detection_mode = "ALL verified humans" if args.multi else "top-1 human only"
    print(f"\nFiltering results:")
    print(f"  BIDS videos to process : {len(video_paths)}")
    print(f"  Skipped (non-BIDS)     : {len(skipped_nonbids)}")
    print(f"  Skipped (BIDS sub A-E) : {len(skipped_bids)}")
    print(f"  Model                  : {args.model_id}")
    print(f"  Backend                : {args.backend}")
    print(f"  Prompt                 : '{args.prompt}'")
    print(f"  Detection mode         : {detection_mode}")
    print(f"  Min confidence         : {args.min_confidence}")
    print(f"  Max aspect ratio       : {args.max_aspect_ratio}")
    print(f"  YOLO person verify     : {'DISABLED' if args.no_yolo else args.yolo_model}")
    print(f"\nNOTE: Original videos are NEVER modified.")
    print(f"      All output is written to: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load YOLO verifier ──
    yolo_verifier = None
    if not args.no_yolo:
        yolo_verifier = YOLOPersonVerifier(
            model_name=args.yolo_model,
            person_conf=args.yolo_person_conf,
        )

    # ── Load SAM3 backend ──
    model_id = args.model_id
    try:
        if args.backend == "native":
            backend = NativeBackend(model_id)
        else:
            backend = TransformersBackend(model_id)
    except Exception as e:
        if model_id == MODEL_ID_BEST:
            print(f"\n⚠ Could not load {MODEL_ID_BEST}: {e}")
            print(f"  Falling back to {MODEL_ID_FALLBACK}...")
            model_id = MODEL_ID_FALLBACK
            if args.backend == "native":
                backend = NativeBackend(model_id)
            else:
                backend = TransformersBackend(model_id)
        else:
            raise

    all_results = []
    failed      = []

    for i, video_path in enumerate(video_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(video_paths)}] {video_path}")
        print(f"{'='*60}")

        if not os.path.exists(video_path):
            print(f"  ⚠ File not found, skipping.")
            failed.append({"video_path": video_path, "error": "File not found"})
            continue

        video_name    = Path(video_path).stem
        video_out_dir = os.path.join(args.output_dir, video_name)
        done_marker   = os.path.join(video_out_dir, "results.json")
        if os.path.exists(done_marker):
            print(f"  ⏭ Already processed, skipping.")
            continue

        try:
            t0     = time.time()
            result = backend.process_video(
                video_path=video_path,
                prompt=args.prompt,
                output_dir=args.output_dir,
                yolo_verifier=yolo_verifier,
                min_confidence=args.min_confidence,
                max_aspect_ratio=args.max_aspect_ratio,
                multi_mode=args.multi,
                max_frames=args.max_frames,
                save_masks=args.save_masks,
                save_bboxes=args.save_bboxes,
                log_every=args.log_every,
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

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append({"video_path": video_path, "error": str(e)})

    global_summary = {
        "prompt":      args.prompt,
        "detection_mode": detection_mode,
        "verification": "YOLOv8 person + aspect ratio",
        "yolo_model":  args.yolo_model if not args.no_yolo else "disabled",
        "min_confidence":    args.min_confidence,
        "max_aspect_ratio":  args.max_aspect_ratio,
        "model_id":    model_id,
        "backend":     args.backend,
        "multi_mode":  args.multi,
        "total_videos_in_csv":     len(video_paths_all),
        "skipped_non_bids":        len(skipped_nonbids),
        "skipped_bids_subject_AE": len(skipped_bids),
        "attempted":   len(video_paths),
        "successful":  len(all_results),
        "failed":      len(failed),
        "failed_videos": failed,
        "per_video_summary": [
            {
                "video_path":              r["video_path"],
                "total_frames":            r.get("total_frames"),
                "frames_processed":        r["frames_processed"],
                "frames_with_detections":  r["frames_with_detections"],
                "frames_rejected_not_human": r.get("frames_rejected_not_human", 0),
                "processing_time_seconds": r.get("processing_time_seconds"),
            }
            for r in all_results
        ],
    }

    with open(os.path.join(args.output_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
    print(f"  Model      : {model_id}")
    print(f"  Prompt     : '{args.prompt}'")
    print(f"  Mode       : {detection_mode}")
    print(f"  YOLO verify: {'disabled' if args.no_yolo else args.yolo_model}")
    print(f"  Processed  : {len(all_results)}/{len(video_paths)} successful")
    print(f"  Skipped    : {len(skipped_nonbids)} non-BIDS, {len(skipped_bids)} BIDS sub A-E")
    print(f"  Failed     : {len(failed)}")
    print(f"  Results    : {args.output_dir}")


if __name__ == "__main__":
    main()