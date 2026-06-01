"""
SAM3.1 Video Inference Script — Human-Only Toddler Detection
=============================================================
Runs SAM3.1 (facebook/sam3.1) on BIDS-formatted videos listed in a CSV file.
Detects and tracks the single highest-confidence HUMAN child per frame.

FIX: Uses a two-pass verification strategy to exclude non-human detections:
  1. Primary prompt: "human toddler" (more specific than "toddler")
  2. Negative prompt: "cat, dog, animal, pet" to detect animals
  3. If the top-1 "toddler" mask overlaps significantly with an animal mask,
     it is rejected as a false positive.
  4. A minimum confidence threshold is enforced.

Only BIDS-formatted videos (filenames containing "sub-") are processed.
Non-BIDS videos are skipped. BIDS subjects starting with A-E are also skipped.

Requirements:
    pip install torch transformers accelerate pandas tqdm opencv-python
    pip install git+https://github.com/facebookresearch/sam3.git

Usage:
    python run_sam3_video.py --csv /path/to/videos.csv --output_dir ./sam3_results --backend native

Output per video  (<output_dir>/<video_name>/):
    masked_video.mp4   — overlay video with mask + bbox + object ID (top-1 only)
    masks/             — per-frame .npy mask arrays  (1, H, W)  uint8
    bboxes/            — per-frame .npy bbox arrays  (1, 4)     int32  [x1,y1,x2,y2]
    detections.csv     — one row per frame: frame_idx, obj_id, x1,y1,x2,y2, score
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

# PRIMARY: More specific prompt to bias toward humans
TEXT_PROMPT = "human toddler"

# NEGATIVE: Prompt to detect animals — used to reject false positives
NEGATIVE_PROMPT = "cat, dog, animal, pet"

# Minimum confidence score to accept a detection (raise if too many FPs)
MIN_CONFIDENCE = 0.25

# IoU threshold: if toddler mask overlaps this much with an animal mask,
# it is considered a false positive and rejected.
ANIMAL_OVERLAP_THRESHOLD = 0.3

DTYPE = torch.bfloat16

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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def is_bids_video(video_path: str) -> bool:
    """Check if a video filename follows BIDS format (contains 'sub-')."""
    fname = Path(video_path).name
    return bool(re.search(r"sub-", fname, re.IGNORECASE))


def should_skip_bids_subject(video_path: str) -> bool:
    """Skip BIDS subjects whose ID starts with A, B, C, D, or E."""
    if not is_bids_video(video_path):
        return False
    fname = Path(video_path).name
    m = re.search(r"sub-([^_/\\]+)", fname, re.IGNORECASE)
    if not m:
        return False
    subject_id = m.group(1)
    return subject_id[0].lower() in BIDS_SKIP_PREFIXES


def convert_xywh_rel_to_xyxy_abs(boxes_xywh, img_width, img_height):
    """Convert normalized [cx, cy, w, h] boxes to absolute [x1, y1, x2, y2]."""
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.int32)

    bboxes = []
    for box in boxes_xywh:
        if hasattr(box, "tolist"):
            box = box.tolist()
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_width)
        y1 = int((cy - h / 2) * img_height)
        x2 = int((cx + w / 2) * img_width)
        y2 = int((cy + h / 2) * img_height)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)
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


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks (both H x W)."""
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_mask_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    What fraction of mask_a's pixels overlap with mask_b?
    Returns intersection / area_of_mask_a.
    """
    area_a = (mask_a > 0).sum()
    if area_a == 0:
        return 0.0
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    return float(intersection) / float(area_a)


def is_animal_false_positive(toddler_mask: np.ndarray,
                              animal_masks: np.ndarray,
                              overlap_threshold: float = ANIMAL_OVERLAP_THRESHOLD) -> bool:
    """
    Check if a toddler detection significantly overlaps with any animal detection.
    If so, the toddler detection is likely a false positive (e.g., a cat).

    Args:
        toddler_mask: (H, W) binary mask of the candidate toddler detection
        animal_masks: (N, H, W) binary masks from the negative/animal prompt
        overlap_threshold: reject if this fraction of the toddler mask overlaps an animal

    Returns:
        True if the detection should be rejected as a false positive
    """
    if animal_masks is None or len(animal_masks) == 0:
        return False

    for animal_mask in animal_masks:
        overlap = compute_mask_overlap_ratio(toddler_mask, animal_mask)
        if overlap >= overlap_threshold:
            return True

    return False


def pick_top1_human_only(obj_ids, scores, masks_np, boxes_xywh,
                          animal_masks=None,
                          min_confidence=MIN_CONFIDENCE,
                          overlap_threshold=ANIMAL_OVERLAP_THRESHOLD):
    """
    Keep only the single highest-confidence detection that:
      1. Meets the minimum confidence threshold
      2. Does NOT significantly overlap with any animal detection

    Returns filtered (obj_ids, scores, masks_np, boxes_xywh) — all with length 1.
    If no valid detection remains, returns empty lists/None.
    """
    if not obj_ids or len(obj_ids) == 0:
        return [], [], None, []

    scores_arr = np.array(scores, dtype=np.float32)

    # Sort by confidence (descending) so we try the best candidate first
    sorted_indices = np.argsort(-scores_arr)

    for idx in sorted_indices:
        idx = int(idx)
        score = float(scores_arr[idx])

        # ── Check 1: Minimum confidence ──
        if score < min_confidence:
            continue  # all remaining will be lower, but check anyway

        # ── Check 2: Animal overlap rejection ──
        if masks_np is not None and idx < len(masks_np) and animal_masks is not None:
            candidate_mask = masks_np[idx]
            if is_animal_false_positive(candidate_mask, animal_masks, overlap_threshold):
                print(f"      ⚠ Rejected detection ID={obj_ids[idx]} "
                      f"(score={score:.3f}) — overlaps with animal mask")
                continue

        # ── This detection passed all checks ──
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

    # No detection passed the filters
    return [], [], None, []


def draw_overlay(frame, masks_np, bboxes, obj_ids, scores):
    out = frame.copy()
    H, W = frame.shape[:2]

    for i in range(len(obj_ids)):
        # ── mask overlay ──
        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i]
            if mask.shape == (H, W) and np.any(mask):
                overlay = out.copy()
                overlay[mask > 0] = MASK_COLOR
                out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)

        # ── bounding box ──
        if i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            if x1 >= 0:
                cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICK)

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

                cv2.rectangle(out, (lx1, ly1), (lx2, ly2), LABEL_BG, -1)
                cv2.putText(out, label, (lx1 + 2, ly2 - baseline - 2),
                            FONT, FONT_SCALE, LABEL_COLOR, FONT_THICK, cv2.LINE_AA)

    return out


def write_masked_video(video_path, per_frame_data, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Could not open video for masked output: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        print(f"    ⚠ ffmpeg exited with code {proc.returncode} for {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3.1 video segmentation (human-only)")
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sam3_results")
    parser.add_argument("--model_id",   type=str, default=MODEL_ID_BEST)
    parser.add_argument("--prompt",     type=str, default=TEXT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT,
                        help="Comma-separated concepts to reject (animals, etc.)")
    parser.add_argument("--min_confidence", type=float, default=MIN_CONFIDENCE,
                        help="Minimum score threshold for accepting a detection")
    parser.add_argument("--animal_overlap_threshold", type=float,
                        default=ANIMAL_OVERLAP_THRESHOLD,
                        help="Reject toddler if this fraction overlaps animal mask")
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

    # ── Bboxes from masks ──
    if masks_np is not None and n_objects > 0 and masks_np.sum() > 0:
        computed_bboxes = masks_to_bboxes(masks_np)
    else:
        computed_bboxes = np.full((n_objects, 4), -1, dtype=np.int32)

    # ── Fallback to model boxes if mask bboxes are invalid ──
    all_invalid = np.all(computed_bboxes == -1) if len(computed_bboxes) > 0 else True
    if all_invalid and boxes_from_model is not None and len(boxes_from_model) > 0:
        final_bboxes = convert_xywh_rel_to_xyxy_abs(boxes_from_model, img_width, img_height)
    else:
        final_bboxes = computed_bboxes

    # ── Save mask .npy ──
    if save_masks and masks_dir and masks_np is not None and n_objects > 0 and masks_np.sum() > 0:
        mask_path = os.path.join(masks_dir, f"frame_{frame_idx:06d}.npy")
        np.save(mask_path, masks_np)

    # ── Save bbox .npy ──
    if save_bboxes and bboxes_dir and n_objects > 0:
        bbox_path = os.path.join(bboxes_dir, f"frame_{frame_idx:06d}.npy")
        np.save(bbox_path, final_bboxes)

    # ── CSV rows ──
    for k, oid in enumerate(obj_ids):
        x1, y1, x2, y2 = final_bboxes[k] if k < len(final_bboxes) else (-1, -1, -1, -1)
        sc = scores[k] if k < len(scores) else None
        csv_rows.append({
            "frame_idx": frame_idx,
            "obj_id":    oid,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "score": round(float(sc), 4) if sc is not None else "",
        })

    # ── results_summary ──
    boxes_model_list = boxes_from_model.tolist() if hasattr(boxes_from_model, 'tolist') else (list(boxes_from_model) if boxes_from_model is not None else [])
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

    # ── per_frame_data for video writing ──
    if n_objects > 0:
        per_frame_data[frame_idx] = {
            "masks":   masks_np if (masks_np is not None and masks_np.sum() > 0) else np.zeros((0, img_height, img_width), dtype=np.uint8),
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

    def _run_single_prompt_pass(self, video_path, prompt, max_frames=None):
        """
        Run a single prompt through the video and collect per-frame masks/scores.
        Returns: dict[frame_idx] -> { "obj_ids", "scores", "masks_np", "boxes" }
        """
        cap = cv2.VideoCapture(video_path)
        img_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        response   = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path))
        session_id = response["session_id"]

        self.predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id,
                         frame_index=0, text=prompt))

        propagate_request = dict(
            type="propagate_in_video",
            session_id=session_id,
        )
        if max_frames is not None:
            propagate_request["max_frame_num_to_track"] = max_frames

        def to_list(x):
            if isinstance(x, list):   return x
            if hasattr(x, "tolist"):  return x.tolist()
            return list(x) if x else []

        frame_results = {}

        for response in self.predictor.handle_stream_request(request=propagate_request):
            frame_idx = response["frame_index"]
            outputs   = response["outputs"]

            obj_ids = to_list(outputs.get("out_obj_ids", []))
            scores  = to_list(outputs.get("out_probs", []))
            boxes   = outputs.get("out_boxes_xywh", None)
            masks   = outputs.get("out_binary_masks", None)

            if masks is not None and len(obj_ids) > 0:
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
                    upsampled = []
                    for m in masks_np:
                        m_resized = cv2.resize(m, (img_width, img_height),
                                               interpolation=cv2.INTER_NEAREST)
                        upsampled.append(m_resized)
                    masks_np = np.array(upsampled, dtype=np.uint8)
            else:
                masks_np = None

            frame_results[frame_idx] = {
                "obj_ids":  obj_ids,
                "scores":   scores,
                "masks_np": masks_np,
                "boxes":    boxes,
            }

        try:
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=session_id))
        except Exception:
            pass

        return frame_results, img_width, img_height

    def process_video(self, video_path, prompt, negative_prompt, output_dir,
                      min_confidence=MIN_CONFIDENCE,
                      animal_overlap_threshold=ANIMAL_OVERLAP_THRESHOLD,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10):
        video_name    = Path(video_path).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        # ══════════════════════════════════════════
        # PASS 1: Primary prompt ("human toddler")
        # ══════════════════════════════════════════
        print(f"    Pass 1: Detecting with prompt '{prompt}'")
        primary_results, img_width, img_height = self._run_single_prompt_pass(
            video_path, prompt, max_frames)

        # ══════════════════════════════════════════
        # PASS 2: Negative prompt ("cat, dog, animal, pet")
        # ══════════════════════════════════════════
        print(f"    Pass 2: Detecting animals with prompt '{negative_prompt}'")
        animal_results, _, _ = self._run_single_prompt_pass(
            video_path, negative_prompt, max_frames)

        # ══════════════════════════════════════════
        # MERGE: Filter primary detections using animal masks
        # ══════════════════════════════════════════
        print(f"    Merging: Rejecting toddler detections that overlap with animals")

        results_summary = {
            "video_path": video_path,
            "video_name": video_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "min_confidence": min_confidence,
            "animal_overlap_threshold": animal_overlap_threshold,
            "frames_processed": 0,
            "frames_with_detections": 0,
            "frames_rejected_as_animal": 0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []

        all_frame_indices = sorted(set(primary_results.keys()))

        for frame_idx in all_frame_indices:
            pr = primary_results[frame_idx]

            # Get animal masks for this frame (if any)
            animal_masks_for_frame = None
            if frame_idx in animal_results:
                ar = animal_results[frame_idx]
                animal_masks_for_frame = ar.get("masks_np")

            # ── FILTER: pick top-1 that is NOT an animal ──
            obj_ids_f, scores_f, masks_f, boxes_f = pick_top1_human_only(
                pr["obj_ids"], pr["scores"], pr["masks_np"], pr["boxes"],
                animal_masks=animal_masks_for_frame,
                min_confidence=min_confidence,
                overlap_threshold=animal_overlap_threshold,
            )

            # Track rejections
            if len(pr["obj_ids"]) > 0 and len(obj_ids_f) == 0:
                results_summary["frames_rejected_as_animal"] += 1

            save_frame_results(
                frame_idx=frame_idx,
                masks_np=masks_f,
                obj_ids=obj_ids_f,
                scores=scores_f,
                boxes_from_model=boxes_f,
                masks_dir=masks_dir,
                bboxes_dir=bboxes_dir,
                save_masks=save_masks,
                save_bboxes=save_bboxes,
                results_summary=results_summary,
                per_frame_data=per_frame_data,
                csv_rows=csv_rows,
                img_width=img_width,
                img_height=img_height,
            )

            if results_summary["frames_processed"] % log_every == 0:
                status = f"    Frame {frame_idx}: {len(obj_ids_f)} objects"
                if scores_f:
                    status += f" (score={scores_f[0]:.3f})"
                print(status)

        # ── Write masked video ──
        masked_video_path = os.path.join(video_out_dir, "masked_video.mp4")
        write_masked_video(video_path, per_frame_data, masked_video_path)

        # ── Save detections CSV ──
        csv_path = os.path.join(video_out_dir, "detections.csv")
        write_detections_csv(csv_rows, csv_path)

        # ── Save JSON summary ──
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

    def _run_single_prompt_pass(self, video_path, prompt, max_frames=None):
        """
        Run a single prompt through the video and collect per-frame masks/scores.
        Returns: dict[frame_idx] -> { "obj_ids", "scores", "masks_np", "boxes" }
        """
        from transformers.video_utils import load_video

        cap = cv2.VideoCapture(video_path)
        img_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

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

        frame_results = {}

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

            frame_results[frame_idx] = {
                "obj_ids":  obj_ids,
                "scores":   scores,
                "masks_np": masks_np,
                "boxes":    boxes,
            }

        return frame_results, img_width, img_height

    def process_video(self, video_path, prompt, negative_prompt, output_dir,
                      min_confidence=MIN_CONFIDENCE,
                      animal_overlap_threshold=ANIMAL_OVERLAP_THRESHOLD,
                      max_frames=None, save_masks=True, save_bboxes=True, log_every=10):

        video_name    = Path(video_path).stem
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        masks_dir  = os.path.join(video_out_dir, "masks")  if save_masks  else None
        bboxes_dir = os.path.join(video_out_dir, "bboxes") if save_bboxes else None
        if masks_dir:  os.makedirs(masks_dir,  exist_ok=True)
        if bboxes_dir: os.makedirs(bboxes_dir, exist_ok=True)

        # ── PASS 1: Primary ──
        print(f"    Pass 1: Detecting with prompt '{prompt}'")
        primary_results, img_width, img_height = self._run_single_prompt_pass(
            video_path, prompt, max_frames)

        # ── PASS 2: Negative (animals) ──
        print(f"    Pass 2: Detecting animals with prompt '{negative_prompt}'")
        animal_results, _, _ = self._run_single_prompt_pass(
            video_path, negative_prompt, max_frames)

        # ── MERGE ──
        print(f"    Merging: Rejecting toddler detections that overlap with animals")

        results_summary = {
            "video_path": video_path,
            "video_name": video_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "min_confidence": min_confidence,
            "animal_overlap_threshold": animal_overlap_threshold,
            "frames_processed": 0,
            "frames_with_detections": 0,
            "frames_rejected_as_animal": 0,
            "per_frame": {},
        }

        per_frame_data = {}
        csv_rows       = []

        all_frame_indices = sorted(set(primary_results.keys()))

        for frame_idx in all_frame_indices:
            pr = primary_results[frame_idx]

            animal_masks_for_frame = None
            if frame_idx in animal_results:
                ar = animal_results[frame_idx]
                animal_masks_for_frame = ar.get("masks_np")

            obj_ids_f, scores_f, masks_f, boxes_f = pick_top1_human_only(
                pr["obj_ids"], pr["scores"], pr["masks_np"], pr["boxes"],
                animal_masks=animal_masks_for_frame,
                min_confidence=min_confidence,
                overlap_threshold=animal_overlap_threshold,
            )

            if len(pr["obj_ids"]) > 0 and len(obj_ids_f) == 0:
                results_summary["frames_rejected_as_animal"] += 1

            save_frame_results(
                frame_idx=frame_idx,
                masks_np=masks_f,
                obj_ids=obj_ids_f,
                scores=scores_f,
                boxes_from_model=boxes_f,
                masks_dir=masks_dir,
                bboxes_dir=bboxes_dir,
                save_masks=save_masks,
                save_bboxes=save_bboxes,
                results_summary=results_summary,
                per_frame_data=per_frame_data,
                csv_rows=csv_rows,
                img_width=img_width,
                img_height=img_height,
            )

            if results_summary["frames_processed"] % log_every == 0:
                status = f"    Frame {frame_idx}: {len(obj_ids_f)} objects"
                if scores_f:
                    status += f" (score={scores_f[0]:.3f})"
                print(status)

        masked_video_path = os.path.join(video_out_dir, "masked_video.mp4")
        write_masked_video(video_path, per_frame_data, masked_video_path)

        csv_path = os.path.join(video_out_dir, "detections.csv")
        write_detections_csv(csv_rows, csv_path)

        summary_path = os.path.join(video_out_dir, "results.json")
        with open(summary_path, "w") as f:
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
    video_paths   = []
    skipped_bids  = []
    skipped_nonbids = []

    for vp in video_paths_all:
        if not is_bids_video(vp):
            skipped_nonbids.append(vp)
            continue
        if should_skip_bids_subject(vp):
            skipped_bids.append(vp)
            continue
        video_paths.append(vp)

    print(f"\nFiltering results:")
    print(f"  BIDS videos to process : {len(video_paths)}")
    print(f"  Skipped (non-BIDS)     : {len(skipped_nonbids)}")
    print(f"  Skipped (BIDS sub A-E) : {len(skipped_bids)}")
    print(f"  Model                  : {args.model_id}")
    print(f"  Backend                : {args.backend}")
    print(f"  Prompt                 : '{args.prompt}'")
    print(f"  Negative prompt        : '{args.negative_prompt}'")
    print(f"  Min confidence         : {args.min_confidence}")
    print(f"  Animal overlap thresh  : {args.animal_overlap_threshold}")
    print(f"  Top-1 HUMAN only       : YES")
    print(f"\nNOTE: Original videos are NEVER modified.")
    print(f"      All output is written exclusively to: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

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
                negative_prompt=args.negative_prompt,
                output_dir=args.output_dir,
                min_confidence=args.min_confidence,
                animal_overlap_threshold=args.animal_overlap_threshold,
                max_frames=args.max_frames,
                save_masks=args.save_masks,
                save_bboxes=args.save_bboxes,
                log_every=args.log_every,
            )
            elapsed = time.time() - t0
            result["processing_time_seconds"] = round(elapsed, 2)
            result["model_used"]              = model_id
            all_results.append(result)
            rejected = result.get("frames_rejected_as_animal", 0)
            print(f"  ✓ Done in {elapsed:.1f}s | "
                  f"{result['frames_with_detections']}/{result['frames_processed']} "
                  f"frames with detections | "
                  f"{rejected} frames rejected (animal overlap)")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append({"video_path": video_path, "error": str(e)})

    global_summary = {
        "prompt":      args.prompt,
        "negative_prompt": args.negative_prompt,
        "min_confidence":  args.min_confidence,
        "animal_overlap_threshold": args.animal_overlap_threshold,
        "model_id":    model_id,
        "backend":     args.backend,
        "top1_human_only": True,
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
                "frames_rejected_as_animal": r.get("frames_rejected_as_animal", 0),
                "processing_time_seconds": r.get("processing_time_seconds"),
            }
            for r in all_results
        ],
    }

    global_path = os.path.join(args.output_dir, "global_summary.json")
    with open(global_path, "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
    print(f"  Model      : {model_id}")
    print(f"  Prompt     : '{args.prompt}'")
    print(f"  Neg. prompt: '{args.negative_prompt}'")
    print(f"  Processed  : {len(all_results)}/{len(video_paths)} successful")
    print(f"  Skipped    : {len(skipped_nonbids)} non-BIDS, {len(skipped_bids)} BIDS sub A-E")
    print(f"  Failed     : {len(failed)}")
    print(f"  Results    : {args.output_dir}")


if __name__ == "__main__":
    main()