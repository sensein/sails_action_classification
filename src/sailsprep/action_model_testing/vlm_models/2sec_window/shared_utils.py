"""Shared utilities for 2-second window locomotion/RMM classification.

Handles:
  - Frame-level annotation loading and 2-sec clip-level label conversion.
  - Video frame sampling from time windows (uniform or random).
  - Comprehensive multi-class and binary evaluation metrics.
  - CSV iteration, prediction saving, and metadata helpers.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
LOCOMOTION_CLASSES: list[str] = [
    "Crawling", "Cruising", "Walking", "Running", "Vehicle",
]
ALL_6_CLASSES: list[str] = LOCOMOTION_CLASSES + ["No_Locomotion"]

RMM_CLASSES: list[str] = [
    "Jumping", "Hands_flapping", "Rocking", "Spinning",
]
ALL_RMM_5_CLASSES: list[str] = RMM_CLASSES + ["No_RMM"]

CLIP_DURATION_SEC: float = 2.0
LABEL_FPS: float = 15.0
FRAMES_PER_CLIP: int = int(CLIP_DURATION_SEC * LABEL_FPS)

TASK_CONFIG: dict[str, dict] = {
    "loco": {
        "label_col_keywords": ["locomotion"],
        "exclude_keywords": ["rep"],
        "active_classes": LOCOMOTION_CLASSES,
        "all_classes": ALL_6_CLASSES,
        "no_action_label": "No_Locomotion",
        "binary_positive": "Locomotion",
    },
    "rmm": {
        "label_col_keywords": ["repetitive", "motor", "rmm"],
        "exclude_keywords": [],
        "active_classes": RMM_CLASSES,
        "all_classes": ALL_RMM_5_CLASSES,
        "no_action_label": "No_RMM",
        "binary_positive": "RMM",
    },
}


# ──────────────────────────────────────────────────────────────
# Ground-truth: frame-level → 2-sec clip-level
# ──────────────────────────────────────────────────────────────
def _find_label_column(df: pd.DataFrame, task: str) -> str:
    cfg = TASK_CONFIG[task]
    keywords = cfg["label_col_keywords"]
    exclude = cfg["exclude_keywords"]

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            if not any(ex in col_lower for ex in exclude):
                return col

    fallback_idx = 1 if task == "loco" else 2
    if len(df.columns) > fallback_idx:
        return df.columns[fallback_idx]

    raise ValueError(
        f"Cannot find {task} column in label CSV. Columns: {list(df.columns)}"
    )


def load_frame_labels(label_csv_path: str, task: str = "loco") -> list[str]:
    """Load per-frame labels from an annotation CSV."""
    cfg = TASK_CONFIG[task]
    active = cfg["active_classes"]
    no_label = cfg["no_action_label"]

    df = pd.read_csv(
        label_csv_path, encoding="utf-8-sig", keep_default_na=False
    )
    df.columns = df.columns.str.strip()
    col = _find_label_column(df, task)

    labels: list[str] = []
    for val in df[col]:
        val_str = str(val).strip()
        labels.append(val_str if val_str in active else no_label)

    return labels


def frame_labels_to_clip_labels(
    frame_labels: list[str],
    task: str = "loco",
    clip_duration_sec: float = CLIP_DURATION_SEC,
    fps: float = LABEL_FPS,
) -> list[dict]:
    """Convert frame-level labels to non-overlapping 2-sec clip labels."""
    cfg = TASK_CONFIG[task]
    active = cfg["active_classes"]
    no_label = cfg["no_action_label"]
    binary_pos = cfg["binary_positive"]

    frames_per_clip = int(clip_duration_sec * fps)
    n_frames = len(frame_labels)
    clips: list[dict] = []

    for start in range(0, n_frames, frames_per_clip):
        end = min(start + frames_per_clip, n_frames)
        window = frame_labels[start:end]

        if len(window) < frames_per_clip * 0.5:
            continue

        counter = Counter(window)
        majority = counter.most_common(1)[0][0]
        is_active = majority in active

        clips.append(
            {
                "start_frame": start,
                "end_frame": end - 1,
                "start_sec": start / fps,
                "end_sec": end / fps,
                "label_full": majority,
                "label_binary": binary_pos if is_active else no_label,
                "label_fine": majority if is_active else None,
                "frame_label_counts": dict(counter),
            }
        )

    return clips


# ──────────────────────────────────────────────────────────────
# Frame sampling from a video time window
# ──────────────────────────────────────────────────────────────
def sample_frames_from_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 6,
    *,
    random_frames: bool = False,
    seed: int = 42,
    clip_index: int = 0,
) -> list[Image.Image]:
    """Sample *num_frames* PIL images from a video time window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 15.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    # Clamp to actual video length
    end_frame = min(end_frame, total_frames)
    start_frame = min(start_frame, max(end_frame - 1, 0))

    total_window = max(end_frame - start_frame, 1)
    n = min(num_frames, total_window)

    if random_frames:
        rng = np.random.default_rng(seed + clip_index)
        indices = sorted(
            rng.choice(total_window, size=n, replace=False).tolist()
        )
        indices = [start_frame + i for i in indices]
    else:
        if total_window <= num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            indices = np.linspace(
                start_frame, end_frame - 1, num_frames, dtype=int
            ).tolist()

    images: list[Image.Image] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
            else:
                print(
                    f"[WARN] Failed to read frame {idx} from {video_path}"
                )
    finally:
        cap.release()

    if not images:
        print(
            f"[ERROR] No frames extracted from {video_path} "
            f"[{start_sec:.2f}s–{end_sec:.2f}s]"
        )

    return images


# ──────────────────────────────────────────────────────────────
# Resume helper
# ──────────────────────────────────────────────────────────────
def get_processed_videos(output_dir: str, prefix: str) -> set[str]:
    """Return video paths already present in partial prediction CSVs."""
    pattern = os.path.join(output_dir, f"{prefix}predictions_*.csv")
    processed: set[str] = set()
    for path in glob.glob(pattern):
        try:
            df = pd.read_csv(path)
            if "video_path" in df.columns:
                processed.update(
                    df["video_path"].dropna().astype(str).tolist()
                )
        except Exception as exc:
            print(f"[WARN] Could not read existing predictions {path}: {exc}")
    if processed:
        print(
            f"[RESUME] {len(processed)} already-processed videos "
            f"in {output_dir}"
        )
    return processed


# ──────────────────────────────────────────────────────────────
# Multi-class evaluation metrics
# ──────────────────────────────────────────────────────────────
def compute_multiclass_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_dir: str,
    prefix: str = "",
) -> dict:
    """Compute and persist comprehensive multi-class metrics."""
    metrics: dict = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true, y_pred)
    )

    for avg in ("micro", "macro", "weighted"):
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )

    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    report = classification_report(
        y_true, y_pred, labels=class_names, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    metrics["classification_report"] = report

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    metrics["confusion_matrix"] = cm.tolist()

    present = sorted(set(y_true) | set(y_pred))
    try:
        y_true_bin = label_binarize(y_true, classes=class_names)
        y_pred_bin = label_binarize(y_pred, classes=class_names)
        if y_true_bin.shape[1] > 1:
            ap: dict[str, float] = {}
            for i, cls in enumerate(class_names):
                if cls in present and y_true_bin[:, i].sum() > 0:
                    ap[cls] = float(
                        average_precision_score(
                            y_true_bin[:, i], y_pred_bin[:, i]
                        )
                    )
            if ap:
                metrics["mAP"] = float(np.mean(list(ap.values())))
            metrics["AP_per_class"] = ap
    except Exception as exc:
        print(f"[WARN] mAP error: {exc}")

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{prefix}evaluation_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, f"{prefix}confusion_matrix.csv"))

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        os.path.join(output_dir, f"{prefix}classification_report.csv")
    )

    print(f"\n{'=' * 60}")
    print(f"{prefix.upper()}EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen's kappa:     {metrics['cohen_kappa']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print(f"  Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:       {metrics['f1_weighted']:.4f}")
    if "mAP" in metrics:
        print(f"  mAP:               {metrics['mAP']:.4f}")
    print(f"\nConfusion matrix:\n{cm_df}")
    print(f"\nPer-class:\n{report_df.to_string()}")

    return metrics


def compute_top2_from_votes(
    frame_preds_list: list[list[str]],
    y_true: list[str],
) -> float:
    """Top-2 accuracy from per-clip frame vote lists."""
    correct = 0
    total = 0
    for preds, true_label in zip(frame_preds_list, y_true):
        if not preds:
            continue
        top2 = [c for c, _ in Counter(preds).most_common(2)]
        if true_label in top2:
            correct += 1
        total += 1
    return correct / total if total else 0.0


# ──────────────────────────────────────────────────────────────
# Binary evaluation metrics
# ──────────────────────────────────────────────────────────────
def compute_binary_metrics(
    y_true_bin: list[int] | np.ndarray,
    y_pred_bin: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    output_dir: str,
    prefix: str = "binary_",
) -> dict:
    """Compute and persist binary classification metrics."""
    yt = np.asarray(y_true_bin)
    yp = np.asarray(y_pred_bin)
    ys = np.asarray(y_scores, dtype=float)

    metrics: dict = {}
    metrics["accuracy"] = float(accuracy_score(yt, yp))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
    metrics["precision"] = float(precision_score(yt, yp, zero_division=0))
    metrics["recall"] = float(recall_score(yt, yp, zero_division=0))
    metrics["f1"] = float(f1_score(yt, yp, zero_division=0))
    metrics["specificity"] = float(
        recall_score(yt, yp, pos_label=0, zero_division=0)
    )
    metrics["cohen_kappa"] = float(cohen_kappa_score(yt, yp))
    metrics["mcc"] = float(matthews_corrcoef(yt, yp))

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    metrics["TP"] = int(tp)
    metrics["TN"] = int(tn)
    metrics["FP"] = int(fp)
    metrics["FN"] = int(fn)
    metrics["NPV"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    metrics["FPR"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["FNR"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(yt, ys))
        fpr, tpr, roc_th = roc_curve(yt, ys)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_th.tolist(),
        }
    except Exception as exc:
        print(f"[WARN] ROC-AUC error: {exc}")
        metrics["roc_auc"] = None

    try:
        prec_c, rec_c, pr_th = precision_recall_curve(yt, ys)
        metrics["pr_auc"] = float(auc(rec_c, prec_c))
        metrics["pr_curve"] = {
            "precision": prec_c.tolist(),
            "recall": rec_c.tolist(),
            "thresholds": pr_th.tolist(),
        }
    except Exception as exc:
        print(f"[WARN] PR-AUC error: {exc}")
        metrics["pr_auc"] = None

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    recall_at: dict[str, float] = {}
    prec_at: dict[str, float] = {}
    for t in thresholds:
        preds_t = (ys >= t).astype(int)
        recall_at[str(t)] = float(recall_score(yt, preds_t, zero_division=0))
        prec_at[str(t)] = float(
            precision_score(yt, preds_t, zero_division=0)
        )
    metrics["recall_at_thresholds"] = recall_at
    metrics["precision_at_thresholds"] = prec_at

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}evaluation_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    labels = ["No_Action", "Action"]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, f"{prefix}confusion_matrix.csv"))

    print(f"\n{'=' * 60}")
    print(f"{prefix.upper()}BINARY EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")
    print(f"  Cohen's kappa:     {metrics['cohen_kappa']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    if metrics.get("pr_auc") is not None:
        print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"\nConfusion matrix:\n{cm_df}")

    return metrics


# ──────────────────────────────────────────────────────────────
# Video iteration helper
# ──────────────────────────────────────────────────────────────
def iterate_videos(
    csv_path: str,
    video_col: str = "video_path",
    label_col: str = "label_path",
) -> list[tuple[str, str]]:
    """Return deduplicated (video_path, label_path) pairs from a CSV."""
    df = pd.read_csv(csv_path)
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        vp = str(row.get(video_col, "")).strip()
        lp = str(row.get(label_col, "")).strip()
        if not vp or not lp or vp == "nan" or lp == "nan":
            continue
        if vp in seen:
            continue
        if not os.path.exists(vp):
            print(f"[WARN] Video not found: {vp}")
            continue
        if not os.path.exists(lp):
            print(f"[WARN] Label not found: {lp}")
            continue
        seen.add(vp)
        pairs.append((vp, lp))

    print(f"[INFO] {len(pairs)} unique video–label pairs from {csv_path}")
    return pairs


def save_predictions_csv(
    results: list[dict],
    output_dir: str,
    prefix: str = "",
    append: bool = False,
) -> pd.DataFrame:
    """Save or append per-clip predictions to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    path = os.path.join(output_dir, f"{prefix}predictions_{array_id}.csv")
    df = pd.DataFrame(results)

    if append and os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
        print(f"[APPENDED] {path}  (+{len(df)} rows)")
    else:
        df.to_csv(path, index=False)
        print(f"[SAVED] {path}  ({len(df)} rows)")

    return df


def add_metadata_to_metrics(json_path: str, **kwargs: object) -> None:
    """Append metadata fields to an existing metrics JSON."""
    with open(json_path) as fh:
        data = json.load(fh)
    data.update(kwargs)
    data["timestamp"] = datetime.now().isoformat()
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)