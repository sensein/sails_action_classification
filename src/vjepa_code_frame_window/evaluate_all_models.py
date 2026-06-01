"""
Comprehensive Action Segmentation Evaluation

Standard action segmentation metrics:
  1. Frame-level: Accuracy, per-class P/R/F1
  2. Segment F1@k: F1 at IoU thresholds {10, 25, 50} (standard: MS-TCN, ASFormer)
  3. Segment mAP@k: mAP at IoU thresholds {0.3, 0.5, 0.7} and average (standard: TAD)
  4. Edit Score: normalized Levenshtein distance on segment sequences

Usage:
    python evaluate_all_models.py --base_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
)


# ============================================================
# 1. SEGMENT UTILITIES
# ============================================================

def frames_to_segments(labels):
    """Convert frame-level label list to segments.
    Returns list of (label, start_frame, end_frame) tuples.
    """
    if len(labels) == 0:
        return []
    segments = []
    current_label = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((current_label, start, i - 1))
            current_label = labels[i]
            start = i
    segments.append((current_label, start, len(labels) - 1))
    return segments


def segment_iou(seg1, seg2):
    """Compute IoU between two segments (label, start, end)."""
    start1, end1 = seg1[1], seg1[2]
    start2, end2 = seg2[1], seg2[2]
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    if inter_end < inter_start:
        return 0.0
    inter = inter_end - inter_start + 1
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - inter
    return inter / union


# ============================================================
# 2. SEGMENT F1 @ IoU THRESHOLD (Action Segmentation standard)
# ============================================================

def segment_f1_at_iou(gt_segments, pred_segments, iou_threshold, bg_class="None"):
    """Segment-level F1 at a given IoU threshold.
    Standard metric from MS-TCN, ASFormer, etc.
    Ignores background segments.
    """
    gt_segs = [s for s in gt_segments if s[0] != bg_class]
    pred_segs = [s for s in pred_segments if s[0] != bg_class]

    if len(gt_segs) == 0 and len(pred_segs) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_segs) == 0:
        return 0.0, 0.0, 1.0
    if len(pred_segs) == 0:
        return 0.0, 1.0, 0.0

    gt_matched = set()
    pred_matched = set()

    for pi, ps in enumerate(pred_segs):
        best_iou = 0.0
        best_gi = -1
        for gi, gs in enumerate(gt_segs):
            if gi in gt_matched:
                continue
            if ps[0] != gs[0]:
                continue
            iou = segment_iou(ps, gs)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            pred_matched.add(pi)
            gt_matched.add(best_gi)

    tp = len(pred_matched)
    precision = tp / len(pred_segs) if len(pred_segs) > 0 else 0.0
    recall = tp / len(gt_segs) if len(gt_segs) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


# ============================================================
# 3. SEGMENT mAP @ IoU THRESHOLD (TAD standard)
# ============================================================

def compute_ap_per_class(gt_segs_class, pred_segs_class, iou_threshold):
    """Compute Average Precision for one class at one IoU threshold.
    pred_segs_class should be sorted by confidence (descending).
    """
    if len(gt_segs_class) == 0:
        return 0.0 if len(pred_segs_class) > 0 else float('nan')

    if len(pred_segs_class) == 0:
        return 0.0

    gt_matched = set()
    tp = np.zeros(len(pred_segs_class))
    fp = np.zeros(len(pred_segs_class))

    for pi, (ps, ps_conf) in enumerate(pred_segs_class):
        best_iou = 0.0
        best_gi = -1
        for gi, gs in enumerate(gt_segs_class):
            if gi in gt_matched:
                continue
            iou = segment_iou(ps, gs)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            tp[pi] = 1
            gt_matched.add(best_gi)
        else:
            fp[pi] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / len(gt_segs_class)

    # Append sentinel values
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [recall[-1] if len(recall) > 0 else 0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP using all unique recall values
    indices = np.where(np.diff(recall))[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return ap


def segment_map_at_iou(gt_segments, pred_segments, pred_confidences,
                        iou_threshold, bg_class="None"):
    """Compute mAP at a given IoU threshold across all classes.
    
    gt_segments: list of (label, start, end)
    pred_segments: list of (label, start, end)
    pred_confidences: list of float, confidence for each pred segment
    """
    gt_segs = [s for s in gt_segments if s[0] != bg_class]
    pred_with_conf = [
        (s, c) for s, c in zip(pred_segments, pred_confidences)
        if s[0] != bg_class
    ]

    all_classes = set(s[0] for s in gt_segs)
    if not all_classes:
        return 1.0 if len(pred_with_conf) == 0 else 0.0

    aps = []
    for cls in sorted(all_classes):
        gt_cls = [s for s in gt_segs if s[0] == cls]
        pred_cls = [(s, c) for s, c in pred_with_conf if s[0] == cls]
        # Sort by confidence descending
        pred_cls.sort(key=lambda x: x[1], reverse=True)

        ap = compute_ap_per_class(gt_cls, pred_cls, iou_threshold)
        if not np.isnan(ap):
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


# ============================================================
# 4. EDIT SCORE
# ============================================================

def edit_score(gt_segments, pred_segments, bg_class="None"):
    """Normalized edit (Levenshtein) distance on segment label sequences.
    1.0 = perfect match, 0.0 = completely different.
    """
    s1 = [s[0] for s in gt_segments if s[0] != bg_class]
    s2 = [s[0] for s in pred_segments if s[0] != bg_class]

    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return 1.0 - dp[n][m] / max(n, m)


# ============================================================
# 5. CONFIDENCE ESTIMATION FOR SEGMENTS
# ============================================================

def get_segment_confidences(df, pred_segments):
    """Estimate confidence per predicted segment.
    Uses the 'confidence' column if available, otherwise uniform.
    """
    has_conf = "confidence" in df.columns
    confidences = []

    for label, start, end in pred_segments:
        if has_conf:
            seg_confs = df.iloc[start:end + 1]["confidence"]
            # Handle mixed types (some CSVs have "N/A")
            numeric_confs = pd.to_numeric(seg_confs, errors="coerce").dropna()
            if len(numeric_confs) > 0:
                confidences.append(numeric_confs.mean())
            else:
                confidences.append(0.5)
        else:
            confidences.append(1.0)

    return confidences


# ============================================================
# 6. PER-VIDEO EVALUATION
# ============================================================

def evaluate_single_video(csv_path, bg_class="None"):
    """Evaluate a single video's predictions."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)

    # Force string and handle NaN
    true_labels = df["true_label"].fillna("None").astype(str).str.strip().tolist()
    pred_labels = df["predicted_label"].fillna("None").astype(str).str.strip().tolist()
    true_labels = ["None" if t in ("nan", "", "N/A") else t for t in true_labels]
    pred_labels = ["None" if p in ("nan", "", "N/A") else p for p in pred_labels]

    gt_segments = frames_to_segments(true_labels)
    pred_segments = frames_to_segments(pred_labels)
    pred_confidences = get_segment_confidences(df, pred_segments)

    results = {
        "num_frames": len(true_labels),
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }

    # Segment F1 @ IoU (action segmentation standard: 10, 25, 50)
    for iou_pct in [10, 25, 50]:
        iou_thresh = iou_pct / 100.0
        f1, p, r = segment_f1_at_iou(gt_segments, pred_segments, iou_thresh, bg_class)
        results[f"seg_f1@{iou_pct}"] = f1
        results[f"seg_p@{iou_pct}"] = p
        results[f"seg_r@{iou_pct}"] = r

    # Segment mAP @ IoU (TAD standard: 0.3, 0.5, 0.7)
    for iou_thresh in [0.3, 0.5, 0.7]:
        key = f"mAP@{iou_thresh}"
        results[key] = segment_map_at_iou(
            gt_segments, pred_segments, pred_confidences,
            iou_thresh, bg_class
        )

    # Average mAP
    results["Avg_mAP"] = np.mean([
        results["mAP@0.3"], results["mAP@0.5"], results["mAP@0.7"]
    ])

    # Edit score
    results["edit_score"] = edit_score(gt_segments, pred_segments, bg_class)

    return results


# ============================================================
# 7. AGGREGATE OVER ONE SEED RUN
# ============================================================

def evaluate_seed_run(pred_dir, bg_class="None"):
    """Evaluate all videos in one seed's per_video_predictions folder."""
    csv_files = sorted(glob.glob(os.path.join(pred_dir, "*_predictions.csv")))
    if not csv_files:
        print(f"  [WARN] No CSVs in {pred_dir}")
        return None

    all_true = []
    all_pred = []
    video_results = []

    for csv_path in csv_files:
        res = evaluate_single_video(csv_path, bg_class)
        all_true.extend(res["true_labels"])
        all_pred.extend(res["pred_labels"])
        video_results.append(res)

    # --- Frame-level metrics ---
    frame_acc = accuracy_score(all_true, all_pred)

    all_classes = sorted(set(all_true + all_pred))
    p_all, r_all, f1_all, sup_all = precision_recall_fscore_support(
        all_true, all_pred, labels=all_classes, zero_division=0
    )
    per_class_all = {
        c: {"precision": p_all[i], "recall": r_all[i], "f1": f1_all[i], "support": sup_all[i]}
        for i, c in enumerate(all_classes)
    }

    # Non-None classes only
    nn_classes = [c for c in all_classes if c != bg_class]
    nn_true = [t for t in all_true if t != bg_class]
    nn_pred = [all_pred[i] for i, t in enumerate(all_true) if t != bg_class]

    if nn_true:
        p_nn, r_nn, f1_nn, _ = precision_recall_fscore_support(
            nn_true, nn_pred, labels=nn_classes, zero_division=0
        )
        macro_p = np.mean(p_nn)
        macro_r = np.mean(r_nn)
        macro_f1 = np.mean(f1_nn)
        nn_acc = accuracy_score(nn_true, nn_pred)
    else:
        macro_p = macro_r = macro_f1 = nn_acc = 0.0

    # --- Segment-level metrics (averaged over videos) ---
    seg_metric_keys = [
        "seg_f1@10", "seg_f1@25", "seg_f1@50",
        "seg_p@10", "seg_p@25", "seg_p@50",
        "seg_r@10", "seg_r@25", "seg_r@50",
        "mAP@0.3", "mAP@0.5", "mAP@0.7", "Avg_mAP",
        "edit_score",
    ]
    seg_metrics = {}
    for key in seg_metric_keys:
        vals = [v[key] for v in video_results if key in v]
        seg_metrics[key] = np.mean(vals) if vals else 0.0

    return {
        "frame_acc": frame_acc,
        "nn_acc": nn_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class_all,
        **seg_metrics,
    }


# ============================================================
# 8. AGGREGATE ACROSS SEEDS
# ============================================================

def fmt(mean, std):
    """Format as 'XX.X ± Y.Y'"""
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def aggregate_seeds(seed_results):
    """Compute mean ± std across seed runs."""
    scalar_keys = [
        "frame_acc", "nn_acc", "macro_precision", "macro_recall", "macro_f1",
        "seg_f1@10", "seg_f1@25", "seg_f1@50",
        "mAP@0.3", "mAP@0.5", "mAP@0.7", "Avg_mAP",
        "edit_score",
    ]

    agg = {}
    for key in scalar_keys:
        vals = [s[key] for s in seed_results if s is not None]
        if vals:
            agg[key] = fmt(np.mean(vals), np.std(vals))
        else:
            agg[key] = "—"

    # Per-class F1 aggregation
    all_classes = set()
    for s in seed_results:
        if s and "per_class" in s:
            all_classes.update(s["per_class"].keys())

    per_class_agg = {}
    for c in sorted(all_classes):
        for metric in ["precision", "recall", "f1"]:
            vals = []
            for s in seed_results:
                if s and "per_class" in s and c in s["per_class"]:
                    vals.append(s["per_class"][c][metric])
            key = f"{c}_{metric}"
            if vals:
                per_class_agg[key] = fmt(np.mean(vals), np.std(vals))
            else:
                per_class_agg[key] = "—"

    agg["per_class"] = per_class_agg
    return agg


# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
    )
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.base_dir, "evaluation_summary.csv")

    models = sorted([
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    tasks = ["locomotion", "rmm"]
    seeds = ["seed_42", "seed_123", "seed_456"]

    all_rows = []
    full_reports = []

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"TASK: {task}")
        print(f"{'='*70}")

        for model in models:
            print(f"\n  Model: {model}")
            seed_results = []

            for seed in seeds:
                pred_dir = os.path.join(
                    args.base_dir, model, task, seed, "per_video_predictions"
                )
                if not os.path.isdir(pred_dir):
                    print(f"    {seed}: [SKIP] no predictions dir")
                    seed_results.append(None)
                    continue

                print(f"    {seed}: evaluating...", end=" ")
                res = evaluate_seed_run(pred_dir)
                if res:
                    print(
                        f"frame_acc={res['frame_acc']:.4f}  "
                        f"seg_f1@50={res['seg_f1@50']:.4f}  "
                        f"mAP@0.5={res['mAP@0.5']:.4f}  "
                        f"edit={res['edit_score']:.4f}"
                    )
                else:
                    print("[EMPTY]")
                seed_results.append(res)

            valid = [s for s in seed_results if s is not None]
            if not valid:
                print(f"    -> No valid seeds, skipping")
                continue

            agg = aggregate_seeds(valid)

            row = {
                "Task": task,
                "Model": model,
                "Seeds": len(valid),
                "Frame Acc": agg["frame_acc"],
                "Action Acc": agg["nn_acc"],
                "Macro P": agg["macro_precision"],
                "Macro R": agg["macro_recall"],
                "Macro F1": agg["macro_f1"],
                "F1@10": agg["seg_f1@10"],
                "F1@25": agg["seg_f1@25"],
                "F1@50": agg["seg_f1@50"],
                "mAP@0.3": agg["mAP@0.3"],
                "mAP@0.5": agg["mAP@0.5"],
                "mAP@0.7": agg["mAP@0.7"],
                "Avg mAP": agg["Avg_mAP"],
                "Edit": agg["edit_score"],
            }

            # Per-class F1
            for key, val in sorted(agg["per_class"].items()):
                if key.endswith("_f1"):
                    class_name = key.replace("_f1", "")
                    row[f"F1({class_name})"] = val

            all_rows.append(row)

            # Detailed report
            full_reports.append(f"\n--- {task} / {model} ({len(valid)} seeds) ---")
            full_reports.append(f"  Frame Acc:     {agg['frame_acc']}")
            full_reports.append(f"  Action Acc:    {agg['nn_acc']}")
            full_reports.append(f"  Macro P/R/F1:  {agg['macro_precision']} / {agg['macro_recall']} / {agg['macro_f1']}")
            full_reports.append(f"  --- Segment F1 (action segmentation) ---")
            full_reports.append(f"  F1@10:         {agg['seg_f1@10']}")
            full_reports.append(f"  F1@25:         {agg['seg_f1@25']}")
            full_reports.append(f"  F1@50:         {agg['seg_f1@50']}")
            full_reports.append(f"  --- Segment mAP (temporal action detection) ---")
            full_reports.append(f"  mAP@0.3:       {agg['mAP@0.3']}")
            full_reports.append(f"  mAP@0.5:       {agg['mAP@0.5']}")
            full_reports.append(f"  mAP@0.7:       {agg['mAP@0.7']}")
            full_reports.append(f"  Avg mAP:       {agg['Avg_mAP']}")
            full_reports.append(f"  --- Other ---")
            full_reports.append(f"  Edit Score:    {agg['edit_score']}")
            full_reports.append(f"  Per-class F1:")
            for key, val in sorted(agg["per_class"].items()):
                if key.endswith("_f1"):
                    full_reports.append(f"    {key}: {val}")

    # Save
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(args.output_csv, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("DETAILED REPORTS")
    print(f"{'='*70}")
    for line in full_reports:
        print(line)

    report_path = args.output_csv.replace(".csv", "_detailed.txt")
    with open(report_path, "w") as f:
        f.write("SUMMARY TABLE\n")
        f.write("=" * 70 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")
        f.write("DETAILED REPORTS\n")
        f.write("=" * 70 + "\n")
        for line in full_reports:
            f.write(line + "\n")

    print(f"\nSummary CSV: {args.output_csv}")
    print(f"Detailed report: {report_path}")


if __name__ == "__main__":
    main()