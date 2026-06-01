"""Aggregate Qwen2.5-VL results across multiple seeds and compute metric spread.

Directory structure expected (created by submit_qwen_clip.sh):
    <results-dir>/
        seed_42/
            clip_predictions_0.csv
            clip_predictions_1.csv
            ...
        seed_123/
            clip_predictions_0.csv
            ...

Also handles a single (no-seed / deterministic) run if clip_predictions_*.csv
files sit directly in <results-dir>.

Usage:
    python aggregate_qwen_results.py --task loco --results-dir /path/to/clips_loco
    python aggregate_qwen_results.py --task rmm  --results-dir /path/to/clips_rmm
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

TASK_CLASSES: dict[str, list[str]] = {
    "loco": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "rmm": ["Jumping", "Hands_flapping", "Rocking", "Spinning"],
}

SPREAD_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "cohen_kappa",
    "mcc",
    "top2_accuracy",
    "mAP",
]


# ---------------------------------------------------------------------------
# Helpers  (identical logic to the Ovis aggregator)
# ---------------------------------------------------------------------------

def _merge_chunks(chunk_dir: str, class_names: list[str]) -> pd.DataFrame | None:
    pattern = os.path.join(chunk_dir, "clip_predictions_*.csv")
    chunk_files = sorted(glob.glob(pattern))
    if not chunk_files:
        return None
    frames = [pd.read_csv(f) for f in chunk_files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["clip_path"], keep="last")
    return merged


def _compute_metrics_for_df(df: pd.DataFrame, class_names: list[str]) -> dict:
    valid = df[df["predicted_label"].isin(class_names)]
    excluded = len(df) - len(valid)

    if valid.empty:
        return {}

    y_true = valid["true_label"].tolist()
    y_pred = valid["predicted_label"].tolist()

    metrics: dict = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    for avg in ("micro", "macro", "weighted"):
        metrics[f"precision_{avg}"] = precision_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"f1_{avg}"] = f1_score(
            y_true, y_pred, average=avg, zero_division=0
        )

    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=class_names,
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
        print(f"  [WARN] Could not compute mAP: {exc}")

    correct_top2 = 0
    total_top2 = 0
    for _, row in valid.iterrows():
        try:
            preds = eval(row["frame_predictions"])  # noqa: S307
        except Exception:
            continue
        if not preds:
            continue
        top2 = [c for c, _ in Counter(preds).most_common(2)]
        if row["true_label"] in top2:
            correct_top2 += 1
        total_top2 += 1
    metrics["top2_accuracy"] = correct_top2 / total_top2 if total_top2 > 0 else 0.0

    metrics["total_clips"] = len(df)
    metrics["valid_clips_evaluated"] = len(valid)
    metrics["failed_clips"] = excluded

    return metrics


def _compute_spread(seed_metrics: dict[str, dict]) -> dict:
    spread: dict = {}
    for metric in SPREAD_METRICS:
        values = []
        for seed_label, m in seed_metrics.items():
            if metric in m:
                values.append(m[metric])
        if not values:
            continue
        arr = np.array(values, dtype=float)
        entry: dict = {
            "values": {
                seed_label: float(seed_metrics[seed_label][metric])
                for seed_label in seed_metrics
                if metric in seed_metrics[seed_label]
            },
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "n_seeds": len(arr),
        }
        if len(arr) >= 2:
            se = stats.sem(arr)
            ci = stats.t.interval(0.95, df=len(arr) - 1, loc=np.mean(arr), scale=se)
            entry["ci95_low"] = float(ci[0])
            entry["ci95_high"] = float(ci[1])
        spread[metric] = entry
    return spread


def _print_spread_table(spread: dict, task: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"METRIC SPREAD ACROSS SEEDS — {task.upper()} (Qwen2.5-VL)")
    print(f"{'=' * 70}")
    header = f"{'Metric':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}"
    print(header)
    print("-" * 70)
    for metric, entry in spread.items():
        print(
            f"{metric:<22} "
            f"{entry['mean']:>8.4f} "
            f"{entry['std']:>8.4f} "
            f"{entry['min']:>8.4f} "
            f"{entry['max']:>8.4f} "
            f"{entry['range']:>8.4f}"
        )
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Qwen2.5-VL results across seeds and compute spread.",
    )
    parser.add_argument("--task", required=True, choices=["loco", "rmm"])
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Base output directory (contains seed_*/ subdirs or chunk CSVs directly).",
    )
    args = parser.parse_args()

    class_names = TASK_CLASSES[args.task]
    results_dir = args.results_dir

    # ---- Discover seed directories ----
    seed_dirs: dict[str, str] = {}

    seed_subdirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")))
    if seed_subdirs:
        for sd in seed_subdirs:
            label = Path(sd).name
            seed_dirs[label] = sd
        print(f"[INFO] Found {len(seed_dirs)} seed directories: {list(seed_dirs)}")
    else:
        seed_dirs["deterministic"] = results_dir
        print("[INFO] No seed_*/ subdirs found — treating results-dir as a single run.")

    agg_root = os.path.join(results_dir, "aggregate_metrics")
    os.makedirs(agg_root, exist_ok=True)

    seed_metrics: dict[str, dict] = {}

    for label, sdir in seed_dirs.items():
        print(f"\n--- Processing {label} ({sdir}) ---")
        merged = _merge_chunks(sdir, class_names)
        if merged is None:
            print(f"  [WARN] No chunk CSVs found, skipping.")
            continue

        merged_path = os.path.join(sdir, "clip_predictions_all.csv")
        merged.to_csv(merged_path, index=False)
        print(f"  Merged {len(merged)} clips → {merged_path}")

        m = _compute_metrics_for_df(merged, class_names)
        if not m:
            print(f"  [WARN] No valid predictions, skipping.")
            continue

        seed_metrics[label] = m

        seed_agg_dir = os.path.join(agg_root, label)
        os.makedirs(seed_agg_dir, exist_ok=True)

        serialisable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in m.items()
        }
        serialisable["seed"] = label
        serialisable["task"] = args.task
        serialisable["model"] = "Qwen2.5-VL-7B-Instruct"
        serialisable["timestamp"] = datetime.now().isoformat()

        with open(os.path.join(seed_agg_dir, "evaluation_metrics.json"), "w") as fh:
            json.dump(serialisable, fh, indent=2)

        cm_arr = np.array(m["confusion_matrix"])
        cm_df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)
        cm_df.to_csv(os.path.join(seed_agg_dir, "confusion_matrix.csv"))

        report_df = pd.DataFrame(m["classification_report"]).transpose()
        report_df.to_csv(os.path.join(seed_agg_dir, "classification_report.csv"))

        print(
            f"  accuracy={m['accuracy']:.4f}  "
            f"balanced_acc={m['balanced_accuracy']:.4f}  "
            f"macro_f1={m['f1_macro']:.4f}  "
            f"kappa={m['cohen_kappa']:.4f}"
        )

    if not seed_metrics:
        print("[ERROR] No seed results could be computed.")
        return

    # ---- Spread across seeds ----
    spread = _compute_spread(seed_metrics)
    _print_spread_table(spread, args.task)

    spread_path = os.path.join(agg_root, "spread_across_seeds.json")
    with open(spread_path, "w") as fh:
        json.dump(
            {
                "task": args.task,
                "model": "Qwen2.5-VL-7B-Instruct",
                "n_seeds": len(seed_metrics),
                "seeds": list(seed_metrics),
                "timestamp": datetime.now().isoformat(),
                "spread": spread,
            },
            fh,
            indent=2,
        )
    print(f"\n[SAVED] Spread metrics → {spread_path}")

    # Flat summary CSV
    rows = []
    for label, m in seed_metrics.items():
        row = {"seed": label}
        for metric in SPREAD_METRICS:
            row[metric] = m.get(metric, float("nan"))
        rows.append(row)

    for stat in ("mean", "std", "min", "max"):
        row = {"seed": stat}
        for metric in SPREAD_METRICS:
            row[metric] = spread.get(metric, {}).get(stat, float("nan"))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(agg_root, "seed_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVED] Per-seed summary CSV → {summary_csv}")

    print(f"\n[DONE] All results in: {agg_root}")


if __name__ == "__main__":
    main()