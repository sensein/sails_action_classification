from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------
def extract_label_from_path(
    clip_path: str, class_names: list[str]
) -> str | None:
    """Extract the ground-truth label from a folder name in *clip_path*."""
    for part in Path(clip_path).parts:
        if part in class_names:
            return part
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_dir: str,
) -> dict:
    """Compute classification metrics and persist results to *output_dir*."""
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
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
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
        print(f"[WARN] Could not compute mAP: {exc}")

    os.makedirs(output_dir, exist_ok=True)

    serialisable: dict = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v

    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
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
    print(f"\nPer-class report:\n{report_df.to_string()}")

    return metrics


def compute_top2_accuracy(
    frame_preds_list: list[list[str]],
    y_true: list[str],
) -> float:
    """Compute top-2 accuracy from per-clip frame vote distributions."""
    correct = 0
    total = 0
    for preds, true_label in zip(frame_preds_list, y_true):
        if not preds:
            continue
        top2 = [cls for cls, _ in Counter(preds).most_common(2)]
        if true_label in top2:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0
