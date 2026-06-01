"""Aggregate InternVideo2 test metrics across seeds.

Reads per-seed test predictions and computes mean, std, and 95%
confidence intervals for accuracy and per-class F1.

Usage::

    python aggregate_iv2_seeds.py --task loco
    python aggregate_iv2_seeds.py --task rmm
    python aggregate_iv2_seeds.py --task loco --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Must match TASK_CONFIG in internvideo2_finetune.py.
TASK_CONFIG: dict[str, dict[str, object]] = {
    "loco": {
        "label_col": "Locomotion",
        "num_classes": 5,
        "output_dir": (
            "/orcd/scratch/bcs/001/sensein/sails/"
            "action_model_outputs/clips_h5/internvideo2/loco_iv2_run2"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 4,
        "output_dir": (
            "/orcd/scratch/bcs/001/sensein/sails/"
            "action_model_outputs/clips_h5/internvideo2/rmm_iv2run2"
        ),
    },
}

DEFAULT_SEEDS: list[int] = [42, 123, 456]


def load_seed_predictions(base_dir: str, seed: int) -> Optional[pd.DataFrame]:
    """Load test predictions CSV for a single seed.

    Args:
        base_dir: Base output directory for the task.
        seed: The seed whose predictions to load.

    Returns:
        DataFrame of predictions, or ``None`` if the file is missing.
    """
    csv_path = os.path.join(base_dir, f"seed_{seed}", "test_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"  seed {seed}: {csv_path} not found, skipping.")
        return None
    df = pd.read_csv(csv_path)
    if "pred_label" not in df.columns or "true_label" not in df.columns:
        print(f"  seed {seed}: CSV missing required columns, skipping.")
        return None
    return df


def compute_per_seed_metrics(
    df: pd.DataFrame,
    all_labels: list[str],
) -> dict[str, object]:
    """Compute accuracy and per-class F1 for one seed's predictions.

    Args:
        df: Predictions DataFrame with ``true_label`` and ``pred_label``.
        all_labels: Sorted list of all class label strings.

    Returns:
        Dict with ``"accuracy"`` (float) and ``"per_class_f1"``
        (dict mapping label to F1 score).
    """
    valid = df[df["pred_label"] != "ERROR"]
    if len(valid) == 0:
        return {
            "accuracy": 0.0,
            "per_class_f1": {lab: 0.0 for lab in all_labels},
        }

    acc = accuracy_score(valid["true_label"], valid["pred_label"])
    report: dict[str, object] = classification_report(
        valid["true_label"],
        valid["pred_label"],
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {
        lab: float(report[lab]["f1-score"])  # type: ignore[index]
        for lab in all_labels
    }
    return {"accuracy": float(acc), "per_class_f1": per_class_f1}


def aggregate(task: str, seeds: list[int]) -> None:
    """Aggregate test metrics across seeds and print summary.

    Computes mean, std, and 95% CI for overall accuracy and per-class
    F1 scores. Saves a JSON summary to the task's base output directory.

    Args:
        task: Task name (``"loco"`` or ``"rmm"``).
        seeds: List of seeds to aggregate over.
    """
    cfg = TASK_CONFIG[task]
    base_dir = str(cfg["output_dir"])

    print(f"\n{'=' * 60}")
    print(f"Aggregating: {task.upper()} | seeds={seeds}")
    print(f"{'=' * 60}")

    # Collect per-seed metrics.
    seed_metrics: list[dict[str, object]] = []
    all_labels: Optional[list[str]] = None

    for seed in seeds:
        df = load_seed_predictions(base_dir, seed)
        if df is None:
            continue

        if all_labels is None:
            raw_labels = sorted(
                set(df["true_label"].unique()) | set(df["pred_label"].unique())
            )
            all_labels = [lab for lab in raw_labels if lab != "ERROR"]

        metrics = compute_per_seed_metrics(df, all_labels)
        seed_metrics.append(metrics)
        print(f"  seed {seed}: accuracy = {metrics['accuracy']:.4f}")

    if not seed_metrics:
        print("  No seed results found.")
        return

    assert all_labels is not None  # guaranteed by the loop above

    # Aggregate accuracy.
    accuracies = np.array([float(m["accuracy"]) for m in seed_metrics])
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    print("\n--- Overall Accuracy ---")
    print(f"  Mean   : {mean_acc:.4f}")
    print(f"  Std    : {std_acc:.4f}")

    ci95: Optional[float] = None
    if len(accuracies) >= 2:
        ci95 = 1.96 * std_acc / float(np.sqrt(len(accuracies)))
        print(f"  95% CI : [{mean_acc - ci95:.4f}, {mean_acc + ci95:.4f}]")

    print(f"  Seeds  : {[round(a, 4) for a in accuracies.tolist()]}")

    # Aggregate per-class F1.
    print("\n--- Per-Class F1 ---")
    print(f"  {'Class':30s}  {'Mean':>8s}  {'Std':>8s}  {'Per-seed':>30s}")
    print(f"  {'-' * 80}")

    summary_per_class: dict[str, dict[str, object]] = {}
    for lab in all_labels:
        f1s = np.array(
            [float(m["per_class_f1"][lab])  # type: ignore[index]
             for m in seed_metrics]
        )
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        summary_per_class[lab] = {
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "per_seed": f1s.tolist(),
        }
        per_seed_str = ", ".join(f"{v:.4f}" for v in f1s)
        print(f"  {lab:30s}  {mean_f1:8.4f}  {std_f1:8.4f}  [{per_seed_str}]")

    # Save summary JSON.
    summary: dict[str, object] = {
        "task": task,
        "seeds": seeds,
        "accuracy": {
            "mean": mean_acc,
            "std": std_acc,
            "per_seed": accuracies.tolist(),
        },
        "per_class_f1": summary_per_class,
    }
    if ci95 is not None:
        accuracy_section = summary["accuracy"]
        assert isinstance(accuracy_section, dict)
        accuracy_section["ci95_lower"] = mean_acc - ci95
        accuracy_section["ci95_upper"] = mean_acc + ci95

    out_path = os.path.join(base_dir, "seed_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved -> {out_path}")


def main() -> None:
    """Parse arguments and run aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate InternVideo2 test metrics across seeds."
    )
    parser.add_argument(
        "--task",
        choices=["loco", "rmm"],
        required=True,
        help="Task to aggregate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to aggregate (default: 42 123 456).",
    )
    args = parser.parse_args()
    aggregate(args.task, args.seeds)


if __name__ == "__main__":
    main()