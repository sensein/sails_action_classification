"""Compute mAP@[0.3,0.4,0.5,0.6,0.7] for every model that has result_detection.json.

For models with 3 seeds, reports mean +/- std and 95% CI.
For models with only 1 run (gpu1_id0 only), reports single-seed result.
Saves a summary table to evaluation_results.csv and evaluation_results.txt.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


# ── mAP implementation ──────────────────────────────────────────────────────


def iou_1d(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute IoU between one prediction [s, e] and an array of GTs [N, 2]."""
    inter = np.maximum(
        0,
        np.minimum(pred[1], gt[:, 1]) - np.maximum(pred[0], gt[:, 0]),
    )
    union = (pred[1] - pred[0]) + (gt[:, 1] - gt[:, 0]) - inter
    return inter / np.maximum(union, 1e-6)


def compute_map(
    predictions: dict,
    ground_truth: dict,
    tiou_thresholds: list[float],
    num_classes: int,
) -> tuple[float, list[float]]:
    """Compute mean mAP over IoU thresholds.

    Parameters
    ----------
    predictions:
        Mapping of video_id to a list of dicts with keys
        ``segment``, ``label``, ``label_id``, and ``score``.
    ground_truth:
        Mapping of video_id to a list of dicts with keys
        ``segment``, ``label``, and ``label_id``.
    tiou_thresholds:
        List of temporal IoU thresholds.
    num_classes:
        Total number of foreground classes.

    Returns
    -------
    tuple[float, list[float]]
        Overall mean mAP and per-threshold mAP values.
    """
    ap_per_thresh: list[float] = []

    for tiou in tiou_thresholds:
        ap_per_class: list[float] = []

        for cls in range(num_classes):
            tp_list: list[int] = []
            fp_list: list[int] = []
            scores_list: list[float] = []
            n_gt = 0

            for vid, gt_list in ground_truth.items():
                gt_segs = np.array(
                    [g["segment"] for g in gt_list if g["label_id"] == cls],
                    dtype=np.float32,
                )
                n_gt += len(gt_segs)
                preds = sorted(
                    [p for p in predictions.get(vid, []) if p["label_id"] == cls],
                    key=lambda x: -x["score"],
                )
                matched = np.zeros(len(gt_segs), dtype=bool)

                for p in preds:
                    scores_list.append(p["score"])
                    if len(gt_segs) == 0:
                        tp_list.append(0)
                        fp_list.append(1)
                        continue
                    ious = iou_1d(np.array(p["segment"]), gt_segs)
                    best = int(np.argmax(ious))
                    if ious[best] >= tiou and not matched[best]:
                        tp_list.append(1)
                        fp_list.append(0)
                        matched[best] = True
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

            if n_gt == 0:
                continue
            if len(scores_list) == 0:
                ap_per_class.append(0.0)
                continue

            order = np.argsort(-np.array(scores_list))
            tp_cum = np.cumsum(np.array(tp_list)[order])
            fp_cum = np.cumsum(np.array(fp_list)[order])
            rec = tp_cum / max(n_gt, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

            ap = 0.0
            for t in np.linspace(0, 1, 11):
                p_at_r = prec[rec >= t]
                ap += (float(np.max(p_at_r)) if len(p_at_r) else 0.0) / 11
            ap_per_class.append(ap)

        ap_per_thresh.append(float(np.mean(ap_per_class)) if ap_per_class else 0.0)

    return float(np.mean(ap_per_thresh)), ap_per_thresh


# ── Data loading ─────────────────────────────────────────────────────────────


def load_gt(anno_path: str, class_map: list[str]) -> dict:
    """Load ground-truth annotations for test-set videos."""
    with open(anno_path) as fh:
        db: dict = json.load(fh)["database"]

    gt: dict = {}
    for vid, info in db.items():
        if info["subset"] != "test":
            continue
        anns = []
        for a in info["annotations"]:
            if a["label"] in class_map:
                anns.append(
                    {
                        "segment": a["segment"],
                        "label": a["label"],
                        "label_id": class_map.index(a["label"]),
                    }
                )
        gt[vid] = anns
    return gt


def load_class_map(path: str) -> list[str]:
    """Load ordered class names from a plain-text file (one name per line)."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_pred(result_path: str, class_map: list[str]) -> dict:
    """Load model predictions from a result_detection.json file."""
    with open(result_path) as fh:
        data: dict = json.load(fh)["results"]

    pred: dict = {}
    for vid, plist in data.items():
        mapped = []
        for p in plist:
            if p["label"] in class_map:
                mapped.append(
                    {
                        "segment": p["segment"],
                        "label": p["label"],
                        "label_id": class_map.index(p["label"]),
                        "score": p["score"],
                    }
                )
        pred[vid] = mapped
    return pred


# ── Configuration ─────────────────────────────────────────────────────────────

TASKS: dict[str, dict[str, str]] = {
    "locomotion": {
        "anno": "data/locomotion/annotations/locomotion_anno.json",
        "cmap": "data/locomotion/annotations/locomotion_category_idx.txt",
    },
    "rmm": {
        "anno": "data/locomotion/annotations/rmm_anno.json",
        "cmap": "data/locomotion/annotations/rmm_category_idx.txt",
    },
}

THRESHOLDS: list[float] = [0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> None:
    """Run mAP evaluation for all discovered result files and save outputs."""
    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    for rfile in sorted(
        glob.glob("exps/*/*/result_detection.json")
        + glob.glob("exps/*/*/*/result_detection.json")
    ):
        parts = rfile.split(os.sep)
        task = parts[1]
        model = parts[2]
        seed_label = parts[3] if parts[3].startswith("seed_") else "seed_single"
        groups[(task, model)].append((seed_label, rfile))

    rows: list[dict] = []

    for (task, model), entries in sorted(groups.items()):
        tcfg = TASKS.get(task)
        if tcfg is None:
            continue

        cmap = load_class_map(tcfg["cmap"])
        gt = load_gt(tcfg["anno"], cmap)
        n_cls = len(cmap)

        seed_maps: dict[str, tuple[float, list[float]]] = {}
        for seed_label, rfile in entries:
            pred = load_pred(rfile, cmap)
            mean_map, per_thresh = compute_map(pred, gt, THRESHOLDS, n_cls)
            seed_maps[seed_label] = (mean_map, per_thresh)
            print(
                f"  {task}/{model}/{seed_label}  mAP@avg={mean_map * 100:.2f}%  "
                f"per-thresh={[f'{v * 100:.1f}' for v in per_thresh]}"
            )

        vals = [v[0] for v in seed_maps.values()]
        n_seeds = len(vals)

        if n_seeds >= 3:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            ci95 = 1.96 * std / np.sqrt(n_seeds)
            summary = f"{mean * 100:.2f} \u00b1 {std * 100:.2f}"
            ci_str = f"[{(mean - ci95) * 100:.2f}, {(mean + ci95) * 100:.2f}]"
        elif n_seeds == 1:
            mean = vals[0]
            std = 0.0
            summary = f"{mean * 100:.2f} (single seed)"
            ci_str = "N/A"
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            summary = f"{mean * 100:.2f} \u00b1 {std * 100:.2f} ({n_seeds} seeds)"
            ci_str = "N/A (<3 seeds)"

        rows.append(
            {
                "task": task,
                "model": model,
                "n_seeds": n_seeds,
                "mean_mAP": mean * 100,
                "std_mAP": std * 100,
                "summary": summary,
                "ci95": ci_str,
                "per_seed": {k: round(v[0] * 100, 2) for k, v in seed_maps.items()},
                "per_thresh_mean": [
                    round(
                        float(np.mean([v[1][i] for v in seed_maps.values()])) * 100,
                        2,
                    )
                    for i in range(len(THRESHOLDS))
                ],
            }
        )

    # ── Print summary table ───────────────────────────────────────────────────

    sep = "=" * 90
    print(f"\n\n{sep}")
    print(
        f"{'Task':<12} {'Model':<25} {'Seeds':>5}  "
        f"{'mAP@avg (mean+/-std)':>28}  {'95% CI':>25}"
    )
    print(sep)
    for r in rows:
        print(
            f"{r['task']:<12} {r['model']:<25} {r['n_seeds']:>5}  "
            f"{r['summary']:>28}  {r['ci95']:>25}"
        )
    print(sep)

    print("\n\nPer-threshold breakdown (averaged over seeds):")
    print(
        f"{'Task':<12} {'Model':<25}  "
        + "  ".join(f"@{t:.1f}" for t in THRESHOLDS)
    )
    print("-" * 90)
    for r in rows:
        thresh_str = "  ".join(f"{v:>6.2f}" for v in r["per_thresh_mean"])
        print(f"{r['task']:<12} {r['model']:<25}  {thresh_str}")

    # ── Save CSV ──────────────────────────────────────────────────────────────

    with open("evaluation_results.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "task", "model", "n_seeds", "mean_mAP", "std_mAP",
                "ci95_lower", "ci95_upper",
                "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7",
            ]
        )
        for r in rows:
            if r["n_seeds"] >= 3:
                ci_vals = r["ci95"].strip("[]").split(", ")
                ci_lo, ci_hi = float(ci_vals[0]), float(ci_vals[1])
            else:
                ci_lo = ci_hi = r["mean_mAP"]
            writer.writerow(
                [
                    r["task"], r["model"], r["n_seeds"],
                    round(r["mean_mAP"], 4), round(r["std_mAP"], 4),
                    round(ci_lo, 4), round(ci_hi, 4),
                    *r["per_thresh_mean"],
                ]
            )

    # ── Save readable text report ─────────────────────────────────────────────

    with open("evaluation_results.txt", "w") as fh:
        fh.write("TAD Evaluation Results\n")
        fh.write("=" * 90 + "\n")
        fh.write(
            f"{'Task':<12} {'Model':<25} {'Seeds':>5}  "
            f"{'mAP@avg (mean+/-std)':>28}  {'95% CI':>25}\n"
        )
        fh.write("=" * 90 + "\n")
        for r in rows:
            fh.write(
                f"{r['task']:<12} {r['model']:<25} {r['n_seeds']:>5}  "
                f"{r['summary']:>28}  {r['ci95']:>25}\n"
            )
        fh.write("\nPer-threshold breakdown:\n")
        fh.write(
            f"{'Task':<12} {'Model':<25}  "
            + "  ".join(f"@{t:.1f}" for t in THRESHOLDS)
            + "\n"
        )
        fh.write("-" * 90 + "\n")
        for r in rows:
            thresh_str = "  ".join(f"{v:>6.2f}" for v in r["per_thresh_mean"])
            fh.write(f"{r['task']:<12} {r['model']:<25}  {thresh_str}\n")
        fh.write("\nPer-seed details:\n")
        for r in rows:
            fh.write(f"\n{r['task']}/{r['model']}:\n")
            for s, v in r["per_seed"].items():
                fh.write(f"  {s}: {v:.2f}%\n")

    print("\nSaved: evaluation_results.csv  evaluation_results.txt")


if __name__ == "__main__":
    main()