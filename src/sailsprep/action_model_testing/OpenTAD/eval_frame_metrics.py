"""Frame-level evaluation of temporal action detection models."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

BASE_EXP_DIR = (
    "/home/aparnabg/orcd/scratch/all_project_files/"
    "action_sota_models/opentad/OpenTAD/exps"
)

DATASETS: dict[str, dict] = {
    "locomotion": {
        "anno_json": "data/locomotion/annotations/locomotion_anno.json",
        "class_names": ["Walking", "Cruising", "Crawling", "Running", "Vehicle"],
        "fps": 15.0,
    },
    "rmm": {
        "anno_json": "data/locomotion/annotations/rmm_anno.json",
        "class_names": ["Hands_flapping", "Jumping", "Rocking", "Spinning"],
        "fps": 15.0,
    },
}

MODELS: list[str] = [
    "actionformer_i3d",
    "actionformer_pose",
    "actionformer_r2plus1d",
    "actionformer_vjepa",
    "dyfadet_i3d",
    "dyfadet_pose",
    "dyfadet_r2plus1d",
    "dyfadet_vjepa",
    "tridet_i3d",
    "tridet_pose",
    "tridet_r2plus1d",
    "tridet_vjepa",
]

BACKGROUND: int = -1
SCORE_THRESH: float = 0.05


def evaluate(
    result_json: str,
    anno: dict,
    class_names: list[str],
    fps: float,
) -> dict | None:
    """Evaluate a single model's detections against ground-truth annotations.

    Parameters
    ----------
    result_json:
        Path to the ``result_detection.json`` produced by the model.
    anno:
        Annotation database loaded from the dataset JSON.
    class_names:
        Ordered list of foreground class names.
    fps:
        Frames per second used to convert time stamps to frame indices.

    Returns
    -------
    dict | None
        Dictionary of metrics, or ``None`` when no test frames are found.
    """
    class_to_id: dict[str, int] = {c: i for i, c in enumerate(class_names)}
    n_classes = len(class_names)

    with open(result_json) as fh:
        detections: dict = json.load(fh)["results"]

    all_gt: list[int] = []
    all_pred: list[int] = []

    for video_id, preds in detections.items():
        if video_id not in anno:
            continue
        meta = anno[video_id]
        if meta["subset"] != "test":
            continue

        num_frames = int(meta["frame"])

        gt_frames = np.full(num_frames, BACKGROUND, dtype=np.int32)
        for seg in meta["annotations"]:
            label = seg["label"]
            if label not in class_to_id:
                continue
            s = max(0, int(seg["segment"][0] * fps))
            e = min(num_frames, int(seg["segment"][1] * fps))
            gt_frames[s:e] = class_to_id[label]

        pred_frames = np.full(num_frames, BACKGROUND, dtype=np.int32)
        preds_filtered = [
            p
            for p in preds
            if p["score"] >= SCORE_THRESH and p["label"] in class_to_id
        ]
        for p in sorted(preds_filtered, key=lambda x: x["score"]):
            cls_id = class_to_id[p["label"]]
            s = max(0, int(p["segment"][0] * fps))
            e = min(num_frames, int(p["segment"][1] * fps))
            if s < e:
                pred_frames[s:e] = cls_id

        mask = gt_frames != BACKGROUND
        all_gt.extend(gt_frames[mask].tolist())
        all_pred.extend(pred_frames[mask].tolist())

    if len(all_gt) == 0:
        return None

    gt_arr = np.array(all_gt)
    pred_arr = np.array(all_pred)
    labels = list(range(n_classes))
    pred_clipped = np.where(pred_arr == BACKGROUND, n_classes, pred_arr)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        top1 = float((gt_arr == pred_clipped).mean() * 100)
        macro_p = float(
            precision_score(gt_arr, pred_clipped, labels=labels, average="macro", zero_division=0)
            * 100
        )
        macro_r = float(
            recall_score(gt_arr, pred_clipped, labels=labels, average="macro", zero_division=0)
            * 100
        )
        macro_f1 = float(
            f1_score(gt_arr, pred_clipped, labels=labels, average="macro", zero_division=0) * 100
        )
        weighted_f1 = float(
            f1_score(gt_arr, pred_clipped, labels=labels, average="weighted", zero_division=0)
            * 100
        )
        bal_acc = float(balanced_accuracy_score(gt_arr, pred_clipped) * 100)
        kappa = float(cohen_kappa_score(gt_arr, pred_clipped))
        per_f: np.ndarray = (
            f1_score(gt_arr, pred_clipped, labels=labels, average=None, zero_division=0) * 100
        )

    return {
        "top1": top1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "bal_acc": bal_acc,
        "kappa": kappa,
        "per_class_f1": per_f,
        "n_frames": len(gt_arr),
    }


def main() -> None:
    """Run evaluation across all datasets and models."""
    for dataset_name, ds_cfg in DATASETS.items():
        anno_path: str = ds_cfg["anno_json"]
        if not os.path.exists(anno_path):
            print(f"\n  [SKIP] Anno not found: {anno_path}")
            continue

        with open(anno_path) as fh:
            anno: dict = json.load(fh)["database"]

        class_names: list[str] = ds_cfg["class_names"]
        fps: float = ds_cfg["fps"]
        results: dict[str, dict] = {}

        for model_name in MODELS:
            result_json = os.path.join(
                BASE_EXP_DIR, dataset_name, model_name, "gpu1_id0", "result_detection.json"
            )
            if not os.path.exists(result_json):
                print(f"  [MISSING] {dataset_name}/{model_name}")
                continue
            print(f"  Evaluating {dataset_name}/{model_name} ...", flush=True)
            res = evaluate(result_json, anno, class_names, fps)
            if res is None:
                print(f"  [NO TEST DATA] {dataset_name}/{model_name}")
                continue
            results[model_name] = res

        sep = "=" * 110
        print(f"\n{sep}")
        print(f"  Dataset: {dataset_name.upper()}  —  Model Comparison")
        print(sep)
        header = (
            f"  {'Model':<25} {'Top-1':>7} {'Mac-P':>7} {'Mac-R':>7}"
            f" {'Mac-F1':>7} {'Wt-F1':>7} {'BalAcc':>7} {'Kappa':>7}"
        )
        print(header)
        print(f"  {'-' * 103}")
        for model_name, r in results.items():
            print(
                f"  {model_name:<25} {r['top1']:>6.2f}% {r['macro_p']:>6.2f}%"
                f" {r['macro_r']:>6.2f}% {r['macro_f1']:>6.2f}%"
                f" {r['weighted_f1']:>6.2f}% {r['bal_acc']:>6.2f}%"
                f" {r['kappa']:>7.4f}"
            )
        print(sep)

        print(f"\n  Per-class F1 scores (%) — {dataset_name.upper()}")
        cls_header = f"  {'Model':<25}" + "".join(f" {c:>14}" for c in class_names)
        print(cls_header)
        print(f"  {'-' * 105}")
        for model_name, r in results.items():
            row = f"  {model_name:<25}"
            for f1_val in r["per_class_f1"]:
                row += f" {f1_val:>13.2f}%"
            print(row)
        print(sep + "\n")


if __name__ == "__main__":
    main()