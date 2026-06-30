"""
Evaluate PySkl sliding window model
Usage:
    python eval_pyskl_sw.py --task rmm
    python eval_pyskl_sw.py --task locomotion
"""
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



class WindowDict(TypedDict):
    start: int
    end: int
    scores: np.ndarray
    
# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV  = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
PYSKL_BASE  = "/home/aparnabg/orcd/pool/pyskl_workspace/pyskl/work_dirs/"
PKL_BASE    = "/home/aparnabg/orcd/pool/pyskl_workspace/data/"
VJEPA_BASE  = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/models_output_seeds/vjepa/window_improved/v2/"
OUTPUT_BASE = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/models_output_seeds/pyskl_sw_standalone/"

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15
SEEDS = [42, 123, 456]

PYSKL_MODELS = {
    "rmm": {
        "model_dir": "ctrgcn_rmm_sw",
        "modality":  "jm",
    },
    "locomotion": {
        "model_dir": "posec3d_locomotion_sw",
        "modality":  "joint",
    },
}

LABEL_MAPS = {
    "rmm": {
        0: "Hands_flapping", 1: "Jumping", 2: "None",
        3: "Rocking",        4: "Spinning",
    },
    "locomotion": {
        0: "Crawling", 1: "Cruising", 2: "None",
        3: "Running",  4: "Vehicle",  5: "Walking",
    },
}

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. LOAD PYSKL WINDOW PREDICTIONS
# ============================================================
def load_pyskl_window_preds(task, seed):
    info     = PYSKL_MODELS[task]
    pred_path = os.path.join(
        PYSKL_BASE, info["model_dir"],
        f"{info['modality']}_s{seed}", "test_pred.pkl"
    )
    pkl_path = os.path.join(PKL_BASE, f"{task}_slidingwindow_pyskl.pkl")

    with open(pred_path, "rb") as f:
        pred_list = pickle.load(f)
    scores = np.stack(pred_list)   # (N, num_classes)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    test_ids = data["split"]["test"]

    assert len(test_ids) == scores.shape[0]

    # Group by video
    windows_by_video = defaultdict(list)
    for i, tid in enumerate(test_ids):
        match = re.match(r"(.+)_(\d+)_(\d+)_w(\d+)$", tid)
        if not match:
            continue
        video_name = match.group(1)
        start      = int(match.group(2))
        end        = int(match.group(3))
        windows_by_video[video_name].append({
            "start":  start,
            "end":    end,
            "scores": scores[i],
        })

    return windows_by_video


# ============================================================
# 2. MAP WINDOWS TO FRAMES (confidence-weighted, same as V-JEPA)
# ============================================================
def windows_to_frame_predictions(windows, T, int_to_label, num_classes):
    """
    Accumulate soft scores per frame, then pick argmax.
    Returns: (predictions list, confidences list)
    """
    frame_scores = np.zeros((T, num_classes), dtype=np.float64)
    frame_counts = np.zeros(T, dtype=np.int32)

    for w in windows:
        start = w["start"]
        end   = w["end"]   # inclusive
        for f in range(start, min(end + 1, T)):
            frame_scores[f] += w["scores"]
            frame_counts[f] += 1

    predictions = []
    confidences = []
    for f in range(T):
        if frame_counts[f] > 0:
            avg = frame_scores[f] / frame_counts[f]
            pred_idx = avg.argmax()
            predictions.append(int_to_label[pred_idx])
            confidences.append(round(float(avg[pred_idx]), 4))
        else:
            predictions.append("None")
            confidences.append(0.0)

    return predictions, confidences


# ============================================================
# 3. LOAD GROUND TRUTH LABELS
# ============================================================
def load_true_labels(label_path, task_column, T):
    try:
        anno = pd.read_csv(label_path)
    except Exception:
        return ["None"] * T

    anno.columns = anno.columns.str.strip()
    if task_column not in anno.columns:
        return ["None"] * T

    labels = (
        anno[task_column]
        .fillna("None").astype(str).str.strip()
        .replace({"": "None", "nan": "None", "N/A": "None"})
        .tolist()
    )
    if len(labels) < T:
        labels += ["None"] * (T - len(labels))
    return labels[:T]


# ============================================================
# 4. RUN ONE SEED
# ============================================================
def run_one_seed(task, seed):
    print(f"\n  Seed {seed}...")

    int_to_label  = LABEL_MAPS[task]
    num_classes   = len(int_to_label)
    task_column   = TASK_COLUMN[task]

    # Load pyskl window predictions
    windows_by_video = load_pyskl_window_preds(task, seed)

    # Output dir
    out_dir = os.path.join(
        OUTPUT_BASE, task, f"seed_{seed}", "per_video_predictions"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load test video list from splits CSV
    splits_df = pd.read_csv(SPLITS_CSV)
    test_df   = splits_df[splits_df["split"] == "test"]

    all_true, all_pred = [], []
    matched = 0

    for _, row in test_df.iterrows():
        feat_path  = row["vjpe_features_full_video_vit_h_features"]
        label_path = row["label_path"]
        video_name = os.path.splitext(os.path.basename(feat_path))[0]

        # Load V-JEPA csv just to get frame count and true labels
        # (we use the same frame count as V-JEPA for consistency)
        vjepa_csv = os.path.join(
            VJEPA_BASE, task, f"seed_{seed}",
            "per_video_predictions", f"{video_name}_predictions.csv"
        )
        if not os.path.exists(vjepa_csv):
            continue

        vjepa_df = pd.read_csv(vjepa_csv).sort_values("frame").reset_index(drop=True)
        T = len(vjepa_df)

        # Get true labels
        true_labels = load_true_labels(label_path, task_column, T)

        # Find matching windows
        windows = []
        for pyskl_name, wins in windows_by_video.items():
            if pyskl_name in video_name or video_name in pyskl_name:
                windows = wins
                matched += 1
                break

        # Map to frame predictions
        predictions, confidences = windows_to_frame_predictions(
            windows, T, int_to_label, num_classes
        )

        # Build output dataframe
        rows = []
        for f in range(T):
            rows.append({
                "frame":            f,
                "true_label":       true_labels[f],
                "predicted_label":  predictions[f],
                "confidence":       confidences[f],
                "correct":          int(true_labels[f] == predictions[f]),
            })
            all_true.append(true_labels[f])
            all_pred.append(predictions[f])

        vid_df = pd.DataFrame(rows)
        vid_df.to_csv(
            os.path.join(out_dir, f"{video_name}_predictions.csv"),
            index=False,
        )

    acc = accuracy_score(all_true, all_pred)
    nn_true = [t for t in all_true if t != "None"]
    nn_pred = [all_pred[i] for i, t in enumerate(all_true) if t != "None"]
    nn_acc  = accuracy_score(nn_true, nn_pred) if nn_true else 0

    print(f"    Matched videos : {matched}")
    print(f"    Frame Acc      : {acc:.4f}")
    print(f"    Action Acc     : {nn_acc:.4f}")

    # Save metrics
    metrics_dir = os.path.join(OUTPUT_BASE, task, f"seed_{seed}")
    with open(os.path.join(metrics_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Frame Accuracy : {acc:.4f}\n")
        f.write(f"Action Accuracy: {nn_acc:.4f}\n\n")
        f.write(classification_report(all_true, all_pred, zero_division=0))

    return {"frame_acc": acc, "action_acc": nn_acc}


# ============================================================
# 5. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["locomotion", "rmm", "both"])
    args = parser.parse_args()

    tasks = ["locomotion", "rmm"] if args.task == "both" else [args.task]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"PySkl Sliding Window — {task.upper()}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            res = run_one_seed(task, seed)
            seed_results.append(res)

        # Aggregate
        frame_accs  = [r["frame_acc"]  for r in seed_results]
        action_accs = [r["action_acc"] for r in seed_results]

        print(f"\n  Aggregated ({len(seed_results)} seeds):")
        print(f"    Frame Acc  : {np.mean(frame_accs)*100:.1f} ± {np.std(frame_accs)*100:.1f}")
        print(f"    Action Acc : {np.mean(action_accs)*100:.1f} ± {np.std(action_accs)*100:.1f}")

    print(f"\nDone! Evaluate with:")
    print(f"  python evaluate_all_models.py --base_dir {OUTPUT_BASE}")


if __name__ == "__main__":
    import numpy as np
    main()

