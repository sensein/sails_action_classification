"""
Fusion: V-JEPA v2 + PySkl Sliding Window (RMM)
================================================
Both models trained on same sliding windows (30 frames, stride 15).
PySkl test_pred.pkl has per-window softmax scores.
V-JEPA has per-frame predictions.

Maps PySkl window scores → per-frame scores (same as V-JEPA inference),
then fuses the two score vectors per frame.

Usage:
    python fusion_sw.py --task rmm --seed 42
    python fusion_sw.py --task rmm  (runs all seeds + aggregates)
"""

import pickle
import numpy as np
import pandas as pd
import json
import os
import re
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV  = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
VJEPA_BASE  = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/window_improved/v2/"
PYSKL_BASE  = "/home/aparnabg/orcd/pool/pyskl_workspace/pyskl/work_dirs/"
PKL_BASE    = "/home/aparnabg/orcd/pool/pyskl_workspace/data/"
OUTPUT_BASE = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/fusion/sw_fusion/"

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15
SEEDS = [42, 123, 456]
ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6]  # weight on pyskl

# Model paths per task
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

# Label maps (must match what was used in build_pyskl_slidingwindow_pkl.py)
LABEL_MAPS = {
    "rmm": {
        0: "Hands_flapping", 1: "Jumping", 2: "None",
        3: "Rocking", 4: "Spinning",
    },
    "locomotion": {
        0: "Crawling", 1: "Cruising", 2: "None",
        3: "Running", 4: "Vehicle", 5: "Walking",
    },
}


# ============================================================
# 1. LOAD PYSKL WINDOW PREDICTIONS
# ============================================================
def load_pyskl_window_preds(task, seed):
    """
    Load test_pred.pkl and sliding window pkl to get
    per-window softmax scores with video name + frame range.

    Returns: dict[video_name] -> list of (start, end, scores_array)
    """
    info = PYSKL_MODELS[task]
    pred_path = os.path.join(
        PYSKL_BASE, info["model_dir"],
        f"{info['modality']}_s{seed}", "test_pred.pkl"
    )
    pkl_path = os.path.join(PKL_BASE, f"{task}_slidingwindow_pyskl.pkl")

    with open(pred_path, "rb") as f:
        pred_list = pickle.load(f)
    scores = np.stack(pred_list)  # (N, num_classes)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    test_ids = data["split"]["test"]

    assert len(test_ids) == scores.shape[0], \
        f"Mismatch: {len(test_ids)} windows vs {scores.shape[0]} predictions"

    # Parse window IDs: videoname_START_END_wN
    windows_by_video = defaultdict(list)
    for i, tid in enumerate(test_ids):
        # Extract start, end, window number from the end
        match = re.match(r"(.+)_(\d+)_(\d+)_w(\d+)$", tid)
        if not match:
            continue
        video_name = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        windows_by_video[video_name].append({
            "start": start,
            "end": end,
            "scores": scores[i],
        })

    print(f"  PySkl: {scores.shape[0]} windows, {len(windows_by_video)} videos, "
          f"mean conf: {scores.max(axis=1).mean():.4f}")
    return windows_by_video


# ============================================================
# 2. MAP PYSKL WINDOWS TO PER-FRAME SCORES
# ============================================================
def pyskl_windows_to_frame_scores(windows, T, num_classes):
    """
    Convert window-level scores to frame-level scores by averaging
    overlapping windows (same approach as V-JEPA inference).

    Returns: (T, num_classes) numpy array
    """
    frame_scores = np.zeros((T, num_classes), dtype=np.float64)
    frame_counts = np.zeros(T, dtype=np.int32)

    for w in windows:
        start = w["start"]
        end = w["end"]  # inclusive
        scores = w["scores"]

        for f in range(start, min(end + 1, T)):
            frame_scores[f] += scores
            frame_counts[f] += 1

    # Average
    for f in range(T):
        if frame_counts[f] > 0:
            frame_scores[f] /= frame_counts[f]

    return frame_scores, frame_counts


# ============================================================
# 3. LOAD V-JEPA PREDICTIONS
# ============================================================
def load_vjepa_predictions(task, seed):
    pred_dir = os.path.join(VJEPA_BASE, task, f"seed_{seed}", "per_video_predictions")
    if not os.path.isdir(pred_dir):
        return {}
    video_preds = {}
    for csv_file in sorted(os.listdir(pred_dir)):
        if not csv_file.endswith("_predictions.csv"):
            continue
        video_name = csv_file.replace("_predictions.csv", "")
        df = pd.read_csv(os.path.join(pred_dir, csv_file))
        df = df.sort_values("frame").reset_index(drop=True)
        video_preds[video_name] = df
    return video_preds


def load_vjepa_label_map(task, seed):
    path = os.path.join(VJEPA_BASE, task, f"seed_{seed}", "label_mapping.json")
    with open(path) as f:
        return json.load(f)


# ============================================================
# 4. BUILD UNIFIED LABEL SPACE
# ============================================================
def build_unified_labels(vjepa_label_map, pyskl_int_to_label):
    """
    Both models include "None". Build a unified action label list.
    """
    vjepa_labels = sorted(vjepa_label_map.keys())
    pyskl_labels = sorted(pyskl_int_to_label.values())
    all_labels = sorted(set(vjepa_labels) | set(pyskl_labels))

    vjepa_str_to_unified = {
        lbl: all_labels.index(lbl) for lbl in vjepa_labels if lbl in all_labels
    }
    pyskl_int_to_unified = {
        idx: all_labels.index(lbl) for idx, lbl in pyskl_int_to_label.items()
        if lbl in all_labels
    }

    return all_labels, vjepa_str_to_unified, pyskl_int_to_unified


# ============================================================
# 5. BUILD V-JEPA SOFT SCORES PER FRAME
# ============================================================
def vjepa_to_frame_scores(vjepa_df, all_labels, vjepa_str_to_unified):
    """
    Convert V-JEPA per-frame predictions (label + confidence)
    to soft score vectors in the unified label space.

    Returns: (T, num_labels) numpy array
    """
    T = len(vjepa_df)
    num_labels = len(all_labels)
    frame_scores = np.zeros((T, num_labels), dtype=np.float64)

    for f in range(T):
        row = vjepa_df.iloc[f]
        pred = str(row["predicted_label"]).strip()
        conf = float(row["confidence"])

        if pred in vjepa_str_to_unified:
            u_idx = vjepa_str_to_unified[pred]
            frame_scores[f, u_idx] = conf

            # Spread remaining probability uniformly
            remaining = 1.0 - conf
            n_other = num_labels - 1
            if n_other > 0:
                for lbl, idx in vjepa_str_to_unified.items():
                    if lbl != pred:
                        frame_scores[f, idx] = remaining / n_other

    return frame_scores


# ============================================================
# 6. FUSE ONE VIDEO
# ============================================================
def fuse_one_video(vjepa_df, pyskl_windows, all_labels,
                    vjepa_str_to_unified, pyskl_int_to_unified,
                    num_pyskl_classes, alpha):
    """
    Fuse V-JEPA + PySkl for one video.
    alpha = weight on pyskl, (1-alpha) = weight on vjepa
    """
    T = len(vjepa_df)
    num_labels = len(all_labels)

    # V-JEPA frame scores
    vjepa_scores = vjepa_to_frame_scores(vjepa_df, all_labels, vjepa_str_to_unified)

    # PySkl frame scores
    pyskl_raw, pyskl_counts = pyskl_windows_to_frame_scores(
        pyskl_windows, T, num_pyskl_classes
    )

    # Map pyskl to unified space
    pyskl_unified = np.zeros((T, num_labels), dtype=np.float64)
    for pyskl_idx, u_idx in pyskl_int_to_unified.items():
        pyskl_unified[:, u_idx] = pyskl_raw[:, pyskl_idx]

    # Normalise pyskl scores per frame
    for f in range(T):
        s = pyskl_unified[f].sum()
        if s > 0:
            pyskl_unified[f] /= s

    # Fuse
    results = []
    for f in range(T):
        true_label = str(vjepa_df.iloc[f]["true_label"]).strip()
        vjepa_pred = str(vjepa_df.iloc[f]["predicted_label"]).strip()

        if pyskl_counts[f] > 0:
            # Both models have predictions — fuse
            fused = alpha * pyskl_unified[f] + (1 - alpha) * vjepa_scores[f]
            fused_idx = fused.argmax()
            fused_pred = all_labels[fused_idx]
            fused_conf = float(fused[fused_idx])
            source = "fused"
        else:
            # No pyskl window covers this frame — use vjepa only
            fused_pred = vjepa_pred
            fused_conf = float(vjepa_df.iloc[f]["confidence"])
            source = "vjepa"

        results.append({
            "frame":            f,
            "true_label":       true_label,
            "predicted_label":  fused_pred,
            "confidence":       round(fused_conf, 4),
            "correct":          int(true_label == fused_pred),
            "source":           source,
        })

    return pd.DataFrame(results)


# ============================================================
# 7. MATCH V-JEPA VIDEO TO PYSKL WINDOWS
# ============================================================
def find_matching_windows(vjepa_video_name, pyskl_windows_by_video):
    """Find pyskl windows for this V-JEPA video."""
    # Direct match
    if vjepa_video_name in pyskl_windows_by_video:
        return pyskl_windows_by_video[vjepa_video_name]

    # Substring match
    for pyskl_name, windows in pyskl_windows_by_video.items():
        if pyskl_name in vjepa_video_name or vjepa_video_name in pyskl_name:
            return windows

    return []


# ============================================================
# 8. RUN ONE SEED
# ============================================================
def run_one_seed(task, seed, alphas):
    print(f"\n{'='*60}")
    print(f"Task: {task}  Seed: {seed}")
    print(f"{'='*60}")

    pyskl_int_to_label = LABEL_MAPS[task]
    num_pyskl_classes = len(pyskl_int_to_label)

    # Load pyskl window predictions
    print("Loading PySkl predictions...")
    pyskl_windows = load_pyskl_window_preds(task, seed)

    # Load V-JEPA predictions
    print("Loading V-JEPA predictions...")
    vjepa_preds = load_vjepa_predictions(task, seed)
    vjepa_label_map = load_vjepa_label_map(task, seed)
    print(f"  V-JEPA: {len(vjepa_preds)} videos")

    # Build unified labels
    all_labels, vjepa_str_to_unified, pyskl_int_to_unified = build_unified_labels(
        vjepa_label_map, pyskl_int_to_label
    )
    print(f"  Unified labels: {all_labels}")

    # Run fusion at each alpha
    results_by_alpha = {}

    for alpha in alphas:
        alpha_str = f"a{int(alpha*10)}"
        out_dir = os.path.join(
            OUTPUT_BASE, alpha_str, task, f"seed_{seed}", "per_video_predictions"
        )
        os.makedirs(out_dir, exist_ok=True)

        all_true, all_pred = [], []
        matched = 0
        unmatched = 0

        for video_name, vjepa_df in vjepa_preds.items():
            windows = find_matching_windows(video_name, pyskl_windows)

            if windows:
                matched += 1
            else:
                unmatched += 1

            fused_df = fuse_one_video(
                vjepa_df, windows, all_labels,
                vjepa_str_to_unified, pyskl_int_to_unified,
                num_pyskl_classes, alpha,
            )

            # Save per-video CSV
            fused_df.to_csv(
                os.path.join(out_dir, f"{video_name}_predictions.csv"),
                index=False,
            )

            all_true.extend(fused_df["true_label"].tolist())
            all_pred.extend(fused_df["predicted_label"].tolist())

        if alpha == alphas[0]:
            print(f"  Matched: {matched}, Unmatched: {unmatched}")

        acc = accuracy_score(all_true, all_pred)
        nn_true = [t for t in all_true if t != "None"]
        nn_pred = [all_pred[i] for i, t in enumerate(all_true) if t != "None"]
        nn_acc = accuracy_score(nn_true, nn_pred) if nn_true else 0

        results_by_alpha[alpha] = {"frame_acc": acc, "action_acc": nn_acc}
        print(f"  α={alpha}: Frame Acc={acc:.4f}  Action Acc={nn_acc:.4f}")

        # Save metrics
        metrics_dir = os.path.join(OUTPUT_BASE, alpha_str, task, f"seed_{seed}")
        with open(os.path.join(metrics_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Alpha: {alpha}\nFrame Acc: {acc:.4f}\nAction Acc: {nn_acc:.4f}\n\n")
            f.write(classification_report(all_true, all_pred, zero_division=0))

    return results_by_alpha


# ============================================================
# 9. AGGREGATE ACROSS SEEDS
# ============================================================
def fmt(mean, std):
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def run_all_seeds(task, alphas):
    seed_results = []
    for seed in SEEDS:
        res = run_one_seed(task, seed, alphas)
        seed_results.append(res)

    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS — {task.upper()}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'Frame Acc':>14} {'Action Acc':>14}")
    print("  " + "-" * 50)

    # Also show V-JEPA only baseline
    vjepa_accs = []
    for seed in SEEDS:
        vjepa_preds = load_vjepa_predictions(task, seed)
        all_true, all_pred = [], []
        for vname, df in vjepa_preds.items():
            all_true.extend(df["true_label"].astype(str).tolist())
            all_pred.extend(df["predicted_label"].astype(str).tolist())
        vjepa_accs.append(accuracy_score(all_true, all_pred))
    print(f"  {'V-JEPA v2 only':<20} {fmt(np.mean(vjepa_accs), np.std(vjepa_accs)):>14}")

    for alpha in alphas:
        frame_accs = [r[alpha]["frame_acc"] for r in seed_results]
        action_accs = [r[alpha]["action_acc"] for r in seed_results]
        print(f"  {'Fused α=' + str(alpha):<20} "
              f"{fmt(np.mean(frame_accs), np.std(frame_accs)):>14} "
              f"{fmt(np.mean(action_accs), np.std(action_accs)):>14}")


# ============================================================
# 10. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["locomotion", "rmm"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.6])
    args = parser.parse_args()

    if args.seed is not None:
        run_one_seed(args.task, args.seed, args.alphas)
    else:
        run_all_seeds(args.task, args.alphas)

    print(f"\nDone! Results in: {OUTPUT_BASE}")
    print(f"Evaluate with:")
    print(f"  python evaluate_all_models.py --base_dir {OUTPUT_BASE}")


if __name__ == "__main__":
    main()