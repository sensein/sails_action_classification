import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV  = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
VJEPA_BASE  = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/window_improved/v2/"
PYSKL_BASE  = "/home/aparnabg/orcd/pool/pyskl_workspace/pyskl/work_dirs/"
POSEC3D_BASE = "/home/aparnabg/orcd/pool/pyskl_workspace/pyskl/work_dirs/"
PKL_BASE    = "/home/aparnabg/orcd/pool/pyskl_workspace/data/"
OUTPUT_BASE = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/fusion/sw_fusion_v3/"

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15
SEEDS  = [42, 123, 456]
ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6]
BETAS  = [0.1, 0.2, 0.3, 0.4, 0.5]

# Best models per task
MODELS = {
    "locomotion": {
        "vjepa": {
            "model_dir": "vjepa/window_improved/v2/",
        },
        "pyskl": {
            "model_dir": "stgcnpp_locomotion_sw/",
            "modality":  "b",
        },
        "posec3d": {
            "model_dir": "posec3d_locomotion_sw/",
            "modality":  "joint",
        },
    },
    "rmm": {
        "vjepa": {
            "model_dir": "vjepa/window_improved/v2/",
        },
        "pyskl": {
            "model_dir": "ctrgcn_rmm_sw/",
            "modality":  "jm",
        },
        "posec3d": {
            "model_dir": "posec3d_rmm_sw/",
            "modality":  "joint",
        },
    },
}

LABEL_MAPS = {
    "locomotion": {
        0: "Crawling", 1: "Cruising", 2: "None",
        3: "Running",  4: "Vehicle",  5: "Walking",
    },
    "rmm": {
        0: "Hands_flapping", 1: "Jumping", 2: "None",
        3: "Rocking",        4: "Spinning",
    },
}


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================
def load_predictions(task, seed, model_type):
    info = MODELS[task][model_type]

    if model_type == "vjepa":
        return load_vjepa_predictions(task, seed)
    elif model_type == "pyskl":
        return load_pyskl_window_preds(task, seed, info["model_dir"], info["modality"])
    elif model_type == "posec3d":
        return load_posec3d_window_preds(task, seed, info["model_dir"], info["modality"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_vjepa_predictions(task, seed):
    pred_dir = os.path.join(
        VJEPA_BASE, task, f"seed_{seed}", "per_video_predictions"
    )
    if not os.path.isdir(pred_dir):
        return {}
    video_preds = {}
    for csv_file in sorted(os.listdir(pred_dir)):
        if not csv_file.endswith("_predictions.csv"):
            continue
        vname = csv_file.replace("_predictions.csv", "")
        df = pd.read_csv(os.path.join(pred_dir, csv_file))
        df = df.sort_values("frame").reset_index(drop=True)
        video_preds[vname] = df
    return video_preds


def load_pyskl_window_preds(task, seed, model_dir, modality):
    pred_path = os.path.join(
        PYSKL_BASE, model_dir,
        f"{modality}_s{seed}", "test_pred.pkl"
    )
    pkl_path = os.path.join(PKL_BASE, f"{task}_slidingwindow_pyskl.pkl")

    with open(pred_path, "rb") as f:
        pred_list = pickle.load(f)
    scores = np.stack(pred_list)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    test_ids = data["split"]["test"]

    assert len(test_ids) == scores.shape[0]

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

    print(f"  PySkl: {scores.shape[0]} windows, {len(windows_by_video)} videos, "
          f"mean conf: {scores.max(axis=1).mean():.4f}")
    return windows_by_video


def load_posec3d_window_preds(task, seed, model_dir, modality):
    pred_path = os.path.join(
        POSEC3D_BASE, model_dir,
        f"{modality}_s{seed}", "test_pred.pkl"
    )
    pkl_path = os.path.join(PKL_BASE, f"{task}_slidingwindow_pyskl.pkl")

    with open(pred_path, "rb") as f:
        pred_list = pickle.load(f)
    scores = np.stack(pred_list)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    test_ids = data["split"]["test"]

    assert len(test_ids) == scores.shape[0]

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

    print(f"  PoseC3D: {scores.shape[0]} windows, {len(windows_by_video)} videos, "
          f"mean conf: {scores.max(axis=1).mean():.4f}")
    return windows_by_video


def load_vjepa_label_map(task, seed):
    path = os.path.join(VJEPA_BASE, task, f"seed_{seed}", "label_mapping.json")
    with open(path) as f:
        return json.load(f)


# ============================================================
# 2. BUILD UNIFIED LABELS
# ============================================================
def build_unified_labels(vjepa_label_map, pyskl_int_to_label, posec3d_int_to_label):
    vjepa_labels   = sorted(vjepa_label_map.keys())
    pyskl_labels   = sorted(pyskl_int_to_label.values())
    posec3d_labels = sorted(posec3d_int_to_label.values())
    all_labels     = sorted(set(vjepa_labels) | set(pyskl_labels) | set(posec3d_labels))

    vjepa_str_to_unified   = {lbl: all_labels.index(lbl) for lbl in vjepa_labels if lbl in all_labels}
    pyskl_int_to_unified   = {idx: all_labels.index(lbl) for idx, lbl in pyskl_int_to_label.items() if lbl in all_labels}
    posec3d_int_to_unified = {idx: all_labels.index(lbl) for idx, lbl in posec3d_int_to_label.items() if lbl in all_labels}

    return all_labels, vjepa_str_to_unified, pyskl_int_to_unified, posec3d_int_to_unified


# ============================================================
# 3. MAP WINDOWS TO PER-FRAME SCORES
# ============================================================
def windows_to_frame_scores(windows, T, num_classes):
    frame_scores = np.zeros((T, num_classes), dtype=np.float64)
    frame_counts = np.zeros(T, dtype=np.int32)

    for w in windows:
        for f in range(w["start"], min(w["end"] + 1, T)):
            frame_scores[f] += w["scores"]
            frame_counts[f] += 1

    for f in range(T):
        if frame_counts[f] > 0:
            frame_scores[f] /= frame_counts[f]

    return frame_scores, frame_counts


# ============================================================
# 4. V-JEPA SOFT SCORES
# ============================================================
def vjepa_to_frame_scores(vjepa_df, all_labels, vjepa_str_to_unified):
    T          = len(vjepa_df)
    num_labels = len(all_labels)
    scores     = np.zeros((T, num_labels), dtype=np.float64)

    for f in range(T):
        row  = vjepa_df.iloc[f]
        pred = str(row["predicted_label"]).strip()
        conf = float(row["confidence"])

        if pred in vjepa_str_to_unified:
            u_idx = vjepa_str_to_unified[pred]
            scores[f, u_idx] = conf
            remaining  = 1.0 - conf
            n_other    = num_labels - 1
            if n_other > 0:
                for lbl, idx in vjepa_str_to_unified.items():
                    if lbl != pred:
                        scores[f, idx] = remaining / n_other
    return scores


# ============================================================
# 5. FUSE ONE VIDEO
# ============================================================
def fuse_one_video(vjepa_df, pyskl_windows, posec3d_windows, all_labels,
                   vjepa_str_to_unified, pyskl_int_to_unified, posec3d_int_to_unified,
                   num_pyskl_classes, num_posec3d_classes, alpha, beta):
    T          = len(vjepa_df)
    num_labels = len(all_labels)

    vjepa_scores                = vjepa_to_frame_scores(vjepa_df, all_labels, vjepa_str_to_unified)
    pyskl_raw, pyskl_counts     = windows_to_frame_scores(pyskl_windows, T, num_pyskl_classes)
    posec3d_raw, posec3d_counts = windows_to_frame_scores(posec3d_windows, T, num_posec3d_classes)

    # Map pyskl and posec3d to unified space
    pyskl_unified   = np.zeros((T, num_labels), dtype=np.float64)
    posec3d_unified = np.zeros((T, num_labels), dtype=np.float64)

    for pyskl_idx, u_idx in pyskl_int_to_unified.items():
        pyskl_unified[:, u_idx] = pyskl_raw[:, pyskl_idx]
    for posec3d_idx, u_idx in posec3d_int_to_unified.items():
        posec3d_unified[:, u_idx] = posec3d_raw[:, posec3d_idx]

    # Normalize per frame
    for f in range(T):
        s_pyskl   = pyskl_unified[f].sum()
        s_posec3d = posec3d_unified[f].sum()
        if s_pyskl > 0:
            pyskl_unified[f] /= s_pyskl
        if s_posec3d > 0:
            posec3d_unified[f] /= s_posec3d

    results = []
    for f in range(T):
        true_label = str(vjepa_df.iloc[f]["true_label"]).strip()
        vjepa_pred = str(vjepa_df.iloc[f]["predicted_label"]).strip()

        if pyskl_counts[f] > 0 and posec3d_counts[f] > 0:
            fused = (
                alpha * vjepa_scores[f] +
                beta * pyskl_unified[f] +
                (1 - alpha - beta) * posec3d_unified[f]
            )
            fused_idx  = fused.argmax()
            fused_pred = all_labels[fused_idx]
            fused_conf = float(fused[fused_idx])
            source     = "vjepa_pyskl_posec3d"
        elif pyskl_counts[f] > 0:
            fused = (
                alpha * vjepa_scores[f] +
                (1 - alpha) * pyskl_unified[f]
            )
            fused_idx  = fused.argmax()
            fused_pred = all_labels[fused_idx]
            fused_conf = float(fused[fused_idx])
            source     = "vjepa_pyskl"
        elif posec3d_counts[f] > 0:
            fused = (
                alpha * vjepa_scores[f] +
                (1 - alpha) * posec3d_unified[f]
            )
            fused_idx  = fused.argmax()
            fused_pred = all_labels[fused_idx]
            fused_conf = float(fused[fused_idx])
            source     = "vjepa_posec3d"
        else:
            fused_pred = vjepa_pred
            fused_conf = float(vjepa_df.iloc[f]["confidence"])
            source     = "vjepa"

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
# 6. MATCH VIDEO
# ============================================================
def find_matching_windows(vjepa_video_name, windows_by_video):
    if vjepa_video_name in windows_by_video:
        return windows_by_video[vjepa_video_name]
    for name, windows in windows_by_video.items():
        if name in vjepa_video_name or vjepa_video_name in name:
            return windows
    return []
    
    
    
# ============================================================
# 7. RUN ONE SEED
# ============================================================
def run_one_seed(task, seed, alphas, betas):
    print(f"\n{'='*60}")
    print(f"Task: {task}  Seed: {seed}")
    print(f"{'='*60}")

    pyskl_int_to_label   = LABEL_MAPS[task]
    posec3d_int_to_label = LABEL_MAPS[task]
    num_pyskl_classes    = len(pyskl_int_to_label)
    num_posec3d_classes  = len(posec3d_int_to_label)

    print("Loading predictions...")
    vjepa_preds     = load_predictions(task, seed, "vjepa")
    pyskl_windows   = load_predictions(task, seed, "pyskl")
    posec3d_windows = load_predictions(task, seed, "posec3d")
    vjepa_label_map = load_vjepa_label_map(task, seed)

    print(f"  V-JEPA: {len(vjepa_preds)} videos")
    print(f"  PySkl:  {len(pyskl_windows)} videos")
    print(f"  PoseC3D: {len(posec3d_windows)} videos")

    all_labels, vjepa_str_to_unified, pyskl_int_to_unified, posec3d_int_to_unified = build_unified_labels(
        vjepa_label_map, pyskl_int_to_label, posec3d_int_to_label
    )
    print(f"  Unified labels: {all_labels}")

    results_by_params = {}

    for alpha in alphas:
        for beta in betas:
            params_str = f"a{int(alpha * 10)}_b{int(beta * 10)}"
            out_dir    = os.path.join(
                OUTPUT_BASE, params_str, task, f"seed_{seed}", "per_video_predictions"
            )
            os.makedirs(out_dir, exist_ok=True)

            all_true, all_pred = [], []

            for video_name, vjepa_df in vjepa_preds.items():
                pyskl_windows_matched   = find_matching_windows(video_name, pyskl_windows)
                posec3d_windows_matched = find_matching_windows(video_name, posec3d_windows)
                fused_df = fuse_one_video(
                    vjepa_df, pyskl_windows_matched, posec3d_windows_matched, all_labels,
                    vjepa_str_to_unified, pyskl_int_to_unified, posec3d_int_to_unified,
                    num_pyskl_classes, num_posec3d_classes, alpha, beta,
                )
                fused_df.to_csv(
                    os.path.join(out_dir, f"{video_name}_predictions.csv"),
                    index=False,
                )
                all_true.extend(fused_df["true_label"].tolist())
                all_pred.extend(fused_df["predicted_label"].tolist())

            acc    = accuracy_score(all_true, all_pred)
            nn_t   = [t for t in all_true if t != "None"]
            nn_p   = [all_pred[i] for i, t in enumerate(all_true) if t != "None"]
            nn_acc = accuracy_score(nn_t, nn_p) if nn_t else 0

            results_by_params[(alpha, beta)] = {"frame_acc": acc, "action_acc": nn_acc}
            print(f"  α={alpha}, β={beta}: Frame Acc={acc:.4f}  Action Acc={nn_acc:.4f}")

            metrics_dir = os.path.join(OUTPUT_BASE, params_str, task, f"seed_{seed}")
            with open(os.path.join(metrics_dir, "test_metrics.txt"), "w") as f:
                f.write(f"Alpha: {alpha}\nBeta: {beta}\nFrame Acc: {acc:.4f}\nAction Acc: {nn_acc:.4f}\n\n")
                f.write(classification_report(all_true, all_pred, zero_division=0))

    return results_by_params


# ============================================================
# 8. AGGREGATE
# ============================================================
def fmt(mean, std):
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def run_all_seeds(task, alphas, betas):
    seed_results = [run_one_seed(task, seed, alphas, betas) for seed in SEEDS]

    print(f"\n{'='*70}")
    print(f"AGGREGATED — {task.upper()}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'Frame Acc':>14} {'Action Acc':>14}")
    print("  " + "-"*50)

    for alpha in alphas:
        for beta in betas:
            fa = [r[(alpha, beta)]["frame_acc"]  for r in seed_results]
            aa = [r[(alpha, beta)]["action_acc"] for r in seed_results]
            print(f"  {'α='+str(alpha)+' β='+str(beta):<20} "
                  f"{fmt(np.mean(fa), np.std(fa)):>14} "
                  f"{fmt(np.mean(aa), np.std(aa)):>14}")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="both",
                        choices=["locomotion", "rmm", "both"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    tasks = ["locomotion", "rmm"] if args.task == "both" else [args.task]

    for task in tasks:
        if args.seed is not None:
            run_one_seed(task, args.seed, args.alphas, args.betas)
        else:
            run_all_seeds(task, args.alphas, args.betas)

    print(f"\nDone! Evaluate with:")
    print(f"  python evaluate_all_models.py --base_dir {OUTPUT_BASE}")


if __name__ == "__main__":
    import numpy as np
    main()
