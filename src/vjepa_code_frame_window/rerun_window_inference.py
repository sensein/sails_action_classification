"""
Re-run inference for window and window_two_st models,
saving per-video, per-FRAME predictions.

Each window's prediction is assigned to all frames in that window.
Overlapping windows are resolved by majority vote per frame.

Usage:
    python rerun_window_inference.py --model window --task locomotion
    python rerun_window_inference.py --model window_two_st --task locomotion
    # Or do all at once:
    python rerun_window_inference.py --all
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import json
import os
import glob
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV    = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
BASE_DIR      = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/"

EMBED_DIM     = 1408
WINDOW_FRAMES = 30
STRIDE_FRAMES = 15
BATCH_SIZE    = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# MODELS (must match training code exactly)
# ============================================================
class AttentiveProbe(nn.Module):
    """Flat window model."""
    def __init__(self, embed_dim, num_classes, num_heads=8, num_queries=1):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim * num_queries, num_classes)

    def forward(self, x):
        B       = x.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        out     = self.norm(out).reshape(B, -1)
        return self.classifier(out)


class AttentiveBackbone(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B       = x.shape[0]
        queries = self.query_token.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        return self.norm(out).squeeze(1)


class HierarchicalProbe(nn.Module):
    """Two-stage window model."""
    def __init__(self, embed_dim, num_stage2_classes, num_heads=8):
        super().__init__()
        self.backbone    = AttentiveBackbone(embed_dim, num_heads)
        self.head_stage1 = nn.Linear(embed_dim, 2)
        self.head_stage2 = nn.Linear(embed_dim, num_stage2_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_stage1(feat), self.head_stage2(feat)


# ============================================================
# VIDEO-LEVEL INFERENCE
# ============================================================
def load_video_data(feat_path, label_path, task_column):
    """Load features and frame-level labels for one video."""
    try:
        feat = np.load(feat_path).T  # (T, D)
    except Exception as e:
        print(f"  [WARN] Cannot load {feat_path}: {e}")
        return None, None

    try:
        anno = pd.read_csv(label_path)
    except Exception as e:
        print(f"  [WARN] Cannot load {label_path}: {e}")
        return None, None

    anno.columns = anno.columns.str.strip()
    if task_column not in anno.columns:
        return None, None

    T = feat.shape[0]
    labels = (
        anno[task_column]
        .fillna("None").astype(str).str.strip()
        .replace({"": "None", "nan": "None", "N/A": "None"})
        .tolist()
    )
    if len(labels) < T:
        labels += ["None"] * (T - len(labels))
    labels = labels[:T]

    return feat, labels


@torch.no_grad()
def infer_one_video_flat(probe, feat, label_map, device):
    """Run flat window model on one video, return per-frame predictions."""
    T = feat.shape[0]
    id_to_label = {v: k for k, v in label_map.items()}
    softmax = nn.Softmax(dim=1)

    # Collect per-frame votes from overlapping windows
    frame_votes = [[] for _ in range(T)]
    frame_confs = [[] for _ in range(T)]

    # Build windows
    windows = []
    window_starts = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        w = torch.tensor(feat[start:start + WINDOW_FRAMES], dtype=torch.float32)
        windows.append(w)
        window_starts.append(start)

    if not windows:
        return None

    # Batch inference
    all_windows = torch.stack(windows).to(device)
    for batch_start in range(0, len(all_windows), BATCH_SIZE):
        batch = all_windows[batch_start:batch_start + BATCH_SIZE]
        logits = probe(batch)
        probs = softmax(logits)

        for i in range(batch.shape[0]):
            wi = batch_start + i
            start = window_starts[wi]
            pred_id = probs[i].argmax().item()
            conf = probs[i, pred_id].item()
            pred_label = id_to_label[pred_id]

            for f in range(start, start + WINDOW_FRAMES):
                if f < T:
                    frame_votes[f].append(pred_label)
                    frame_confs[f].append(conf)

    # Majority vote per frame
    predictions = []
    confidences = []
    for f in range(T):
        if frame_votes[f]:
            vote = Counter(frame_votes[f]).most_common(1)[0][0]
            avg_conf = np.mean([c for c, v in zip(frame_confs[f], frame_votes[f]) if v == vote])
            predictions.append(vote)
            confidences.append(round(avg_conf, 4))
        else:
            predictions.append("None")
            confidences.append(0.0)

    return predictions, confidences


@torch.no_grad()
def infer_one_video_hierarchical(probe, feat, stage2_map, device):
    """Run hierarchical window model on one video, return per-frame predictions."""
    T = feat.shape[0]
    id_to_s2 = {v: k for k, v in stage2_map.items()}
    softmax = nn.Softmax(dim=1)

    frame_votes = [[] for _ in range(T)]
    frame_s1_confs = [[] for _ in range(T)]

    windows = []
    window_starts = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        w = torch.tensor(feat[start:start + WINDOW_FRAMES], dtype=torch.float32)
        windows.append(w)
        window_starts.append(start)

    if not windows:
        return None

    all_windows = torch.stack(windows).to(device)
    for batch_start in range(0, len(all_windows), BATCH_SIZE):
        batch = all_windows[batch_start:batch_start + BATCH_SIZE]
        logits1, logits2 = probe(batch)
        probs1 = softmax(logits1)
        probs2 = softmax(logits2)

        for i in range(batch.shape[0]):
            wi = batch_start + i
            start = window_starts[wi]
            s1_pred = probs1[i].argmax().item()
            s1_conf = probs1[i, s1_pred].item()

            if s1_pred == 0:
                pred_label = "None"
            else:
                s2_pred = probs2[i].argmax().item()
                pred_label = id_to_s2[s2_pred]

            for f in range(start, start + WINDOW_FRAMES):
                if f < T:
                    frame_votes[f].append(pred_label)
                    frame_s1_confs[f].append(s1_conf)

    predictions = []
    confidences = []
    for f in range(T):
        if frame_votes[f]:
            vote = Counter(frame_votes[f]).most_common(1)[0][0]
            avg_conf = np.mean(frame_s1_confs[f])
            predictions.append(vote)
            confidences.append(round(avg_conf, 4))
        else:
            predictions.append("None")
            confidences.append(0.0)

    return predictions, confidences


# ============================================================
# MAIN
# ============================================================
def run_model_task_seed(model_name, task, seed, device):
    """Process one model/task/seed combination."""
    task_column = TASK_COLUMN[task]
    seed_dir = os.path.join(BASE_DIR, model_name, task, f"seed_{seed}")

    if not os.path.isdir(seed_dir):
        print(f"  [SKIP] {seed_dir} not found")
        return

    probe_path = os.path.join(seed_dir, "best_probe.pt")
    mapping_path = os.path.join(seed_dir, "label_mapping.json")

    if not os.path.exists(probe_path):
        print(f"  [SKIP] No best_probe.pt in {seed_dir}")
        return

    with open(mapping_path) as f:
        label_mapping = json.load(f)

    is_hierarchical = model_name == "window_two_st"

    # Load model
    if is_hierarchical:
        stage2_map = label_mapping["stage2"]
        num_s2 = len(stage2_map)
        probe = HierarchicalProbe(EMBED_DIM, num_s2)
    else:
        label_map = label_mapping
        num_classes = len(label_map)
        probe = AttentiveProbe(EMBED_DIM, num_classes)

    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    probe.eval()
    probe = probe.to(device)

    # Output directory
    pred_dir = os.path.join(seed_dir, "per_video_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Load test videos
    splits_df = pd.read_csv(SPLITS_CSV)
    test_df = splits_df[splits_df["split"] == "test"]

    for _, row in test_df.iterrows():
        feat_path = row["vjpe_features_full_video_vit_h_features"]
        label_path = row["label_path"]
        video_name = os.path.splitext(os.path.basename(feat_path))[0]

        feat, true_labels = load_video_data(feat_path, label_path, task_column)
        if feat is None:
            continue

        if is_hierarchical:
            result = infer_one_video_hierarchical(probe, feat, stage2_map, device)
        else:
            result = infer_one_video_flat(probe, feat, label_map, device)

        if result is None:
            continue

        predictions, confidences = result
        T = len(true_labels)

        rows = []
        for f in range(T):
            rows.append({
                "frame": f,
                "true_label": true_labels[f],
                "predicted_label": predictions[f],
                "confidence": confidences[f],
                "correct": int(true_labels[f] == predictions[f]),
            })

        vid_df = pd.DataFrame(rows)
        out_csv = os.path.join(pred_dir, f"{video_name}_predictions.csv")
        vid_df.to_csv(out_csv, index=False)
        acc = vid_df["correct"].mean()
        print(f"    {video_name} | frames={T} | acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=["window", "window_two_st"])
    parser.add_argument("--task", type=str, default=None,
                        choices=["locomotion", "rmm"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run all models/tasks/seeds")
    args = parser.parse_args()

    device = torch.device(DEVICE)

    if args.all:
        models = ["window", "window_two_st"]
        tasks = ["locomotion", "rmm"]
        seeds = [42, 123, 456]
    else:
        models = [args.model] if args.model else ["window", "window_two_st"]
        tasks = [args.task] if args.task else ["locomotion", "rmm"]
        seeds = [args.seed] if args.seed else [42, 123, 456]

    for model in models:
        for task in tasks:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Model: {model}  Task: {task}  Seed: {seed}")
                print(f"{'='*60}")
                run_model_task_seed(model, task, seed, device)

    print("\nDone! Now re-run evaluate_all_models.py to get full metrics.")


if __name__ == "__main__":
    main()