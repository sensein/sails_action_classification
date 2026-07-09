"""
Per-Frame Action Recognition — Flat Attentive Probe
====================================================
For every frame t, uses a local ±5 frame context window (11 frames).
Predicts one label per frame. NaN → "None" class.

Usage:
    python train_probe_framelevel.py --task locomotion --seed 42
    python train_probe_framelevel.py --task rmm --seed 42
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV    = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/framelevel_onest/"

EMBED_DIM     = 1408
CONTEXT       = 5          # ±5 frames → window of 11
WINDOW_SIZE   = 2 * CONTEXT + 1   # 11
BATCH_SIZE    = 256        # larger batch fine — frames are cheap
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
PATIENCE      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. PER-FRAME WINDOW BUILDER
# ============================================================
def build_frames_from_video(feat_path, label_path, task_column):
    """
    For every frame t in the video:
        - extract context window [t-5 .. t .. t+5] with edge padding
        - assign label from annotation at frame t
    Returns list of (window_tensor [11, 1024], label_str, frame_idx).
    """
    try:
        feat = np.load(feat_path)           # (1024, T)
    except Exception as e:
        print(f"  [WARN] Cannot load features {feat_path}: {e}")
        return []

    feat = feat.T                           # (T, 1024)
    T    = feat.shape[0]

    try:
        anno = pd.read_csv(label_path)
    except Exception as e:
        print(f"  [WARN] Cannot load annotations {label_path}: {e}")
        return []

    anno.columns = anno.columns.str.strip()
    if task_column not in anno.columns:
        print(f"  [WARN] Column '{task_column}' not in {label_path}")
        return []

    # Frame-level label array
    labels_raw = (
        anno[task_column]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace({"": "None", "nan": "None", "N/A": "None"})
        .tolist()
    )
    if len(labels_raw) < T:
        labels_raw += ["None"] * (T - len(labels_raw))
    labels_raw = labels_raw[:T]

    # Pad feature array at edges so every frame has full ±5 context
    # Edge padding: repeat first/last frame
    pad_feat = np.pad(
        feat,
        pad_width=((CONTEXT, CONTEXT), (0, 0)),
        mode="edge"
    )   # (T + 2*CONTEXT, 1024)

    frames = []
    for t in range(T):
        window = pad_feat[t: t + WINDOW_SIZE]       # (11, 1024)
        label  = labels_raw[t]
        frames.append((
            torch.tensor(window, dtype=torch.float32),
            label,
            t,
        ))
    return frames


def build_all_frames(splits_csv, task_column, split_names):
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(split_names)]

    feat_col  = "vjpe_features_full_video_vit_h_features"
    label_col = "label_path"

    all_frames = []
    for _, row in df.iterrows():
        frames = build_frames_from_video(
            row[feat_col], row[label_col], task_column
        )
        all_frames.extend(frames)

    print(f"  Total frames ({', '.join(split_names)}): {len(all_frames)}")
    return all_frames


# ============================================================
# 2. DATASET
# ============================================================
class FrameDataset(Dataset):
    def __init__(self, frames_raw, label_map):
        self.data = []
        for window, lbl, t in frames_raw:
            enc = label_map.get(lbl, label_map["None"])
            self.data.append((window, enc, lbl, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, enc, lbl, t = self.data[idx]
        return (
            window,
            torch.tensor(enc,  dtype=torch.long),
            lbl,
            t,
        )


# ============================================================
# 3. ATTENTIVE PROBE  (same as before, window=11 frames)
# ============================================================
class AttentiveProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm        = nn.LayerNorm(embed_dim)
        self.classifier  = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 11, 1024)
        B       = x.shape[0]
        queries = self.query_token.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        out     = self.norm(out).squeeze(1)     # (B, 1024)
        return self.classifier(out)             # (B, num_classes)


# ============================================================
# 4. TRAINING LOOP
# ============================================================
def train_probe(probe, train_loader, val_loader, device, out_dir):
    probe     = probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    best_val_acc     = 0.0
    best_state       = None
    patience_counter = 0
    log_rows         = []

    for epoch in range(MAX_EPOCHS):
        # ── Train ──────────────────────────────────────────
        probe.train()
        tr_loss = tr_correct = tr_total = 0

        for windows, labels, _, _ in train_loader:
            windows = windows.to(device)
            labels  = labels.to(device)
            logits  = probe(windows)
            loss    = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += labels.size(0)

        scheduler.step()
        tr_acc  = tr_correct / tr_total
        tr_loss /= tr_total

        # ── Validate ───────────────────────────────────────
        probe.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for windows, labels, _, _ in val_loader:
                windows = windows.to(device)
                labels  = labels.to(device)
                logits  = probe(windows)
                loss    = F.cross_entropy(logits, labels)
                val_loss    += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        log_rows.append({
            "epoch":      epoch + 1,
            "train_loss": round(tr_loss,  6),
            "train_acc":  round(tr_acc,   6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc,  6),
        })

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(log_rows).to_csv(
        os.path.join(out_dir, "training_log.csv"), index=False
    )
    if best_state:
        probe.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return probe


# ============================================================
# 5. INFERENCE — one CSV per video
# ============================================================
@torch.no_grad()
def run_inference_per_video(probe, splits_csv, task_column,
                             label_map, device, out_dir):
    print("\n" + "=" * 60)
    print("INFERENCE — per-frame, one CSV per video")
    print("=" * 60)

    probe.eval()
    probe       = probe.to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    softmax     = nn.Softmax(dim=1)

    pred_dir = os.path.join(out_dir, "per_video_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    df = pd.read_csv(splits_csv)
    df = df[df["split"] == "test"]

    feat_col  = "vjpe_features_full_video_vit_h_features"
    label_col = "label_path"

    all_true, all_pred = [], []

    for _, row in df.iterrows():
        video_name = os.path.splitext(
            os.path.basename(row[feat_col])
        )[0]

        frames_raw = build_frames_from_video(
            row[feat_col], row[label_col], task_column
        )
        if not frames_raw:
            continue

        # Build dataset for this video
        vid_dataset = FrameDataset(frames_raw, label_map)
        vid_loader  = DataLoader(
            vid_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        rows = []
        for windows, _labels, lbl_strs, frame_idxs in vid_loader:
            windows = windows.to(device)
            logits  = probe(windows)
            probs   = softmax(logits)

            for i in range(windows.shape[0]):
                pred_enc  = probs[i].argmax().item()
                confidence= probs[i, pred_enc].item()
                true_lbl  = lbl_strs[i]
                pred_lbl  = id_to_label[pred_enc]

                rows.append({
                    "frame":            frame_idxs[i].item(),
                    "true_label":       true_lbl,
                    "predicted_label":  pred_lbl,
                    "confidence":       round(confidence, 4),
                    "correct":          int(true_lbl == pred_lbl),
                })
                all_true.append(true_lbl)
                all_pred.append(pred_lbl)

        vid_df = pd.DataFrame(rows).sort_values("frame")
        out_csv = os.path.join(pred_dir, f"{video_name}_predictions.csv")
        vid_df.to_csv(out_csv, index=False)
        vid_acc = vid_df["correct"].mean()
        print(f"  {video_name} | frames={len(vid_df)} | acc={vid_acc:.4f}")

    # ── Aggregate metrics across all test videos ───────────
    print("\n── Aggregate Test Metrics ──")
    overall_acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"Overall Accuracy: {overall_acc:.4f}  ({len(all_true)} frames)")

    report      = classification_report(all_true, all_pred, zero_division=0)
    labels_list = sorted(set(all_true))
    cm          = confusion_matrix(all_true, all_pred, labels=labels_list)
    cm_df       = pd.DataFrame(cm, index=labels_list, columns=labels_list)

    print(report)
    print("Confusion Matrix:")
    print(cm_df)

    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Overall Accuracy : {overall_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
    print(f"Metrics saved to: {metrics_path}")


# ============================================================
# 6. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["locomotion", "rmm"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    task_column = TASK_COLUMN[args.task]
    seed        = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = os.path.join(OUTPUT_BASE, args.task, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Task       : {args.task}  ({task_column})")
    print(f"Seed       : {seed}")
    print(f"Device     : {device}")
    print(f"Output dir : {out_dir}")
    print(f"Context    : ±{CONTEXT} frames → window of {WINDOW_SIZE}")

    # ── Build frame datasets ───────────────────────────────
    print("\nBuilding train frames...")
    train_raw = build_all_frames(SPLITS_CSV, task_column, ["train"])
    print("Building val frames...")
    val_raw   = build_all_frames(SPLITS_CSV, task_column, ["val"])

    # Label map from train only
    all_labels = sorted(set(lbl for _, lbl, _ in train_raw))
    if "None" not in all_labels:
        all_labels.append("None")
    label_map = {lbl: i for i, lbl in enumerate(sorted(all_labels))}

    print(f"\nLabel map ({len(label_map)} classes): {label_map}")
    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    train_counts = Counter(lbl for _, lbl, _ in train_raw)
    print(f"\nTrain class distribution:\n{train_counts}")

    train_loader = DataLoader(
        FrameDataset(train_raw, label_map),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        FrameDataset(val_raw, label_map),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    print(f"\nTrain frames : {len(train_raw)}")
    print(f"Val frames   : {len(val_raw)}")

    # ── Model ──────────────────────────────────────────────
    probe = AttentiveProbe(
        embed_dim=EMBED_DIM, num_classes=len(label_map)
    )
    print(f"\nProbe parameters : {sum(p.numel() for p in probe.parameters()):,}")

    # ── Train ──────────────────────────────────────────────
    probe = train_probe(probe, train_loader, val_loader, device, out_dir)
    torch.save(probe.state_dict(), os.path.join(out_dir, "best_probe.pt"))

    # ── Inference ──────────────────────────────────────────
    run_inference_per_video(
        probe, SPLITS_CSV, task_column, label_map, device, out_dir
    )

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()