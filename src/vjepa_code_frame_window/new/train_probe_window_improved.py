"""
Improved Sliding-Window Attentive Probe
========================================
Builds on the flat window baseline with:
  1. Sinusoidal positional encoding (temporal order awareness)
  2. Lightweight 2-layer transformer encoder (frame interaction)
  3. Multi-query cross-attention (richer pooling)
  4. Focal loss (handles class imbalance)
  5. Dropout regularisation
  6. Confidence-weighted frame voting at inference
  7. Per-video CSV predictions saved during inference

Usage:
    python train_probe_window_improved.py --task locomotion --seed 42
    python train_probe_window_improved.py --task rmm --seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV    = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/window_improved/v1"

EMBED_DIM     = 1408
WINDOW_FRAMES = 30          # 2s at 15fps
STRIDE_FRAMES = 15          # 1s stride
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
PATIENCE      = 10
DROPOUT       = 0.1
NUM_QUERIES   = 4           # multi-query pooling
NUM_ENC_LAYERS = 2          # transformer encoder depth
NUM_HEADS     = 8
FOCAL_GAMMA   = 2.0         # focal loss gamma (0 = standard CE)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. SLIDING WINDOW BUILDER
# ============================================================
def build_windows_from_video(feat_path, label_path, task_column):
    """
    Slide a window over one video's features + annotations.
    Returns list of (window_tensor [WINDOW_FRAMES, EMBED_DIM], label_str).
    """
    try:
        feat = np.load(feat_path)           # (EMBED_DIM, T)
    except Exception as e:
        print(f"  [WARN] Cannot load features {feat_path}: {e}")
        return []

    feat = feat.T                           # (T, EMBED_DIM)
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

    windows = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        end         = start + WINDOW_FRAMES
        window_feat = feat[start:end]
        window_lbl  = labels_raw[start:end]
        label       = Counter(window_lbl).most_common(1)[0][0]
        windows.append((
            torch.tensor(window_feat, dtype=torch.float32),
            label,
        ))
    return windows


def build_all_windows(splits_csv, task_column, split_names):
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(split_names)]

    feat_col  = "vjpe_features_full_video_vit_h_features"
    label_col = "label_path"

    all_windows = []
    for _, row in df.iterrows():
        wins = build_windows_from_video(
            row[feat_col], row[label_col], task_column
        )
        all_windows.extend(wins)

    print(f"  Total windows ({', '.join(split_names)}): {len(all_windows)}")
    return all_windows


# ============================================================
# 2. DATASET
# ============================================================
class WindowDataset(Dataset):
    def __init__(self, windows_raw, label_map):
        self.data = []
        for feat, lbl in windows_raw:
            enc = label_map.get(lbl, label_map["None"])
            self.data.append((feat, enc))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat, enc = self.data[idx]
        return feat, torch.tensor(enc, dtype=torch.long)


# ============================================================
# 3. POSITIONAL ENCODING
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for temporal order."""
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 4. IMPROVED ATTENTIVE PROBE
# ============================================================
class ImprovedAttentiveProbe(nn.Module):
    """
    Improvements over the baseline AttentiveProbe:
      - Sinusoidal positional encoding
      - Lightweight transformer encoder (frame-to-frame interaction)
      - Multi-query cross-attention pooling
      - Dropout regularisation
    """
    def __init__(self, embed_dim, num_classes, num_heads=NUM_HEADS,
                 num_queries=NUM_QUERIES, num_enc_layers=NUM_ENC_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=WINDOW_FRAMES)

        # Lightweight transformer encoder — lets frames attend to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # smaller than typical 4x to stay lightweight
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_enc_layers
        )

        # Multi-query cross-attention pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim * num_queries, num_classes)

    def forward(self, x):
        # x: (B, WINDOW_FRAMES, EMBED_DIM)
        B = x.shape[0]

        # Add positional encoding
        x = self.pos_enc(x)

        # Frame-to-frame self-attention (transformer encoder)
        x = self.encoder(x)                          # (B, T, D)

        # Multi-query cross-attention pooling
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, D)
        pooled, _ = self.cross_attn(queries, x, x)     # (B, Q, D)
        pooled = self.cross_norm(pooled)

        # Flatten queries and classify
        pooled = self.dropout(pooled.reshape(B, -1))   # (B, Q*D)
        return self.classifier(pooled)                  # (B, num_classes)


# ============================================================
# 5. FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal loss to down-weight easy (majority class) examples
    and focus on hard (minority class) examples.
    gamma=0 is equivalent to standard cross-entropy.
    """
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        # Gather the log-prob and prob for the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)         # p_t
        log_pt = (log_probs * targets_one_hot).sum(dim=1) # log(p_t)

        focal_weight = (1.0 - pt) ** self.gamma

        loss = -focal_weight * log_pt

        if self.weight is not None:
            class_weights = self.weight[targets]
            loss = loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# 6. TRAINING LOOP
# ============================================================
def train_probe(probe, train_loader, val_loader, label_counts,
                num_classes, device, out_dir):
    probe = probe.to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    # Inverse-frequency class weights for focal loss
    total = sum(label_counts.values())
    class_weights = torch.ones(num_classes, device=device)
    for cls_idx, count in label_counts.items():
        class_weights[cls_idx] = total / (num_classes * count)
    # Normalise so mean weight = 1
    class_weights = class_weights / class_weights.mean()

    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)

    best_val_acc     = 0.0
    best_state       = None
    patience_counter = 0
    log_rows         = []

    for epoch in range(MAX_EPOCHS):
        # ── Train ──────────────────────────────────────────
        probe.train()
        tr_loss = tr_correct = tr_total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = probe(feats)
            loss   = criterion(logits, labels)
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
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = probe(feats)
                loss   = criterion(logits, labels)
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
# 7. PER-VIDEO INFERENCE  (confidence-weighted frame voting)
# ============================================================
def load_video_data(feat_path, label_path, task_column):
    """Load features and frame-level ground truth for one video."""
    try:
        feat = np.load(feat_path).T         # (T, D)
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
def infer_one_video(probe, feat, label_map, device):
    """
    Run sliding window inference on one video.
    Each window produces a prediction; overlapping windows vote per frame.
    Uses CONFIDENCE-WEIGHTED voting instead of naive majority vote.

    Returns: (predictions_list, confidences_list) both length T.
    """
    T = feat.shape[0]
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    softmax = nn.Softmax(dim=1)

    # Accumulate soft votes per frame: (T, num_classes)
    frame_scores = np.zeros((T, num_classes), dtype=np.float64)
    frame_counts = np.zeros(T, dtype=np.int32)

    # Build windows
    windows = []
    window_starts = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        w = torch.tensor(feat[start:start + WINDOW_FRAMES], dtype=torch.float32)
        windows.append(w)
        window_starts.append(start)

    if not windows:
        # Video shorter than one window — predict None for all frames
        return ["None"] * T, [0.0] * T

    # Batch inference
    all_windows = torch.stack(windows).to(device)
    for batch_start in range(0, len(all_windows), BATCH_SIZE):
        batch = all_windows[batch_start:batch_start + BATCH_SIZE]
        logits = probe(batch)
        probs = softmax(logits).cpu().numpy()  # (batch, num_classes)

        for i in range(batch.shape[0]):
            wi = batch_start + i
            start = window_starts[wi]
            # Add full probability distribution as soft vote to each frame
            for f in range(start, min(start + WINDOW_FRAMES, T)):
                frame_scores[f] += probs[i]
                frame_counts[f] += 1

    # Resolve per frame: pick class with highest accumulated probability
    predictions = []
    confidences = []
    for f in range(T):
        if frame_counts[f] > 0:
            avg_probs = frame_scores[f] / frame_counts[f]
            pred_id   = avg_probs.argmax()
            predictions.append(id_to_label[pred_id])
            confidences.append(round(float(avg_probs[pred_id]), 4))
        else:
            # Frames beyond last window start (tail frames)
            predictions.append("None")
            confidences.append(0.0)

    return predictions, confidences


@torch.no_grad()
def run_inference_per_video(probe, splits_csv, task_column,
                             label_map, device, out_dir):
    """Run inference on every test video, save one CSV per video."""
    print("\n" + "=" * 60)
    print("INFERENCE — per-frame (confidence-weighted), one CSV per video")
    print("=" * 60)

    probe.eval()
    probe = probe.to(device)

    pred_dir = os.path.join(out_dir, "per_video_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    df = pd.read_csv(splits_csv)
    df = df[df["split"] == "test"]

    feat_col  = "vjpe_features_full_video_vit_h_features"
    label_col = "label_path"

    all_true, all_pred = [], []

    for _, row in df.iterrows():
        feat_path  = row[feat_col]
        label_path = row[label_col]
        video_name = os.path.splitext(os.path.basename(feat_path))[0]

        feat, true_labels = load_video_data(feat_path, label_path, task_column)
        if feat is None:
            continue

        predictions, confidences = infer_one_video(probe, feat, label_map, device)
        T = len(true_labels)

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
        out_csv = os.path.join(pred_dir, f"{video_name}_predictions.csv")
        vid_df.to_csv(out_csv, index=False)
        vid_acc = vid_df["correct"].mean()
        print(f"  {video_name} | frames={T} | acc={vid_acc:.4f}")

    # ── Aggregate test metrics ─────────────────────────────
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
# 8. MAIN
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
    print(f"Task        : {args.task}  ({task_column})")
    print(f"Seed        : {seed}")
    print(f"Device      : {device}")
    print(f"Output dir  : {out_dir}")
    print(f"Window      : {WINDOW_FRAMES} frames, stride {STRIDE_FRAMES}")
    print(f"Queries     : {NUM_QUERIES}")
    print(f"Enc layers  : {NUM_ENC_LAYERS}")
    print(f"Focal gamma : {FOCAL_GAMMA}")

    # ── Build windows ──────────────────────────────────────
    print("\nBuilding train windows...")
    train_raw = build_all_windows(SPLITS_CSV, task_column, ["train"])
    print("Building val windows...")
    val_raw   = build_all_windows(SPLITS_CSV, task_column, ["val"])

    # Label map from train only
    all_train_labels = sorted(set(lbl for _, lbl in train_raw))
    if "None" not in all_train_labels:
        all_train_labels.append("None")
    label_map = {lbl: i for i, lbl in enumerate(sorted(all_train_labels))}
    num_classes = len(label_map)

    print(f"\nLabel map ({num_classes} classes): {label_map}")
    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Class distribution (for focal loss weights)
    train_label_counts_str = Counter(lbl for _, lbl in train_raw)
    train_label_counts_enc = {
        label_map[lbl]: count for lbl, count in train_label_counts_str.items()
    }
    print(f"\nTrain class distribution:\n{train_label_counts_str}")

    train_loader = DataLoader(
        WindowDataset(train_raw, label_map),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        WindowDataset(val_raw, label_map),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    print(f"\nTrain windows : {len(train_raw)}")
    print(f"Val windows   : {len(val_raw)}")

    # ── Model ──────────────────────────────────────────────
    probe = ImprovedAttentiveProbe(
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
    )
    total_params = sum(p.numel() for p in probe.parameters())
    print(f"\nProbe parameters : {total_params:,}")
    print(f"Num classes      : {num_classes}")

    # ── Train ──────────────────────────────────────────────
    probe = train_probe(
        probe, train_loader, val_loader,
        train_label_counts_enc, num_classes, device, out_dir,
    )

    probe_path = os.path.join(out_dir, "best_probe.pt")
    torch.save(probe.state_dict(), probe_path)
    print(f"Probe saved to: {probe_path}")

    # ── Per-video inference ────────────────────────────────
    run_inference_per_video(
        probe, SPLITS_CSV, task_column, label_map, device, out_dir
    )

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()