"""
Untrimmed Video Action Recognition — Sliding Window Attentive Probe
====================================================================
Loads pre-extracted V-JEPA ViT-H features (1024, T) from full videos,
slides a 2s window (30 frames) with 1s stride (15 frames) over each video,
assigns majority-vote labels per window, trains an AttentiveProbe.

Usage:
    python train_probe_untrimmed.py --task locomotion --seed 42
    python train_probe_untrimmed.py --task rmm --seed 42
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
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/window/"

EMBED_DIM     = 1024       # ViT-H feature dimension
WINDOW_FRAMES = 30         # 2s at 15fps
STRIDE_FRAMES = 15         # 1s at 15fps
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
PATIENCE      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Map task name -> annotation column
TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. SLIDING WINDOW DATASET BUILDER
# ============================================================
def build_windows_from_video(feat_path, label_path, task_column):
    """
    Load one video's features + annotations, slide window,
    return list of (window_tensor [30, 1024], label_str).
    Features on disk: (1024, T) → we transpose to (T, 1024).
    """
    # Load features
    try:
        feat = np.load(feat_path)           # (1024, T)
    except Exception as e:
        print(f"  [WARN] Failed to load features {feat_path}: {e}")
        return []

    feat = feat.T                           # (T, 1024)
    T    = feat.shape[0]

    # Load annotations
    try:
        anno = pd.read_csv(label_path)
    except Exception as e:
        print(f"  [WARN] Failed to load annotations {label_path}: {e}")
        return []

    # Normalise column names (strip whitespace)
    anno.columns = anno.columns.str.strip()

    if task_column not in anno.columns:
        print(f"  [WARN] Column '{task_column}' not found in {label_path}")
        return []

    # Build frame-level label array (length = T)
    # Annotation may have fewer rows than feature frames → pad with "None"
    labels_raw = anno[task_column].fillna("None").astype(str).str.strip()
    labels_raw = labels_raw.replace({"": "None", "nan": "None", "N/A": "None"})
    frame_labels = labels_raw.tolist()

    # Pad or trim to match feature length T
    if len(frame_labels) < T:
        frame_labels += ["None"] * (T - len(frame_labels))
    frame_labels = frame_labels[:T]

    # Slide window
    windows = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        end        = start + WINDOW_FRAMES
        window_feat = feat[start:end]                          # (30, 1024)
        window_lbl  = frame_labels[start:end]

        # Majority vote
        count = Counter(window_lbl)
        label = count.most_common(1)[0][0]

        windows.append((
            torch.tensor(window_feat, dtype=torch.float32),   # (30, 1024)
            label
        ))

    return windows


def build_all_windows(splits_csv, task_column, split_names):
    """
    Iterate over all videos in the given splits,
    return flat list of (tensor, label_str) and the label_map.
    Split is done at VIDEO level using the 'split' column.
    """
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(split_names)]

    feat_col  = "vjpe_features_full_video_vit_h_features"
    label_col = "label_path"

    all_windows = []
    for _, row in df.iterrows():
        windows = build_windows_from_video(
            row[feat_col], row[label_col], task_column
        )
        all_windows.extend(windows)

    print(f"  Total windows ({', '.join(split_names)}): {len(all_windows)}")
    return all_windows


def encode_labels(windows):
    """Assign integer codes to string labels, return encoded list + map."""
    all_labels  = sorted(set(lbl for _, lbl in windows))
    label_map   = {lbl: i for i, lbl in enumerate(all_labels)}
    encoded     = [(feat, label_map[lbl]) for feat, lbl in windows]
    return encoded, label_map


# ============================================================
# 2. DATASET
# ============================================================
class WindowDataset(Dataset):
    def __init__(self, windows):
        # windows: list of (tensor [30,1024], int)
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        feat, label = self.windows[idx]
        return feat, torch.tensor(label, dtype=torch.long)


# ============================================================
# 3. ATTENTIVE PROBE
# ============================================================
class AttentiveProbe(nn.Module):
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
        # x: (B, 30, 1024)
        B       = x.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        out     = self.norm(out).reshape(B, -1)
        return self.classifier(out)


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
        # --- Train ---
        probe.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = probe(feats)
            loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += labels.size(0)
        scheduler.step()
        tr_acc  = tr_correct / tr_total
        tr_loss /= tr_total

        # --- Validate ---
        probe.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = probe(feats)
                loss   = F.cross_entropy(logits, labels)
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
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
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
# 5. INFERENCE & METRICS
# ============================================================
@torch.no_grad()
def run_inference(probe, test_windows_raw, label_map, device, out_dir):
    """
    test_windows_raw: list of (tensor, label_str)  ← un-encoded, for readable output
    """
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET")
    print("=" * 60)

    probe.eval()
    probe       = probe.to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    softmax     = nn.Softmax(dim=1)

    # Encode with the same label_map
    test_enc = [
        (feat, label_map.get(lbl, label_map.get("None", 0)))
        for feat, lbl in test_windows_raw
    ]
    dataset = WindowDataset(test_enc)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    true_labels_str, pred_labels_str, confidences = [], [], []
    sample_idx = 0

    for feats, labels in loader:
        feats  = feats.to(device)
        logits = probe(feats)
        probs  = softmax(logits)
        for i in range(logits.shape[0]):
            top_pred  = probs[i].argmax().item()
            top_score = probs[i, top_pred].item()
            true_enc  = labels[i].item()
            true_labels_str.append(id_to_label[true_enc])
            pred_labels_str.append(id_to_label[top_pred])
            confidences.append(round(top_score, 4))
            sample_idx += 1

    results_df = pd.DataFrame({
        "true_label":      true_labels_str,
        "predicted_label": pred_labels_str,
        "confidence":      confidences,
        "correct":         [int(t == p) for t, p in
                            zip(true_labels_str, pred_labels_str)],
    })
    results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    accuracy = results_df["correct"].mean()
    print(f"\nOverall Accuracy : {accuracy:.4f}  "
          f"({int(results_df['correct'].sum())}/{len(results_df)})")

    report      = classification_report(
        true_labels_str, pred_labels_str, zero_division=0
    )
    labels_list = sorted(set(true_labels_str))
    cm          = confusion_matrix(
        true_labels_str, pred_labels_str, labels=labels_list
    )
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm_df)

    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
    print(f"Metrics saved to: {metrics_path}")
    return results_df


# ============================================================
# 6. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["locomotion", "rmm"],
        help="Which annotation column to classify"
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Random seed (42, 456, 123)"
    )
    args = parser.parse_args()

    task_column = TASK_COLUMN[args.task]
    seed        = args.seed

    # Reproducibility
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

    # ── Build windows ──────────────────────────────────────
    print("\nBuilding train windows...")
    train_windows_raw = build_all_windows(
        SPLITS_CSV, task_column, ["train"]
    )
    print("Building val windows...")
    val_windows_raw   = build_all_windows(
        SPLITS_CSV, task_column, ["val"]
    )
    print("Building test windows...")
    test_windows_raw  = build_all_windows(
        SPLITS_CSV, task_column, ["test"]
    )

    # Build label map from TRAIN only, then apply to all splits
    # (so val/test unseen labels fall back to "None")
    all_train_labels = sorted(set(lbl for _, lbl in train_windows_raw))
    # Make sure "None" is always present
    if "None" not in all_train_labels:
        all_train_labels.append("None")
    label_map = {lbl: i for i, lbl in enumerate(sorted(all_train_labels))}

    print(f"\nLabel map ({len(label_map)} classes): {label_map}")

    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    def encode(windows_raw):
        return [
            (feat, label_map.get(lbl, label_map["None"]))
            for feat, lbl in windows_raw
        ]

    train_enc = encode(train_windows_raw)
    val_enc   = encode(val_windows_raw)

    print(f"\nTrain windows : {len(train_enc)}")
    print(f"Val windows   : {len(val_enc)}")
    print(f"Test windows  : {len(test_windows_raw)}")

    # ── Class distribution ─────────────────────────────────
    train_label_counts = Counter(lbl for _, lbl in train_windows_raw)
    print(f"\nTrain class distribution:\n{train_label_counts}")

    # ── Train ──────────────────────────────────────────────
    num_classes  = len(label_map)
    probe        = AttentiveProbe(
        embed_dim=EMBED_DIM, num_classes=num_classes
    )
    print(f"\nProbe parameters: {sum(p.numel() for p in probe.parameters()):,}")
    print(f"Num classes     : {num_classes}")

    train_loader = DataLoader(
        WindowDataset(train_enc),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        WindowDataset(val_enc),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    probe = train_probe(probe, train_loader, val_loader, device, out_dir)

    probe_path = os.path.join(out_dir, "best_probe.pt")
    torch.save(probe.state_dict(), probe_path)
    print(f"Probe saved to: {probe_path}")

    # ── Test inference ─────────────────────────────────────
    run_inference(probe, test_windows_raw, label_map, device, out_dir)

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()