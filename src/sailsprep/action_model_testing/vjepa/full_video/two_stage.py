"""
Untrimmed Video Action Recognition — Hierarchical Attentive Probe
=================================================================
Stage 1: None vs Not-None (binary)
Stage 2: Specific action classification (non-None windows only)
Both stages share a backbone and are trained jointly.

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
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/window_two_st/"

EMBED_DIM     = 1024
WINDOW_FRAMES = 30         # 2s at 15fps
STRIDE_FRAMES = 15         # 1s stride
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
PATIENCE      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weighting — tune if Stage 1 or Stage 2 dominates
LAMBDA_STAGE1 = 1.0
LAMBDA_STAGE2 = 1.0

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. SLIDING WINDOW BUILDER
# ============================================================
def build_windows_from_video(feat_path, label_path, task_column):
    """
    Returns list of (window_tensor [30, 1024], label_str).
    NaN / empty → "None".
    """
    try:
        feat = np.load(feat_path)       # (1024, T)
    except Exception as e:
        print(f"  [WARN] Cannot load features {feat_path}: {e}")
        return []

    feat = feat.T                       # (T, 1024)
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

    # Pad / trim to feature length
    if len(labels_raw) < T:
        labels_raw += ["None"] * (T - len(labels_raw))
    labels_raw = labels_raw[:T]

    windows = []
    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        end         = start + WINDOW_FRAMES
        window_feat = feat[start:end]                           # (30, 1024)
        window_lbl  = labels_raw[start:end]
        label       = Counter(window_lbl).most_common(1)[0][0] # majority vote
        windows.append((
            torch.tensor(window_feat, dtype=torch.float32),
            label
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
# 2. LABEL MAPS
#    stage1_map : {label_str -> 0=None, 1=NotNone}
#    stage2_map : {label_str -> 0..K}  (non-None classes only)
# ============================================================
def build_label_maps(train_windows_raw):
    all_labels = sorted(set(lbl for _, lbl in train_windows_raw))
    if "None" not in all_labels:
        all_labels.append("None")

    stage1_map = {lbl: (0 if lbl == "None" else 1) for lbl in all_labels}

    non_none = sorted(lbl for lbl in all_labels if lbl != "None")
    stage2_map = {lbl: i for i, lbl in enumerate(non_none)}

    return stage1_map, stage2_map


# ============================================================
# 3. DATASET
# ============================================================
class WindowDataset(Dataset):
    """
    Each item returns:
        feat        : (30, 1024)
        stage1_lbl  : 0=None, 1=NotNone
        stage2_lbl  : class index if NotNone, else -1 (ignored in loss)
        label_str   : original string label (for inference readability)
    """
    def __init__(self, windows_raw, stage1_map, stage2_map):
        self.data = []
        for feat, lbl in windows_raw:
            s1 = stage1_map.get(lbl, 0)
            s2 = stage2_map.get(lbl, -1)   # -1 for None windows
            self.data.append((feat, s1, s2, lbl))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat, s1, s2, lbl = self.data[idx]
        return (
            feat,
            torch.tensor(s1, dtype=torch.long),
            torch.tensor(s2, dtype=torch.long),
            lbl,
        )


# ============================================================
# 4. HIERARCHICAL PROBE
#    Shared AttentiveProbe backbone → Stage1 head + Stage2 head
# ============================================================
class AttentiveBackbone(nn.Module):
    """Shared cross-attention pooling over the 30-frame window."""
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
        # x: (B, 30, 1024)
        B       = x.shape[0]
        queries = self.query_token.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        out     = self.norm(out)            # (B, 1, 1024)
        return out.squeeze(1)              # (B, 1024)


class HierarchicalProbe(nn.Module):
    def __init__(self, embed_dim, num_stage2_classes, num_heads=8):
        super().__init__()
        self.backbone    = AttentiveBackbone(embed_dim, num_heads)
        # Stage 1: binary — None vs NotNone
        self.head_stage1 = nn.Linear(embed_dim, 2)
        # Stage 2: fine-grained action classes
        self.head_stage2 = nn.Linear(embed_dim, num_stage2_classes)

    def forward(self, x):
        feat    = self.backbone(x)          # (B, 1024)
        logits1 = self.head_stage1(feat)    # (B, 2)
        logits2 = self.head_stage2(feat)    # (B, num_stage2_classes)
        return logits1, logits2


# ============================================================
# 5. JOINT LOSS
#    Stage 2 loss is computed ONLY on non-None windows in the batch
# ============================================================
def hierarchical_loss(logits1, logits2, stage1_labels, stage2_labels):
    """
    logits1      : (B, 2)
    logits2      : (B, K)
    stage1_labels: (B,)  0=None, 1=NotNone
    stage2_labels: (B,)  class index or -1 for None windows
    """
    loss1 = F.cross_entropy(logits1, stage1_labels)

    # Stage 2 loss only on non-None windows (stage2_labels != -1)
    mask = stage2_labels != -1
    if mask.sum() > 0:
        loss2 = F.cross_entropy(
            logits2[mask], stage2_labels[mask]
        )
    else:
        loss2 = torch.tensor(0.0, device=logits1.device)

    total = LAMBDA_STAGE1 * loss1 + LAMBDA_STAGE2 * loss2
    return total, loss1, loss2


# ============================================================
# 6. TRAINING LOOP
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
        tr_loss = tr_s1_correct = tr_s2_correct = tr_s2_total = tr_total = 0

        for feats, s1_lbls, s2_lbls, _ in train_loader:
            feats   = feats.to(device)
            s1_lbls = s1_lbls.to(device)
            s2_lbls = s2_lbls.to(device)

            logits1, logits2 = probe(feats)
            loss, l1, l2     = hierarchical_loss(
                logits1, logits2, s1_lbls, s2_lbls
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B             = s1_lbls.size(0)
            tr_loss      += loss.item() * B
            tr_total     += B
            tr_s1_correct += (logits1.argmax(1) == s1_lbls).sum().item()

            mask = s2_lbls != -1
            if mask.sum() > 0:
                tr_s2_correct += (
                    logits2[mask].argmax(1) == s2_lbls[mask]
                ).sum().item()
                tr_s2_total += mask.sum().item()

        scheduler.step()
        tr_loss   /= tr_total
        tr_s1_acc  = tr_s1_correct / tr_total
        tr_s2_acc  = tr_s2_correct / tr_s2_total if tr_s2_total > 0 else 0.0

        # ── Validate ───────────────────────────────────────
        probe.eval()
        val_loss = val_s1_correct = val_s2_correct = val_s2_total = val_total = 0

        with torch.no_grad():
            for feats, s1_lbls, s2_lbls, _ in val_loader:
                feats   = feats.to(device)
                s1_lbls = s1_lbls.to(device)
                s2_lbls = s2_lbls.to(device)

                logits1, logits2 = probe(feats)
                loss, _, _       = hierarchical_loss(
                    logits1, logits2, s1_lbls, s2_lbls
                )

                B              = s1_lbls.size(0)
                val_loss      += loss.item() * B
                val_total     += B
                val_s1_correct += (logits1.argmax(1) == s1_lbls).sum().item()

                mask = s2_lbls != -1
                if mask.sum() > 0:
                    val_s2_correct += (
                        logits2[mask].argmax(1) == s2_lbls[mask]
                    ).sum().item()
                    val_s2_total += mask.sum().item()

        val_loss   /= val_total
        val_s1_acc  = val_s1_correct / val_total
        val_s2_acc  = val_s2_correct / val_s2_total if val_s2_total > 0 else 0.0

        # Use Stage 2 val acc as the primary metric for early stopping
        # (Stage 1 is easier; Stage 2 is what we care about)
        monitor_acc = val_s2_acc

        print(
            f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
            f"Loss: {tr_loss:.4f} → {val_loss:.4f} | "
            f"S1 acc: {tr_s1_acc:.4f} → {val_s1_acc:.4f} | "
            f"S2 acc: {tr_s2_acc:.4f} → {val_s2_acc:.4f}"
        )

        log_rows.append({
            "epoch":       epoch + 1,
            "train_loss":  round(tr_loss,   6),
            "train_s1_acc":round(tr_s1_acc, 6),
            "train_s2_acc":round(tr_s2_acc, 6),
            "val_loss":    round(val_loss,   6),
            "val_s1_acc":  round(val_s1_acc, 6),
            "val_s2_acc":  round(val_s2_acc, 6),
        })

        if monitor_acc > best_val_acc:
            best_val_acc     = monitor_acc
            best_state       = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best Stage2 val acc: {best_val_acc:.4f}")
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
    print(f"\nBest Stage2 val accuracy: {best_val_acc:.4f}")
    return probe


# ============================================================
# 7. INFERENCE
#    Hard decision: Stage1 → None stops here, NotNone → Stage2
# ============================================================
@torch.no_grad()
def run_inference(probe, test_windows_raw, stage1_map, stage2_map,
                  device, out_dir):
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET  (hierarchical hard decision)")
    print("=" * 60)

    probe.eval()
    probe = probe.to(device)

    id_to_stage2 = {v: k for k, v in stage2_map.items()}
    softmax      = nn.Softmax(dim=1)

    dataset = WindowDataset(test_windows_raw, stage1_map, stage2_map)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    rows = []
    for feats, _s1_lbls, _s2_lbls, lbl_strs in loader:
        feats = feats.to(device)

        logits1, logits2 = probe(feats)
        probs1           = softmax(logits1)   # (B, 2)
        probs2           = softmax(logits2)   # (B, K)

        for i in range(feats.shape[0]):
            true_lbl    = lbl_strs[i]
            s1_pred     = probs1[i].argmax().item()   # 0=None, 1=NotNone
            s1_conf     = probs1[i, s1_pred].item()

            if s1_pred == 0:
                # Stage 1 says None — hard stop
                final_pred = "None"
                s2_conf    = None
            else:
                # Stage 1 says NotNone — go to Stage 2
                s2_pred    = probs2[i].argmax().item()
                s2_conf    = probs2[i, s2_pred].item()
                final_pred = id_to_stage2[s2_pred]

            rows.append({
                "true_label":       true_lbl,
                "predicted_label":  final_pred,
                "stage1_pred":      "None" if s1_pred == 0 else "NotNone",
                "stage1_conf":      round(s1_conf, 4),
                "stage2_pred":      final_pred if s1_pred == 1 else "N/A",
                "stage2_conf":      round(s2_conf, 4) if s2_conf else "N/A",
                "correct":          int(true_lbl == final_pred),
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # ── Overall metrics ────────────────────────────────────
    accuracy = results_df["correct"].mean()
    print(f"\nOverall Accuracy : {accuracy:.4f}  "
          f"({int(results_df['correct'].sum())}/{len(results_df)})")

    # ── Stage 1 metrics ────────────────────────────────────
    true_s1  = results_df["true_label"].apply(
        lambda x: "None" if x == "None" else "NotNone"
    )
    print("\n── Stage 1 (None vs NotNone) ──")
    print(classification_report(true_s1, results_df["stage1_pred"],
                                 zero_division=0))

    # ── Stage 2 metrics (only on truly NotNone windows) ───
    non_none_df = results_df[results_df["true_label"] != "None"]
    print("── Stage 2 (action classes, on NotNone windows only) ──")
    if len(non_none_df) > 0:
        s2_report = classification_report(
            non_none_df["true_label"],
            non_none_df["predicted_label"],
            zero_division=0,
        )
        print(s2_report)

        labels_list = sorted(non_none_df["true_label"].unique())
        cm = confusion_matrix(
            non_none_df["true_label"],
            non_none_df["predicted_label"],
            labels=labels_list,
        )
        cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
        print("Confusion Matrix (Stage 2):")
        print(cm_df)
    else:
        s2_report = "No non-None windows in test set."
        cm_df     = pd.DataFrame()
        print(s2_report)

    # ── Save metrics ───────────────────────────────────────
    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Overall Accuracy : {accuracy:.4f}\n\n")
        f.write("── Stage 1 (None vs NotNone) ──\n")
        f.write(classification_report(true_s1, results_df["stage1_pred"],
                                       zero_division=0))
        f.write("\n── Stage 2 (action classes) ──\n")
        f.write(s2_report)
        if len(cm_df) > 0:
            f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
    print(f"\nMetrics saved to: {metrics_path}")
    return results_df


# ============================================================
# 8. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["locomotion", "rmm"],
    )
    parser.add_argument(
        "--seed", type=int, required=True,
    )
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

    # ── Build windows ──────────────────────────────────────
    print("\nBuilding train windows...")
    train_raw = build_all_windows(SPLITS_CSV, task_column, ["train"])
    print("Building val windows...")
    val_raw   = build_all_windows(SPLITS_CSV, task_column, ["val"])
    print("Building test windows...")
    test_raw  = build_all_windows(SPLITS_CSV, task_column, ["test"])

    # ── Label maps (from train only) ───────────────────────
    stage1_map, stage2_map = build_label_maps(train_raw)
    num_stage2_classes     = len(stage2_map)

    print(f"\nStage 1 map : {{None: 0, NotNone: 1}}")
    print(f"Stage 2 map : {stage2_map}  ({num_stage2_classes} classes)")

    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump({"stage1": stage1_map, "stage2": stage2_map}, f, indent=2)

    # ── Class distribution ─────────────────────────────────
    train_counts = Counter(lbl for _, lbl in train_raw)
    print(f"\nTrain class distribution:\n{train_counts}")

    # ── Datasets & loaders ─────────────────────────────────
    train_dataset = WindowDataset(train_raw, stage1_map, stage2_map)
    val_dataset   = WindowDataset(val_raw,   stage1_map, stage2_map)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4
    )

    print(f"\nTrain windows : {len(train_dataset)}")
    print(f"Val windows   : {len(val_dataset)}")
    print(f"Test windows  : {len(test_raw)}")

    # ── Model ──────────────────────────────────────────────
    probe = HierarchicalProbe(
        embed_dim=EMBED_DIM,
        num_stage2_classes=num_stage2_classes,
    )
    total_params = sum(p.numel() for p in probe.parameters())
    print(f"\nProbe parameters : {total_params:,}")
    print(f"Stage2 classes   : {num_stage2_classes}")

    # ── Train ──────────────────────────────────────────────
    probe = train_probe(probe, train_loader, val_loader, device, out_dir)

    probe_path = os.path.join(out_dir, "best_probe.pt")
    torch.save(probe.state_dict(), probe_path)
    print(f"Probe saved to: {probe_path}")

    # ── Inference ──────────────────────────────────────────
    run_inference(probe, test_raw, stage1_map, stage2_map, device, out_dir)

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()