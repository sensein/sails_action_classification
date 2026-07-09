"""
Per-Frame Action Recognition — Hierarchical Attentive Probe
============================================================
Stage 1: None vs NotNone  (binary, every frame)
Stage 2: Specific action  (only NotNone frames)
Shared backbone, joint training, hard decision at inference.
±5 frame context window around each frame.

Usage:
    python train_probe_framelevel_hierarchical.py --task locomotion --seed 42
    python train_probe_framelevel_hierarchical.py --task rmm --seed 42
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
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/vjepa/framelevel_hierarchical/"

EMBED_DIM     = 1408
CONTEXT       = 5
WINDOW_SIZE   = 2 * CONTEXT + 1    # 11
BATCH_SIZE    = 256
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
PATIENCE      = 10
LAMBDA_S1     = 1.0
LAMBDA_S2     = 1.0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}


# ============================================================
# 1. FRAME BUILDER  (same as flat version)
# ============================================================
def build_frames_from_video(feat_path, label_path, task_column):
    try:
        feat = np.load(feat_path)
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

    pad_feat = np.pad(feat, ((CONTEXT, CONTEXT), (0, 0)), mode="edge")

    frames = []
    for t in range(T):
        window = pad_feat[t: t + WINDOW_SIZE]       # (11, 1024)
        frames.append((
            torch.tensor(window, dtype=torch.float32),
            labels_raw[t],
            t,
        ))
    return frames


def build_all_frames(splits_csv, task_column, split_names):
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(split_names)]

    all_frames = []
    for _, row in df.iterrows():
        frames = build_frames_from_video(
            row["vjpe_features_full_video_vit_h_features"],
            row["label_path"],
            task_column,
        )
        all_frames.extend(frames)

    print(f"  Total frames ({', '.join(split_names)}): {len(all_frames)}")
    return all_frames


# ============================================================
# 2. LABEL MAPS
# ============================================================
def build_label_maps(train_raw):
    all_labels = sorted(set(lbl for _, lbl, _ in train_raw))
    if "None" not in all_labels:
        all_labels.append("None")

    stage1_map = {lbl: (0 if lbl == "None" else 1) for lbl in all_labels}
    non_none   = sorted(lbl for lbl in all_labels if lbl != "None")
    stage2_map = {lbl: i for i, lbl in enumerate(non_none)}
    return stage1_map, stage2_map


# ============================================================
# 3. DATASET
# ============================================================
class FrameDataset(Dataset):
    def __init__(self, frames_raw, stage1_map, stage2_map):
        self.data = []
        for window, lbl, t in frames_raw:
            s1 = stage1_map.get(lbl, 0)
            s2 = stage2_map.get(lbl, -1)   # -1 for None frames
            self.data.append((window, s1, s2, lbl, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, s1, s2, lbl, t = self.data[idx]
        return (
            window,
            torch.tensor(s1, dtype=torch.long),
            torch.tensor(s2, dtype=torch.long),
            lbl,
            t,
        )


# ============================================================
# 4. MODEL
# ============================================================
class HierarchicalProbe(nn.Module):
    def __init__(self, embed_dim, num_stage2_classes, num_heads=8):
        super().__init__()
        # Shared backbone
        self.query_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm        = nn.LayerNorm(embed_dim)
        # Two heads
        self.head_s1     = nn.Linear(embed_dim, 2)
        self.head_s2     = nn.Linear(embed_dim, num_stage2_classes)

    def forward(self, x):
        B       = x.shape[0]
        queries = self.query_token.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        feat    = self.norm(out).squeeze(1)     # (B, 1024)
        return self.head_s1(feat), self.head_s2(feat)


# ============================================================
# 5. JOINT LOSS
# ============================================================
def hierarchical_loss(logits1, logits2, s1_lbls, s2_lbls):
    loss1 = F.cross_entropy(logits1, s1_lbls)
    mask  = s2_lbls != -1
    loss2 = (
        F.cross_entropy(logits2[mask], s2_lbls[mask])
        if mask.sum() > 0
        else torch.tensor(0.0, device=logits1.device)
    )
    return LAMBDA_S1 * loss1 + LAMBDA_S2 * loss2, loss1, loss2


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
        tr_loss = tr_s1c = tr_s2c = tr_s2t = tr_total = 0

        for windows, s1, s2, _, _ in train_loader:
            windows = windows.to(device)
            s1      = s1.to(device)
            s2      = s2.to(device)
            l1, l2  = probe(windows)
            loss, _, _ = hierarchical_loss(l1, l2, s1, s2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B        = s1.size(0)
            tr_loss += loss.item() * B
            tr_total+= B
            tr_s1c  += (l1.argmax(1) == s1).sum().item()
            mask     = s2 != -1
            if mask.sum() > 0:
                tr_s2c += (l2[mask].argmax(1) == s2[mask]).sum().item()
                tr_s2t += mask.sum().item()

        scheduler.step()
        tr_loss  /= tr_total
        tr_s1_acc = tr_s1c / tr_total
        tr_s2_acc = tr_s2c / tr_s2t if tr_s2t > 0 else 0.0

        # ── Validate ───────────────────────────────────────
        probe.eval()
        val_loss = val_s1c = val_s2c = val_s2t = val_total = 0

        with torch.no_grad():
            for windows, s1, s2, _, _ in val_loader:
                windows = windows.to(device)
                s1      = s1.to(device)
                s2      = s2.to(device)
                l1, l2  = probe(windows)
                loss, _, _ = hierarchical_loss(l1, l2, s1, s2)

                B         = s1.size(0)
                val_loss += loss.item() * B
                val_total+= B
                val_s1c  += (l1.argmax(1) == s1).sum().item()
                mask      = s2 != -1
                if mask.sum() > 0:
                    val_s2c += (l2[mask].argmax(1) == s2[mask]).sum().item()
                    val_s2t += mask.sum().item()

        val_loss  /= val_total
        val_s1_acc = val_s1c / val_total
        val_s2_acc = val_s2c / val_s2t if val_s2t > 0 else 0.0

        print(
            f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
            f"Loss: {tr_loss:.4f}→{val_loss:.4f} | "
            f"S1: {tr_s1_acc:.4f}→{val_s1_acc:.4f} | "
            f"S2: {tr_s2_acc:.4f}→{val_s2_acc:.4f}"
        )

        log_rows.append({
            "epoch":        epoch + 1,
            "train_loss":   round(tr_loss,   6),
            "train_s1_acc": round(tr_s1_acc, 6),
            "train_s2_acc": round(tr_s2_acc, 6),
            "val_loss":     round(val_loss,   6),
            "val_s1_acc":   round(val_s1_acc, 6),
            "val_s2_acc":   round(val_s2_acc, 6),
        })

        # Monitor Stage 2 val acc for early stopping
        if val_s2_acc > best_val_acc:
            best_val_acc     = val_s2_acc
            best_state       = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best S2 val acc: {best_val_acc:.4f}")
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
# 7. INFERENCE — hard decision, one CSV per video
# ============================================================
@torch.no_grad()
def run_inference_per_video(probe, splits_csv, task_column,
                             stage1_map, stage2_map, device, out_dir):
    print("\n" + "=" * 60)
    print("INFERENCE — per-frame hierarchical, one CSV per video")
    print("=" * 60)

    probe.eval()
    probe       = probe.to(device)
    id_to_s2    = {v: k for k, v in stage2_map.items()}
    softmax     = nn.Softmax(dim=1)

    pred_dir = os.path.join(out_dir, "per_video_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    df = pd.read_csv(splits_csv)
    df = df[df["split"] == "test"]

    all_true, all_pred = [], []

    for _, row in df.iterrows():
        video_name = os.path.splitext(
            os.path.basename(row["vjpe_features_full_video_vit_h_features"])
        )[0]

        frames_raw = build_frames_from_video(
            row["vjpe_features_full_video_vit_h_features"],
            row["label_path"],
            task_column,
        )
        if not frames_raw:
            continue

        vid_dataset = FrameDataset(frames_raw, stage1_map, stage2_map)
        vid_loader  = DataLoader(
            vid_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        rows = []
        for windows, _s1_lbls, _s2_lbls, lbl_strs, frame_idxs in vid_loader:
            windows = windows.to(device)
            l1, l2  = probe(windows)
            p1      = softmax(l1)
            p2      = softmax(l2)

            for i in range(windows.shape[0]):
                s1_pred   = p1[i].argmax().item()   # 0=None, 1=NotNone
                s1_conf   = p1[i, s1_pred].item()
                true_lbl  = lbl_strs[i]

                if s1_pred == 0:
                    final_pred = "None"
                    s2_pred_str = "N/A"
                    s2_conf     = "N/A"
                else:
                    s2_pred     = p2[i].argmax().item()
                    s2_conf     = round(p2[i, s2_pred].item(), 4)
                    final_pred  = id_to_s2[s2_pred]
                    s2_pred_str = final_pred

                rows.append({
                    "frame":           frame_idxs[i].item(),
                    "true_label":      true_lbl,
                    "predicted_label": final_pred,
                    "stage1_pred":     "None" if s1_pred == 0 else "NotNone",
                    "stage1_conf":     round(s1_conf, 4),
                    "stage2_pred":     s2_pred_str,
                    "stage2_conf":     s2_conf,
                    "correct":         int(true_lbl == final_pred),
                })
                all_true.append(true_lbl)
                all_pred.append(final_pred)

        vid_df  = pd.DataFrame(rows).sort_values("frame")
        out_csv = os.path.join(pred_dir, f"{video_name}_predictions.csv")
        vid_df.to_csv(out_csv, index=False)
        vid_acc = vid_df["correct"].mean()
        print(f"  {video_name} | frames={len(vid_df)} | acc={vid_acc:.4f}")

    # ── Aggregate metrics ──────────────────────────────────
    print("\n── Aggregate Test Metrics ──")
    overall_acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"Overall Accuracy: {overall_acc:.4f}  ({len(all_true)} frames)")

    # Stage 1 aggregate
    true_s1 = ["None" if t == "None" else "NotNone" for t in all_true]
    pred_s1 = ["None" if p == "None" else "NotNone" for p in all_pred]
    print("\n── Stage 1 (None vs NotNone) ──")
    print(classification_report(true_s1, pred_s1, zero_division=0))

    # Stage 2 aggregate (NotNone frames only)
    nn_true = [t for t in all_true if t != "None"]
    nn_pred = [all_pred[i] for i, t in enumerate(all_true) if t != "None"]
    print("── Stage 2 (action classes) ──")
    s2_report   = classification_report(nn_true, nn_pred, zero_division=0)
    labels_list = sorted(set(nn_true))
    cm          = confusion_matrix(nn_true, nn_pred, labels=labels_list)
    cm_df       = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    print(s2_report)
    print("Confusion Matrix:")
    print(cm_df)

    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Overall Accuracy : {overall_acc:.4f}\n\n")
        f.write("── Stage 1 ──\n")
        f.write(classification_report(true_s1, pred_s1, zero_division=0))
        f.write("\n── Stage 2 ──\n")
        f.write(s2_report)
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
    print(f"Task       : {args.task}  ({task_column})")
    print(f"Seed       : {seed}")
    print(f"Device     : {device}")
    print(f"Output dir : {out_dir}")
    print(f"Context    : ±{CONTEXT} frames → window of {WINDOW_SIZE}")

    # ── Build frames ───────────────────────────────────────
    print("\nBuilding train frames...")
    train_raw = build_all_frames(SPLITS_CSV, task_column, ["train"])
    print("Building val frames...")
    val_raw   = build_all_frames(SPLITS_CSV, task_column, ["val"])

    stage1_map, stage2_map = build_label_maps(train_raw)
    num_s2 = len(stage2_map)

    print(f"\nStage1 map : {{None:0, NotNone:1}}")
    print(f"Stage2 map : {stage2_map}  ({num_s2} classes)")

    with open(os.path.join(out_dir, "label_mapping.json"), "w") as f:
        json.dump({"stage1": stage1_map, "stage2": stage2_map}, f, indent=2)

    train_counts = Counter(lbl for _, lbl, _ in train_raw)
    print(f"\nTrain class distribution:\n{train_counts}")

    train_loader = DataLoader(
        FrameDataset(train_raw, stage1_map, stage2_map),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        FrameDataset(val_raw, stage1_map, stage2_map),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    print(f"\nTrain frames : {len(train_raw)}")
    print(f"Val frames   : {len(val_raw)}")

    # ── Model ──────────────────────────────────────────────
    probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=num_s2)
    print(f"\nProbe parameters : {sum(p.numel() for p in probe.parameters()):,}")

    # ── Train ──────────────────────────────────────────────
    probe = train_probe(probe, train_loader, val_loader, device, out_dir)
    torch.save(probe.state_dict(), os.path.join(out_dir, "best_probe.pt"))

    # ── Inference ──────────────────────────────────────────
    run_inference_per_video(
        probe, SPLITS_CSV, task_column,
        stage1_map, stage2_map, device, out_dir
    )

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()