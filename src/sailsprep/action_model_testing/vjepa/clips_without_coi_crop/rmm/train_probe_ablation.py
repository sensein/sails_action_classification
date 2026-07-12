"""
VJEPA2 ViT-G — Ablation Probe Training — RMM (Per Seed, Per Head)
==================================================================
Loads pre-extracted RMM features (from extract_features.py),
does train/test split with the given seed, trains the chosen
classification head, saves results.

Classes: hands_flapping / jumping / rocking / spinning

Usage:
    python train_probe_ablation.py --seed 42  --head attentive
    python train_probe_ablation.py --seed 456 --head transformer
    python train_probe_ablation.py --seed 123 --head linear

Output per (seed, head):
    .../rmm_vjepa/seed_42/attentive/
        ├── best_probe.pt
        ├── label_mapping.json
        ├── test_split.csv
        ├── predictions.csv
        ├── training_log.csv
        └── test_metrics.txt
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from clips_without_coi_crop.common.datasets import FeatureDataset

# ============================================================
# CONFIG  (matches RMM train_probe.py)
# ============================================================
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/rmm_vjepa/"
FEAT_PATH     = os.path.join(OUTPUT_BASE, "extracted_features.pt")
META_PATH     = os.path.join(OUTPUT_BASE, "dataset_meta.json")

EMBED_DIM     = 1408
NUM_CLASSES   = 4          # hands_flapping, jumping, rocking, spinning
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
TEST_SPLIT    = 0.30
PATIENCE      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_CHOICES  = ["linear", "mlp_small", "mlp_large", "attentive", "transformer"]


# ============================================================
# 1. ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="VJEPA2 RMM ablation probe — one seed, one head"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed (e.g. 42, 456, 123)")
    parser.add_argument("--head", type=str, required=True,
                        choices=HEAD_CHOICES,
                        help="Classification head to train")
    return parser.parse_args()


# ============================================================
# 2. CLASSIFICATION HEADS
# ============================================================

class LinearProbe(nn.Module):
    """
    Mean-pool all patch tokens -> LayerNorm -> single Linear layer.
    Simplest baseline; tests linear separability of VJEPA features.
    """
    def __init__(self, embed_dim, num_classes, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        return self.head(self.norm(x.mean(dim=1)))


class MLPSmallProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> 512-dim hidden -> GELU -> Dropout -> Linear.
    Adds nonlinearity over LinearProbe with minimal parameters.
    """
    def __init__(self, embed_dim, num_classes, hidden=512, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))


class MLPLargeProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> 1024 -> GELU -> 512 -> GELU -> Dropout -> Linear.
    Deeper MLP with more capacity.
    """
    def __init__(self, embed_dim, num_classes, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))


class AttentiveProbe(nn.Module):
    """
    Cross-attention: a single learned query attends over all patch tokens.
    Same as the original RMM train_probe.py head.
    """
    def __init__(self, embed_dim, num_classes, num_heads=8, **kwargs):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm  = nn.LayerNorm(embed_dim)
        self.head  = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.head(self.norm(out).squeeze(1))


class TransformerProbe(nn.Module):
    """
    Prepend a learnable CLS token, run 2-layer Transformer encoder,
    classify from CLS output. Most powerful head.
    """
    def __init__(self, embed_dim, num_classes, num_heads=8,
                 num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)        # [B, N+1, D]
        x   = self.transformer(x)
        return self.head(self.norm(x[:, 0]))     # CLS token


# ---- Factory -------------------------------------------------------
def build_probe(head_name, embed_dim, num_classes):
    registry = {
        "linear":      LinearProbe,
        "mlp_small":   MLPSmallProbe,
        "mlp_large":   MLPLargeProbe,
        "attentive":   AttentiveProbe,
        "transformer": TransformerProbe,
    }
    probe    = registry[head_name](embed_dim=embed_dim, num_classes=num_classes)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  Head '{head_name}': {n_params:,} parameters", flush=True)
    return probe


# ============================================================
# 4. TRAINING LOOP
# ============================================================
def train_probe(probe, train_loader, val_loader, device, out_dir, head_name):
    probe     = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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

        tr_acc   = tr_correct  / tr_total
        val_acc  = val_correct / val_total
        tr_loss  /= tr_total
        val_loss /= val_total

        print(f"  [{head_name}] Ep {epoch+1:3d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}", flush=True)

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
            print(f"  [{head_name}] -> best val acc: {best_val_acc:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{head_name}] Early stop at epoch {epoch+1}", flush=True)
                break

    pd.DataFrame(log_rows).to_csv(
        os.path.join(out_dir, "training_log.csv"), index=False
    )

    if best_state:
        probe.load_state_dict(best_state)
    print(f"  [{head_name}] Best val acc: {best_val_acc:.4f}", flush=True)
    return probe, best_val_acc


# ============================================================
# 5. INFERENCE
# ============================================================
@torch.no_grad()
def run_inference(probe, test_features, test_labels_enc, original_labels,
                  video_paths, label_map, device, out_dir, head_name):
    print("\n" + "=" * 60)
    print(f"INFERENCE ON 30% TEST SET  [{head_name}]")
    print("=" * 60)

    probe.eval().to(device)
    id2lab  = {v: k for k, v in label_map.items()}
    softmax = nn.Softmax(dim=1)

    dataset = FeatureDataset(
        test_features,
        torch.tensor(test_labels_enc, dtype=torch.long)
    )
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results, idx = [], 0
    for feats, labels in loader:
        probs = softmax(probe(feats.to(device)))
        for i in range(probs.shape[0]):
            top  = int(probs[i].argmax().item())
            conf = float(probs[i, top].item())
            results.append({
                "video_path":              video_paths[idx],
                "true_label":              original_labels[idx],
                "true_label_encoded":      labels[i].item(),
                "predicted_label":         id2lab[top],
                "predicted_label_encoded": top,
                "confidence":              round(conf, 4),
                "correct":                 int(id2lab[top] == original_labels[idx]),
            })
            idx += 1

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    acc = df["correct"].mean()
    print(f"\n[{head_name}] Accuracy: {acc:.4f}  "
          f"({int(df['correct'].sum())}/{len(df)})", flush=True)

    report = classification_report(df["true_label"], df["predicted_label"],
                                   zero_division=0)
    print(report)

    labs = sorted(df["true_label"].unique())
    cm   = pd.DataFrame(
        confusion_matrix(df["true_label"], df["predicted_label"], labels=labs),
        index=labs, columns=labs,
    )
    print("Confusion Matrix:\n", cm)

    with open(os.path.join(out_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Head     : {head_name}\n")
        f.write(f"Model    : facebook/vjepa2-vitg-fpc64-256\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm.to_string()}\n")

    return acc


# ============================================================
# 6. MAIN
# ============================================================
def main():
    args = parse_args()
    seed = args.seed
    head = args.head

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output dir: .../rmm_vjepa/seed_42/attentive/
    seed_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}", head)
    os.makedirs(seed_dir, exist_ok=True)

    device = torch.device(DEVICE)
    print("=" * 60,                          flush=True)
    print(f"  RMM Ablation",                 flush=True)
    print(f"  Seed   : {seed}",              flush=True)
    print(f"  Head   : {head}",              flush=True)
    print(f"  Device : {device}",            flush=True)
    print(f"  Output : {seed_dir}",          flush=True)
    print("=" * 60,                          flush=True)

    # --- Load shared features ---
    print(f"\nLoading features from: {FEAT_PATH}", flush=True)
    saved    = torch.load(FEAT_PATH, map_location="cpu")
    features = saved["features"]   # [N, N_tokens, 1408]
    labels   = saved["labels"]     # [N]

    with open(META_PATH) as f:
        meta = json.load(f)

    video_paths     = meta["video_paths"]
    original_labels = meta["labels"]
    label_map       = meta["label_map"]

    print(f"  Features : {features.shape}", flush=True)
    print(f"  Clips    : {len(labels)}",    flush=True)
    print(f"  Labels   : {label_map}",      flush=True)

    with open(os.path.join(seed_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # --- Train / test split (identical to original train_probe.py) ---
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SPLIT,
        random_state=seed,
        stratify=labels.numpy(),
    )

    train_features = features[train_idx]
    train_labels   = labels[train_idx]
    test_features  = features[test_idx]
    test_labels    = labels[test_idx]

    pd.DataFrame({
        "video_path":    [video_paths[i]     for i in test_idx],
        "label":         [original_labels[i] for i in test_idx],
        "label_encoded": test_labels.tolist(),
    }).to_csv(os.path.join(seed_dir, "test_split.csv"), index=False)

    print(f"\n  Train: {len(train_idx)}  |  Test: {len(test_idx)} clips", flush=True)

    # --- Build & train probe ---
    probe = build_probe(head, EMBED_DIM, NUM_CLASSES)

    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
    )
    val_loader = DataLoader(
        FeatureDataset(test_features, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    probe, best_val_acc = train_probe(
        probe, train_loader, val_loader, device, seed_dir, head
    )
    torch.save(probe.state_dict(), os.path.join(seed_dir, "best_probe.pt"))

    # --- Inference ---
    run_inference(
        probe           = probe,
        test_features   = test_features,
        test_labels_enc = test_labels.tolist(),
        original_labels = [original_labels[i] for i in test_idx],
        video_paths     = [video_paths[i]     for i in test_idx],
        label_map       = label_map,
        device          = device,
        out_dir         = seed_dir,
        head_name       = head,
    )

    print(f"\nDone — seed={seed}  head={head}", flush=True)
    print(f"Results saved to: {seed_dir}",      flush=True)


if __name__ == "__main__":
    main()