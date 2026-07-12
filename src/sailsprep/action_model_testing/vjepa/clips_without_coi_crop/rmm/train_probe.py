"""
VJEPA2 ViT-G — Attentive Probe Training — RMM (Per Seed)
=========================================================
Loads pre-extracted RMM features (shared across seeds), does train/test split
with the given seed, trains an AttentiveProbe, saves results.

Classes: hands_flapping / jumping / rocking / spinning

Usage:
    python train_probe.py --seed 42
    python train_probe.py --seed 456
    python train_probe.py --seed 123

Output per seed:
    /orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/rmm_vjepa/
    └── seed_42/
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
from clips_without_coi_crop.common.probes import AttentiveProbe
from clips_without_coi_crop.common.datasets import FeatureDataset

# ============================================================
# CONFIG
# ============================================================
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/rmm_vjepa/"
FEAT_PATH     = os.path.join(OUTPUT_BASE, "extracted_features.pt")
META_PATH     = os.path.join(OUTPUT_BASE, "dataset_meta.json")

EMBED_DIM     = 1408       # ViT-G
NUM_CLASSES   = 4          # hands_flapping, jumping, rocking, spinning
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.05
TEST_SPLIT    = 0.30
PATIENCE      = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. TRAINING LOOP
# ============================================================
def train_probe(probe, train_loader, val_loader, device, seed_dir):
    probe     = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", flush=True)

        log_rows.append({
            "epoch"     : epoch + 1,
            "train_loss": round(tr_loss, 6),
            "train_acc" : round(tr_acc, 6),
            "val_loss"  : round(val_loss, 6),
            "val_acc"   : round(val_acc, 6),
        })

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_state       = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best val accuracy: {best_val_acc:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    pd.DataFrame(log_rows).to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)

    if best_state:
        probe.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_val_acc:.4f}", flush=True)
    return probe


# ============================================================
# 4. INFERENCE & METRICS
# ============================================================
@torch.no_grad()
def run_inference(probe, test_features, test_labels_enc, original_labels,
                  video_paths, label_map, device, seed_dir):
    print("\n" + "=" * 60)
    print("INFERENCE ON 30% TEST SET")
    print("=" * 60)

    probe.eval()
    probe       = probe.to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    softmax     = nn.Softmax(dim=1)

    dataset = FeatureDataset(
        test_features,
        torch.tensor(test_labels_enc, dtype=torch.long)
    )
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results, sample_idx = [], 0
    for feats, labels in loader:
        feats  = feats.to(device)
        logits = probe(feats)
        probs  = softmax(logits)
        for i in range(logits.shape[0]):
            top_pred  = probs[i].argmax().item()
            top_score = probs[i, top_pred].item()
            true_enc  = labels[i].item()
            results.append({
                "video_path"             : video_paths[sample_idx],
                "true_label"             : original_labels[sample_idx],
                "true_label_encoded"     : true_enc,
                "predicted_label"        : id_to_label[top_pred],
                "predicted_label_encoded": top_pred,
                "confidence"             : round(top_score, 4),
                "correct"                : int(top_pred == true_enc),
            })
            sample_idx += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(seed_dir, "predictions.csv"), index=False)

    accuracy = results_df["correct"].mean()
    print(f"\nOverall Accuracy : {accuracy:.4f}  "
          f"({int(results_df['correct'].sum())}/{len(results_df)})", flush=True)

    print("\nClassification Report:")
    report = classification_report(
        results_df["true_label"], results_df["predicted_label"], zero_division=0
    )
    print(report)

    labels_list = sorted(results_df["true_label"].unique())
    cm    = confusion_matrix(results_df["true_label"],
                             results_df["predicted_label"], labels=labels_list)
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    print("Confusion Matrix:")
    print(cm_df)

    metrics_path = os.path.join(seed_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Model      : facebook/vjepa2-vitg-fpc64-256\n")
        f.write(f"Probe      : AttentiveProbe\n")
        f.write(f"Accuracy   : {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
    print(f"Metrics saved to: {metrics_path}", flush=True)
    return results_df


# ============================================================
# 5. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed (42, 456, or 123)")
    args = parser.parse_args()
    seed = args.seed

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seed_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Seed        : {seed}",       flush=True)
    print(f"Device      : {device}",     flush=True)
    print(f"Output dir  : {seed_dir}",   flush=True)

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

    print(f"Features shape : {features.shape}", flush=True)
    print(f"Total clips    : {len(labels)}",    flush=True)
    print(f"Label map      : {label_map}",      flush=True)

    with open(os.path.join(seed_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # --- Train/test split with this seed ---
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

    # Save test split info
    test_df = pd.DataFrame({
        "video_path"   : [video_paths[i] for i in test_idx],
        "label"        : [original_labels[i] for i in test_idx],
        "label_encoded": test_labels.tolist(),
    })
    test_df.to_csv(os.path.join(seed_dir, "test_split.csv"), index=False)

    print(f"\nTrain: {len(train_idx)} | Test: {len(test_idx)} clips", flush=True)

    # --- Train probe ---
    print("\n" + "=" * 60)
    print(f"TRAINING PROBE  (seed={seed})")
    print("=" * 60, flush=True)

    probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=NUM_CLASSES)
    print(f"Probe parameters: {sum(p.numel() for p in probe.parameters()):,}", flush=True)

    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        FeatureDataset(test_features, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    probe = train_probe(probe, train_loader, val_loader, device, seed_dir)

    probe_path = os.path.join(seed_dir, "best_probe.pt")
    torch.save(probe.state_dict(), probe_path)
    print(f"Probe saved to: {probe_path}", flush=True)

    # --- Inference ---
    run_inference(
        probe           = probe,
        test_features   = test_features,
        test_labels_enc = test_labels.tolist(),
        original_labels = [original_labels[i] for i in test_idx],
        video_paths     = [video_paths[i] for i in test_idx],
        label_map       = label_map,
        device          = device,
        seed_dir        = seed_dir,
    )

    print(f"\nSeed {seed} complete! Results in: {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
