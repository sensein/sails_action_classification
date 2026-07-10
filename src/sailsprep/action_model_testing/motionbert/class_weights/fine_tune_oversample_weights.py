"""
MotionBERT Pipeline: 2D Pose Extraction -> 3D Pose Lifting -> Action Recognition
=================================================================================

Uses master CSV for train/val/test splits instead of random splitting.

Run: python pipeline.py --step all --device cuda
"""

import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================



# Pose outputs (2D/3D .npy files)
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    "/orcd/data/satra/002/projects/SAILS/feature_processing/pipeline_outputs/single_child_videos_motion_bert/"
)

# Action recognition outputs (checkpoints, predictions, logs)
ACTION_OUTPUT_ROOT = os.environ.get(
    "ACTION_OUTPUT_ROOT",
    "/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/output_no_weights"
)

# Master CSV for splits
MASTER_CSV = os.environ.get(
    "MASTER_CSV",
    "/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v3_balanced.csv"
)
CLIP_COL = "cut_clip_path"
SPLIT_COL = "split"

# Working directory (where MotionBERT repo is cloned and code runs from)
WORK_DIR = "/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert"
MOTIONBERT_ROOT = os.path.join(WORK_DIR, "MotionBERT")

# Classes: CSV label name -> internal name
# CSV uses folder names like "Walking", code uses lowercase "walk"
CSV_CLASS_TO_INTERNAL = {
    "Walking": "walk",
    "Cruising": "cruise",
    "Crawling": "crawl",
    "Vehicle": "vehicle",
    "Running": "run",
}
ACTION_CLASSES = ["walk", "cruise", "crawl", "vehicle", "run"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# Pose config
MAX_FRAMES = 243
NUM_KEYPOINTS = 17
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Training config
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-4
FINETUNE_BATCH_SIZE = 8
RANDOM_SEED = 42


def setup_output_dirs():
    """Create output directory structure.
    Pose outputs go to OUTPUT_ROOT, action recognition outputs go to ACTION_OUTPUT_ROOT.
    """
    dirs = {
        # Pose outputs -> OUTPUT_ROOT
        "pose_2d": os.path.join(OUTPUT_ROOT, "pose_2d"),
        "pose_3d": os.path.join(OUTPUT_ROOT, "pose_3d"),
        # Action recognition outputs -> ACTION_OUTPUT_ROOT
        "predictions": os.path.join(ACTION_OUTPUT_ROOT, "predictions"),
        "checkpoints": os.path.join(ACTION_OUTPUT_ROOT, "checkpoints"),
        "logs": os.path.join(ACTION_OUTPUT_ROOT, "logs"),
        "metadata": os.path.join(ACTION_OUTPUT_ROOT, "metadata"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================================
# LOAD SPLITS FROM MASTER CSV
# ============================================================================

def load_splits_from_csv(master_csv, clip_col, split_col):
    """
    Read master CSV and return splits dict with 'train', 'val', 'test' lists.
    Each entry has: path, label, class_name, filename, video_id.
    """
    print(f"\n[INFO] Loading splits from: {master_csv}")
    df = pd.read_csv(master_csv)
    print(f"  Total rows: {len(df)}")

    # Drop rows with missing clip paths
    df = df.dropna(subset=[clip_col])
    df = df[df[clip_col].str.strip() != ""]
    print(f"  Rows with valid clip paths: {len(df)}")

    # Extract class name from clip path (parent folder name)
    df["csv_label"] = df[clip_col].apply(lambda p: os.path.basename(os.path.dirname(p)))

    # Map CSV class names to internal names
    df["class_name"] = df["csv_label"].map(CSV_CLASS_TO_INTERNAL)
    df = df[df["class_name"].notna()]
    print(f"  Rows matching known classes: {len(df)}")

    # Build video info dicts
    splits = {"train": [], "val": [], "test": []}

    for _, row in df.iterrows():
        split = row[split_col]
        if split not in splits:
            continue

        cls_name = row["class_name"]
        clip_path = row[clip_col]
        filename = os.path.basename(clip_path)
        video_id = f"{cls_name}_{Path(filename).stem}"

        vid_info = {
            "path": clip_path,
            "label": CLASS_TO_IDX[cls_name],
            "class_name": cls_name,
            "filename": filename,
            "video_id": video_id,
        }
        splits[split].append(vid_info)

    # Print summary
    print(f"\n  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("  " + "-" * 44)
    for cls in ACTION_CLASSES:
        n_tr = sum(1 for v in splits["train"] if v["class_name"] == cls)
        n_va = sum(1 for v in splits["val"] if v["class_name"] == cls)
        n_te = sum(1 for v in splits["test"] if v["class_name"] == cls)
        print(f"  {cls:<12} {n_tr:>8} {n_va:>8} {n_te:>8} {n_tr+n_va+n_te:>8}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL':<12} {len(splits['train']):>8} {len(splits['val']):>8} {len(splits['test']):>8} "
          f"{len(splits['train'])+len(splits['val'])+len(splits['test']):>8}")

    # Verify no overlap
    train_ids = {v["video_id"] for v in splits["train"]}
    val_ids = {v["video_id"] for v in splits["val"]}
    test_ids = {v["video_id"] for v in splits["test"]}
    assert len(train_ids & val_ids) == 0, "Train/Val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train/Test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val/Test overlap detected!"
    print("  Overlap check: PASSED (no overlap between splits)")

    return splits


def get_all_videos(splits):
    """Get all videos across all splits (for pose extraction which processes everything)."""
    all_videos = []
    seen = set()
    for split_name in ["train", "val", "test"]:
        for v in splits[split_name]:
            if v["video_id"] not in seen:
                all_videos.append(v)
                seen.add(v["video_id"])
    return all_videos


# ============================================================================
# STEP 1: Extract 2D Poses using YOLO11-Pose
# ============================================================================

def extract_2d_poses(videos, output_dir_2d, device="cuda"):
    """
    Extract 2D keypoints from each video using YOLO11-Pose.
    Saves per-video .npy files with shape (T, 17, 3) -> [x, y, confidence].
    Maps YOLO's COCO 17 keypoints to H36M 17 keypoints for MotionBERT.
    """
    from ultralytics import YOLO

    print("\n" + "="*70)
    print("STEP 1: Extracting 2D Poses (YOLO11-Pose)")
    print("="*70)

    model = YOLO("yolo11n-pose.pt")

    COCO_TO_H36M = {
        0: 9, 5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
        11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
    }

    results_log = []

    for vid_info in tqdm(videos, desc="Extracting 2D poses"):
        video_path = vid_info["path"]
        video_id = vid_info["video_id"]
        out_path = os.path.join(output_dir_2d, f"{video_id}.npy")

        if os.path.exists(out_path):
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            h36m_kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kps_data = results[0].keypoints.data.cpu().numpy()
                if len(kps_data) > 0:
                    avg_conf = kps_data[:, :, 2].mean(axis=1)
                    best_person = kps_data[np.argmax(avg_conf)]

                    for coco_idx, h36m_idx in COCO_TO_H36M.items():
                        h36m_kps[h36m_idx] = best_person[coco_idx]

                    h36m_kps[0] = (h36m_kps[1] + h36m_kps[4]) / 2.0
                    h36m_kps[8] = (h36m_kps[11] + h36m_kps[14]) / 2.0
                    h36m_kps[7] = (h36m_kps[0] + h36m_kps[8]) / 2.0
                    h36m_kps[10] = h36m_kps[9].copy()
                    h36m_kps[10][1] -= 15

            all_keypoints.append(h36m_kps)

        cap.release()

        keypoints_array = np.array(all_keypoints, dtype=np.float32)
        np.save(out_path, keypoints_array)

        results_log.append({
            "video_id": video_id,
            "num_frames": len(all_keypoints),
            "fps": fps,
            "shape": list(keypoints_array.shape),
        })
        print(f"  [OK] {video_id}: {keypoints_array.shape}")

    return results_log


# ============================================================================
# STEP 2: Lift 2D Poses to 3D using MotionBERT
# ============================================================================

def lift_2d_to_3d(videos, dir_2d, dir_3d, device="cuda"):
    """
    Use MotionBERT pretrained pose estimation model to lift 2D -> 3D.
    """
    print("\n" + "="*70)
    print("STEP 2: Lifting 2D -> 3D Poses (MotionBERT)")
    print("="*70)

    sys.path.insert(0, MOTIONBERT_ROOT)
    from lib.utils.tools import get_config
    from lib.model.DSTformer import DSTformer

    cfg_path = os.path.join(MOTIONBERT_ROOT, "configs/pose3d/MB_ft_h36m.yaml")
    ckpt_path = os.path.join(MOTIONBERT_ROOT, "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        print("  Download with:")
        print("    cd MotionBERT")
        print("    huggingface-cli download walterzhu/MotionBERT checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin --local-dir .")
        return

    cfg = get_config(cfg_path)

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=cfg.dim_feat, dim_rep=cfg.dim_rep,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        norm_layer=nn.LayerNorm, maxlen=cfg.maxlen, num_joints=NUM_KEYPOINTS,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_pos"], strict=False)
    model.to(device)
    model.eval()

    for vid_info in tqdm(videos, desc="Lifting to 3D"):
        video_id = vid_info["video_id"]
        in_path = os.path.join(dir_2d, f"{video_id}.npy")
        out_path = os.path.join(dir_3d, f"{video_id}.npy")

        if os.path.exists(out_path):
            continue
        if not os.path.exists(in_path):
            print(f"  [WARN] 2D poses not found for {video_id}, skipping.")
            continue

        kps_2d = np.load(in_path)
        T = kps_2d.shape[0]

        xy = kps_2d[:, :, :2].copy()
        conf = kps_2d[:, :, 2:3].copy()
        root = xy[:, 0:1, :]
        xy = xy - root
        scale = np.abs(xy).max() + 1e-6
        xy = xy / scale
        input_2d = np.concatenate([xy, conf], axis=-1)

        all_3d = []
        for start in range(0, T, MAX_FRAMES):
            end = min(start + MAX_FRAMES, T)
            chunk = input_2d[start:end]
            chunk_len = chunk.shape[0]

            if chunk_len < MAX_FRAMES:
                pad = np.zeros((MAX_FRAMES - chunk_len, NUM_KEYPOINTS, 3), dtype=np.float32)
                chunk_padded = np.concatenate([chunk, pad], axis=0)
            else:
                chunk_padded = chunk

            inp = torch.FloatTensor(chunk_padded).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_3d = model(inp)
            pred_3d = pred_3d.squeeze(0).cpu().numpy()[:chunk_len]
            all_3d.append(pred_3d)

        kps_3d = np.concatenate(all_3d, axis=0)
        np.save(out_path, kps_3d)
        print(f"  [OK] {video_id}: {kps_3d.shape}")


# ============================================================================
# STEP 3: Action Recognition - Dataset, Finetuning, Inference
# ============================================================================

class SkeletonActionDataset(Dataset):
    """Dataset for skeleton-based action recognition with MotionBERT."""

    def __init__(self, samples, pose_dir, max_frames=243):
        self.samples = samples
        self.pose_dir = pose_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        npy_path = os.path.join(self.pose_dir, f"{s['video_id']}.npy")
        kps = np.load(npy_path).astype(np.float32)
        T = kps.shape[0]

        if T >= self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            kps = kps[indices]
        else:
            pad_len = self.max_frames - T
            pad = np.tile(kps[-1:], (pad_len, 1, 1))
            kps = np.concatenate([kps, pad], axis=0)

        root = kps[:, 0:1, :2]
        kps[:, :, :2] = kps[:, :, :2] - root
        scale = np.abs(kps[:, :, :2]).max() + 1e-6
        kps[:, :, :2] /= scale

        label = s["label"]
        return torch.FloatTensor(kps), torch.LongTensor([label]).squeeze()


def build_action_recognition_model(motionbert_root, num_classes, device, ckpt_path=None):
    """Build MotionBERT encoder + action recognition head."""
    sys.path.insert(0, motionbert_root)
    from lib.model.DSTformer import DSTformer

    encoder = DSTformer(
        dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
        num_heads=8, mlp_ratio=4, norm_layer=nn.LayerNorm,
        maxlen=MAX_FRAMES, num_joints=NUM_KEYPOINTS,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Loading pretrained encoder from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt.get("model", ckpt), strict=False)

    class ActionRecognitionModel(nn.Module):
        def __init__(self, encoder, dim_rep, num_classes):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj = nn.Linear(NUM_KEYPOINTS * 3, dim_rep)
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim_rep),
                nn.Dropout(0.3),
                nn.Linear(dim_rep, num_classes),
            )

        def forward(self, x):
            B, T, J, C = x.shape
            feat = self.encoder(x)
            feat = feat.reshape(B, T, -1)
            feat = feat.permute(0, 2, 1)
            feat = self.pool(feat).squeeze(-1)
            feat = self.proj(feat)
            out = self.classifier(feat)
            return out

    model = ActionRecognitionModel(encoder, dim_rep=512, num_classes=num_classes)
    model.to(device)
    return model


def compute_class_weights(train_samples, num_classes, action_classes):
    """
    Compute inverse-frequency class weights from training data.
    Same method as VideoMAE: weight_i = total / (num_classes * count_i)
    """
    from collections import Counter

    label_counts = Counter(s["label"] for s in train_samples)
    total = len(train_samples)

    weights = []
    print("\n  Class weights (inverse frequency):")
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # avoid div by zero
        w = total / (num_classes * count)
        weights.append(w)
        print(f"    {action_classes[i]:<12} count={count:>6}  weight={w:.4f}")

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor


def finetune_action_recognition(splits, pose_dir, dirs, device="cuda"):
    """
    Finetune MotionBERT for action recognition.
    Uses train split for training, val split for validation.
    Test split is NEVER touched here.
    Uses inverse-frequency class weights for imbalanced data.
    """
    print("\n" + "="*70)
    print("STEP 3a: Finetuning Action Recognition (with class weights)")
    print("="*70)

    train_samples = [s for s in splits["train"]
                     if os.path.exists(os.path.join(pose_dir, f"{s['video_id']}.npy"))]
    val_samples = [s for s in splits["val"]
                   if os.path.exists(os.path.join(pose_dir, f"{s['video_id']}.npy"))]

    print(f"  Train (with pose data): {len(train_samples)}")
    print(f"  Val   (with pose data): {len(val_samples)}")
    print(f"  Test  (held out):       {len(splits['test'])}")

    if len(train_samples) == 0:
        print("[ERROR] No training samples with pose data found. Run Steps 1-2 first.")
        return None

    # Compute class weights from training set
    class_weights = compute_class_weights(train_samples, len(ACTION_CLASSES), ACTION_CLASSES)
    class_weights = class_weights.to(device)

    # Save weights for reference
    weights_path = os.path.join(dirs["checkpoints"], "class_weights.pt")
    torch.save(class_weights.cpu(), weights_path)
    print(f"  [OK] Saved class weights to {weights_path}")

    train_ds = SkeletonActionDataset(train_samples, pose_dir, MAX_FRAMES)
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4)

    val_loader = None
    if len(val_samples) > 0:
        val_ds = SkeletonActionDataset(val_samples, pose_dir, MAX_FRAMES)
        val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4)

    # Build model
    pretrained_ckpt = os.path.join(MOTIONBERT_ROOT, "checkpoint/pretrain/MB_release/latest_epoch.bin")
    model = build_action_recognition_model(
        MOTIONBERT_ROOT, len(ACTION_CLASSES), device, pretrained_ckpt
    )

    # Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=0.01
    )
    # Weighted loss for training (handles class imbalance)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    best_val_acc = 0.0
    start_epoch = 0
    log_history = []

    # Resume from checkpoint if exists
    latest_ckpt_path = os.path.join(dirs["checkpoints"], "latest_action_model.pth")
    if os.path.exists(latest_ckpt_path):
        print(f"  [RESUME] Found existing checkpoint, resuming training...")
        resume_ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"]
        best_val_acc = resume_ckpt.get("best_val_acc", 0.0)
        print(f"  [RESUME] Resuming from epoch {start_epoch + 1}, best_val_acc={best_val_acc:.1f}%")

        log_path = os.path.join(dirs["logs"], "training_log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_history = json.load(f)

        if start_epoch >= FINETUNE_EPOCHS:
            print(f"  [SKIP] Training already completed ({start_epoch}/{FINETUNE_EPOCHS} epochs).")
            return model

    # Adjust frozen/unfrozen based on epoch
    if start_epoch >= 10:
        for param in model.encoder.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR if start_epoch < 10 else FINETUNE_LR * 0.1,
        weight_decay=0.01
    )
    # Weighted loss for training, unweighted for validation (fair evaluation)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion_val = nn.CrossEntropyLoss()  # unweighted for fair eval

    if os.path.exists(latest_ckpt_path) and "optimizer_state_dict" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  [WARN] Could not restore optimizer state: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1
    )

    for epoch in range(start_epoch, FINETUNE_EPOCHS):
        if epoch == 10:
            print("  [INFO] Unfreezing encoder for end-to-end finetuning.")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR * 0.1, weight_decay=0.01)

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_kps, batch_labels in train_loader:
            batch_kps = batch_kps.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_kps)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_kps.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_kps.size(0)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1) * 100

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_acc,
        }

        # --- Validate (unweighted loss for fair evaluation) ---
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch_kps, batch_labels in val_loader:
                    batch_kps = batch_kps.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = model(batch_kps)
                    loss = criterion_val(logits, batch_labels)
                    val_loss += loss.item() * batch_kps.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == batch_labels).sum().item()
                    val_total += batch_kps.size(0)
            val_acc = val_correct / max(val_total, 1) * 100
            log_entry["val_loss"] = val_loss / max(val_total, 1)
            log_entry["val_acc"] = val_acc
        else:
            val_acc = train_acc  # fallback if no val set

        log_history.append(log_entry)

        val_str = f"Val Acc: {val_acc:.1f}%" if val_loader else ""
        print(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS} | "
              f"Train Loss: {log_entry['train_loss']:.4f} Acc: {train_acc:.1f}% | {val_str}")

        # Save best model based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(dirs["checkpoints"], "best_action_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
                "class_mapping": CLASS_TO_IDX,
            }, best_ckpt_path)
            print(f"    -> Saved best model (val_acc={val_acc:.1f}%)")

        # Save latest checkpoint every epoch
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "class_mapping": CLASS_TO_IDX,
        }, latest_ckpt_path)

        log_path = os.path.join(dirs["logs"], "training_log.json")
        with open(log_path, "w") as f:
            json.dump(log_history, f, indent=2)

    # Save final model
    final_ckpt_path = os.path.join(dirs["checkpoints"], "final_action_model.pth")
    torch.save({
        "epoch": FINETUNE_EPOCHS,
        "model_state_dict": model.state_dict(),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "class_mapping": CLASS_TO_IDX,
    }, final_ckpt_path)

    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    return model


def run_action_inference(splits, pose_dir, dirs, device="cuda"):
    """
    Run action recognition inference ONLY on the held-out test set.
    """
    print("\n" + "="*70)
    print("STEP 3b: Action Recognition Inference (TEST SET ONLY)")
    print("="*70)

    test_samples = splits["test"]
    print(f"  Test samples: {len(test_samples)}")

    ckpt_path = os.path.join(dirs["checkpoints"], "best_action_model.pth")
    if not os.path.exists(ckpt_path):
        print("[ERROR] No finetuned model found. Run finetuning first.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_action_recognition_model(MOTIONBERT_ROOT, len(ACTION_CLASSES), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    results = []
    results_path = os.path.join(dirs["predictions"], "action_predictions.json")
    already_done = set()
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        already_done = {r["video_id"] for r in results}
        print(f"  [RESUME] Found {len(already_done)} existing predictions.")

    for vid_info in tqdm(test_samples, desc="Running action inference (test set)"):
        video_id = vid_info["video_id"]
        if video_id in already_done:
            continue

        npy_path = os.path.join(pose_dir, f"{video_id}.npy")
        if not os.path.exists(npy_path):
            print(f"  [WARN] Pose data not found for {video_id}, skipping.")
            continue

        kps = np.load(npy_path).astype(np.float32)
        T = kps.shape[0]

        if T >= MAX_FRAMES:
            indices = np.linspace(0, T - 1, MAX_FRAMES, dtype=int)
            kps = kps[indices]
        else:
            pad_len = MAX_FRAMES - T
            pad = np.tile(kps[-1:], (pad_len, 1, 1))
            kps = np.concatenate([kps, pad], axis=0)

        root = kps[:, 0:1, :2]
        kps[:, :, :2] -= root
        scale = np.abs(kps[:, :, :2]).max() + 1e-6
        kps[:, :, :2] /= scale

        inp = torch.FloatTensor(kps).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probs))

        result = {
            "video_id": video_id,
            "source_path": vid_info["path"],
            "ground_truth": vid_info["class_name"],
            "predicted_class": IDX_TO_CLASS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
            "correct": vid_info["class_name"] == IDX_TO_CLASS[pred_idx],
        }
        results.append(result)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n  Overall accuracy: {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r["ground_truth"]
        per_class[gt]["total"] += 1
        if r["correct"]:
            per_class[gt]["correct"] += 1

    print("\n  Per-class accuracy:")
    for cls_name in ACTION_CLASSES:
        stats = per_class[cls_name]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"    {cls_name:>10}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MotionBERT Pose + Action Recognition Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "pose2d", "pose3d", "finetune", "inference"],
                        help="Which step to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-2d-for-action", action="store_true",
                        help="Use 2D poses instead of 3D for action recognition")
    args = parser.parse_args()

    print(f"[CONFIG] Device:            {args.device}")
    print(f"[CONFIG] Pose Output:       {OUTPUT_ROOT}")
    print(f"[CONFIG] Action Output:     {ACTION_OUTPUT_ROOT}")
    print(f"[CONFIG] Master CSV:        {MASTER_CSV}")

    # Setup
    dirs = setup_output_dirs()

    # Load splits from master CSV
    splits = load_splits_from_csv(MASTER_CSV, CLIP_COL, SPLIT_COL)

    # All videos (for pose extraction — processes everything regardless of split)
    all_videos = get_all_videos(splits)
    print(f"\n[INFO] Total unique videos: {len(all_videos)}")

    # Save metadata
    metadata = {
        "action_classes": ACTION_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "csv_class_mapping": CSV_CLASS_TO_INTERNAL,
        "master_csv": MASTER_CSV,
        "num_videos": len(all_videos),
        "max_frames": MAX_FRAMES,
        "pose_output_root": OUTPUT_ROOT,
        "action_output_root": ACTION_OUTPUT_ROOT,
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
    }
    with open(os.path.join(dirs["metadata"], "dataset_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save split info
    split_record = {
        "train": [v["video_id"] for v in splits["train"]],
        "val": [v["video_id"] for v in splits["val"]],
        "test": [v["video_id"] for v in splits["test"]],
        "source": MASTER_CSV,
    }
    with open(os.path.join(dirs["metadata"], "data_splits.json"), "w") as f:
        json.dump(split_record, f, indent=2)

    # Pose dir for action recognition
    pose_dir_for_action = dirs["pose_2d"] if args.use_2d_for_action else dirs["pose_3d"]

    # Run steps (Steps 1 & 2 process ALL videos)
    if args.step in ("all", "pose2d"):
        extract_2d_poses(all_videos, dirs["pose_2d"], device=args.device)

    if args.step in ("all", "pose3d"):
        lift_2d_to_3d(all_videos, dirs["pose_2d"], dirs["pose_3d"], device=args.device)

    if args.step in ("all", "finetune"):
        if not any(os.path.exists(os.path.join(dirs["pose_3d"], f"{v['video_id']}.npy")) for v in all_videos):
            print("[INFO] No 3D poses found, using 2D poses for action recognition.")
            pose_dir_for_action = dirs["pose_2d"]
        finetune_action_recognition(splits, pose_dir_for_action, dirs, device=args.device)

    if args.step in ("all", "inference"):
        if not any(os.path.exists(os.path.join(pose_dir_for_action, f"{v['video_id']}.npy")) for v in all_videos):
            print("[INFO] No 3D poses found, using 2D poses for action recognition.")
            pose_dir_for_action = dirs["pose_2d"]
        run_action_inference(splits, pose_dir_for_action, dirs, device=args.device)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print(f"Pose results saved to:   {OUTPUT_ROOT}")
    print(f"Action results saved to: {ACTION_OUTPUT_ROOT}")
    print("="*70)


if __name__ == "__main__":
    main()