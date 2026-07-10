"""
SlowFast R50 Fine-tuning for Locomotion Classification
Uses master CSV for train/val/test splits + inverse-frequency class weights.

Run: python slowfast_finetune.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import glob
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIG
# ============================================================
# Master CSV for splits
MASTER_CSV = os.environ.get(
    "MASTER_CSV",
    "/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v3_balanced.csv"
)
CLIP_COL = "cut_clip_path"
SPLIT_COL = "split"

# Classes: CSV folder names -> internal names
CSV_CLASS_TO_INTERNAL = {
    "Walking": "walk",
    "Cruising": "cruise",
    "Crawling": "crawl",
    "Vehicle": "vehicle",
    "Running": "run",
}
ACTION_CLASSES = ["walk", "cruise", "crawl", "vehicle", "run"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

NUM_CLASSES = len(ACTION_CLASSES)
BATCH_SIZE = 4
NUM_WORKERS = 4
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# Save paths
MODEL_SAVE_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/output_finetune_clip/"
)
OUTPUT_CSV = os.path.join(MODEL_SAVE_DIR, "finetune_clip_predictions.csv")

# SlowFast-specific params
NUM_FRAMES = 32
SAMPLING_RATE = 2
FPS = 30
ALPHA = 4
SIDE_SIZE = 256
CROP_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ============================================================
# 1. PACK PATHWAY (SlowFast-specific)
# ============================================================
class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames, 1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // ALPHA).long(),
        )
        return [slow_pathway, fast_pathway]


# ============================================================
# 2. LOAD SPLITS FROM MASTER CSV
# ============================================================
def load_splits_from_csv(master_csv, clip_col, split_col):
    """
    Read master CSV and return splits dict with 'train', 'val', 'test' DataFrames.
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

    # Encode labels
    df["label_encoded"] = df["class_name"].map(CLASS_TO_IDX)
    df["video_path"] = df[clip_col]

    # Split
    df_train = df[df[split_col] == "train"].reset_index(drop=True)
    df_val = df[df[split_col] == "val"].reset_index(drop=True)
    df_test = df[df[split_col] == "test"].reset_index(drop=True)

    # Print summary
    print(f"\n  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("  " + "-" * 44)
    for cls in ACTION_CLASSES:
        n_tr = len(df_train[df_train["class_name"] == cls])
        n_va = len(df_val[df_val["class_name"] == cls])
        n_te = len(df_test[df_test["class_name"] == cls])
        print(f"  {cls:<12} {n_tr:>8} {n_va:>8} {n_te:>8} {n_tr+n_va+n_te:>8}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL':<12} {len(df_train):>8} {len(df_val):>8} {len(df_test):>8} "
          f"{len(df_train)+len(df_val)+len(df_test):>8}")

    # Verify no overlap
    train_paths = set(df_train["video_path"])
    val_paths = set(df_val["video_path"])
    test_paths = set(df_test["video_path"])
    assert len(train_paths & val_paths) == 0, "Train/Val overlap!"
    assert len(train_paths & test_paths) == 0, "Train/Test overlap!"
    assert len(val_paths & test_paths) == 0, "Val/Test overlap!"
    print("  Overlap check: PASSED")

    return df_train, df_val, df_test


# ============================================================
# 3. COMPUTE CLASS WEIGHTS
# ============================================================
def compute_class_weights(df_train, num_classes):
    """Inverse-frequency class weights from training set."""
    label_counts = Counter(df_train["label_encoded"].values)
    total = len(df_train)

    weights = []
    print("\n  Class weights (inverse frequency):")
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        w = total / (num_classes * count)
        weights.append(w)
        cls_name = IDX_TO_CLASS[i]
        print(f"    {cls_name:<12} count={count:>6}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 4. CUSTOM VIDEO DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            video = EncodedVideo.from_path(video_path, decode_audio=False, decoder="decord")
            video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)

            if video_data["video"] is None:
                raise ValueError("Decoder returned None")

            if self.transform:
                video_data = self.transform(video_data)

            return video_data["video"], label

        except Exception as e:
            # Try next valid video instead of dummy tensor
            for attempt in range(10):
                fallback_idx = (idx + attempt + 1) % len(self.video_paths)
                try:
                    fb_video = EncodedVideo.from_path(self.video_paths[fallback_idx], decode_audio=False, decoder="decord")
                    fb_data = fb_video.get_clip(start_sec=0, end_sec=CLIP_DURATION)
                    if fb_data["video"] is None:
                        continue
                    if self.transform:
                        fb_data = self.transform(fb_data)
                    return fb_data["video"], self.labels[fallback_idx]
                except:
                    continue
            # Should never reach here, but just in case
            dummy = [torch.zeros(3, NUM_FRAMES // ALPHA, CROP_SIZE, CROP_SIZE),
                     torch.zeros(3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)]
            return dummy, label


# ============================================================
# 5. CUSTOM COLLATE
# ============================================================
def slowfast_collate(batch):
    videos, labels = zip(*batch)
    slow = torch.stack([v[0] for v in videos])
    fast = torch.stack([v[1] for v in videos])
    labels = torch.tensor(labels, dtype=torch.long)
    return [slow, fast], labels


# ============================================================
# 6. DATA MODULE (from master CSV)
# ============================================================
class ClipDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.label_map = CLASS_TO_IDX
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self, stage=None):
        self.train_df, self.val_df, self.test_df = load_splits_from_csv(
            MASTER_CSV, CLIP_COL, SPLIT_COL
        )

        # Save label mapping and split info
        map_path = os.path.join(MODEL_SAVE_DIR, "label_mapping.json")
        with open(map_path, "w") as f:
            json.dump(self.label_map, f, indent=2)

        self.test_df.to_csv(os.path.join(MODEL_SAVE_DIR, "test_split.csv"), index=False)

        split_info = {
            "master_csv": MASTER_CSV,
            "train": len(self.train_df),
            "val": len(self.val_df),
            "test": len(self.test_df),
        }
        with open(os.path.join(MODEL_SAVE_DIR, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # Train transforms (with augmentation)
        self.train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(NUM_FRAMES),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(MEAN, STD),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCropVideo(CROP_SIZE),
                RandomHorizontalFlipVideo(p=0.5),
                PackPathway(),
            ]),
        )

        # Val/Test transforms (no augmentation)
        self.eval_transform = ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(NUM_FRAMES),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(MEAN, STD),
                ShortSideScale(size=SIDE_SIZE),
                CenterCropVideo(CROP_SIZE),
                PackPathway(),
            ]),
        )

    def train_dataloader(self):
        ds = VideoDataset(
            self.train_df["video_path"].tolist(),
            self.train_df["label_encoded"].tolist(),
            transform=self.train_transform,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=slowfast_collate)

    def val_dataloader(self):
        ds = VideoDataset(
            self.val_df["video_path"].tolist(),
            self.val_df["label_encoded"].tolist(),
            transform=self.eval_transform,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=slowfast_collate)


# ============================================================
# 7. LIGHTNING MODULE (Fine-tuning with class weights)
# ============================================================
class SlowFastFineTune(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.class_weights = class_weights

        # Load pretrained SlowFast
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=True,
        )

        # Replace the final classification head
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            print("Freezing backbone - only training the classification head")
            for name, param in self.model.named_parameters():
                if "blocks.6" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        # Weighted loss for training
        if self.class_weights is not None:
            weight = self.class_weights.to(preds.device)
            loss = F.cross_entropy(preds, labels, weight=weight)
        else:
            loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        # Unweighted loss for fair validation
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ============================================================
# 8. INFERENCE (TEST SET ONLY)
# ============================================================
def run_inference(model, test_df, device):
    """Run inference ONLY on the held-out test set (never seen during training)."""
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE ON TEST SET (held-out, never seen during training)")
    print("=" * 60)

    model.eval()
    model = model.to(device)

    eval_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(MEAN, STD),
            ShortSideScale(size=SIDE_SIZE),
            CenterCropVideo(CROP_SIZE),
            PackPathway(),
        ]),
    )

    video_paths = test_df["video_path"].tolist()
    true_labels = test_df["label_encoded"].tolist()
    original_labels = test_df["class_name"].tolist()

    softmax = torch.nn.Softmax(dim=1)
    results = []

    for idx, video_path in enumerate(video_paths):
        print(f"[{idx+1}/{len(video_paths)}] Predicting: {os.path.basename(video_path)}")

        try:
            video = EncodedVideo.from_path(video_path, decode_audio=False, decoder="decord")
            video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)

            # If clip is too short, use full video duration
            if video_data["video"] is None:
                vid_duration = video.duration
                if vid_duration and vid_duration > 0:
                    video_data = video.get_clip(start_sec=0, end_sec=vid_duration)

            if video_data["video"] is None:
                raise ValueError("Decoder returned None")

            video_data = eval_transform(video_data)
            inputs = video_data["video"]
            inputs = [i.to(device)[None, ...] for i in inputs]

            with torch.no_grad():
                preds = model(inputs)

            preds = softmax(preds)
            top_pred = preds.argmax(dim=1).item()
            top_score = preds[0, top_pred].item()
            pred_label = IDX_TO_CLASS[top_pred]

            results.append({
                "video_path": video_path,
                "true_label": original_labels[idx],
                "true_label_encoded": true_labels[idx],
                "predicted_label": pred_label,
                "predicted_label_encoded": top_pred,
                "confidence": round(top_score, 4),
                "correct": int(top_pred == true_labels[idx]),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "video_path": video_path,
                "true_label": original_labels[idx],
                "true_label_encoded": true_labels[idx],
                "predicted_label": "ERROR",
                "predicted_label_encoded": -1,
                "confidence": 0.0,
                "correct": 0,
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to {OUTPUT_CSV}")

    # Print metrics
    valid = results_df[results_df["predicted_label"] != "ERROR"]
    if len(valid) > 0:
        accuracy = valid["correct"].mean()
        print(f"\n{'=' * 60}")
        print(f"TEST SET RESULTS")
        print(f"{'=' * 60}")
        print(f"Total test clips: {len(results_df)}")
        print(f"Successfully processed: {len(valid)}")
        print(f"Errors: {len(results_df) - len(valid)}")
        print(f"Overall Accuracy: {accuracy:.4f} ({int(valid['correct'].sum())}/{len(valid)})")

        print(f"\nClassification Report:")
        print(classification_report(
            valid["true_label"], valid["predicted_label"], zero_division=0
        ))

        print(f"Confusion Matrix:")
        labels = sorted(valid["true_label"].unique())
        cm = confusion_matrix(valid["true_label"], valid["predicted_label"], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df)

        metrics_path = os.path.join(MODEL_SAVE_DIR, "test_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Master CSV: {MASTER_CSV}\n")
            f.write(f"Train: {len(test_df)} test clips\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                valid["true_label"], valid["predicted_label"], zero_division=0
            ))
            f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
        print(f"Metrics saved to {metrics_path}")

    return results_df


# ============================================================
# 9. MAIN: TRAIN + INFERENCE
# ============================================================
def main():
    print(f"[CONFIG] MASTER_CSV: {MASTER_CSV}")
    print(f"[CONFIG] OUTPUT_DIR: {MODEL_SAVE_DIR}")

    data_module = ClipDataModule()
    data_module.setup()

    # Compute class weights from training set
    class_weights = compute_class_weights(data_module.train_df, NUM_CLASSES)

    # Save weights
    weights_path = os.path.join(MODEL_SAVE_DIR, "class_weights.pt")
    torch.save(class_weights, weights_path)
    print(f"  [OK] Saved class weights to {weights_path}")

    model = SlowFastFineTune(
        num_classes=NUM_CLASSES,
        freeze_backbone=FREEZE_BACKBONE,
        class_weights=class_weights,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="slowfast-clip-{epoch:02d}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    best_path = checkpoint_cb.best_model_path
    print(f"\nBest model saved at: {best_path}")

    # Load best model for inference
    print("\nLoading best model for inference...")
    best_model = SlowFastFineTune.load_from_checkpoint(
        best_path,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference on TEST SET ONLY (never seen during training or validation)
    run_inference(
        model=best_model,
        test_df=data_module.test_df,
        device=device,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()