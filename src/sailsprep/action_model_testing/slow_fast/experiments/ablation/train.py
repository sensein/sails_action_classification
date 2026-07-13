"""
SlowFast Ablation Study — All Versions in One Script
=====================================================
Run with: python train.py --version v2

Versions:
  v1  : Baseline (frozen backbone, no class weights, LR=1e-4) — already done
  v2  : Class-weighted loss
  v3  : Unfreeze backbone (full fine-tune), LR=1e-5
  v4  : Oversample minority classes
  v5  : Class weights + Unfreeze backbone
  v6  : Class weights + Oversample + Unfreeze backbone
  v7  : Higher LR (1e-3) frozen backbone
  v8  : Larger batch size (8) frozen backbone
  v9  : More frames (64 instead of 32)
  v10 : Different model (Slow R50)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)

# Make the shared `experiments/common/` package importable when this script is
# run directly (e.g. `python train.py --version v2`) from inside this folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.data import load_splits_from_csv, slow_collate, slowfast_collate  # noqa: E402
from common.labels import ACTION_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS  # noqa: E402
from common.pack_pathway import PackPathway  # noqa: E402
from common.video_dataset import VideoDataset  # noqa: E402

# ============================================================
# PARSE VERSION
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default="v2", help="Ablation version: v2-v10")
args = parser.parse_args()
VERSION = args.version

print(f"\n{'='*60}")
print(f"  ABLATION STUDY — VERSION: {VERSION}")
print(f"{'='*60}\n")

# ============================================================
# VERSION-SPECIFIC CONFIGS
# ============================================================
# Defaults (same as V1)
MASTER_CSV = os.environ.get(
    "MASTER_CSV",
    "/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v3_balanced.csv",
)
CLIP_COL = "cut_clip_path"
SPLIT_COL = "split"

NUM_CLASSES = len(ACTION_CLASSES)
BATCH_SIZE = 4
NUM_WORKERS = 4
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
USE_CLASS_WEIGHTS = False
USE_OVERSAMPLING = False
NUM_FRAMES = 32
SAMPLING_RATE = 2
MODEL_NAME = "slowfast_r50"
FPS = 30
ALPHA = 4
SIDE_SIZE = 256
CROP_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

# Override per version
if VERSION == "v1":
    # Baseline: frozen backbone, no class weights, LR=1e-4 (all defaults above)
    pass

elif VERSION == "v2":
    # Class-weighted loss only
    USE_CLASS_WEIGHTS = True

elif VERSION == "v3":
    # Unfreeze backbone, lower LR
    FREEZE_BACKBONE = False
    LEARNING_RATE = 1e-5

elif VERSION == "v4":
    # Oversample minority classes
    USE_OVERSAMPLING = True

elif VERSION == "v5":
    # Class weights + Unfreeze backbone
    USE_CLASS_WEIGHTS = True
    FREEZE_BACKBONE = False
    LEARNING_RATE = 1e-5

elif VERSION == "v6":
    # Class weights + Oversample + Unfreeze backbone
    USE_CLASS_WEIGHTS = True
    USE_OVERSAMPLING = True
    FREEZE_BACKBONE = False
    LEARNING_RATE = 1e-5

elif VERSION == "v7":
    # Higher LR
    LEARNING_RATE = 1e-3

elif VERSION == "v8":
    # Larger batch size
    BATCH_SIZE = 8

elif VERSION == "v9":
    # Smaller crop size (224 instead of 256)
    CROP_SIZE = 224
    SIDE_SIZE = 224

elif VERSION == "v10":
    # Different model: Slow R50
    MODEL_NAME = "slow_r50"

else:
    raise ValueError(f"Unknown version: {VERSION}")

CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS

# Save paths per version
MODEL_SAVE_DIR = f"/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/output_ablation/{VERSION}/"
OUTPUT_CSV = os.path.join(MODEL_SAVE_DIR, f"predictions_{VERSION}.csv")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# Print config
print(f"Config for {VERSION}:")
print(f"  Model:            {MODEL_NAME}")
print(f"  Freeze backbone:  {FREEZE_BACKBONE}")
print(f"  Learning rate:    {LEARNING_RATE}")
print(f"  Batch size:       {BATCH_SIZE}")
print(f"  Num frames:       {NUM_FRAMES}")
print(f"  Class weights:    {USE_CLASS_WEIGHTS}")
print(f"  Oversampling:     {USE_OVERSAMPLING}")
print(f"  Clip duration:    {CLIP_DURATION:.2f}s")
print()


# ============================================================
# 1. DATA MODULE
# ============================================================
class ClipDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.label_map = CLASS_TO_IDX
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.class_weights = None

    def setup(self, stage=None):
        self.train_df, self.val_df, self.test_df = load_splits_from_csv(
            MASTER_CSV, CLIP_COL, SPLIT_COL
        )

        print(
            f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | "
            f"Test: {len(self.test_df)} clips"
        )

        # Compute class weights (inverse frequency)
        if USE_CLASS_WEIGHTS:
            counts = self.train_df["label_encoded"].value_counts().sort_index()
            total = len(self.train_df)
            weights = total / (len(counts) * counts.values)
            self.class_weights = torch.FloatTensor(weights)
            print(f"\nClass weights: {dict(zip(ACTION_CLASSES, weights.round(2)))}")

        # Save
        map_path = os.path.join(MODEL_SAVE_DIR, "label_mapping.json")
        with open(map_path, "w") as f:
            json.dump(self.label_map, f, indent=2)
        self.test_df.to_csv(os.path.join(MODEL_SAVE_DIR, "test_split.csv"), index=False)

        # Transforms
        if MODEL_NAME == "slow_r50":
            train_post = []
            eval_post = []
        else:
            train_post = [PackPathway(alpha=ALPHA)]
            eval_post = [PackPathway(alpha=ALPHA)]

        self.train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(NUM_FRAMES),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(MEAN, STD),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCropVideo(CROP_SIZE),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
                + train_post
            ),
        )

        self.eval_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(NUM_FRAMES),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(MEAN, STD),
                    ShortSideScale(size=SIDE_SIZE),
                    CenterCropVideo(CROP_SIZE),
                ]
                + eval_post
            ),
        )

    def train_dataloader(self):
        ds = VideoDataset(
            self.train_df["video_path"].tolist(),
            self.train_df["label_encoded"].tolist(),
            clip_duration=CLIP_DURATION,
            num_frames=NUM_FRAMES,
            crop_size=CROP_SIZE,
            alpha=ALPHA,
            transform=self.train_transform,
            model_name=MODEL_NAME,
        )
        collate = slow_collate if MODEL_NAME == "slow_r50" else slowfast_collate

        if USE_OVERSAMPLING:
            # Weighted random sampler — oversample minority classes
            labels = self.train_df["label_encoded"].values
            class_counts = np.bincount(labels)
            sample_weights = 1.0 / class_counts[labels]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(labels), replacement=True
            )
            return DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                collate_fn=collate,
            )
        else:
            return DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate
            )

    def val_dataloader(self):
        ds = VideoDataset(
            self.val_df["video_path"].tolist(),
            self.val_df["label_encoded"].tolist(),
            clip_duration=CLIP_DURATION,
            num_frames=NUM_FRAMES,
            crop_size=CROP_SIZE,
            alpha=ALPHA,
            transform=self.eval_transform,
            model_name=MODEL_NAME,
        )
        collate = slow_collate if MODEL_NAME == "slow_r50" else slowfast_collate
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate
        )


# ============================================================
# 2. LIGHTNING MODULE
# ============================================================
class SlowFastFineTune(pl.LightningModule):
    def __init__(
        self, num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE, class_weights=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Load pretrained model
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model=MODEL_NAME,
            pretrained=True,
        )

        # Replace classification head
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)

        # Freeze backbone if needed
        if freeze_backbone:
            print("Freezing backbone - only training the classification head")
            # Find the last block name
            block_names = [name for name, _ in self.model.named_parameters()]
            last_block = "blocks." + str(len(self.model.blocks) - 1)
            print(f"  Last block identifier: {last_block}")
            for name, param in self.model.named_parameters():
                if last_block not in name:
                    param.requires_grad = False
            trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            print(f"  Trainable params: {trainable}")

        # Class weights for loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        loss = F.cross_entropy(preds, labels, weight=self.class_weights)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        # Unweighted loss for fair validation, regardless of USE_CLASS_WEIGHTS
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
# 3. INFERENCE FUNCTION
# ============================================================
def run_inference(model, test_df, label_map, device):
    print("\n" + "=" * 60)
    print(f"RUNNING INFERENCE — {VERSION}")
    print("=" * 60)

    model.eval()
    model = model.to(device)

    if MODEL_NAME == "slow_r50":
        post_transforms = []
    else:
        post_transforms = [PackPathway(alpha=ALPHA)]

    eval_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(NUM_FRAMES),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(MEAN, STD),
                ShortSideScale(size=SIDE_SIZE),
                CenterCropVideo(CROP_SIZE),
            ]
            + post_transforms
        ),
    )

    id_to_label = {v: k for k, v in label_map.items()}
    video_paths = test_df["video_path"].tolist()
    true_labels = test_df["label_encoded"].tolist()
    original_labels = test_df["class_name"].tolist()

    softmax = torch.nn.Softmax(dim=1)
    results = []

    for idx, video_path in enumerate(video_paths):
        print(f"[{idx+1}/{len(video_paths)}] {os.path.basename(video_path)}")

        try:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)

            if video_data["video"] is None:
                raise ValueError("Decoder returned None")

            video_data = eval_transform(video_data)
            inputs = video_data["video"]

            if MODEL_NAME == "slow_r50":
                inputs = inputs.to(device)[None, ...]
            else:
                inputs = [i.to(device)[None, ...] for i in inputs]

            with torch.no_grad():
                preds = model(inputs)

            preds = softmax(preds)
            top_pred = preds.argmax(dim=1).item()
            top_score = preds[0, top_pred].item()
            pred_label = id_to_label[top_pred]

            results.append(
                {
                    "video_path": video_path,
                    "true_label": original_labels[idx],
                    "true_label_encoded": true_labels[idx],
                    "predicted_label": pred_label,
                    "predicted_label_encoded": top_pred,
                    "confidence": round(top_score, 4),
                    "correct": int(top_pred == true_labels[idx]),
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "video_path": video_path,
                    "true_label": original_labels[idx],
                    "true_label_encoded": true_labels[idx],
                    "predicted_label": "ERROR",
                    "predicted_label_encoded": -1,
                    "confidence": 0.0,
                    "correct": 0,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to {OUTPUT_CSV}")

    valid = results_df[results_df["predicted_label"] != "ERROR"]
    if len(valid) > 0:
        accuracy = valid["correct"].mean()
        print(f"\n{'=' * 60}")
        print(f"TEST RESULTS — {VERSION}")
        print(f"{'=' * 60}")
        print(
            f"Total: {len(results_df)} | Processed: {len(valid)} | Errors: {len(results_df) - len(valid)}"
        )
        print(f"Accuracy: {accuracy:.4f} ({int(valid['correct'].sum())}/{len(valid)})")

        report = classification_report(
            valid["true_label"], valid["predicted_label"], zero_division=0
        )
        print(f"\nClassification Report:\n{report}")

        labels = sorted(valid["true_label"].unique())
        cm = confusion_matrix(valid["true_label"], valid["predicted_label"], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(f"Confusion Matrix:\n{cm_df}")

        metrics_path = os.path.join(MODEL_SAVE_DIR, f"test_metrics_{VERSION}.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Version: {VERSION}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Freeze backbone: {FREEZE_BACKBONE}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Num frames: {NUM_FRAMES}\n")
            f.write(f"Class weights: {USE_CLASS_WEIGHTS}\n")
            f.write(f"Oversampling: {USE_OVERSAMPLING}\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"Confusion Matrix:\n{cm_df.to_string()}\n")
        print(f"Metrics saved to {metrics_path}")

    return results_df


# ============================================================
# 4. MAIN
# ============================================================
def main():
    data_module = ClipDataModule()
    data_module.setup()

    model = SlowFastFineTune(
        num_classes=NUM_CLASSES,
        freeze_backbone=FREEZE_BACKBONE,
        class_weights=data_module.class_weights,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename=f"slowfast-{VERSION}-{{epoch:02d}}-{{val_loss:.3f}}",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    best_path = checkpoint_cb.best_model_path
    print(f"\nBest model: {best_path}")

    # Load best model
    best_model = SlowFastFineTune.load_from_checkpoint(
        best_path,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        class_weights=data_module.class_weights,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inference
    run_inference(
        model=best_model,
        test_df=data_module.test_df,
        label_map=data_module.label_map,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"  {VERSION} COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
