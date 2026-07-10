import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import glob
import pandas as pd
import pytorch_lightning as pl
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
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================
CLIPS_DIR = "/home/aparnabg/orcd/scratch/all_project_files/cut_locomotion_clips/"
# Subfolders: walk/, run/, crawl/, cruise/, vehicle/

NUM_CLASSES = 5  # walk, run, crawl, cruise, vehicle
BATCH_SIZE = 4
NUM_WORKERS = 4
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# 70% train, 30% test
TEST_SPLIT = 0.30

# Save paths
MODEL_SAVE_DIR = "/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/output_finetune_clip/"
OUTPUT_CSV = os.path.join(MODEL_SAVE_DIR, "finetune_clip_predictions.csv")

# SlowFast-specific params
NUM_FRAMES = 16       # Reduced from 32 to handle short clips
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
# 2. BUILD DATASET FROM FOLDER STRUCTURE
# ============================================================
def build_dataset_from_folders(clips_dir):
    """
    Read clips from folder structure:
      clips_dir/walk/*.mp4
      clips_dir/run/*.mp4
      etc.
    Returns: list of (video_path, label_string)
    """
    data = []
    classes = sorted([d for d in os.listdir(clips_dir)
                      if os.path.isdir(os.path.join(clips_dir, d))])

    print(f"Found classes: {classes}")

    for cls in classes:
        cls_dir = os.path.join(clips_dir, cls)
        clips = glob.glob(os.path.join(cls_dir, "*.mp4"))
        for clip_path in clips:
            data.append({"video_path": clip_path, "label": cls})
        print(f"  {cls}: {len(clips)} clips")

    df = pd.DataFrame(data)
    print(f"Total clips: {len(df)}")
    return df


# ============================================================
# 3. CUSTOM VIDEO DATASET
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
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)

            if video_data["video"] is None:
                raise ValueError("Decoder returned None")

            if self.transform:
                video_data = self.transform(video_data)

            return video_data["video"], label

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            dummy = [torch.zeros(3, NUM_FRAMES // ALPHA, CROP_SIZE, CROP_SIZE),
                     torch.zeros(3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)]
            return dummy, label


# ============================================================
# 4. CUSTOM COLLATE
# ============================================================
def slowfast_collate(batch):
    videos, labels = zip(*batch)
    slow = torch.stack([v[0] for v in videos])
    fast = torch.stack([v[1] for v in videos])
    labels = torch.tensor(labels, dtype=torch.long)
    return [slow, fast], labels


# ============================================================
# 5. DATA MODULE (70% train, 30% test)
# ============================================================
class ClipDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.label_map = None
        self.test_df = None

    def setup(self, stage=None):
        # Build dataset from folder structure
        df = build_dataset_from_folders(CLIPS_DIR)

        # Create label mapping
        classes = sorted(df["label"].unique())
        self.label_map = {cls: i for i, cls in enumerate(classes)}
        df["label_encoded"] = df["label"].map(self.label_map)
        print(f"\nLabel mapping: {self.label_map}")

        # Stratified split: 70% train, 30% test
        self.train_df, self.test_df = train_test_split(
            df, test_size=TEST_SPLIT, random_state=42, stratify=df["label"]
        )
        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        print(f"Train: {len(self.train_df)} | Test: {len(self.test_df)} clips")
        print(f"\nTrain distribution:")
        print(self.train_df["label"].value_counts().to_string())
        print(f"\nTest distribution:")
        print(self.test_df["label"].value_counts().to_string())

        # Save label mapping and test split
        map_path = os.path.join(MODEL_SAVE_DIR, "label_mapping.json")
        with open(map_path, "w") as f:
            json.dump(self.label_map, f, indent=2)
        print(f"\nLabel mapping saved to {map_path}")

        self.test_df.to_csv(os.path.join(MODEL_SAVE_DIR, "test_split.csv"), index=False)

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

        # Test transforms (no augmentation)
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
        # Use a portion of train for validation during training
        ds = VideoDataset(
            self.test_df["video_path"].tolist(),
            self.test_df["label_encoded"].tolist(),
            transform=self.eval_transform,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=slowfast_collate)


# ============================================================
# 6. LIGHTNING MODULE (Fine-tuning)
# ============================================================
class SlowFastFineTune(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE):
        super().__init__()
        self.save_hyperparameters()

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
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
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
# 7. INFERENCE FUNCTION
# ============================================================
def run_inference(model, test_df, label_map, device):
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE ON 30% TEST SET")
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

    id_to_label = {v: k for k, v in label_map.items()}

    video_paths = test_df["video_path"].tolist()
    true_labels = test_df["label_encoded"].tolist()
    original_labels = test_df["label"].tolist()

    softmax = torch.nn.Softmax(dim=1)
    results = []

    for idx, video_path in enumerate(video_paths):
        print(f"[{idx+1}/{len(video_paths)}] Predicting: {os.path.basename(video_path)}")

        try:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)

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
            pred_label = id_to_label[top_pred]

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
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                valid["true_label"], valid["predicted_label"], zero_division=0
            ))
            f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
        print(f"Metrics saved to {metrics_path}")

    return results_df


# ============================================================
# 8. MAIN: TRAIN + INFERENCE
# ============================================================
def main():
    data_module = ClipDataModule()
    model = SlowFastFineTune(num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE)

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

    # Run inference on 30% test set
    data_module.setup()
    run_inference(
        model=best_model,
        test_df=data_module.test_df,
        label_map=data_module.label_map,
        device=device,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()