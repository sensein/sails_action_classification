"""Video Swin fine-tuning for Loco and RMM."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, TypedDict, cast

import cv2
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

class TaskConfig(TypedDict):
    label_col: str
    num_classes: int
    output_dir: str


TASK_CONFIG: dict[str, TaskConfig] = {
    "loco": {
        "label_col": "Locomotion",
        "num_classes": 5,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/clips_h5/video_swin/loco_swin_seeds"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 4,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/clips_h5/video_swin/rmm_swin_seeds"
        ),
    },
}

SPLIT_CSV: str = (
    "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
)

# ---------------------------------------------------------------------------
# Global hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 4
ACCUM_STEPS: int = 8
NUM_WORKERS: int = 16
MAX_EPOCHS: int = 20
LEARNING_RATE: float = 1e-4
DEFAULT_SEEDS: list[int] = [42, 123, 456]

NUM_FRAMES: int = 32
CROP_SIZE: int = 224
MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
STD: tuple[float, ...] = (0.229, 0.224, 0.225)

ANN_FPS: float = 15.0
MIN_FRAMES: int = 15
CLIP_FRAMES: int = 30

SWIN_CKPT_URL: str = (
    "https://github.com/SwinTransformer/storage/releases/"
    "download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth"
)
SWIN_CKPT_LOCAL: str = os.path.expanduser(
    "~/.cache/video_swin/swin_base_k400.pth"
)


# ---------------------------------------------------------------------------
# Sample dict type alias — used everywhere instead of bare dict
# ---------------------------------------------------------------------------

# A sample produced by build_samples / chunk_run.
Sample = dict[str, str | int | float]


# ---------------------------------------------------------------------------
# 1. Bounding-box loading
# ---------------------------------------------------------------------------

def load_bbox_map(h5_path: str) -> dict[int, tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {
        int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1
    }


# ---------------------------------------------------------------------------
# 2. Action-run extraction and clipping
# ---------------------------------------------------------------------------

def find_action_runs(
    ann: pd.DataFrame, label_col: str
) -> list[tuple[int, int, str]]:
    """Identify contiguous runs of the same action label."""
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()
    runs: list[tuple[int, int, str]] = []
    i, n = 0, len(df)
    while i < n:
        lab = labels[i].strip()
        if lab in ("N/A", ""):
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and labels[j + 1].strip() == lab
            and frames[j + 1] == frames[j] + 1
        ):
            j += 1
        runs.append((frames[i], frames[j], lab))
        i = j + 1
    return runs


def chunk_run(start: int, end: int) -> list[tuple[int, int]]:
    """Split an action run into fixed-length clips."""
    total = end - start + 1
    if total < MIN_FRAMES:
        return []
    if total < CLIP_FRAMES * 2:
        if total < 45:
            return [(start, end)]
        split_pt = start + CLIP_FRAMES
        return [(start, split_pt - 1), (split_pt, end)]
    clips: list[tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(s + CLIP_FRAMES - 1, end)
        if (e - s + 1) >= MIN_FRAMES:
            clips.append((s, e))
        s += CLIP_FRAMES
    return clips


def build_samples(
    split_csv: str, label_col: str
) -> dict[str, list[Sample]]:
    """Build per-split sample lists from the CSV."""
    split_df = pd.read_csv(split_csv)
    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    for col in required:
        if col not in split_df.columns:
            raise ValueError(f"Split CSV missing column: {col}")

    by_split: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    for _, row in split_df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_anno_h5"]).strip()
        sp = str(row["split"]).strip().lower()

        if sp not in by_split:
            continue
        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            continue

        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  skip ({exc}): {lp}")
            continue

        if label_col not in ann.columns:
            continue

        runs = find_action_runs(ann, label_col)
        for sf, ef, lab in runs:
            for cs, ce in chunk_run(sf, ef):
                by_split[sp].append(
                    {
                        "video_path": vp,
                        "h5_path": hp,
                        "start_frame": int(cs),
                        "end_frame": int(ce),
                        "label_str": lab,
                        "ann_fps": ANN_FPS,
                    }
                )
    return by_split


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class BBoxCropVideoDataset(Dataset[tuple[torch.Tensor, int]]):
    """Reads video segments, crops to subject bounding box, and normalizes."""

    def __init__(
        self,
        samples: list[Sample],
        label_map: dict[str, int],
        num_frames: int = NUM_FRAMES,
        crop_size: int = CROP_SIZE,
        *,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.training = training
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_segment(self, sample: Sample) -> torch.Tensor:
        """Decode, bbox-crop, resize, and normalize a single clip."""
        cap = cv2.VideoCapture(str(sample["video_path"]))
        if not cap.isOpened():
            raise OSError(f"cannot open {sample['video_path']}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vid_fps / float(sample["ann_fps"]))))

        bbox_map = load_bbox_map(str(sample["h5_path"]))
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(
            int(sample["start_frame"]), int(sample["end_frame"]) + 1
        )
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]
        bbox_keys = np.array(sorted(bbox_map.keys()))

        frames: list[np.ndarray] = []
        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                frames.append(
                    np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                )
                continue

            h, w = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                x1, y1, x2, y2 = bbox_map[nearest]

            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(crop)

        cap.release()
        arr = np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label = self.label_map[str(sample["label_str"])]
        try:
            frames = self._read_segment(sample)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as exc:  # noqa: BLE001
            print(
                f"  load error {os.path.basename(str(sample['video_path']))}"
                f" [{sample['start_frame']}-{sample['end_frame']}]: {exc}"
            )
            return (
                torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
                label,
            )


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack (video, label) pairs into batched tensors."""
    videos, labels = zip(*batch, strict=True)
    return torch.stack(list(videos)), torch.tensor(list(labels), dtype=torch.long)


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------

class H5BBoxDataModule(pl.LightningDataModule):
    """DataModule that builds bbox-cropped clip samples from the split CSV."""

    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.label_map: dict[str, int] = {}
        self.class_weights: torch.Tensor | None = None
        self.train_samples: list[Sample] = []
        self.val_samples: list[Sample] = []
        self.test_samples: list[Sample] = []

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Build clip samples and compute inverse-frequency class weights."""
        print(f"Building samples (label_col={self.label_col})...")
        by_split = build_samples(SPLIT_CSV, self.label_col)
        n_train = len(by_split["train"])
        n_val = len(by_split["val"])
        n_test = len(by_split["test"])
        print(f"Clips  train={n_train}  val={n_val}  test={n_test}")
        if n_train == 0:
            raise RuntimeError("No training clips built — check paths and label column.")

        all_labels: list[str] = sorted(
            {str(s["label_str"]) for split in by_split.values() for s in split}
        )
        self.label_map = {lab: i for i, lab in enumerate(all_labels)}
        print(f"Label map: {self.label_map}")

        for sp, samps in by_split.items():
            dist = Counter(str(s["label_str"]) for s in samps)
            print(f"  {sp} distribution: {dict(dist)}")

        self.train_samples = by_split["train"]
        self.val_samples = by_split["val"]
        self.test_samples = by_split["test"]

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "label_mapping.json"), "w") as fh:
            json.dump(self.label_map, fh, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False
        )

        n_classes = len(self.label_map)
        counts = np.zeros(n_classes, dtype=np.float64)
        for s in self.train_samples:
            counts[self.label_map[str(s["label_str"])]] += 1
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights (train):")
        for lab, idx in self.label_map.items():
            print(
                f"  {lab:30s} count={int(counts[idx]):4d}"
                f"  weight={weights[idx]:.3f}"
            )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the training DataLoader."""
        assert self.label_map, "Call setup() first."
        ds = BBoxCropVideoDataset(self.train_samples, self.label_map, training=True)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the validation DataLoader."""
        assert self.label_map, "Call setup() first."
        ds = BBoxCropVideoDataset(self.val_samples, self.label_map, training=False)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# 5. Video Swin-B classifier
# ---------------------------------------------------------------------------

class VideoSwinClassifier(nn.Module):
    """Classification head wrapper around the Video Swin-B backbone."""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        return cast(torch.Tensor, self.head(feats))


def build_video_swin_b(
    num_classes: int, *, freeze_all_but_last_stage: bool = True
) -> VideoSwinClassifier:
    """Instantiate Video Swin-B with K400 weights and a classification head."""
    try:
        from sailsprep.action_model_testing.Video_Swin.video_swin_transformer import (
            SwinTransformer3D,
        )
    except ImportError as exc:
        raise ImportError("Cannot import SwinTransformer3D.") from exc

    backbone = SwinTransformer3D(  # type: ignore[no-untyped-call]
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
    )

    os.makedirs(os.path.dirname(SWIN_CKPT_LOCAL), exist_ok=True)
    if not os.path.exists(SWIN_CKPT_LOCAL):
        print(f"Downloading Swin-B K400 checkpoint -> {SWIN_CKPT_LOCAL}")
        torch.hub.download_url_to_file(SWIN_CKPT_URL, SWIN_CKPT_LOCAL)

    ckpt = torch.load(SWIN_CKPT_LOCAL, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if not k.startswith(("cls_head.", "head."))
    }
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"Loaded Swin-B K400. missing={len(missing)}  unexpected={len(unexpected)}")

    clf = VideoSwinClassifier(backbone, feat_dim=1024, num_classes=num_classes)

    if freeze_all_but_last_stage:
        for name, param in clf.named_parameters():
            param.requires_grad = "backbone.layers.3" in name or name.startswith("head.")
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        print(f"  Trainable: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M")

    return clf


# ---------------------------------------------------------------------------
# 6. Lightning training module
# ---------------------------------------------------------------------------

class VideoSwinFineTune(pl.LightningModule):
    """PyTorch Lightning module for Video Swin-B fine-tuning."""

    def __init__(
        self,
        num_classes: int,
        freeze: bool = True,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_video_swin_b(num_classes, freeze_all_but_last_stage=freeze)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss: torch.Tensor = F.cross_entropy(logits, y, weight=self.class_weights)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss: torch.Tensor = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss
    def configure_optimizers(self) -> Any:  # Lightning accepts dict at runtime
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=3, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# 7. Test-set inference
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    test_samples: list[Sample],
    label_map: dict[str, int],
    device: torch.device,
    output_csv: str,
    output_dir: str,
) -> None:
    """Run inference on the held-out test set and write predictions and metrics."""
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET")
    print("=" * 60)

    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax = nn.Softmax(dim=1)
    rows: list[dict[str, str | int | float]] = []

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            probs = softmax(logits)
            top = int(probs.argmax(1).item())
            conf = float(probs[0, top].item())
            rows.append(
                {
                    "video_path": str(s["video_path"]),
                    "start_frame": int(s["start_frame"]),
                    "end_frame": int(s["end_frame"]),
                    "true_label": str(s["label_str"]),
                    "pred_label": id_to_label[top],
                    "confidence": round(conf, 4),
                    "correct": int(id_to_label[top] == str(s["label_str"])),
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(ds)}] processed")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR on sample {i}: {exc}")
            rows.append(
                {
                    "video_path": str(s["video_path"]),
                    "start_frame": int(s["start_frame"]),
                    "end_frame": int(s["end_frame"]),
                    "true_label": str(s["label_str"]),
                    "pred_label": "ERROR",
                    "confidence": 0.0,
                    "correct": 0,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved -> {output_csv}")

    valid = df[df["pred_label"] != "ERROR"]
    if len(valid) == 0:
        print("No valid predictions to evaluate.")
        return

    acc = valid["correct"].mean()
    print(f"\nAccuracy: {acc:.4f} ({int(valid['correct'].sum())}/{len(valid)})")
    all_labels = sorted(label_map.keys())
    print("\nClassification report:")
    print(
        classification_report(
            valid["true_label"], valid["pred_label"],
            labels=all_labels, zero_division=0,
        )
    )
    cm = confusion_matrix(valid["true_label"], valid["pred_label"], labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("Confusion matrix:")
    print(cm_df)

    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as fh:
        fh.write(f"Accuracy: {acc:.4f}\n\n")
        fh.write(
            classification_report(
                valid["true_label"], valid["pred_label"],
                labels=all_labels, zero_division=0,
            )
        )
        fh.write(f"\n{cm_df.to_string()}\n")
    print(f"Metrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video Swin-B fine-tuning for action recognition."
    )
    parser.add_argument("--task", choices=["loco", "rmm"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg: TaskConfig = TASK_CONFIG[args.task]
    label_col: str = cfg["label_col"]
    output_dir: str = os.path.join(cfg["output_dir"], f"seed_{args.seed}")
    output_csv: str = os.path.join(output_dir, "test_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"TASK: {args.task.upper()}  label_col={label_col}  seed={args.seed}")
    print(f"{'=' * 60}")

    pl.seed_everything(args.seed)
    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes, found {n_classes}."
    )

    model = VideoSwinFineTune(
        num_classes=n_classes, freeze=True, class_weights=dm.class_weights
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, monitor="val_loss", mode="min", save_top_k=2,
        filename=f"videoswin-{args.task}-s{args.seed}-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=ACCUM_STEPS,
    )
    trainer.fit(model, dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}")
    best_model = VideoSwinFineTune.load_from_checkpoint(
        best, num_classes=n_classes, freeze=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best_model, dm.test_samples, dm.label_map, device, output_csv, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()