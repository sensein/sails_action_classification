"""Video Swin binary classifier: N/A vs non-N/A on 2-sec sliding windows.

Stage 1 of a two-stage pipeline. Slides a 2-second window (1-second
stride) across full annotated videos and trains a binary classifier to
distinguish N/A (no activity) from non-N/A (some activity present).


Usage::

    python video_swin_binary_sliding.py --task loco
    python video_swin_binary_sliding.py --task loco --seed 123

"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Optional

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

from common.utils import VideoSwinClassifier, collate_fn, load_bbox_map

# ---------------------------------------------------------------------------
# Task configuration (binary: 2 classes for all tasks)
# ---------------------------------------------------------------------------
TASK_CONFIG: dict[str, dict] = {
    "loco": {
        "label_col": "Locomotion",
        "num_classes": 2,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/loco_swin_binary"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 2,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/rmm_swin_binary"
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
NUM_WORKERS: int = 16
MAX_EPOCHS: int = 20
LEARNING_RATE: float = 1e-4
DEFAULT_SEEDS: list[int] = [42, 123, 456]

NUM_FRAMES: int = 32
CROP_SIZE: int = 224
MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
STD: tuple[float, ...] = (0.229, 0.224, 0.225)

# Annotation timing and windowing.
ANN_FPS: float = 15.0
WINDOW_SEC: float = 2.0
WINDOW_STRIDE: float = 1.0
MIN_WIN_FRAMES: int = 5

# Binary labels.
NA_LABEL: str = "N/A"
NON_NA_LABEL: str = "non-N/A"

# Pretrained Swin-B Kinetics-400 checkpoint.
SWIN_CKPT_URL: str = (
    "https://github.com/SwinTransformer/storage/releases/"
    "download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth"
)
SWIN_CKPT_LOCAL: str = os.path.expanduser(
    "~/.cache/video_swin/swin_base_k400.pth"
)


# ---------------------------------------------------------------------------
# 1. Bounding-box loading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. Sliding-window builder (binary: N/A vs non-N/A)
# ---------------------------------------------------------------------------
def get_window_binary_label(
    frame_to_label: dict[int, str], ann_start: int, ann_end: int
) -> str:
    """Determine binary label for a window via majority vote.

    Each frame is mapped to either N/A or non-N/A. The majority across
    the window determines the window label.

    Args:
        frame_to_label: Mapping from frame index to original label string.
        ann_start: First annotation frame (inclusive).
        ann_end: Last annotation frame (exclusive).

    Returns:
        ``"N/A"`` or ``"non-N/A"`` based on majority vote.
    """
    na_count = 0
    non_na_count = 0
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, NA_LABEL)
        if lbl in ("", "nan", "None", NA_LABEL):
            na_count += 1
        else:
            non_na_count += 1
    if non_na_count >= na_count:
        return NON_NA_LABEL
    return NA_LABEL


def build_samples(
    split_csv: str, label_col: str
) -> dict[str, list[dict]]:
    """Slide 2-sec windows across full videos with binary labels.

    Each window is labeled N/A or non-N/A based on majority vote of its
    frames. Videos that are entirely N/A are still included (they
    contribute N/A training examples for the binary classifier).

    Args:
        split_csv: Path to the master split CSV.
        label_col: Annotation column to use for labels.

    Returns:
        Dict mapping ``"train"``/``"val"``/``"test"`` to lists of sample
        dicts.

    Raises:
        ValueError: If a required column is missing from ``split_csv``.
    """
    split_df = pd.read_csv(split_csv)
    required_cols = [
        "video_path", "label_path", "interpolated_full_h5", "split",
    ]
    for c in required_cols:
        if c not in split_df.columns:
            raise ValueError(f"Split CSV missing column: '{c}'")

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    window_ann_frames = int(WINDOW_SEC * ANN_FPS)
    stride_ann_frames = int(WINDOW_STRIDE * ANN_FPS)

    for _, row in split_df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_full_h5"]).strip()
        sp = str(row["split"]).strip().lower()

        if sp not in by_split:
            continue
        if not (
            os.path.exists(vp)
            and os.path.exists(lp)
            and os.path.exists(hp)
        ):
            print(f"  [skip] missing file(s) for: {os.path.basename(vp)}")
            continue

        # Load frame-to-label map.
        try:
            ann = pd.read_csv(
                lp, encoding="utf-8-sig", keep_default_na=False
            )
            ann.columns = ann.columns.str.strip()
        except Exception as e:
            print(f"  [skip] bad label CSV ({e}): {lp}")
            continue

        if label_col not in ann.columns or "Frame" not in ann.columns:
            print(f"  [skip] missing columns in: {lp}")
            continue

        ann = ann.sort_values("Frame").reset_index(drop=True)
        frame_to_label: dict[int, str] = {}
        for _, r in ann.iterrows():
            fn = int(r["Frame"])
            lbl = str(r[label_col]).strip()
            if lbl in ("", "nan", "None"):
                lbl = NA_LABEL
            frame_to_label[fn] = lbl

        if not frame_to_label:
            continue

        max_ann_frame = max(frame_to_label.keys())
        total_ann_frames = max_ann_frame + 1

        # Slide windows across the full video.
        start = 0
        while start + window_ann_frames <= total_ann_frames + stride_ann_frames:
            end = min(start + window_ann_frames, total_ann_frames)
            n_valid = end - start
            if n_valid < MIN_WIN_FRAMES:
                start += stride_ann_frames
                continue

            binary_label = get_window_binary_label(
                frame_to_label, start, end
            )
            by_split[sp].append({
                "video_path": vp,
                "h5_path": hp,
                "start_frame": int(start),
                "end_frame": int(end - 1),
                "label_str": binary_label,
                "ann_fps": ANN_FPS,
            })
            start += stride_ann_frames

    return by_split


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class BBoxCropVideoDataset(Dataset):
    """Reads sliding-window segments with bbox crop and normalization.

    Args:
        samples: List of sample dicts from ``build_samples``.
        label_map: Mapping from string label to integer class index.
        num_frames: Number of frames to uniformly sample per window.
        crop_size: Spatial size after resizing the bbox crop.
        training: If ``True``, applies random horizontal flip augmentation.
    """

    def __init__(
        self,
        samples: list[dict],
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

    def _read_segment(self, sample: dict) -> torch.Tensor:
        """Decode, crop, and normalize frames for a single window.

        Args:
            sample: A sample dict with video path, bbox h5 path, and
                frame range.

        Returns:
            Tensor of shape ``(C, T, H, W)`` with ImageNet normalization.

        Raises:
            IOError: If the video file cannot be opened.
            ValueError: If the bbox map is empty.
        """
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {sample['video_path']}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vid_fps / sample["ann_fps"])))

        bbox_map = load_bbox_map(sample["h5_path"])
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(sample["start_frame"], sample["end_frame"] + 1)
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(
            int
        )
        chosen = ann_frames[idxs]
        bbox_keys = np.array(sorted(bbox_map.keys()))

        frames: list[np.ndarray] = []
        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                frames.append(
                    np.zeros(
                        (self.crop_size, self.crop_size, 3), dtype=np.uint8
                    )
                )
                continue

            h, w = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                nearest = int(
                    bbox_keys[np.argmin(np.abs(bbox_keys - af))]
                )
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
        arr = (
            np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        )
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label = self.label_map[sample["label_str"]]
        try:
            frames = self._read_segment(sample)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as e:
            print(
                f"  load error {os.path.basename(sample['video_path'])} "
                f"[{sample['start_frame']}-{sample['end_frame']}]: {e}"
            )
            return (
                torch.zeros(
                    3, self.num_frames, self.crop_size, self.crop_size
                ),
                label,
            )


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------
class H5BBoxDataModule(pl.LightningDataModule):
    """DataModule for binary sliding-window samples from full videos.

    Args:
        label_col: Annotation column to use (e.g. ``"Locomotion"``).
        output_dir: Directory to save label mappings and test split CSV.
    """

    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.label_map: Optional[dict[str, int]] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.train_samples: list[dict] = []
        self.val_samples: list[dict] = []
        self.test_samples: list[dict] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Build binary sliding-window samples and compute class weights."""
        print(
            f"\nBuilding binary sliding-window samples"
            f" (label_col={self.label_col})..."
        )
        by_split = build_samples(SPLIT_CSV, self.label_col)

        n_tr = len(by_split["train"])
        n_v = len(by_split["val"])
        n_te = len(by_split["test"])
        print(f"Windows  train={n_tr}  val={n_v}  test={n_te}")
        if n_tr == 0:
            raise RuntimeError(
                "No training windows built -- check CSV paths/columns."
            )

        # Fixed binary label map: N/A=0, non-N/A=1.
        self.label_map = {NA_LABEL: 0, NON_NA_LABEL: 1}
        print(f"Label map (binary): {self.label_map}")

        for sp, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
            print(f"  {sp} distribution: {dict(sorted(dist.items()))}")

        self.train_samples = by_split["train"]
        self.val_samples = by_split["val"]
        self.test_samples = by_split["test"]

        os.makedirs(self.output_dir, exist_ok=True)
        with open(
            os.path.join(self.output_dir, "label_mapping.json"), "w"
        ) as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False
        )

        # Compute inverse-frequency class weights from training windows.
        n_classes = len(self.label_map)
        counts = np.zeros(n_classes, dtype=np.float64)
        for s in self.train_samples:
            counts[self.label_map[s["label_str"]]] += 1
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print("\nClass weights (train):")
        for lab, idx in sorted(self.label_map.items(), key=lambda x: x[1]):
            print(
                f"  {lab:15s}  count={int(counts[idx]):6d}"
                f"  weight={weights[idx]:.4f}"
            )

    def train_dataloader(self) -> DataLoader:
        ds = BBoxCropVideoDataset(
            self.train_samples, self.label_map, training=True
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = BBoxCropVideoDataset(
            self.val_samples, self.label_map, training=False
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# 5. Video Swin-B model
# ---------------------------------------------------------------------------
def build_video_swin_b(
    num_classes: int,
    *,
    freeze_all_but_last_stage: bool = True,
) -> VideoSwinClassifier:
    """Load Video Swin-B with K400 pretrained weights and attach a head.

    Args:
        num_classes: Number of output classes (2 for binary).
        freeze_all_but_last_stage: If ``True``, freeze all parameters
            except ``backbone.layers.3`` and the classification head.

    Returns:
        A ``VideoSwinClassifier`` module ready for training.

    Raises:
        ImportError: If ``video_swin_transformer`` is not installed.
    """
    try:
        from common.video_swin_transformer import SwinTransformer3D
    except ImportError as e:
        raise ImportError(
            "Please install: pip install "
            "git+https://github.com/haofanwang/"
            "video-swin-transformer-pytorch.git"
        ) from e

    model = SwinTransformer3D(
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded Swin-B K400: missing={len(missing)}"
        f" unexpected={len(unexpected)}"
    )

    clf = VideoSwinClassifier(model, feat_dim=1024, num_classes=num_classes)

    if freeze_all_but_last_stage:
        print("Freezing all but last stage (layers.3) + head")
        for name, param in clf.named_parameters():
            if "backbone.layers.3" in name or name.startswith("head."):
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(
            p.numel() for p in clf.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in clf.parameters())
        print(
            f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M"
        )

    return clf


# ---------------------------------------------------------------------------
# 6. Lightning Module
# ---------------------------------------------------------------------------
class VideoSwinFineTune(pl.LightningModule):
    """PyTorch Lightning module for binary Video Swin-B fine-tuning.

    Args:
        num_classes: Number of output classes (2 for binary).
        freeze: Whether to freeze all but the last stage.
        class_weights: Optional per-class weights for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        freeze: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_video_swin_b(
            num_classes, freeze_all_but_last_stage=freeze
        )
        if class_weights is not None:
            self.register_buffer(
                "class_weights", class_weights.float(), persistent=False
            )
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=LEARNING_RATE, weight_decay=0.05
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=3, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# 7. Inference — window-level + video-level aggregation
# ---------------------------------------------------------------------------
def run_inference(
    model: nn.Module,
    test_samples: list[dict],
    label_map: dict[str, int],
    device: torch.device,
    output_dir: str,
) -> None:
    """Run binary inference on test windows and aggregate to video level.

    Args:
        model: Trained model (will be moved to eval mode).
        test_samples: List of test sample dicts.
        label_map: Binary label mapping (N/A=0, non-N/A=1).
        device: Torch device to run inference on.
        output_dir: Directory to save prediction CSVs and metrics.
    """
    print("\n" + "=" * 60)
    print("INFERENCE -- BINARY (N/A vs non-N/A)")
    print("=" * 60)

    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax = nn.Softmax(dim=1)
    window_rows: list[dict] = []

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
            row = {
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "pred_label": id_to_label[top],
                "confidence": round(conf, 4),
                "correct": int(id_to_label[top] == s["label_str"]),
                "prob_NA": round(float(probs[0, 0].item()), 4),
                "prob_nonNA": round(float(probs[0, 1].item()), 4),
            }
            window_rows.append(row)
            if (i + 1) % 50 == 0:
                print(
                    f"  [{i + 1}/{len(ds)}]"
                    f" {os.path.basename(s['video_path'])}"
                    f" [{s['start_frame']}-{s['end_frame']}]"
                    f" true={s['label_str']} pred={id_to_label[top]}"
                    f" ({conf:.2f})"
                )
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            window_rows.append({
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "pred_label": "ERROR",
                "confidence": 0.0,
                "correct": 0,
                "prob_NA": 0.0,
                "prob_nonNA": 0.0,
            })

    # Save window-level predictions.
    win_df = pd.DataFrame(window_rows)
    win_csv = os.path.join(output_dir, "test_predictions_window.csv")
    win_df.to_csv(win_csv, index=False)
    print(f"\nWindow predictions -> {win_csv}")

    valid_win = win_df[win_df["pred_label"] != "ERROR"]
    binary_labels = [NA_LABEL, NON_NA_LABEL]

    if len(valid_win):
        win_acc = valid_win["correct"].mean()
        print(
            f"Window-level accuracy: {win_acc:.4f}"
            f" ({int(valid_win['correct'].sum())}/{len(valid_win)})"
        )
        print("\n--- Window-level classification report ---")
        print(
            classification_report(
                valid_win["true_label"],
                valid_win["pred_label"],
                labels=binary_labels,
                zero_division=0,
            )
        )
        cm = confusion_matrix(
            valid_win["true_label"],
            valid_win["pred_label"],
            labels=binary_labels,
        )
        cm_df = pd.DataFrame(
            cm, index=binary_labels, columns=binary_labels
        )
        print("Confusion matrix (window):")
        print(cm_df)

    # Video-level aggregation via average softmax probabilities.
    print("\n" + "=" * 60)
    print("VIDEO-LEVEL AGGREGATION (binary)")
    print("=" * 60)

    video_rows: list[dict] = []
    for vpath, grp in valid_win.groupby("video_path"):
        avg_na = grp["prob_NA"].mean()
        avg_non_na = grp["prob_nonNA"].mean()
        pred_label = NON_NA_LABEL if avg_non_na >= avg_na else NA_LABEL
        true_lab = Counter(
            grp["true_label"].tolist()
        ).most_common(1)[0][0]
        video_rows.append({
            "video_path": vpath,
            "true_label": true_lab,
            "pred_label": pred_label,
            "correct": int(pred_label == true_lab),
            "n_windows": len(grp),
            "avg_prob_NA": round(float(avg_na), 4),
            "avg_prob_nonNA": round(float(avg_non_na), 4),
        })

    vid_df = pd.DataFrame(video_rows)
    vid_csv = os.path.join(output_dir, "test_predictions_video.csv")
    vid_df.to_csv(vid_csv, index=False)
    print(f"Video predictions -> {vid_csv}")

    if len(vid_df):
        vid_acc = vid_df["correct"].mean()
        print(
            f"Video-level accuracy: {vid_acc:.4f}"
            f" ({int(vid_df['correct'].sum())}/{len(vid_df)})"
        )
        print("\n--- Video-level classification report ---")
        print(
            classification_report(
                vid_df["true_label"],
                vid_df["pred_label"],
                labels=binary_labels,
                zero_division=0,
            )
        )
        cm_v = confusion_matrix(
            vid_df["true_label"],
            vid_df["pred_label"],
            labels=binary_labels,
        )
        cm_vdf = pd.DataFrame(
            cm_v, index=binary_labels, columns=binary_labels
        )
        print("Confusion matrix (video):")
        print(cm_vdf)

        # Save combined metrics.
        metrics_path = os.path.join(output_dir, "test_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("=== WINDOW-LEVEL (binary) ===\n")
            if len(valid_win):
                f.write(
                    f"Accuracy: {valid_win['correct'].mean():.4f}\n\n"
                )
                f.write(
                    classification_report(
                        valid_win["true_label"],
                        valid_win["pred_label"],
                        labels=binary_labels,
                        zero_division=0,
                    )
                )
                f.write(f"\n{cm_df.to_string()}\n\n")
            f.write("\n=== VIDEO-LEVEL (binary) ===\n")
            f.write(f"Accuracy: {vid_acc:.4f}\n\n")
            f.write(
                classification_report(
                    vid_df["true_label"],
                    vid_df["pred_label"],
                    labels=binary_labels,
                    zero_division=0,
                )
            )
            f.write(f"\n{cm_vdf.to_string()}\n")
        print(f"\nMetrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for single-seed binary sliding-window training.

    Parses ``--task`` and ``--seed`` from the command line. Each seed
    writes outputs to a ``seed_<N>`` subdirectory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Video Swin-B binary (N/A vs non-N/A) sliding-window"
            " classifier (2s window, 1s stride)."
        )
    )
    parser.add_argument(
        "--task",
        choices=["loco", "rmm"],
        required=True,
        help="Task context: 'loco' or 'rmm'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    label_col = cfg["label_col"]
    output_dir = os.path.join(cfg["output_dir"], f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"TASK : {args.task.upper()} | BINARY (N/A vs non-N/A)"
        f" | seed={args.seed}"
    )
    print(
        f"MODE : full-video sliding window"
        f" ({WINDOW_SEC}s window, {WINDOW_STRIDE}s stride)"
    )
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    pl.seed_everything(args.seed)

    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes, found {n_classes}."
    )
    print(f"\nNum classes: {n_classes} (binary)")

    model = VideoSwinFineTune(
        num_classes=n_classes,
        freeze=True,
        class_weights=dm.class_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename=(
            f"videoswin-{args.task}-binary-s{args.seed}"
            "-{epoch:02d}-{val_loss:.3f}"
        ),
    )
    early_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.fit(model, dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}")
    best_model = VideoSwinFineTune.load_from_checkpoint(
        best, num_classes=n_classes, freeze=False
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    run_inference(
        best_model, dm.test_samples, dm.label_map, device, output_dir
    )
    print("\nDone.")


if __name__ == "__main__":
    main()