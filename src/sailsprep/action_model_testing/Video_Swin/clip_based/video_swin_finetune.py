"""Video Swin fine-tuning for Loco and RMM.
Usage::

    # Single seed (default 42):
    python video_swin_finetune.py --task loco

    # Specific seed:
    python video_swin_finetune.py --task loco --seed 123

"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

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

TASK_CONFIG: Dict[str, Dict] = {
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

# Per-GPU batch size. Reduce if OOM on the H100.
BATCH_SIZE: int = 4
ACCUM_STEPS: int = 8 
# DataLoader worker processes per GPU.
NUM_WORKERS: int = 16

MAX_EPOCHS: int = 20
LEARNING_RATE: float = 1e-4
DEFAULT_SEEDS: List[int] = [42, 123, 456]

# Video Swin-B expects 32 frames at 224x224.
NUM_FRAMES: int = 32
CROP_SIZE: int = 224
MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

# Annotation timing and clipping constants.
ANN_FPS: float = 15.0
MIN_FRAMES: int = 15
CLIP_FRAMES: int = 30

# Official Kinetics-400 pretrained Swin-B checkpoint.
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


def load_bbox_map(h5_path: str) -> Dict[int, Tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from an HDF5 annotations file.

    Args:
        h5_path: Path to the interpolated annotation HDF5 file.

    Returns:
        Mapping from annotation frame index to ``(x1, y1, x2, y2)`` bbox.
    """
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
    ann: pd.DataFrame,
    label_col: str,
) -> List[Tuple[int, int, str]]:
    """Identify contiguous runs of the same action label.

    Skips rows where the label is ``"N/A"`` or empty. A run breaks whenever
    the label changes or frames are non-consecutive.

    Args:
        ann: Annotation DataFrame with ``Frame`` and ``label_col`` columns.
        label_col: Name of the label column, e.g. ``"Locomotion"``.

    Returns:
        List of ``(start_frame, end_frame, label)`` tuples in frame order.
    """
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()
    runs: List[Tuple[int, int, str]] = []
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


def chunk_run(start: int, end: int) -> List[Tuple[int, int]]:
    """Split an action run into fixed-length clips.

    Clipping rules:

    - Fewer than ``MIN_FRAMES`` (15) frames: skip entirely.
    - 15–44 frames: single clip spanning the full run.
    - 45–59 frames: two clips split at frame 30.
    - 60+ frames: non-overlapping 30-frame chunks; the final chunk is kept
      only if it contains at least ``MIN_FRAMES`` frames.

    Args:
        start: First annotation frame of the run (inclusive).
        end: Last annotation frame of the run (inclusive).

    Returns:
        List of ``(clip_start, clip_end)`` tuples.
    """
    total = end - start + 1
    if total < MIN_FRAMES:
        return []
    if total < CLIP_FRAMES * 2:
        if total < 45:
            return [(start, end)]
        split_pt = start + CLIP_FRAMES
        return [(start, split_pt - 1), (split_pt, end)]
    clips: List[Tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(s + CLIP_FRAMES - 1, end)
        if (e - s + 1) >= MIN_FRAMES:
            clips.append((s, e))
        s += CLIP_FRAMES
    return clips


def build_samples(
    split_csv: str,
    label_col: str,
) -> Dict[str, List[Dict]]:
    """Build per-split sample lists from the master split CSV.

    Reads ``video_path``, ``label_path``, ``interpolated_anno_h5``, and
    ``split`` from the CSV. Only the ``Locomotion`` and
    ``Repetitive_Motor_Movements`` label columns are used; all other
    annotation columns are ignored.

    Args:
        split_csv: Path to the master split CSV
            (``latest_split_csv.csv``).
        label_col: Annotation column to extract labels from.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
        a list of sample dicts containing ``video_path``, ``h5_path``,
        ``start_frame``, ``end_frame``, ``label_str``, and ``ann_fps``.

    Raises:
        ValueError: If a required column is absent from ``split_csv``.
    """
    split_df = pd.read_csv(split_csv)
    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    for col in required:
        if col not in split_df.columns:
            raise ValueError(f"Split CSV missing column: {col}")

    by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

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
# 3. Dataset — on-the-fly bbox crop, output shape ``(C, T, H, W)``
# ---------------------------------------------------------------------------


class BBoxCropVideoDataset(Dataset):
    """Reads video segments, crops to subject bounding box, and normalizes.

    Each item decodes ``num_frames`` uniformly-sampled frames from the
    annotation frame range, crops to the nearest available bounding box,
    resizes to ``crop_size x crop_size``, and applies ImageNet normalization.

    Args:
        samples: List of sample dicts produced by ``build_samples``.
        label_map: Mapping from string label to integer class index.
        num_frames: Number of frames to uniformly sample per clip.
        crop_size: Spatial size (height and width) after resizing the bbox crop.
        training: When ``True``, applies random horizontal flip augmentation
            with probability 0.5.
    """

    def __init__(
        self,
        samples: List[Dict],
        label_map: Dict[str, int],
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
        # Pre-built normalization tensors for broadcast subtraction/division.
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_segment(self, sample: Dict) -> torch.Tensor:
        """Decode, bbox-crop, resize, and normalize a single clip.

        Maps annotation frame indices to video frame indices using the ratio
        of video FPS to annotation FPS, then crops each frame to the subject
        bounding box (falling back to the nearest annotated frame if the
        exact frame is absent from the bbox map).

        Args:
            sample: Sample dict with ``video_path``, ``h5_path``,
                ``start_frame``, ``end_frame``, and ``ann_fps`` keys.

        Returns:
            Float tensor of shape ``(C, T, H, W)`` with ImageNet
            normalization applied.

        Raises:
            IOError: If the video file cannot be opened by OpenCV.
            ValueError: If the HDF5 bbox map is empty.
        """
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {sample['video_path']}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Step converts annotation-frame index to video-frame index.
        step = max(1, int(round(vid_fps / sample["ann_fps"])))

        bbox_map = load_bbox_map(sample["h5_path"])
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(sample["start_frame"], sample["end_frame"] + 1)
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]
        bbox_keys = np.array(sorted(bbox_map.keys()))

        frames: List[np.ndarray] = []
        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                # Replace unreadable frames with a black patch.
                frames.append(
                    np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                )
                continue

            h, w = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                # Fall back to temporally nearest annotated bbox.
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                x1, y1, x2, y2 = bbox_map[nearest]

            # Clamp coordinates to valid frame bounds.
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
        # Stack to (T, H, W, C) then permute to (C, T, H, W).
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label = self.label_map[sample["label_str"]]
        try:
            frames = self._read_segment(sample)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as exc:  # noqa: BLE001
            print(
                f"  load error {os.path.basename(sample['video_path'])}"
                f" [{sample['start_frame']}-{sample['end_frame']}]: {exc}"
            )
            return (
                torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
                label,
            )


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(video, label)`` pairs into batched tensors.

    Args:
        batch: List of ``(video_tensor, class_index)`` tuples from the
            dataset.

    Returns:
        Tuple of ``(videos, labels)`` tensors with shapes
        ``(B, C, T, H, W)`` and ``(B,)`` respectively.
    """
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------


class H5BBoxDataModule(pl.LightningDataModule):
    """DataModule that builds bbox-cropped clip samples from the split CSV.

    Reads the master split CSV, constructs clip samples for each split,
    computes inverse-frequency class weights from the training set, and
    exposes standard DataLoaders.

    Args:
        label_col: Annotation column to use, e.g. ``"Locomotion"`` or
            ``"Repetitive_Motor_Movements"``.
        output_dir: Directory to write ``label_mapping.json`` and
            ``test_split.csv`` for later evaluation.
    """

    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.label_map: Optional[Dict[str, int]] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.train_samples: List[Dict] = []
        self.val_samples: List[Dict] = []
        self.test_samples: List[Dict] = []

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """Build clip samples and compute inverse-frequency class weights.

        Args:
            stage: Unused Lightning stage argument (``"fit"`` / ``"test"``).
                Kept for API compatibility.
        """
        print(f"Building samples (label_col={self.label_col})...")
        by_split = build_samples(SPLIT_CSV, self.label_col)
        n_train = len(by_split["train"])
        n_val = len(by_split["val"])
        n_test = len(by_split["test"])
        print(f"Clips  train={n_train}  val={n_val}  test={n_test}")
        if n_train == 0:
            raise RuntimeError("No training clips built — check paths and label column.")

        all_labels = sorted(
            {s["label_str"] for split in by_split.values() for s in split}
        )
        self.label_map = {lab: i for i, lab in enumerate(all_labels)}
        print(f"Label map: {self.label_map}")

        for sp, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
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

        # Inverse-frequency weights: rarer classes receive higher weight.
        n_classes = len(self.label_map)
        counts = np.zeros(n_classes, dtype=np.float64)
        for s in self.train_samples:
            counts[self.label_map[s["label_str"]]] += 1
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights (train):")
        for lab, idx in self.label_map.items():
            print(
                f"  {lab:30s} count={int(counts[idx]):4d}"
                f"  weight={weights[idx]:.3f}"
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with shuffle and augmentation."""
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
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader without shuffle."""
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
# 5. Video Swin-B classifier
# ---------------------------------------------------------------------------


class VideoSwinClassifier(nn.Module):
    """Classification head wrapper around the Video Swin-B backbone.

    Applies global adaptive average pooling over the spatiotemporal feature
    volume produced by the backbone and projects to ``num_classes`` logits
    via a single linear layer.

    Args:
        backbone: Video Swin-B trunk (``SwinTransformer3D``).
        feat_dim: Channel dimension of the backbone output.
        num_classes: Number of target action classes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the backbone and classification head.

        Args:
            x: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Unnormalized logits of shape ``(B, num_classes)``.
        """
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        return self.head(feats)


def build_video_swin_b(
    num_classes: int,
    *,
    freeze_all_but_last_stage: bool = True,
) -> VideoSwinClassifier:
    """Instantiate Video Swin-B with K400 weights and a classification head.

    Downloads the official Kinetics-400 checkpoint on first call and caches
    it to ``~/.cache/video_swin/``. Requires the
    ``video-swin-transformer-pytorch`` package::

        pip install git+https://github.com/haofanwang/video-swin-transformer-pytorch.git

    Args:
        num_classes: Number of output action classes.
        freeze_all_but_last_stage: When ``True``, freezes all backbone
            parameters except those in ``backbone.layers.3`` and the
            classification head. This trains ~14 % of total parameters.

    Returns:
        A :class:`VideoSwinClassifier` ready for fine-tuning.

    Raises:
        ImportError: If ``video_swin_transformer`` cannot be imported.
    """
    try:
        from video_swin_transformer import SwinTransformer3D
    except ImportError as exc:
        raise ImportError(
            "Install the backbone: pip install "
            "git+https://github.com/haofanwang/video-swin-transformer-pytorch.git"
        ) from exc

    # Architecture matches swin_base_patch244_window877_kinetics400_22k.pth.
    backbone = SwinTransformer3D(
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
    )

    # Download K400 checkpoint if not already cached.
    os.makedirs(os.path.dirname(SWIN_CKPT_LOCAL), exist_ok=True)
    if not os.path.exists(SWIN_CKPT_LOCAL):
        print(f"Downloading Swin-B K400 checkpoint -> {SWIN_CKPT_LOCAL}")
        torch.hub.download_url_to_file(SWIN_CKPT_URL, SWIN_CKPT_LOCAL)

    ckpt = torch.load(SWIN_CKPT_LOCAL, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Strip "backbone." prefix and discard the K400 classification head.
    state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if not k.startswith(("cls_head.", "head."))
    }
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"Loaded Swin-B K400 weights. missing={len(missing)}  unexpected={len(unexpected)}")
    
    # ADD HERE:
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    # Swin-B: embed_dim=128, 4 stages -> feat_dim = 128 * 2^3 = 1024.
    clf = VideoSwinClassifier(backbone, feat_dim=1024, num_classes=num_classes)

    if freeze_all_but_last_stage:
        print("Freezing all parameters except backbone.layers.3 and head.")
        for name, param in clf.named_parameters():
            trainable = "backbone.layers.3" in name or name.startswith("head.")
            param.requires_grad = trainable
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        print(
            f"  Trainable: {n_trainable / 1e6:.2f}M"
            f" / {n_total / 1e6:.2f}M"
            f" ({100.0 * n_trainable / n_total:.1f}%)"
        )

    return clf


# ---------------------------------------------------------------------------
# 6. Lightning training module
# ---------------------------------------------------------------------------


class VideoSwinFineTune(pl.LightningModule):
    """PyTorch Lightning module for Video Swin-B fine-tuning.

    Wraps :func:`build_video_swin_b`, adds weighted cross-entropy loss,
    and configures AdamW with a ``ReduceLROnPlateau`` scheduler.

    Args:
        num_classes: Number of output action classes.
        freeze: Whether to freeze all backbone layers except the last stage.
        class_weights: Optional 1-D tensor of per-class loss weights.
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
        """Delegate to the inner classifier.

        Args:
            x: Video tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for a training batch.

        Args:
            batch: ``(videos, labels)`` tuple.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute unweighted cross-entropy loss for a validation batch.

        Args:
            batch: ``(videos, labels)`` tuple.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict:
        """Set up AdamW optimiser with ReduceLROnPlateau scheduler.

        Returns:
            Lightning optimizer/scheduler configuration dict.
        """
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
    test_samples: List[Dict],
    label_map: Dict[str, int],
    device: torch.device,
    output_csv: str,
    output_dir: str,
) -> None:
    """Run inference on the held-out test set and write predictions + metrics.

    Saves a per-sample CSV with columns ``video_path``, ``start_frame``,
    ``end_frame``, ``true_label``, ``pred_label``, ``confidence``, and
    ``correct``. Also writes ``test_metrics.txt`` with accuracy,
    classification report, and confusion matrix.

    Args:
        model: Trained model; will be switched to eval mode on ``device``.
        test_samples: List of test sample dicts from the DataModule.
        label_map: Label string to integer class-index mapping.
        device: Torch device for inference.
        output_csv: Destination path for the per-sample predictions CSV.
        output_dir: Directory for ``test_metrics.txt``.
    """
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET")
    print("=" * 60)

    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax = nn.Softmax(dim=1)
    rows: List[Dict] = []

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
                    "video_path": s["video_path"],
                    "start_frame": s["start_frame"],
                    "end_frame": s["end_frame"],
                    "true_label": s["label_str"],
                    "pred_label": id_to_label[top],
                    "confidence": round(conf, 4),
                    "correct": int(id_to_label[top] == s["label_str"]),
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(ds)}] processed")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR on sample {i}: {exc}")
            rows.append(
                {
                    "video_path": s["video_path"],
                    "start_frame": s["start_frame"],
                    "end_frame": s["end_frame"],
                    "true_label": s["label_str"],
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
            valid["true_label"],
            valid["pred_label"],
            labels=all_labels,
            zero_division=0,
        )
    )
    cm = confusion_matrix(
        valid["true_label"], valid["pred_label"], labels=all_labels
    )
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("Confusion matrix:")
    print(cm_df)

    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as fh:
        fh.write(f"Accuracy: {acc:.4f}\n\n")
        fh.write(
            classification_report(
                valid["true_label"],
                valid["pred_label"],
                labels=all_labels,
                zero_division=0,
            )
        )
        fh.write(f"\n{cm_df.to_string()}\n")
    print(f"Metrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run training followed by test-set inference.

    Each seed writes outputs to a ``seed_<N>`` subdirectory under the
    task's base output directory so that runs with different seeds do not
    overwrite each other.
    """
    parser = argparse.ArgumentParser(
        description="Video Swin-B fine-tuning for action recognition."
    )
    parser.add_argument(
        "--task",
        choices=["loco", "rmm"],
        required=True,
        help="Task: 'loco' for Locomotion, 'rmm' for Repetitive Motor Movements.",
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
    # Each seed gets its own output subdirectory to avoid collisions.
    output_dir = os.path.join(cfg["output_dir"], f"seed_{args.seed}")
    output_csv = os.path.join(output_dir, "test_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"TASK: {args.task.upper()}"
        f"  label_col={label_col}"
        f"  seed={args.seed}"
    )
    print(f"{'=' * 60}")
    print(f"Output dir: {output_dir}")

    pl.seed_everything(args.seed)
    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes for {args.task},"
        f" found {n_classes}. Check the label column and CSV."
    )
    print(f"\nNum classes: {n_classes}")

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
            f"videoswin-{args.task}-s{args.seed}"
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
        gradient_clip_val=1.0,
        accumulate_grad_batches=ACCUM_STEPS,  # ADD THIS LINE
    )
    trainer.fit(model, dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}")
    best_model = VideoSwinFineTune.load_from_checkpoint(
        best, num_classes=n_classes, freeze=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(
        best_model,
        dm.test_samples,
        dm.label_map,
        device,
        output_csv,
        output_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()