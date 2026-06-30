"""Video Swin fine-tuning with 2-sec sliding windows over full videos.

Slides a 2-second window (1-second stride) across full annotated videos,
assigns each window a majority label (N/A included as a class), and trains
Video Swin-B on these windows. At inference, window-level predictions are
aggregated to video-level via average softmax probabilities.
Usage::

    python video_swin_fullvideo_sliding.py --task loco
    python video_swin_fullvideo_sliding.py --task loco --seed 123

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
# Task configuration (N/A included as a class -> num_classes is +1 vs clip)
# ---------------------------------------------------------------------------

TASK_CONFIG: Dict[str, Dict] = {
    "loco": {
        "label_col": "Locomotion",
        "num_classes": 6,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/loco_swin_run1"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 5,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/rmm_swin_run1"
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
BATCH_SIZE: int = 32

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

# Annotation timing and windowing constants.
ANN_FPS: float = 15.0
WINDOW_SEC: float = 2.0
WINDOW_STRIDE: float = 1.0
MIN_WIN_FRAMES: int = 5

# N/A sentinel — kept as a real class in this script.
NA_LABEL: str = "N/A"

# Official Kinetics-400 pretrained Swin-B checkpoint.
SWIN_CKPT_URL: str = (
    "https://github.com/SwinTransformer/storage/releases/"
    "download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth"
)
SWIN_CKPT_LOCAL: str = os.path.expanduser(
    "~/.cache/video_swin/swin_base_k400.pth"
)


# ---------------------------------------------------------------------------
# 1. Bounding-box loading (full-video HDF5)
# ---------------------------------------------------------------------------


def load_bbox_map(h5_path: str) -> Dict[int, Tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from a full-video HDF5 file.

    Args:
        h5_path: Path to the interpolated full-video HDF5 file.

    Returns:
        Mapping from annotation frame index to ``(x1, y1, x2, y2)`` bbox.
    """
    with h5py.File(h5_path, "r") as fh:
        table = fh["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {
        int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1
    }


# ---------------------------------------------------------------------------
# 2. Sliding-window builder
# ---------------------------------------------------------------------------


def get_window_label(
    frame_to_label: Dict[int, str],
    ann_start: int,
    ann_end: int,
) -> str:
    """Determine the majority label for a window of annotation frames.

    Frames absent from ``frame_to_label`` or with empty/null string values
    are treated as ``NA_LABEL``.

    Args:
        frame_to_label: Mapping from frame index to label string.
        ann_start: First annotation frame (inclusive).
        ann_end: Last annotation frame (exclusive).

    Returns:
        The most common label string in the window, or ``NA_LABEL`` if the
        window is empty.
    """
    labels: List[str] = []
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, NA_LABEL)
        if lbl in ("", "nan", "None"):
            lbl = NA_LABEL
        labels.append(lbl)
    if not labels:
        return NA_LABEL
    return Counter(labels).most_common(1)[0][0]


def build_samples(
    split_csv: str,
    label_col: str,
) -> Dict[str, List[Dict]]:
    """Slide 2-sec windows across full videos and collect sample dicts.

    Only the ``Locomotion`` and ``Repetitive_Motor_Movements`` columns are
    used from the annotation CSVs; all other columns are ignored. Videos
    where every window carries a majority label of ``N/A`` are skipped
    entirely.

    Args:
        split_csv: Path to the master split CSV
            (``latest_split_csv.csv``).
        label_col: Annotation column to extract labels from.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
        a list of sample dicts with keys ``video_path``, ``h5_path``,
        ``start_frame``, ``end_frame``, ``label_str``, and ``ann_fps``.

    Raises:
        ValueError: If a required column is absent from ``split_csv``.
    """
    split_df = pd.read_csv(split_csv)
    required_cols = [
        "video_path",
        "label_path",
        "interpolated_full_h5",
        "split",
    ]
    for col in required_cols:
        if col not in split_df.columns:
            raise ValueError(f"Split CSV missing column: '{col}'")

    by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    window_ann_frames = int(WINDOW_SEC * ANN_FPS)
    stride_ann_frames = int(WINDOW_STRIDE * ANN_FPS)

    for _, row in split_df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_full_h5"]).strip()
        sp = str(row["split"]).strip().lower()

        if sp not in by_split:
            continue
        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            print(f"  [skip] missing file(s) for: {os.path.basename(vp)}")
            continue

        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] bad label CSV ({exc}): {lp}")
            continue

        if label_col not in ann.columns or "Frame" not in ann.columns:
            print(f"  [skip] missing columns in: {lp}")
            continue

        ann = ann.sort_values("Frame").reset_index(drop=True)
        frame_to_label: Dict[int, str] = {}
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

        # Slide windows across the full annotation range.
        video_samples: List[Dict] = []
        start = 0
        while start + window_ann_frames <= total_ann_frames + stride_ann_frames:
            end = min(start + window_ann_frames, total_ann_frames)
            n_valid = end - start
            if n_valid < MIN_WIN_FRAMES:
                start += stride_ann_frames
                continue

            label_str = get_window_label(frame_to_label, start, end)
            video_samples.append(
                {
                    "video_path": vp,
                    "h5_path": hp,
                    "start_frame": int(start),
                    "end_frame": int(end - 1),
                    "label_str": label_str,
                    "ann_fps": ANN_FPS,
                }
            )
            start += stride_ann_frames

        # Skip videos whose windows are entirely N/A — no useful signal.
        has_real_label = any(s["label_str"] != NA_LABEL for s in video_samples)
        if not has_real_label:
            print(f"  [skip -- all N/A] {os.path.basename(vp)}")
            continue

        by_split[sp].extend(video_samples)

    return by_split


# ---------------------------------------------------------------------------
# 3. Dataset — bbox crop from full-video HDF5, output shape ``(C, T, H, W)``
# ---------------------------------------------------------------------------


class BBoxCropVideoDataset(Dataset):
    """Reads sliding-window segments with bbox crop and ImageNet normalization.

    Each item decodes ``num_frames`` uniformly-sampled frames from the
    annotation frame range defined by the sample dict, crops each frame to
    the nearest available bounding box, resizes to ``crop_size x crop_size``,
    and applies ImageNet normalization.

    Args:
        samples: List of sample dicts from ``build_samples``.
        label_map: Mapping from string label to integer class index.
        num_frames: Number of frames to uniformly sample per window.
        crop_size: Spatial size (height and width) after resizing.
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
        """Decode, bbox-crop, resize, and normalize a single window.

        Maps annotation frame indices to video frame indices via the ratio
        of video FPS to annotation FPS. Falls back to the temporally nearest
        annotated bbox when the exact frame is absent from the bbox map.

        Args:
            sample: Sample dict with ``video_path``, ``h5_path``,
                ``start_frame``, ``end_frame``, and ``ann_fps``.

        Returns:
            Float tensor of shape ``(C, T, H, W)`` with ImageNet
            normalization applied.

        Raises:
            IOError: If OpenCV cannot open the video file.
            ValueError: If the HDF5 bbox map is empty.
        """
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {sample['video_path']}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Convert annotation-frame index to video-frame index.
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
                # Fall back to the temporally nearest annotated bbox.
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
        # Stack gives (T, H, W, C); permute to (C, T, H, W).
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
        batch: List of ``(video_tensor, class_index)`` tuples.

    Returns:
        Tuple of ``(videos, labels)`` with shapes ``(B, C, T, H, W)``
        and ``(B,)`` respectively.
    """
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------


class H5BBoxDataModule(pl.LightningDataModule):
    """DataModule for sliding-window samples extracted from full videos.

    Reads the master split CSV, builds 2-second sliding-window samples for
    each split, and computes inverse-frequency class weights from the
    training set.

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
        """Build sliding-window samples and compute inverse-frequency weights.

        Args:
            stage: Unused Lightning stage argument. Kept for API
                compatibility.
        """
        print(
            f"\nBuilding sliding-window samples"
            f" (label_col={self.label_col})..."
        )
        by_split = build_samples(SPLIT_CSV, self.label_col)

        n_tr = len(by_split["train"])
        n_v = len(by_split["val"])
        n_te = len(by_split["test"])
        print(f"Windows  train={n_tr}  val={n_v}  test={n_te}")
        if n_tr == 0:
            raise RuntimeError(
                "No training windows built — check CSV paths and label column."
            )

        # Build label map; always ensure N/A is present.
        all_labels = sorted(
            {s["label_str"] for split in by_split.values() for s in split}
        )
        if NA_LABEL not in all_labels:
            all_labels = sorted(all_labels + [NA_LABEL])
        self.label_map = {lab: i for i, lab in enumerate(all_labels)}
        print(f"Label map ({len(self.label_map)} classes): {self.label_map}")

        for sp, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
            print(f"  {sp} distribution: {dict(sorted(dist.items()))}")

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

        print("\nClass weights (train):")
        for lab, idx in sorted(self.label_map.items(), key=lambda x: x[1]):
            print(
                f"  {lab:35s}  count={int(counts[idx]):5d}"
                f"  weight={weights[idx]:.4f}"
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
    volume and projects to ``num_classes`` logits via a linear layer.

    Args:
        backbone: Video Swin-B trunk (``SwinTransformer3D``).
        feat_dim: Channel dimension of the backbone output features.
        num_classes: Number of target action classes (including N/A).
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
        """Run a forward pass through backbone and classification head.

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

    Downloads the official checkpoint on first call and caches it to
    ``~/.cache/video_swin/``. Requires ``video-swin-transformer-pytorch``::

        pip install git+https://github.com/haofanwang/video-swin-transformer-pytorch.git

    Args:
        num_classes: Number of output action classes (including N/A).
        freeze_all_but_last_stage: When ``True``, freezes all backbone
            parameters except those in ``backbone.layers.3`` and the
            classification head.

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
    print(
        f"Loaded Swin-B K400 weights. "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )

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
    """PyTorch Lightning module for sliding-window Video Swin-B fine-tuning.

    Wraps :func:`build_video_swin_b`, adds weighted cross-entropy loss,
    and configures AdamW with a ``ReduceLROnPlateau`` scheduler.

    Args:
        num_classes: Number of output classes (including N/A).
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
# 7. Inference — window-level predictions + video-level aggregation
# ---------------------------------------------------------------------------


def run_inference(
    model: nn.Module,
    test_samples: List[Dict],
    label_map: Dict[str, int],
    device: torch.device,
    output_dir: str,
) -> None:
    """Run inference on test windows and aggregate predictions to video level.

    Saves three output files:

    - ``test_predictions_window.csv`` — one row per sliding window.
    - ``test_predictions_video.csv`` — one row per video, aggregated via
      average softmax probabilities across its windows.
    - ``test_metrics.txt`` — accuracy, classification report, and confusion
      matrix at both window and video level.

    Args:
        model: Trained model; will be switched to eval mode on ``device``.
        test_samples: List of test sample dicts from the DataModule.
        label_map: Label string to integer class-index mapping.
        device: Torch device for inference.
        output_dir: Directory to write all output files.
    """
    print("\n" + "=" * 60)
    print("INFERENCE -- TEST SET (window + video level)")
    print("=" * 60)

    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax = nn.Softmax(dim=1)
    window_rows: List[Dict] = []

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
            row: Dict = {
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "pred_label": id_to_label[top],
                "confidence": round(conf, 4),
                "correct": int(id_to_label[top] == s["label_str"]),
            }
            # Per-class probabilities are stored for video-level averaging.
            for j in range(len(id_to_label)):
                row[f"prob_{id_to_label[j]}"] = round(
                    float(probs[0, j].item()), 4
                )
            window_rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(ds)}] processed")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR on sample {i}: {exc}")
            window_rows.append(
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

    # Save window-level predictions.
    win_df = pd.DataFrame(window_rows)
    win_csv = os.path.join(output_dir, "test_predictions_window.csv")
    win_df.to_csv(win_csv, index=False)
    print(f"\nWindow predictions -> {win_csv}")

    valid_win = win_df[win_df["pred_label"] != "ERROR"]
    all_labels_sorted = sorted(label_map.keys())
    cm_df: Optional[pd.DataFrame] = None

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
                labels=all_labels_sorted,
                zero_division=0,
            )
        )
        cm = confusion_matrix(
            valid_win["true_label"],
            valid_win["pred_label"],
            labels=all_labels_sorted,
        )
        cm_df = pd.DataFrame(cm, index=all_labels_sorted, columns=all_labels_sorted)
        print("Confusion matrix (window):")
        print(cm_df)

    # Video-level aggregation via average softmax probabilities.
    print("\n" + "=" * 60)
    print("VIDEO-LEVEL AGGREGATION")
    print("=" * 60)

    prob_cols = [c for c in win_df.columns if c.startswith("prob_")]
    video_rows: List[Dict] = []

    for vpath, grp in valid_win.groupby("video_path"):
        true_lab = Counter(grp["true_label"].tolist()).most_common(1)[0][0]
        if not prob_cols:
            # Fallback to majority vote when prob columns are absent.
            pred = Counter(grp["pred_label"].tolist()).most_common(1)[0][0]
            video_rows.append(
                {
                    "video_path": vpath,
                    "true_label": true_lab,
                    "pred_label": pred,
                    "correct": int(pred == true_lab),
                    "n_windows": len(grp),
                }
            )
        else:
            avg_probs = grp[prob_cols].mean(axis=0).values
            best_idx = int(np.argmax(avg_probs))
            pred_label = prob_cols[best_idx].replace("prob_", "")
            row = {
                "video_path": vpath,
                "true_label": true_lab,
                "pred_label": pred_label,
                "correct": int(pred_label == true_lab),
                "n_windows": len(grp),
            }
            for pc, v in zip(prob_cols, avg_probs):
                row[pc] = round(float(v), 4)
            video_rows.append(row)

    vid_df = pd.DataFrame(video_rows)
    vid_csv = os.path.join(output_dir, "test_predictions_video.csv")
    vid_df.to_csv(vid_csv, index=False)
    print(f"Video predictions -> {vid_csv}")

    if len(vid_df) == 0:
        print("No video-level predictions to evaluate.")
        return

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
            labels=all_labels_sorted,
            zero_division=0,
        )
    )
    cm_v = confusion_matrix(
        vid_df["true_label"],
        vid_df["pred_label"],
        labels=all_labels_sorted,
    )
    cm_vdf = pd.DataFrame(cm_v, index=all_labels_sorted, columns=all_labels_sorted)
    print("Confusion matrix (video):")
    print(cm_vdf)

    # Write combined metrics file.
    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as fh:
        fh.write("=== WINDOW-LEVEL ===\n")
        if len(valid_win) and cm_df is not None:
            fh.write(f"Accuracy: {valid_win['correct'].mean():.4f}\n\n")
            fh.write(
                classification_report(
                    valid_win["true_label"],
                    valid_win["pred_label"],
                    labels=all_labels_sorted,
                    zero_division=0,
                )
            )
            fh.write(f"\n{cm_df.to_string()}\n\n")
        fh.write("\n=== VIDEO-LEVEL ===\n")
        fh.write(f"Accuracy: {vid_acc:.4f}\n\n")
        fh.write(
            classification_report(
                vid_df["true_label"],
                vid_df["pred_label"],
                labels=all_labels_sorted,
                zero_division=0,
            )
        )
        fh.write(f"\n{cm_vdf.to_string()}\n")
    print(f"\nMetrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run sliding-window training then test inference.

    Each seed writes outputs to a ``seed_<N>`` subdirectory under the
    task's base output directory so that multi-seed runs do not overwrite
    each other.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Video Swin-B full-video sliding-window fine-tuning "
            f"({WINDOW_SEC}s window, {WINDOW_STRIDE}s stride, N/A as class)."
        )
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
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"TASK : {args.task.upper()}"
        f"  label_col={label_col}"
        f"  seed={args.seed}"
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
        f"Expected {cfg['num_classes']} classes for task '{args.task}',"
        f" found {n_classes}. Check data or update TASK_CONFIG."
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
            f"videoswin-{args.task}-fullvideo-s{args.seed}"
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
    )
    trainer.fit(model, dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}")
    best_model = VideoSwinFineTune.load_from_checkpoint(
        best, num_classes=n_classes, freeze=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(
        best_model, dm.test_samples, dm.label_map, device, output_dir
    )
    print("\nDone.")


if __name__ == "__main__":
    main()