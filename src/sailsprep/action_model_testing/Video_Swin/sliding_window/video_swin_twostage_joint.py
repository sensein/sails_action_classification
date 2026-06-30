"""Video Swin joint two-stage classifier on 2-sec sliding windows.

Single model with shared backbone and two heads:
  - Binary head: N/A vs non-N/A (every window).
  - Action head: specific class (only non-N/A windows).

Trained jointly with a combined loss. At inference, the binary head
gates the action head: if binary predicts N/A, the final prediction
is N/A; otherwise the action head's prediction is used.

Usage::

    python video_swin_twostage_joint.py --task loco
    python video_swin_twostage_joint.py --task loco --seed 123

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

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
TASK_CONFIG: dict[str, dict] = {
    "loco": {
        "label_col": "Locomotion",
        "num_action_classes": 5,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/loco_swin_twostage"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_action_classes": 4,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/window_2sec_slide_1sec/"
            "video_swin/rmm_swin_twostage"
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

ANN_FPS: float = 15.0
WINDOW_SEC: float = 2.0
WINDOW_STRIDE: float = 1.0
MIN_WIN_FRAMES: int = 5

NA_LABEL: str = "N/A"

# Weight balancing binary vs action loss.
# binary_weight=1.0 and action_weight=1.0 means equal importance.
BINARY_LOSS_WEIGHT: float = 1.0
ACTION_LOSS_WEIGHT: float = 1.0

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
def load_bbox_map(h5_path: str) -> dict[int, tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from a full-video HDF5 file.

    Args:
        h5_path: Path to the interpolated full-video HDF5 file.

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
# 2. Sliding-window builder (returns BOTH binary and action labels)
# ---------------------------------------------------------------------------
def get_window_labels(
    frame_to_label: dict[int, str], ann_start: int, ann_end: int
) -> tuple[str, str]:
    """Determine binary and action labels for a window via majority vote.

    Args:
        frame_to_label: Mapping from frame index to original label string.
        ann_start: First annotation frame (inclusive).
        ann_end: Last annotation frame (exclusive).

    Returns:
        Tuple of ``(original_majority_label, binary_label)`` where
        ``binary_label`` is ``"N/A"`` or ``"non-N/A"``.
    """
    labels: list[str] = []
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, NA_LABEL)
        if lbl in ("", "nan", "None"):
            lbl = NA_LABEL
        labels.append(lbl)
    if not labels:
        return NA_LABEL, NA_LABEL
    majority = Counter(labels).most_common(1)[0][0]
    binary = NA_LABEL if majority == NA_LABEL else "non-N/A"
    return majority, binary


def build_samples(
    split_csv: str, label_col: str
) -> dict[str, list[dict]]:
    """Slide 2-sec windows across full videos with both label types.

    Each sample contains both the original action label (for multi-class
    head) and a binary label (for the binary head).

    Args:
        split_csv: Path to the master split CSV.
        label_col: Annotation column to use for labels.

    Returns:
        Dict mapping ``"train"``/``"val"``/``"test"`` to lists of sample
        dicts. Each dict has ``label_str`` (original action or N/A) and
        ``binary_label`` (N/A or non-N/A).

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
            continue

        try:
            ann = pd.read_csv(
                lp, encoding="utf-8-sig", keep_default_na=False
            )
            ann.columns = ann.columns.str.strip()
        except Exception as e:
            print(f"  [skip] bad label CSV ({e}): {lp}")
            continue

        if label_col not in ann.columns or "Frame" not in ann.columns:
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

        video_samples: list[dict] = []
        start = 0
        while start + window_ann_frames <= total_ann_frames + stride_ann_frames:
            end = min(start + window_ann_frames, total_ann_frames)
            n_valid = end - start
            if n_valid < MIN_WIN_FRAMES:
                start += stride_ann_frames
                continue

            action_label, binary_label = get_window_labels(
                frame_to_label, start, end
            )
            video_samples.append({
                "video_path": vp,
                "h5_path": hp,
                "start_frame": int(start),
                "end_frame": int(end - 1),
                "label_str": action_label,
                "binary_label": binary_label,
                "ann_fps": ANN_FPS,
            })
            start += stride_ann_frames

        # Keep video if at least one window has a real label.
        has_real_label = any(
            s["binary_label"] != NA_LABEL for s in video_samples
        )
        if not has_real_label:
            continue

        by_split[sp].extend(video_samples)

    return by_split


# ---------------------------------------------------------------------------
# 3. Dataset — returns (video_tensor, binary_label, action_label)
# ---------------------------------------------------------------------------
class TwoStageVideoDataset(Dataset):
    """Reads sliding-window segments and returns both label types.

    Args:
        samples: List of sample dicts from ``build_samples``.
        binary_map: Mapping for binary labels (N/A=0, non-N/A=1).
        action_map: Mapping for action labels (e.g. Walking=0, ...).
        num_frames: Number of frames to uniformly sample per window.
        crop_size: Spatial size after resizing the bbox crop.
        training: If ``True``, applies random horizontal flip.
    """

    def __init__(
        self,
        samples: list[dict],
        binary_map: dict[str, int],
        action_map: dict[str, int],
        num_frames: int = NUM_FRAMES,
        crop_size: int = CROP_SIZE,
        *,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.binary_map = binary_map
        self.action_map = action_map
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
            sample: Sample dict with video path, h5 path, frame range.

        Returns:
            Tensor of shape ``(C, T, H, W)``.
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

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, int]:
        """Return video tensor, binary label, and action label.

        For N/A windows, the action label is set to -1 (ignored in loss).
        """
        sample = self.samples[idx]
        binary_label = self.binary_map[sample["binary_label"]]

        # Action label is -1 for N/A windows (will be masked in loss).
        if sample["label_str"] in self.action_map:
            action_label = self.action_map[sample["label_str"]]
        else:
            action_label = -1

        try:
            frames = self._read_segment(sample)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, binary_label, action_label
        except Exception as e:
            print(
                f"  load error {os.path.basename(sample['video_path'])} "
                f"[{sample['start_frame']}-{sample['end_frame']}]: {e}"
            )
            return (
                torch.zeros(
                    3, self.num_frames, self.crop_size, self.crop_size
                ),
                binary_label,
                action_label,
            )


def collate_fn(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack video tensors, binary labels, and action labels."""
    videos, binary_labels, action_labels = zip(*batch)
    return (
        torch.stack(videos),
        torch.tensor(binary_labels, dtype=torch.long),
        torch.tensor(action_labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------
class TwoStageDataModule(pl.LightningDataModule):
    """DataModule for joint two-stage sliding-window training.

    Args:
        label_col: Annotation column to use (e.g. ``"Locomotion"``).
        output_dir: Directory to save label mappings and test split CSV.
    """

    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.binary_map: dict[str, int] = {NA_LABEL: 0, "non-N/A": 1}
        self.action_map: Optional[dict[str, int]] = None
        self.binary_weights: Optional[torch.Tensor] = None
        self.action_weights: Optional[torch.Tensor] = None
        self.train_samples: list[dict] = []
        self.val_samples: list[dict] = []
        self.test_samples: list[dict] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Build samples and compute class weights for both heads."""
        print(
            f"\nBuilding two-stage sliding-window samples"
            f" (label_col={self.label_col})..."
        )
        by_split = build_samples(SPLIT_CSV, self.label_col)

        n_tr = len(by_split["train"])
        n_v = len(by_split["val"])
        n_te = len(by_split["test"])
        print(f"Windows  train={n_tr}  val={n_v}  test={n_te}")
        if n_tr == 0:
            raise RuntimeError("No training windows built.")

        # Build action label map from non-N/A labels only.
        action_labels = sorted({
            s["label_str"]
            for split in by_split.values()
            for s in split
            if s["label_str"] != NA_LABEL
        })
        self.action_map = {lab: i for i, lab in enumerate(action_labels)}
        print(f"Binary map: {self.binary_map}")
        print(
            f"Action map ({len(self.action_map)} classes): {self.action_map}"
        )

        # Print distributions.
        for sp, samps in by_split.items():
            bin_dist = Counter(s["binary_label"] for s in samps)
            act_dist = Counter(
                s["label_str"] for s in samps
                if s["label_str"] != NA_LABEL
            )
            print(f"  {sp} binary: {dict(sorted(bin_dist.items()))}")
            print(f"  {sp} action: {dict(sorted(act_dist.items()))}")

        self.train_samples = by_split["train"]
        self.val_samples = by_split["val"]
        self.test_samples = by_split["test"]

        os.makedirs(self.output_dir, exist_ok=True)
        mappings = {
            "binary_map": self.binary_map,
            "action_map": self.action_map,
        }
        with open(
            os.path.join(self.output_dir, "label_mapping.json"), "w"
        ) as f:
            json.dump(mappings, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False
        )

        # Binary class weights.
        bin_counts = np.zeros(2, dtype=np.float64)
        for s in self.train_samples:
            bin_counts[self.binary_map[s["binary_label"]]] += 1
        bin_counts = np.maximum(bin_counts, 1.0)
        bin_w = bin_counts.sum() / (2 * bin_counts)
        self.binary_weights = torch.tensor(bin_w, dtype=torch.float32)

        print("\nBinary weights (train):")
        for lab, idx in sorted(
            self.binary_map.items(), key=lambda x: x[1]
        ):
            print(
                f"  {lab:15s}  count={int(bin_counts[idx]):6d}"
                f"  weight={bin_w[idx]:.4f}"
            )

        # Action class weights (non-N/A windows only).
        n_act = len(self.action_map)
        act_counts = np.zeros(n_act, dtype=np.float64)
        for s in self.train_samples:
            if s["label_str"] in self.action_map:
                act_counts[self.action_map[s["label_str"]]] += 1
        act_counts = np.maximum(act_counts, 1.0)
        act_w = act_counts.sum() / (n_act * act_counts)
        self.action_weights = torch.tensor(act_w, dtype=torch.float32)

        print("\nAction weights (train, non-N/A only):")
        for lab, idx in sorted(
            self.action_map.items(), key=lambda x: x[1]
        ):
            print(
                f"  {lab:30s}  count={int(act_counts[idx]):5d}"
                f"  weight={act_w[idx]:.4f}"
            )

    def train_dataloader(self) -> DataLoader:
        ds = TwoStageVideoDataset(
            self.train_samples,
            self.binary_map,
            self.action_map,
            training=True,
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
        ds = TwoStageVideoDataset(
            self.val_samples,
            self.binary_map,
            self.action_map,
            training=False,
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
# 5. Two-headed Video Swin-B model
# ---------------------------------------------------------------------------
class VideoSwinTwoStage(nn.Module):
    """Video Swin-B with shared backbone and two classification heads.

    Args:
        backbone: The Video Swin-B backbone module.
        feat_dim: Dimensionality of the backbone's output features.
        num_action_classes: Number of action classes (excluding N/A).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_action_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.binary_head = nn.Linear(feat_dim, 2)
        self.action_head = nn.Linear(feat_dim, num_action_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through shared backbone and both heads.

        Args:
            x: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Tuple of ``(features, binary_logits, action_logits)`` where
            features has shape ``(B, feat_dim)``, binary_logits has shape
            ``(B, 2)``, and action_logits has shape
            ``(B, num_action_classes)``.
        """
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        binary_logits = self.binary_head(feats)
        action_logits = self.action_head(feats)
        return feats, binary_logits, action_logits


def build_video_swin_twostage(
    num_action_classes: int,
    *,
    freeze_all_but_last_stage: bool = True,
) -> VideoSwinTwoStage:
    """Load Video Swin-B and attach binary + action heads.

    Args:
        num_action_classes: Number of action classes (excluding N/A).
        freeze_all_but_last_stage: If ``True``, freeze all parameters
            except ``backbone.layers.3`` and both heads.

    Returns:
        A ``VideoSwinTwoStage`` module ready for training.

    Raises:
        ImportError: If ``video_swin_transformer`` is not installed.
    """
    try:
        from video_swin_transformer import SwinTransformer3D
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

    clf = VideoSwinTwoStage(
        model, feat_dim=1024, num_action_classes=num_action_classes
    )

    if freeze_all_but_last_stage:
        print("Freezing all but last stage (layers.3) + both heads")
        for name, param in clf.named_parameters():
            if (
                "backbone.layers.3" in name
                or name.startswith("binary_head.")
                or name.startswith("action_head.")
            ):
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
# 6. Lightning Module with joint loss
# ---------------------------------------------------------------------------
class VideoSwinTwoStageModule(pl.LightningModule):
    """Lightning module for joint binary + action training.

    The loss is::

        loss = w_bin * CE(binary_logits, binary_labels)
             + w_act * CE(action_logits[non_na_mask], action_labels[non_na_mask])

    The action loss is only computed on windows where the ground truth
    is non-N/A (action_label != -1).

    Args:
        num_action_classes: Number of action classes (excluding N/A).
        freeze: Whether to freeze all but the last stage.
        binary_weights: Per-class weights for binary cross-entropy.
        action_weights: Per-class weights for action cross-entropy.
    """

    def __init__(
        self,
        num_action_classes: int,
        freeze: bool = True,
        binary_weights: Optional[torch.Tensor] = None,
        action_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["binary_weights", "action_weights"]
        )
        self.model = build_video_swin_twostage(
            num_action_classes, freeze_all_but_last_stage=freeze
        )
        if binary_weights is not None:
            self.register_buffer(
                "binary_weights", binary_weights.float(), persistent=False
            )
        else:
            self.binary_weights = None
        if action_weights is not None:
            self.register_buffer(
                "action_weights", action_weights.float(), persistent=False
            )
        else:
            self.action_weights = None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _compute_loss(
        self,
        binary_logits: torch.Tensor,
        action_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        action_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint loss from both heads.

        Args:
            binary_logits: Shape ``(B, 2)``.
            action_logits: Shape ``(B, num_action_classes)``.
            binary_labels: Shape ``(B,)`` with values 0 or 1.
            action_labels: Shape ``(B,)`` with values 0..K-1 or -1.

        Returns:
            Tuple of ``(total_loss, binary_loss, action_loss)``.
        """
        binary_loss = F.cross_entropy(
            binary_logits, binary_labels, weight=self.binary_weights
        )

        # Action loss only on non-N/A windows (action_label != -1).
        non_na_mask = action_labels >= 0
        if non_na_mask.any():
            action_loss = F.cross_entropy(
                action_logits[non_na_mask],
                action_labels[non_na_mask],
                weight=self.action_weights,
            )
        else:
            action_loss = torch.tensor(
                0.0, device=binary_logits.device, requires_grad=True
            )

        total_loss = (
            BINARY_LOSS_WEIGHT * binary_loss
            + ACTION_LOSS_WEIGHT * action_loss
        )
        return total_loss, binary_loss, action_loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, binary_labels, action_labels = batch
        _, binary_logits, action_logits = self.model(x)

        total_loss, bin_loss, act_loss = self._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )

        bin_acc = (binary_logits.argmax(1) == binary_labels).float().mean()
        non_na = action_labels >= 0
        if non_na.any():
            act_acc = (
                action_logits[non_na].argmax(1) == action_labels[non_na]
            ).float().mean()
        else:
            act_acc = torch.tensor(0.0)

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_bin_loss", bin_loss, prog_bar=False)
        self.log("train_act_loss", act_loss, prog_bar=False)
        self.log("train_bin_acc", bin_acc, prog_bar=True)
        self.log("train_act_acc", act_acc, prog_bar=True)
        return total_loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, binary_labels, action_labels = batch
        _, binary_logits, action_logits = self.model(x)

        total_loss, bin_loss, act_loss = self._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )

        bin_acc = (binary_logits.argmax(1) == binary_labels).float().mean()
        non_na = action_labels >= 0
        if non_na.any():
            act_acc = (
                action_logits[non_na].argmax(1) == action_labels[non_na]
            ).float().mean()
        else:
            act_acc = torch.tensor(0.0)

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_bin_loss", bin_loss, prog_bar=False)
        self.log("val_act_loss", act_loss, prog_bar=False)
        self.log("val_bin_acc", bin_acc, prog_bar=True)
        self.log("val_act_acc", act_acc, prog_bar=True)
        return total_loss

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
# 7. Inference — gated two-stage prediction
# ---------------------------------------------------------------------------
def run_inference(
    model: nn.Module,
    test_samples: list[dict],
    binary_map: dict[str, int],
    action_map: dict[str, int],
    device: torch.device,
    output_dir: str,
) -> None:
    """Run gated two-stage inference on test windows.

    For each window:
      1. Binary head predicts N/A or non-N/A.
      2. If non-N/A, action head predicts the specific class.
      3. If N/A, final prediction is N/A regardless of action head.

    Results are saved at both window and video level.

    Args:
        model: Trained two-stage model.
        test_samples: List of test sample dicts.
        binary_map: Binary label mapping.
        action_map: Action label mapping.
        device: Torch device.
        output_dir: Directory to save outputs.
    """
    print("\n" + "=" * 60)
    print("INFERENCE -- TWO-STAGE (binary gate + action)")
    print("=" * 60)

    model.eval().to(device)
    bin_id_to_label = {v: k for k, v in binary_map.items()}
    act_id_to_label = {v: k for k, v in action_map.items()}
    ds = TwoStageVideoDataset(
        test_samples, binary_map, action_map, training=False
    )
    softmax = nn.Softmax(dim=1)
    window_rows: list[dict] = []

    # Build the full label set for reporting (action labels + N/A).
    all_labels = sorted(action_map.keys()) + [NA_LABEL]

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            x, _, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                _, binary_logits, action_logits = model(x)

            bin_probs = softmax(binary_logits)
            act_probs = softmax(action_logits)

            bin_pred_idx = int(bin_probs.argmax(1).item())
            bin_pred = bin_id_to_label[bin_pred_idx]

            # Gated prediction: if binary says N/A, final = N/A.
            if bin_pred == NA_LABEL:
                final_pred = NA_LABEL
                final_conf = float(bin_probs[0, 0].item())
            else:
                act_pred_idx = int(act_probs.argmax(1).item())
                final_pred = act_id_to_label[act_pred_idx]
                final_conf = float(act_probs[0, act_pred_idx].item())

            # True label for comparison.
            true_label = s["label_str"]

            row = {
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": true_label,
                "true_binary": s["binary_label"],
                "pred_binary": bin_pred,
                "pred_action": final_pred,
                "confidence": round(final_conf, 4),
                "correct_binary": int(bin_pred == s["binary_label"]),
                "correct_final": int(final_pred == true_label),
                "prob_NA": round(float(bin_probs[0, 0].item()), 4),
                "prob_nonNA": round(float(bin_probs[0, 1].item()), 4),
            }
            # Add per-action probabilities.
            for j, lab in act_id_to_label.items():
                row[f"prob_{lab}"] = round(
                    float(act_probs[0, j].item()), 4
                )
            window_rows.append(row)

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i + 1}/{len(ds)}]"
                    f" {os.path.basename(s['video_path'])}"
                    f" [{s['start_frame']}-{s['end_frame']}]"
                    f" true={true_label} bin={bin_pred}"
                    f" final={final_pred} ({final_conf:.2f})"
                )
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            window_rows.append({
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "true_binary": s["binary_label"],
                "pred_binary": "ERROR",
                "pred_action": "ERROR",
                "confidence": 0.0,
                "correct_binary": 0,
                "correct_final": 0,
            })

    # Save window-level predictions.
    win_df = pd.DataFrame(window_rows)
    win_csv = os.path.join(output_dir, "test_predictions_window.csv")
    win_df.to_csv(win_csv, index=False)
    print(f"\nWindow predictions -> {win_csv}")

    valid_win = win_df[win_df["pred_action"] != "ERROR"]

    if len(valid_win):
        # Binary accuracy.
        bin_acc = valid_win["correct_binary"].mean()
        print(f"\nBinary accuracy: {bin_acc:.4f}")
        binary_labels_list = [NA_LABEL, "non-N/A"]
        print("\n--- Binary classification report ---")
        print(
            classification_report(
                valid_win["true_binary"],
                valid_win["pred_binary"],
                labels=binary_labels_list,
                zero_division=0,
            )
        )
        bin_cm = confusion_matrix(
            valid_win["true_binary"],
            valid_win["pred_binary"],
            labels=binary_labels_list,
        )
        bin_cm_df = pd.DataFrame(
            bin_cm,
            index=binary_labels_list,
            columns=binary_labels_list,
        )
        print("Binary confusion matrix:")
        print(bin_cm_df)

        # Final (gated) accuracy.
        final_acc = valid_win["correct_final"].mean()
        print(f"\nFinal gated accuracy: {final_acc:.4f}")
        print("\n--- Final gated classification report ---")
        print(
            classification_report(
                valid_win["true_label"],
                valid_win["pred_action"],
                labels=all_labels,
                zero_division=0,
            )
        )
        final_cm = confusion_matrix(
            valid_win["true_label"],
            valid_win["pred_action"],
            labels=all_labels,
        )
        final_cm_df = pd.DataFrame(
            final_cm, index=all_labels, columns=all_labels
        )
        print("Final confusion matrix:")
        print(final_cm_df)

    # Video-level aggregation.
    print("\n" + "=" * 60)
    print("VIDEO-LEVEL AGGREGATION (two-stage)")
    print("=" * 60)

    video_rows: list[dict] = []
    for vpath, grp in valid_win.groupby("video_path"):
        pred_counts = Counter(grp["pred_action"].tolist())
        pred_label = pred_counts.most_common(1)[0][0]
        true_lab = Counter(
            grp["true_label"].tolist()
        ).most_common(1)[0][0]
        video_rows.append({
            "video_path": vpath,
            "true_label": true_lab,
            "pred_label": pred_label,
            "correct": int(pred_label == true_lab),
            "n_windows": len(grp),
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

    # Save metrics.
    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== BINARY (window) ===\n")
        if len(valid_win):
            f.write(f"Accuracy: {bin_acc:.4f}\n\n")
            f.write(
                classification_report(
                    valid_win["true_binary"],
                    valid_win["pred_binary"],
                    labels=binary_labels_list,
                    zero_division=0,
                )
            )
            f.write(f"\n{bin_cm_df.to_string()}\n\n")

        f.write("\n=== FINAL GATED (window) ===\n")
        if len(valid_win):
            f.write(f"Accuracy: {final_acc:.4f}\n\n")
            f.write(
                classification_report(
                    valid_win["true_label"],
                    valid_win["pred_action"],
                    labels=all_labels,
                    zero_division=0,
                )
            )
            f.write(f"\n{final_cm_df.to_string()}\n\n")

        f.write("\n=== VIDEO-LEVEL ===\n")
        if len(vid_df):
            f.write(f"Accuracy: {vid_acc:.4f}\n\n")
            f.write(
                classification_report(
                    vid_df["true_label"],
                    vid_df["pred_label"],
                    labels=all_labels,
                    zero_division=0,
                )
            )
    print(f"\nMetrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for joint two-stage training + inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Video Swin-B joint two-stage (binary + action) classifier"
            " on 2s sliding windows."
        )
    )
    parser.add_argument(
        "--task",
        choices=["loco", "rmm"],
        required=True,
        help="Task: 'loco' for Locomotion, 'rmm' for RMM.",
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
        f"TASK : {args.task.upper()} | TWO-STAGE JOINT"
        f" | seed={args.seed}"
    )
    print(
        f"MODE : sliding window"
        f" ({WINDOW_SEC}s window, {WINDOW_STRIDE}s stride)"
    )
    print(f"LOSS : {BINARY_LOSS_WEIGHT} * binary + {ACTION_LOSS_WEIGHT} * action")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    pl.seed_everything(args.seed)

    dm = TwoStageDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_action = len(dm.action_map)
    assert n_action == cfg["num_action_classes"], (
        f"Expected {cfg['num_action_classes']} action classes,"
        f" found {n_action}."
    )
    print(f"\nAction classes: {n_action}  |  Binary classes: 2")

    model = VideoSwinTwoStageModule(
        num_action_classes=n_action,
        freeze=True,
        binary_weights=dm.binary_weights,
        action_weights=dm.action_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename=(
            f"videoswin-{args.task}-twostage-s{args.seed}"
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
    best_model = VideoSwinTwoStageModule.load_from_checkpoint(
        best, num_action_classes=n_action, freeze=False
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    run_inference(
        best_model,
        dm.test_samples,
        dm.binary_map,
        dm.action_map,
        device,
        output_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()