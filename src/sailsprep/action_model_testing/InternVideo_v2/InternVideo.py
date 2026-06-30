"""End-to-end InternVideo2-6B fine-tuning for Locomotion and RMM tasks.

Usage::

    # Single seed (default 42), 1 GPU:
    torchrun --standalone --nproc_per_node=1 \\
        internvideo2_finetune.py --task loco

    # Specific seed:
    torchrun --standalone --nproc_per_node=1 \\
        internvideo2_finetune.py --task loco --seed 123

"""

# HF cache redirect must be set before importing transformers.
import os

os.environ.setdefault("HF_HOME", "/orcd/data/satra/002/huggingface")
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", "/orcd/data/satra/002/huggingface/hub"
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE", "/orcd/data/satra/002/huggingface/hub"
)


import argparse
import json
from collections import Counter

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
        "num_classes": 5,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/clips_h5/internvideo2/loco_iv2_seeds"
        ),
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 4,
        "output_dir": (
            "/orcd/data/satra/002/projects/SAILS/"
            "action_model_outputs/clips_h5/internvideo2/rmm_iv2_seeds"
        ),
    },
}


SPLIT_CSV: str = (
    "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
)

# ---------------------------------------------------------------------------
# Global hyperparameters
# ---------------------------------------------------------------------------

#: Per-GPU mini-batch size. InternVideo2-6B is large; keep at 2 unless
BATCH_SIZE: int = 2

#: Gradient accumulation steps.
#: Effective batch = BATCH_SIZE * num_gpus * ACCUM_STEPS.
ACCUM_STEPS: int = 8

#: DataLoader worker count per GPU.
NUM_WORKERS: int = 8

#: Maximum training epochs before early stopping intervenes.
MAX_EPOCHS: int = 20

#: Peak learning rate for AdamW.  Lower than typical because the backbone
#: is huge and only the last block is unfrozen.
LEARNING_RATE: float = 5e-5

#: Seeds used for multi-run reproducibility analysis.
DEFAULT_SEEDS: list[int] = [42, 123, 456]

#: Number of frames uniformly sampled per clip.
NUM_FRAMES: int = 4

#: Spatial size (H=W) after resizing the person bounding-box crop.
CROP_SIZE: int = 224

#: ImageNet RGB channel means for normalisation.
MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)

#: ImageNet RGB channel standard deviations for normalisation.
STD: tuple[float, ...] = (0.229, 0.224, 0.225)

#: Frame rate of the behaviour annotation CSVs.
ANN_FPS: float = 15.0

#: Shortest action run (in annotation frames) to keep as a training clip.
MIN_FRAMES: int = 15

#: Target clip length in annotation frames used when chunking long runs.
CLIP_FRAMES: int = 30

#: HuggingFace model ID for InternVideo2 Stage-2 6B.
IV2_MODEL_ID: str = "OpenGVLab/InternVideo2-Stage2_6B"


# ---------------------------------------------------------------------------
# 1. Bounding-box loading
# ---------------------------------------------------------------------------


def load_bbox_map(h5_path: str) -> dict[int, tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from an interpolated annotation HDF5.

    The HDF5 stores a Pandas-style table under ``bboxes/table``.  Column
    layout inside ``values_block_1``: [frame_idx, ?, x1, y1, x2, y2].

    Args:
        h5_path: Path to the interpolated annotation HDF5 file produced
            by the SAILS preprocessing pipeline.

    Returns:
        Mapping from annotation frame index to an ``(x1, y1, x2, y2)``
        bounding box in pixel coordinates.
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
) -> list[tuple[int, int, str]]:
    """Identify contiguous frame runs sharing the same action label.

    Skips rows where the label is ``"N/A"`` or empty.  Only the columns
    ``Frame`` and ``label_col`` are read; all other annotation columns
    (Gestures_Functional_Actions, Visual_Attention, etc.) are ignored.

    Args:
        ann: Annotation DataFrame loaded from a per-session CSV under
            ``/home/aparnabg/orcd/scratch/app/all_annoation/``.
        label_col: Name of the target column, either ``"Locomotion"``
            or ``"Repetitive_Motor_Movements"``.

    Returns:
        List of ``(start_frame, end_frame, label)`` tuples in frame order.
    """
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames: list[int] = df["Frame"].astype(int).tolist()
    labels: list[str] = df[label_col].astype(str).tolist()
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
    """Split a single action run into fixed-length training clips.

    Short runs (< ``MIN_FRAMES``) are discarded.  Runs shorter than
    ``2 * CLIP_FRAMES`` are kept whole or split once at ``CLIP_FRAMES``.
    Longer runs are tiled with a stride of ``CLIP_FRAMES``.

    Args:
        start: First annotation frame index of the run (inclusive).
        end: Last annotation frame index of the run (inclusive).

    Returns:
        List of ``(clip_start, clip_end)`` tuples, each at least
        ``MIN_FRAMES`` long.
    """
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
    split_csv: str,
    label_col: str,
) -> dict[str, list[dict]]:
    """Build per-split sample dicts from the master split CSV.

    Reads ``latest_split_csv.csv`` (columns: video_path, label_path,
    interpolated_anno_h5, split, plus unused feature-path columns), locates
    each session's annotation CSV, extracts action runs from ``label_col``,
    and chunks them into clips.

    Args:
        split_csv: Absolute path to the master split CSV.
        label_col: Annotation column to use; must be ``"Locomotion"``
            or ``"Repetitive_Motor_Movements"``.

    Returns:
        Dict mapping ``"train"`` / ``"val"`` / ``"test"`` to lists of
        sample dicts.  Each dict has keys: ``video_path``, ``h5_path``,
        ``start_frame``, ``end_frame``, ``label_str``, ``ann_fps``.

    Raises:
        ValueError: If a required column is absent from ``split_csv``.
    """
    split_df = pd.read_csv(split_csv)
    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    for col in required:
        if col not in split_df.columns:
            raise ValueError(f"Split CSV missing column: {col}")

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for _, row in split_df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_anno_h5"]).strip()
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
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as exc:
            print(f"  skip ({exc}): {lp}")
            continue

        if label_col not in ann.columns:
            continue

        for sf, ef, lab in find_action_runs(ann, label_col):
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
# 3. Dataset  (on-the-fly bbox crop)  —  output shape ``(C, T, H, W)``
# ---------------------------------------------------------------------------


class BBoxCropVideoDataset(Dataset):
    """Video dataset that crops frames to the subject bounding box.

    Reads frames on-the-fly via OpenCV, crops to the nearest available
    bounding box from the interpolated HDF5, resizes to ``crop_size x
    crop_size``, and normalises to ImageNet statistics.  Output tensors
    have shape ``(C, T, H, W)`` with ``C=3``, ``T=num_frames``.

    Args:
        samples: Sample dicts produced by :func:`build_samples`.
        label_map: Mapping from label string to integer class index.
        num_frames: Number of frames uniformly sampled from each clip.
        crop_size: Spatial height and width after resizing the bbox crop.
        training: When ``True`` applies random horizontal-flip augmentation.
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
        # Pre-build normalisation tensors on CPU; moved to device by collate.
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self) -> int:  # noqa: D105
        return len(self.samples)

    def _read_segment(self, sample: dict) -> torch.Tensor:
        """Decode, crop, resize, and normalise frames for one sample.

        Seeks to each chosen annotation frame, converts the annotation-frame
        index to a video-frame index using the FPS ratio, and crops the
        frame to the nearest bounding box in the HDF5 map.

        Args:
            sample: A sample dict with keys ``video_path``, ``h5_path``,
                ``start_frame``, ``end_frame``, and ``ann_fps``.

        Returns:
            Float32 tensor of shape ``(C, T, H, W)`` normalised to
            ImageNet mean/std.

        Raises:
            IOError: If the video file cannot be opened by OpenCV.
            ValueError: If the bbox HDF5 contains no entries.
        """
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {sample['video_path']}")
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Ratio of video frames per annotation frame.
        step = max(1, int(round(vid_fps / sample["ann_fps"])))

        bbox_map = load_bbox_map(sample["h5_path"])
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(sample["start_frame"], sample["end_frame"] + 1)
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]
        bbox_keys = np.array(sorted(bbox_map.keys()))

        frames: list[np.ndarray] = []
        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                # Substitute a black frame rather than failing the whole clip.
                frames.append(
                    np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                )
                continue

            h, w = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                # Fall back to the temporally nearest available bbox.
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                x1, y1, x2, y2 = bbox_map[nearest]

            # Clamp to frame boundaries.
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
        # (T, H, W, C) -> (C, T, H, W).
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # noqa: D105
        sample = self.samples[idx]
        label = self.label_map[sample["label_str"]]
        try:
            frames = self._read_segment(sample)
            if self.training and torch.rand(1).item() < 0.5:
                # Random horizontal flip along the width dimension.
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as exc:
            print(
                f"  load error {os.path.basename(sample['video_path'])} "
                f"[{sample['start_frame']}-{sample['end_frame']}]: {exc}"
            )
            return (
                torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
                label,
            )


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(video, label)`` pairs into batched tensors.

    Args:
        batch: List of ``(video_tensor, label_int)`` pairs from the dataset.

    Returns:
        Tuple of ``(videos, labels)`` where ``videos`` has shape
        ``(B, C, T, H, W)`` and ``labels`` has shape ``(B,)``.
    """
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# 4. Lightning DataModule
# ---------------------------------------------------------------------------


class H5BBoxDataModule(pl.LightningDataModule):
    """Lightning DataModule for bbox-cropped behavioural video clips.

    Reads the master split CSV, builds train/val/test clip lists, computes
    inverse-frequency class weights, and exposes standard DataLoaders.

    Args:
        label_col: Annotation column to use as supervision signal.
        output_dir: Directory where ``label_mapping.json`` and
            ``test_split.csv`` are written during :meth:`setup`.
    """

    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.label_map: dict[str, int] = {}
        self.class_weights: torch.Tensor | None = None
        self.train_samples: list[dict] = []
        self.val_samples: list[dict] = []
        self.test_samples: list[dict] = []

    def setup(self, stage: str | None = None) -> None:
        """Build clip sample lists and compute class weights.

        Called by Lightning before any dataloader is created.  Safe to
        call on every rank because it only does CPU/disk work.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``None``.
                Not used here; all splits are always prepared.
        """
        print(f"Building samples (label_col={self.label_col})...")
        by_split = build_samples(SPLIT_CSV, self.label_col)
        n_train, n_val, n_test = (
            len(by_split["train"]),
            len(by_split["val"]),
            len(by_split["test"]),
        )
        print(f"Clips  train={n_train}  val={n_val}  test={n_test}")
        if n_train == 0:
            raise RuntimeError("No training clips built — check split CSV paths.")

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
        with open(os.path.join(self.output_dir, "label_mapping.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False
        )

        # Inverse-frequency weights so rare classes are not overwhelmed.
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

    def train_dataloader(self) -> DataLoader:  # noqa: D102
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

    def val_dataloader(self) -> DataLoader:  # noqa: D102
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
# 5. InternVideo2-6B backbone and classifier
# ---------------------------------------------------------------------------


class InternVideo2Classifier(nn.Module):
    """Linear classifier head on top of an InternVideo2 vision encoder.

    Extracts the CLS token (index 0) from a ``(B, N, D)`` sequence output,
    or mean-pools if the output is not three-dimensional, then applies
    LayerNorm and a linear projection to class logits.

    Args:
        backbone: InternVideo2 vision encoder ``nn.Module``.
        feat_dim: Embedding dimensionality of the backbone output.
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
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from a video clip tensor.

        Args:
            x: Float tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        out = self.backbone(x)
        # Handle tuple or list backbone outputs.
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 3:
            # Sequence output (B, N, D): take CLS token at position 0.
            feat = out[:, 0]
        elif out.dim() == 2:
            feat = out
        else:
            feat = out.flatten(2).mean(-1)
        return self.head(self.norm(feat))


def _freeze_except_last_block(clf: InternVideo2Classifier) -> None:
    """Freeze every parameter except the last transformer block and head.

    Searches for the block container under ``backbone.blocks``,
    ``backbone.layers``, or ``backbone.encoder.blocks`` and marks all
    parameters outside the last block, ``norm``, and ``head`` as
    non-trainable.

    Args:
        clf: Classifier whose backbone parameters will be (mostly) frozen.

    Raises:
        RuntimeError: If no transformer blocks list can be located.
    """
    print("Freezing all but the LAST transformer block + head...")
    block_module = None
    block_attr = ""
    for cand in ("blocks", "layers", "encoder.blocks"):
        mod = clf.backbone
        ok = True
        for part in cand.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                ok = False
                break
        if ok and hasattr(mod, "__len__"):
            block_module = mod
            block_attr = cand
            break

    if block_module is None:
        raise RuntimeError(
            "Could not locate transformer blocks list on vision encoder."
        )

    last_idx = len(block_module) - 1
    last_prefix = f"backbone.{block_attr}.{last_idx}"
    print(f"  Last block prefix: {last_prefix}  (depth={len(block_module)})")

    for name, param in clf.named_parameters():
        param.requires_grad = (
            name.startswith(last_prefix)
            or name.startswith("head.")
            or name.startswith("norm.")
        )

    trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    total = sum(p.numel() for p in clf.parameters())
    print(
        f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
        f"({100 * trainable / total:.2f}%)"
    )


def build_internvideo2_6b(
    num_classes: int,
    *,
    freeze_all_but_last_block: bool = True,
) -> InternVideo2Classifier:
    """Download InternVideo2-Stage2_6B and attach a classification head.

    This function **must** be called after Lightning has set the CUDA
    device for the current process (i.e. from inside
    ``LightningModule.setup()``, not from ``__init__``).  Calling it
    earlier triggers ``cudaGetDeviceCount`` Error 101 in multi-GPU
    ``torchrun`` launches.

    Args:
        num_classes: Number of output classes.
        freeze_all_but_last_block: Whether to freeze all parameters except
            the last transformer block, LayerNorm, and classification head.

    Returns:
        A ready-to-train :class:`InternVideo2Classifier`.

    Raises:
        AttributeError: If the vision encoder attribute cannot be found
            on the loaded model.
        RuntimeError: Propagated from :func:`_freeze_except_last_block` if
            transformer blocks cannot be located.
    """
    from transformers import AutoModel  # noqa: PLC0415  (late import intentional)

    print(
        f"Loading {IV2_MODEL_ID} "
        "(this is ~12 GB, may take a few minutes)..."
    )
    full_model = AutoModel.from_pretrained(
            IV2_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    # Extract only the vision tower from the multimodal model.
    if hasattr(full_model, "vision_encoder"):
        vision = full_model.vision_encoder
    elif hasattr(full_model, "visual"):
        vision = full_model.visual
    else:
        raise AttributeError(
            "Could not find vision encoder. "
            f"Available attrs: {list(full_model._modules.keys())}"
        )

    # Delete unused components to free CPU RAM before moving to GPU.
    for attr in (
        "text_encoder",
        "text_decoder",
        "vision_align",
        "text_align",
        "itm_head",
    ):
        if hasattr(full_model, attr):
            delattr(full_model, attr)
    import gc
    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    # Enable gradient checkpointing to reduce activation memory.
    if hasattr(vision, "gradient_checkpointing_enable"):
        vision.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")
    elif hasattr(vision, "set_grad_checkpointing"):
        vision.set_grad_checkpointing(True)
        print("  Gradient checkpointing: ENABLED (set_grad_checkpointing)")
    else:
        print("  WARNING: gradient checkpointing not auto-detected on backbone")

    feat_dim: int = getattr(vision, "embed_dim", 3200)
    print(f"  Vision encoder feat_dim = {feat_dim}")

    clf = InternVideo2Classifier(vision, feat_dim, num_classes)
    if freeze_all_but_last_block:
        _freeze_except_last_block(clf)
    return clf


# ---------------------------------------------------------------------------
# 6. Lightning Module
# ---------------------------------------------------------------------------


class InternVideo2FineTune(pl.LightningModule):
    """PyTorch Lightning module wrapping InternVideo2 fine-tuning.

    The backbone is loaded lazily inside :meth:`setup` (called by the
    Trainer *after* CUDA device initialisation) to avoid the
    ``cudaGetDeviceCount Error 101`` crash that occurs when ``torchrun``
    spawns multiple processes and the model is loaded in ``__init__``.

    Args:
        num_classes: Number of target action classes.
        freeze: Whether to freeze all but the last transformer block.
        class_weights: Optional inverse-frequency weights for cross-entropy.
    """

    def __init__(
        self,
        num_classes: int,
        freeze: bool = True,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        # Store class_weights as a plain attribute before setup() so that
        # register_buffer can be called there once self.device is valid.
        self._class_weights_init = class_weights
        # model is built in setup() — not here.
        self.model: InternVideo2Classifier | None = None

    def setup(self, stage: str) -> None:
        """Build the InternVideo2 backbone after CUDA device is ready.

        Lightning calls this once per process after ``set_device``, so it
        is safe to call ``AutoModel.from_pretrained`` here.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        if self.model is not None:
            # Already built (e.g. called twice in some Lightning versions).
            return
        self.model = build_internvideo2_6b(
            self.hparams.num_classes,
            freeze_all_but_last_block=self.hparams.freeze,
        )
        if self._class_weights_init is not None:
            self.register_buffer(
                "class_weights",
                self._class_weights_init.float(),
                persistent=False,
            )
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss and log train metrics.

        Args:
            batch: ``(videos, labels)`` tensors from the DataLoader.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar loss tensor for backpropagation.
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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute unweighted cross-entropy and log validation metrics.

        Args:
            batch: ``(videos, labels)`` tensors from the DataLoader.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar validation loss tensor.
        """
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict:
        """Set up AdamW with ``ReduceLROnPlateau`` on ``val_loss``.

        Returns:
            Lightning optimizer/scheduler config dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=LEARNING_RATE,
            weight_decay=0.05,
            betas=(0.9, 0.98),
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# 7. Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: nn.Module,
    test_samples: list[dict],
    label_map: dict[str, int],
    device: torch.device,
    output_csv: str,
    output_dir: str,
) -> None:
    """Run per-clip inference on the test set and write predictions + metrics.

    Loads each test clip individually (no batching) to avoid OOM on large
    models, runs under ``torch.cuda.amp.autocast`` with bfloat16, and writes
    both a CSV of per-clip predictions and a plain-text classification report.

    Args:
        model: Trained model moved to ``device`` and set to eval mode here.
        test_samples: List of sample dicts from the DataModule.
        label_map: String label to integer class-index mapping.
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
    rows: list[dict] = []

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x).float()
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
                print(f"[{i + 1}/{len(ds)}] processed")
        except Exception as exc:
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
    print(f"\nPredictions -> {output_csv}")

    valid = df[df["pred_label"] != "ERROR"]
    if len(valid) == 0:
        print("  No valid predictions — skipping metrics.")
        return

    acc = valid["correct"].mean()
    print(
        f"\nAccuracy: {acc:.4f} ({int(valid['correct'].sum())}/{len(valid)})"
    )
    all_labels = sorted(label_map.keys())
    report_str = classification_report(
        valid["true_label"],
        valid["pred_label"],
        labels=all_labels,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report_str)

    cm = confusion_matrix(
        valid["true_label"], valid["pred_label"], labels=all_labels
    )
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("Confusion matrix:")
    print(cm_df)

    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_str)
        f.write(f"\n{cm_df.to_string()}\n")
    print(f"Metrics -> {metrics_path}")


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run single-seed training + inference.

    Each seed writes all outputs to a ``seed_<N>`` subdirectory under the
    task's base output directory, so multiple seeds can run concurrently
    as independent SLURM array tasks without path collisions.
    """
    parser = argparse.ArgumentParser(
        description="InternVideo2-6B fine-tuning for behavioural action recognition."
    )
    parser.add_argument(
        "--task",
        choices=["loco", "rmm"],
        required=True,
        help=(
            "Task to train: 'loco' for Locomotion (5 classes), "
            "'rmm' for Repetitive Motor Movements (4 classes)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help=(
            "Number of GPUs per node (default: 1).  "
            "InternVideo2-6B fits on a single H100 80 GB; "
        ),
    )
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    label_col: str = cfg["label_col"]
    # Seed-scoped subdirectory prevents cross-seed output collisions.
    output_dir = os.path.join(cfg["output_dir"], f"seed_{args.seed}")
    output_csv = os.path.join(output_dir, "test_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"TASK: {args.task.upper()} | label_col={label_col} | seed={args.seed}"
    )
    print(f"{'=' * 60}")
    print(
        f"GPUs: {args.gpus}  | per-GPU batch: {BATCH_SIZE}"
        f"  | accum: {ACCUM_STEPS}"
    )
    print(f"Effective batch: {BATCH_SIZE * args.gpus * ACCUM_STEPS}")
    print(f"Output dir: {output_dir}")

    pl.seed_everything(args.seed)

    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes for {args.task}, "
        f"found {n_classes}"
    )
    print(f"\nNum classes: {n_classes}")

    # Pass class_weights here; the module stores them and registers the
    # buffer inside setup() once the CUDA device is available.
    model = InternVideo2FineTune(
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
            f"iv2-{args.task}-s{args.seed}"
            "-{epoch:02d}-{val_loss:.3f}"
        ),
    )
    early_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=args.gpus,
        # ddp_find_unused_parameters_true is required because only the last
        # transformer block is unfrozen; frozen params produce no gradients.
        strategy=(
            "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
        ),
        accumulate_grad_batches=ACCUM_STEPS,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
        # bf16 is more numerically stable than fp16 for very large models.
        precision="bf16-mixed",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dm)

    best_ckpt = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")

    # Inference runs only on rank 0 to avoid duplicate CSV writes.
    if trainer.is_global_zero:
        best_model = InternVideo2FineTune.load_from_checkpoint(
            best_ckpt,
            num_classes=n_classes,
            freeze=False,
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