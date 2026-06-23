"""
End-to-end SlowFast fine-tuning pipeline.

Usage:
    python slow_fast.py --label loco
    python slow_fast.py --label rmm

"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, cast

import cv2
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SlowFast fine-tuning: loco or rmm")
    parser.add_argument(
        "--label",
        type=str,
        choices=["loco", "rmm"],
        required=True,
        help="Which label column to train on: 'loco' (Locomotion) or 'rmm' (Repetitive_Motor_Movements)",
    )
    return parser.parse_args()


# ============================================================
# CONFIG (static)
# ============================================================
SPLIT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

LABEL_CONFIGS: dict[str, dict[str, object]] = {
    "loco": {
        "label_col": "Locomotion",
        "num_classes": 5,
        "output_dir": "/orcd/data/satra/002/projects/SAILS/action_outputs_features/action_model_outputs/clips_h5/slowfast/loco/",
    },
    "rmm": {
        "label_col": "Repetitive_Motor_Movements",
        "num_classes": 4,
        "output_dir": "/orcd/data/satra/002/projects/SAILS/action_outputs_features/action_model_outputs/clips_h5/slowfast/rmm/",
    },
}

BATCH_SIZE = 32
NUM_WORKERS = 16
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True
SEED = 42

# SlowFast params
NUM_FRAMES = 32
ALPHA = 4
CROP_SIZE = 224
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

# Annotation timing / clipping
ANN_FPS = 15.0
MIN_FRAMES = 15     # < 15 annotation-frames -> skip
CLIP_FRAMES = 30    # 2 s at 15 FPS


# ============================================================
# 1. CLIPPING LOGIC  (mirrors check_clips.py exactly)
# ============================================================
def chunk_run(start: int, end: int) -> list[tuple[int, int]]:
    """
    Split a consecutive run [start, end] into clips.

    Rules:
      < 15 frames              -> []
      15-44 frames (1-2.99 s)  -> [(start, end)]
      45-59 frames (3-3.99 s)  -> [(start, start+29), (start+30, end)]
      >= 60 frames (>= 4 s)    -> 30-frame chunks; last kept if >= 15 frames
    """
    total = end - start + 1

    if total < MIN_FRAMES:
        return []

    if total < CLIP_FRAMES * 2:          # 15-59 frames
        if total < 45:                   # 15-44 -> 1 clip
            return [(start, end)]
        else:                            # 45-59 -> 2 clips
            split_pt = start + CLIP_FRAMES
            return [(start, split_pt - 1), (split_pt, end)]

    # >= 60 frames -> fixed 30-frame chunks
    clips: list[tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(s + CLIP_FRAMES - 1, end)
        if (e - s + 1) >= MIN_FRAMES:
            clips.append((s, e))
        s += CLIP_FRAMES
    return clips


def find_action_runs(
    ann: pd.DataFrame, label_col: str
) -> list[tuple[int, int, str]]:
    """
    Return list of (start_frame, end_frame, label_str) for consecutive
    same-label runs. Rows with NA or empty labels break the run.
    """
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


# ============================================================
# 2. H5 BBOX LOADING
# ============================================================
def load_bbox_map(h5_path: str) -> dict[int, tuple[int, int, int, int]]:
    """Return {ann_frame_idx: (x1, y1, x2, y2)} from interpolated H5."""
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


# ============================================================
# 3. BUILD SAMPLES  (uses 'split' column from CSV)
# ============================================================
def build_samples(
    split_csv: str, label_col: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """
    Return three lists (train_samples, val_samples, test_samples).
    Each sample is a dict: video_path, h5_path, start_frame, end_frame,
                           label_str, ann_fps, split.
    The train/val/test assignment comes from the 'split' column in the CSV.
    Clipping rules from chunk_run() are applied to every run.
    """
    df_csv = pd.read_csv(split_csv)

    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    missing = [c for c in required if c not in df_csv.columns]
    if missing:
        raise ValueError(f"Split CSV missing columns: {missing}")

    split_buckets: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for _, row in df_csv.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_anno_h5"]).strip()
        sp = str(row["split"]).strip().lower()

        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            print(f"  skip (missing file): {os.path.basename(vp)}")
            continue

        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as e:
            print(f"  skip ({e}): {lp}")
            continue

        if label_col not in ann.columns:
            print(f"  skip (no column '{label_col}'): {lp}")
            continue

        runs = find_action_runs(ann, label_col)
        for sf, ef, lab in runs:
            clips = chunk_run(sf, ef)
            for cs, ce in clips:
                split_buckets[sp].append({
                    "video_path": vp,
                    "h5_path": hp,
                    "start_frame": int(cs),
                    "end_frame": int(ce),
                    "label_str": lab,
                    "ann_fps": ANN_FPS,
                    "split": sp,
                })

    train_s = split_buckets.get("train", [])
    val_s = split_buckets.get("val", [])
    test_s = split_buckets.get("test", [])

    print(f"  Clips  ->  train: {len(train_s)} | val: {len(val_s)} | test: {len(test_s)}")
    return train_s, val_s, test_s


# ============================================================
# 4. DATASET
# ============================================================
class BBoxCropVideoDataset(Dataset[tuple[list[torch.Tensor], int]]):
    def __init__(
        self,
        samples: list[dict[str, object]],
        label_map: dict[str, int],
        num_frames: int = NUM_FRAMES,
        crop_size: int = CROP_SIZE,
        alpha: int = ALPHA,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.alpha = alpha
        self.training = training
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_segment(self, s: dict[str, object]) -> torch.Tensor:
        cap = cv2.VideoCapture(str(s["video_path"]))
        if not cap.isOpened():
            raise OSError(f"cannot open {s['video_path']}")
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vid_fps / float(str(s["ann_fps"])))))

        bbox_map = load_bbox_map(str(s["h5_path"]))
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(int(str(s["start_frame"])), int(str(s["end_frame"])) + 1)
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]

        bbox_keys = np.array(sorted(bbox_map.keys()))
        frames: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()    
            if not ret:
                frames.append(
                    cast(
                        np.ndarray[Any, np.dtype[np.uint8]],
                        np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8),
                    )
                )
                continue

            H, W = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                x1, y1, x2, y2 = bbox_map[nearest]

            x1 = max(0, min(x1, W - 1))
            x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(y1 + 1, min(y2, H))

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(cast(np.ndarray[Any, np.dtype[np.uint8]], crop))

        cap.release()
        arr = np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _pack_pathway(self, frames: torch.Tensor) -> list[torch.Tensor]:
        fast = frames
        slow_idx = torch.linspace(
            0, frames.shape[1] - 1, frames.shape[1] // self.alpha
        ).long()
        slow = torch.index_select(frames, 1, slow_idx)
        return [slow, fast]

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], int]:
        s = self.samples[idx]
        label = self.label_map[str(s["label_str"])]
        try:
            frames = self._read_segment(s)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return self._pack_pathway(frames), label
        except Exception as e:
            print(
                f"  load error {os.path.basename(str(s['video_path']))} "
                f"[{s['start_frame']}-{s['end_frame']}]: {e}"
            )
            dummy = [
                torch.zeros(3, self.num_frames // self.alpha, self.crop_size, self.crop_size),
                torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
            ]
            return dummy, label


def slowfast_collate(
    batch: list[tuple[list[torch.Tensor], int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    videos, labels = zip(*batch, strict=False)
    slow = torch.stack([v[0] for v in videos])
    fast = torch.stack([v[1] for v in videos])
    return [slow, fast], torch.tensor(labels, dtype=torch.long)


# ============================================================
# 5. DATA MODULE
# ============================================================
class H5BBoxDataModule(pl.LightningDataModule):
    def __init__(self, label_col: str, output_dir: str) -> None:
        super().__init__()
        self.label_col = label_col
        self.output_dir = output_dir
        self.label_map: dict[str, int] = {}
        self.train_samples: list[dict[str, object]] = []
        self.val_samples: list[dict[str, object]] = []
        self.test_samples: list[dict[str, object]] = []
        self.class_weights: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        print(f"\nBuilding samples  [label_col={self.label_col}] ...")
        train_s, val_s, test_s = build_samples(SPLIT_CSV, self.label_col)

        all_samples = train_s + val_s + test_s
        if not all_samples:
            raise RuntimeError("No samples built. Check split CSV / annotations.")

        labels = sorted({str(s["label_str"]) for s in all_samples})
        self.label_map = {lab: i for i, lab in enumerate(labels)}
        print(f"Label map: {self.label_map}")

        dist = Counter(str(s["label_str"]) for s in all_samples)
        print("Class distribution (all splits):")
        for k, v in sorted(dist.items()):
            print(f"  {k}: {v}")

        self.train_samples = train_s
        self.val_samples = val_s
        self.test_samples = test_s

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "label_mapping.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(test_s).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False
        )

        n_classes = len(self.label_map)
        counts = np.zeros(n_classes, dtype=np.float64)
        for s in train_s:
            counts[self.label_map[str(s["label_str"])]] += 1
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print("Class weights (from train split):")
        for lab, idx in self.label_map.items():
            print(f"  {lab:30s}  count={int(counts[idx]):4d}  weight={weights[idx]:.3f}")

    def train_dataloader(self) -> DataLoader[tuple[list[torch.Tensor], int]]:
        ds = BBoxCropVideoDataset(self.train_samples, self.label_map, training=True)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=slowfast_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[list[torch.Tensor], int]]:
        ds = BBoxCropVideoDataset(self.val_samples, self.label_map, training=False)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=slowfast_collate,
            pin_memory=True,
        )


# ============================================================
# 6. LIGHTNING MODULE
# ============================================================
class SlowFastFineTune(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = FREEZE_BACKBONE,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=True,
        )
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            print("Freezing backbone — training head only")
            for name, p in self.model.named_parameters():
                if "blocks.6" not in name:
                    p.requires_grad = False

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights: torch.Tensor | None = None

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]



    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.Adam(params, lr=LEARNING_RATE)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=3, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ============================================================
# 7. INFERENCE
# ============================================================
def run_inference(
    model: SlowFastFineTune,
    test_samples: list[dict[str, object]],
    label_map: dict[str, int],
    device: torch.device,
    output_dir: str,
) -> None:
    print("\n" + "=" * 60)
    print("INFERENCE ON TEST SET")
    print("=" * 60)

    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax = nn.Softmax(dim=1)
    rows = []

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            inputs, label = ds[i]
            inputs = [x.unsqueeze(0).to(device) for x in inputs]
            with torch.no_grad():
                logits = model(inputs)
            probs = softmax(logits)
            top = int(probs.argmax(1).item())
            conf = float(probs[0, top].item())
            rows.append({
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "pred_label": id_to_label[top],
                "confidence": round(conf, 4),
                "correct": int(id_to_label[top] == str(s["label_str"])),
            })
            print(
                f"[{i + 1}/{len(ds)}] {os.path.basename(str(s['video_path']))} "
                f"[{s['start_frame']}-{s['end_frame']}]  "
                f"true={s['label_str']}  pred={id_to_label[top]}  ({conf:.2f})"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "video_path": s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "true_label": s["label_str"],
                "pred_label": "ERROR",
                "confidence": 0.0,
                "correct": 0,
            })

    output_csv = os.path.join(output_dir, "test_predictions.csv")
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved -> {output_csv}")

    valid = df[df["pred_label"] != "ERROR"]
    if len(valid):
        acc = valid["correct"].mean()
        print(f"\nAccuracy: {acc:.4f}  ({int(valid['correct'].sum())}/{len(valid)})")
        print("\nClassification report:")
        print(classification_report(valid["true_label"], valid["pred_label"], zero_division=0))
        label_names = sorted(valid["true_label"].unique())
        cm = confusion_matrix(valid["true_label"], valid["pred_label"], labels=label_names)
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        print("Confusion matrix:")
        print(cm_df)

        with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(
                classification_report(valid["true_label"], valid["pred_label"], zero_division=0)
            )
            f.write(f"\n{cm_df.to_string()}\n")


# ============================================================
# 8. MAIN
# ============================================================
def main() -> None:
    args = parse_args()
    cfg = LABEL_CONFIGS[args.label]

    label_col = str(cfg["label_col"])
    num_classes = int(str(cfg["num_classes"]))
    output_dir = str(cfg["output_dir"])

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Mode       : {args.label.upper()}")
    print(f"  Label col  : {label_col}")
    print(f"  Num classes: {num_classes}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    pl.seed_everything(SEED)

    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    actual_classes = len(dm.label_map)
    if actual_classes != num_classes:
        print(
            f"  WARNING: config NUM_CLASSES={num_classes} but found "
            f"{actual_classes} classes in data. Using {actual_classes}."
        )
    num_classes = actual_classes

    model = SlowFastFineTune(
        num_classes=num_classes,
        freeze_backbone=FREEZE_BACKBONE,
        class_weights=dm.class_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename=f"slowfast-{args.label}-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}")

    best_model = SlowFastFineTune.load_from_checkpoint(
        best,
        num_classes=num_classes,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best_model, dm.test_samples, dm.label_map, device, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()