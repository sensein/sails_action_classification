"""
VideoMAE V2 ViT-Base — 2-Stage Full-Video Sliding Window Fine-Tuning
====================================================================
Stage 1 (binary head)  : N/A  vs  non-N/A
Stage 2 (fine-grained) : classify among real activity classes only

Architecture
------------
  Shared ViT-B backbone (frozen except last block blocks.11 + fc_norm)
  ├── binary_head : Linear(768 → 2)   trained on ALL windows
  └── fg_head     : Linear(768 → K)   trained on non-N/A windows only
  Combined loss: loss = loss_binary + λ * loss_fg   (λ=1.0)

Data / windowing
----------------
  • Full video via interpolated_full_h5 (bbox crop per window)
  • 2-sec sliding window, 1-sec stride, across entire annotated span
  • VIDEO-LEVEL FILTER: video must have >=1 non-N/A window to be included
  • N/A windows used for Stage-1 training but excluded from Stage-2 loss

Metrics saved
-------------
  Stage 1 (binary):
    Accuracy, precision, recall, F1, ROC-AUC,
    active-recall @ multiple thresholds
    → window-level + video-level CSVs + txt/json

  Stage 2 (fine-grained, non-N/A windows only):
    Top-1, Top-2, balanced acc, Kappa, MCC, mAP,
    micro/macro/weighted P/R/F1, per-class AP,
    confusion matrix, full classification report
    → window-level + video-level CSVs + txt/json

Usage:
    python videomae2_twostage_sliding.py --task loco
    python videomae2_twostage_sliding.py --task rmm

Setup (one-time):
    wget -O modeling_finetune.py \
      "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py"
    mkdir -p ~/.cache/videomae2
    wget -O ~/.cache/videomae2/vit_b_k710_dl_from_giant.pth \
      "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth"
    pip install timm
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import cv2
import pytorch_lightning as pl
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

# ============================================================
# TASK CONFIG
# ============================================================
TASK_CONFIG = {
    "loco": {
        "label_col":      "Locomotion",
        "num_fg_classes": 5,
        "output_dir":     "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco_twostage",
    },
    "rmm": {
        "label_col":      "Repetitive_Motor_Movements",
        "num_fg_classes": 4,
        "output_dir":     "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm_twostage",
    },
}

SPLIT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

# ============================================================
# GLOBAL CONFIG
# ============================================================
BATCH_SIZE     = 8
NUM_WORKERS    = 4
MAX_EPOCHS     = 50
LEARNING_RATE  = 1e-4
SEED           = 42
FG_LOSS_LAMBDA = 1.0

NUM_FRAMES  = 16
CROP_SIZE   = 224
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)

ANN_FPS        = 15.0
WINDOW_SEC     = 2.0
WINDOW_STRIDE  = 1.0
MIN_WIN_FRAMES = 5

NA_LABEL   = "N/A"
BIN_NA     = 0
BIN_ACTIVE = 1

VMAE2_CKPT = os.path.expanduser("~/.cache/videomae2/vit_b_k710_dl_from_giant.pth")

BINARY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ============================================================
# 1. BBOX LOADING
# ============================================================
def load_bbox_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


# ============================================================
# 2. SLIDING WINDOW BUILDER
# ============================================================
def get_window_label(frame_to_label, ann_start, ann_end):
    labels = []
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, NA_LABEL)
        if lbl in ("", "nan", "None"):
            lbl = NA_LABEL
        labels.append(lbl)
    if not labels:
        return NA_LABEL
    return Counter(labels).most_common(1)[0][0]


def build_samples(split_csv, label_col, fg_label_map):
    """
    Build sliding-window samples.

    VIDEO-LEVEL FILTER: only include videos with >=1 non-N/A window.

    Each sample carries:
        video_path, h5_path, start_frame, end_frame, ann_fps,
        label_str   : original string label
        bin_label   : 0=N/A, 1=active
        fg_label    : int index in fg_label_map  (-1 for N/A windows)
    """
    split_df = pd.read_csv(split_csv)
    for c in ["video_path", "label_path", "interpolated_full_h5", "split"]:
        if c not in split_df.columns:
            raise ValueError(f"Split CSV missing column: '{c}'")

    by_split          = {"train": [], "val": [], "test": []}
    window_ann_frames = int(WINDOW_SEC    * ANN_FPS)
    stride_ann_frames = int(WINDOW_STRIDE * ANN_FPS)
    skipped_files     = 0
    skipped_allna     = 0

    for _, row in split_df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_full_h5"]).strip()
        sp = str(row["split"]).strip().lower()

        if sp not in by_split:
            continue
        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            skipped_files += 1
            print(f"  [skip-missing] {os.path.basename(vp)}")
            continue

        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as e:
            skipped_files += 1
            print(f"  [skip-csv] {lp}: {e}")
            continue

        if label_col not in ann.columns or "Frame" not in ann.columns:
            skipped_files += 1
            print(f"  [skip-cols] {lp}")
            continue

        ann = ann.sort_values("Frame").reset_index(drop=True)
        frame_to_label = {}
        for _, r in ann.iterrows():
            fn  = int(r["Frame"])
            lbl = str(r[label_col]).strip()
            if lbl in ("", "nan", "None"):
                lbl = NA_LABEL
            frame_to_label[fn] = lbl

        if not frame_to_label:
            continue

        total_ann_frames = max(frame_to_label.keys()) + 1
        video_samples    = []
        start            = 0

        while start + window_ann_frames <= total_ann_frames + stride_ann_frames:
            end     = min(start + window_ann_frames, total_ann_frames)
            n_valid = end - start
            if n_valid < MIN_WIN_FRAMES:
                start += stride_ann_frames
                continue

            lbl_str = get_window_label(frame_to_label, start, end)
            bin_lbl = BIN_NA if lbl_str == NA_LABEL else BIN_ACTIVE
            fg_lbl  = fg_label_map.get(lbl_str, -1)

            video_samples.append({
                "video_path":  vp,
                "h5_path":     hp,
                "start_frame": int(start),
                "end_frame":   int(end - 1),
                "ann_fps":     ANN_FPS,
                "label_str":   lbl_str,
                "bin_label":   bin_lbl,
                "fg_label":    fg_lbl,
            })
            start += stride_ann_frames

        # VIDEO-LEVEL FILTER
        if not any(s["bin_label"] == BIN_ACTIVE for s in video_samples):
            skipped_allna += 1
            print(f"  [skip-allNA] {os.path.basename(vp)}")
            continue

        by_split[sp].extend(video_samples)

    print(f"\nBuild done. skipped_files={skipped_files}  skipped_all_NA={skipped_allna}")
    return by_split


# ============================================================
# 3. DATASET
# ============================================================
class TwoStageVideoDataset(Dataset):
    """Returns (video_tensor, bin_label, fg_label). fg_label=-1 for N/A windows."""
    def __init__(self, samples, num_frames=NUM_FRAMES,
                 crop_size=CROP_SIZE, training=False):
        self.samples    = samples
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.training   = training
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std  = torch.tensor(STD ).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _read_segment(self, s):
        cap = cv2.VideoCapture(s["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {s['video_path']}")

        vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step     = max(1, int(round(vid_fps / s["ann_fps"])))
        bbox_map = load_bbox_map(s["h5_path"])
        if not bbox_map:
            cap.release()
            raise ValueError("empty bbox map")

        ann_frames = np.arange(s["start_frame"], s["end_frame"] + 1)
        idxs       = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen     = ann_frames[idxs]
        bbox_keys  = np.array(sorted(bbox_map.keys()))
        frames     = []

        for af in chosen:
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((self.crop_size, self.crop_size, 3), np.uint8))
                continue

            H, W = frame.shape[:2]
            if af in bbox_map:
                x1, y1, x2, y2 = bbox_map[af]
            else:
                nearest = int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])
                x1, y1, x2, y2 = bbox_map[nearest]

            x1 = max(0, min(x1, W-1)); x2 = max(x1+1, min(x2, W))
            y1 = max(0, min(y1, H-1)); y2 = max(y1+1, min(y2, H))

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(crop)

        cap.release()
        arr    = np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            frames = self._read_segment(s)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
        except Exception as e:
            print(f"  load error {os.path.basename(s['video_path'])} "
                  f"[{s['start_frame']}-{s['end_frame']}]: {e}")
            frames = torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)

        return (
            frames,
            torch.tensor(s["bin_label"], dtype=torch.long),
            torch.tensor(s["fg_label"],  dtype=torch.long),
        )


def collate_fn(batch):
    videos, bin_labels, fg_labels = zip(*batch)
    return torch.stack(videos), torch.stack(bin_labels), torch.stack(fg_labels)


# ============================================================
# 4. DATA MODULE
# ============================================================
class TwoStageDataModule(pl.LightningDataModule):
    def __init__(self, label_col, num_fg_classes, output_dir):
        super().__init__()
        self.label_col      = label_col
        self.num_fg_classes = num_fg_classes
        self.output_dir     = output_dir
        self.fg_label_map   = None
        self.id_to_fg       = None
        self.train_samples  = None
        self.val_samples    = None
        self.test_samples   = None
        self.bin_weights    = None
        self.fg_weights     = None

    def setup(self, stage=None):
        print(f"\nBuilding 2-stage sliding-window samples "
              f"(label_col={self.label_col})...")

        # Pass 1: discover all non-N/A label strings
        split_df = pd.read_csv(SPLIT_CSV)
        all_fg   = set()
        for lp in split_df["label_path"].dropna().unique():
            lp = str(lp).strip()
            if not os.path.exists(lp):
                continue
            try:
                df = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
                df.columns = df.columns.str.strip()
                if self.label_col not in df.columns:
                    continue
                for v in df[self.label_col].astype(str).str.strip().unique():
                    if v not in ("", "nan", "None", NA_LABEL, "N/A"):
                        all_fg.add(v)
            except Exception:
                continue

        fg_classes        = sorted(all_fg)
        self.fg_label_map = {c: i for i, c in enumerate(fg_classes)}
        self.id_to_fg     = {i: c for c, i in self.fg_label_map.items()}
        print(f"Fine-grained classes ({len(fg_classes)}): {fg_classes}")

        assert len(fg_classes) == self.num_fg_classes, (
            f"Expected {self.num_fg_classes} fg classes, discovered {len(fg_classes)}. "
            f"Update TASK_CONFIG or check your data.")

        # Pass 2: build windowed samples
        by_split = build_samples(SPLIT_CSV, self.label_col, self.fg_label_map)

        n_tr, n_v, n_te = (len(by_split["train"]),
                           len(by_split["val"]),
                           len(by_split["test"]))
        print(f"Windows  train={n_tr}  val={n_v}  test={n_te}")
        if n_tr == 0:
            raise RuntimeError("No training windows built.")

        self.train_samples = by_split["train"]
        self.val_samples   = by_split["val"]
        self.test_samples  = by_split["test"]

        for sp_name, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
            print(f"  {sp_name}: {dict(sorted(dist.items()))}")

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "fg_label_mapping.json"), "w") as f:
            json.dump(self.fg_label_map, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False)

        # Binary class weights
        bin_counts = np.zeros(2, dtype=np.float64)
        for s in self.train_samples:
            bin_counts[s["bin_label"]] += 1
        bin_counts      = np.maximum(bin_counts, 1.0)
        bin_w           = bin_counts.sum() / (2 * bin_counts)
        self.bin_weights = torch.tensor(bin_w, dtype=torch.float32)

        # Fine-grained class weights (non-N/A windows only)
        fg_counts = np.zeros(self.num_fg_classes, dtype=np.float64)
        for s in self.train_samples:
            if s["fg_label"] >= 0:
                fg_counts[s["fg_label"]] += 1
        fg_counts       = np.maximum(fg_counts, 1.0)
        fg_w            = fg_counts.sum() / (self.num_fg_classes * fg_counts)
        self.fg_weights = torch.tensor(fg_w, dtype=torch.float32)

        print("\nBinary weights (train):")
        for lbl, idx in [("N/A (0)", 0), ("active (1)", 1)]:
            print(f"  {lbl:12s}  count={int(bin_counts[idx]):5d}  "
                  f"weight={bin_w[idx]:.4f}")
        print("\nFine-grained weights (train):")
        for cls, idx in sorted(self.fg_label_map.items(), key=lambda x: x[1]):
            print(f"  {cls:35s}  count={int(fg_counts[idx]):5d}  "
                  f"weight={fg_w[idx]:.4f}")

    def train_dataloader(self):
        ds = TwoStageVideoDataset(self.train_samples, training=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn,
                          pin_memory=True)

    def val_dataloader(self):
        ds = TwoStageVideoDataset(self.val_samples, training=False)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn,
                          pin_memory=True)


# ============================================================
# 5. MODEL — shared ViT-B backbone + binary head + fg head
# ============================================================
class TwoStageVideoMAE2(nn.Module):
    """
    Shared ViT-B backbone (features from fc_norm output, 768-d)
    ├── binary_head : Linear(768 → 2)
    └── fg_head     : Linear(768 → K)
    """
    FEAT_DIM = 768   # ViT-B embed_dim

    def __init__(self, num_fg_classes):
        super().__init__()
        self.backbone    = self._load_backbone()
        self.binary_head = nn.Linear(self.FEAT_DIM, 2)
        self.fg_head     = nn.Linear(self.FEAT_DIM, num_fg_classes)

        for head in [self.binary_head, self.fg_head]:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _load_backbone():
        try:
            from modeling_finetune import vit_base_patch16_224
        except ImportError as e:
            raise ImportError(
                "Download modeling_finetune.py:\n"
                "  wget -O modeling_finetune.py "
                "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/"
                "master/models/modeling_finetune.py"
            ) from e

        # Build with K710 classes to load checkpoint, then strip the head
        backbone = vit_base_patch16_224(num_classes=710)

        if not os.path.exists(VMAE2_CKPT):
            print(f"Downloading VideoMAE V2 ViT-B K710 → {VMAE2_CKPT}")
            os.makedirs(os.path.dirname(VMAE2_CKPT), exist_ok=True)
            torch.hub.download_url_to_file(
                "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/"
                "vit_b_k710_dl_from_giant.pth",
                VMAE2_CKPT,
            )

        ckpt  = torch.load(VMAE2_CKPT, map_location="cpu")
        state = ckpt.get("module", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"Loaded ViT-B K710: missing={len(missing)} unexpected={len(unexpected)}")

        # Remove the original head — we attach our own heads externally
        backbone.head = nn.Identity()
        return backbone

    def get_features(self, x):
        """Run backbone up to fc_norm, return (B, 768) feature vector."""
        # vit_base_patch16_224 forward returns logits via self.head.
        # With head=Identity it returns the 768-d fc_norm output directly.
        return self.backbone(x)   # (B, 768)

    def forward(self, x):
        feat       = self.get_features(x)     # (B, 768)
        bin_logits = self.binary_head(feat)   # (B, 2)
        fg_logits  = self.fg_head(feat)       # (B, K)
        return bin_logits, fg_logits

    def freeze_except_last_block(self):
        print("Freezing all but blocks.11 + fc_norm + both heads")
        for name, p in self.named_parameters():
            p.requires_grad = (
                "blocks.11." in name
                or "fc_norm" in name
                or name.startswith("binary_head.")
                or name.startswith("fg_head.")
            )
        tr  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {tr/1e6:.2f}M / {tot/1e6:.2f}M")


# ============================================================
# 6. LIGHTNING MODULE
# ============================================================
class TwoStageFineTune(pl.LightningModule):
    def __init__(self, num_fg_classes, freeze=True,
                 bin_weights=None, fg_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["bin_weights", "fg_weights"])
        self.model = TwoStageVideoMAE2(num_fg_classes)
        if freeze:
            self.model.freeze_except_last_block()

        self.bin_weights = None
        self.fg_weights  = None
        if bin_weights is not None:
            self.register_buffer("bin_weights", bin_weights.float(), persistent=False)
        if fg_weights is not None:
            self.register_buffer("fg_weights",  fg_weights.float(),  persistent=False)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, bin_logits, fg_logits, bin_labels, fg_labels, stage):
        loss_bin = F.cross_entropy(
            bin_logits, bin_labels,
            weight=self.bin_weights if stage == "train" else None)

        mask    = fg_labels >= 0
        loss_fg = torch.tensor(0.0, device=bin_logits.device)
        if mask.any():
            loss_fg = F.cross_entropy(
                fg_logits[mask], fg_labels[mask],
                weight=self.fg_weights if stage == "train" else None)

        return loss_bin + FG_LOSS_LAMBDA * loss_fg, loss_bin, loss_fg

    def training_step(self, batch, _):
        x, bin_y, fg_y = batch
        bin_l, fg_l    = self.model(x)
        loss, l_bin, l_fg = self._compute_loss(bin_l, fg_l, bin_y, fg_y, "train")

        bin_acc = (bin_l.argmax(1) == bin_y).float().mean()
        mask    = fg_y >= 0
        fg_acc  = (fg_l[mask].argmax(1) == fg_y[mask]).float().mean() \
                  if mask.any() else torch.tensor(0.0)

        self.log_dict({"train_loss": loss, "train_bin_loss": l_bin,
                       "train_fg_loss": l_fg, "train_bin_acc": bin_acc,
                       "train_fg_acc": fg_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, bin_y, fg_y = batch
        bin_l, fg_l    = self.model(x)
        loss, l_bin, l_fg = self._compute_loss(bin_l, fg_l, bin_y, fg_y, "val")

        bin_acc = (bin_l.argmax(1) == bin_y).float().mean()
        mask    = fg_y >= 0
        fg_acc  = (fg_l[mask].argmax(1) == fg_y[mask]).float().mean() \
                  if mask.any() else torch.tensor(0.0)

        self.log_dict({"val_loss": loss, "val_bin_loss": l_bin,
                       "val_fg_loss": l_fg, "val_bin_acc": bin_acc,
                       "val_fg_acc": fg_acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt    = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.05)
        sch    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     opt, mode="min", patience=3, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ============================================================
# 7. METRICS HELPERS
# ============================================================
def binary_metrics_report(y_true_bin, y_prob_active, output_dir, tag):
    lines = []

    def log(s=""):
        print(s); lines.append(s)

    y_pred   = (y_prob_active >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_bin == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_bin == 0)).sum())

    acc     = (y_pred == y_true_bin).mean()
    prec    = tp / max(tp + fp, 1)
    rec     = tp / max(tp + fn, 1)
    f1      = 2 * prec * rec / max(prec + rec, 1e-9)
    spec    = tn / max(tn + fp, 1)
    roc_auc = roc_auc_score(y_true_bin, y_prob_active) \
              if len(np.unique(y_true_bin)) > 1 else float("nan")

    log("\n" + "=" * 60)
    log(f"STAGE-1 BINARY METRICS  [{tag}]")
    log("=" * 60)
    log(f"  Accuracy          : {acc:.4f}")
    log(f"  Precision (active): {prec:.4f}")
    log(f"  Recall (active)   : {rec:.4f}")
    log(f"  Specificity (N/A) : {spec:.4f}")
    log(f"  F1 (active)       : {f1:.4f}")
    log(f"  ROC-AUC           : {roc_auc:.4f}")
    log(f"\n  Confusion matrix (rows=actual, cols=pred):")
    log(f"              pred_NA  pred_active")
    log(f"  actual_NA   {tn:8d}  {fp:10d}")
    log(f"  actual_act  {fn:8d}  {tp:10d}")

    log(f"\n  Active-Recall @ thresholds:")
    log(f"  {'Thresh':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged%':>10}")
    thresh_rows = []
    for thr in BINARY_THRESHOLDS:
        yp   = (y_prob_active >= thr).astype(int)
        tp_t = int(((yp == 1) & (y_true_bin == 1)).sum())
        fp_t = int(((yp == 1) & (y_true_bin == 0)).sum())
        fn_t = int(((yp == 0) & (y_true_bin == 1)).sum())
        p_t  = tp_t / max(tp_t + fp_t, 1)
        r_t  = tp_t / max(tp_t + fn_t, 1)
        f1_t = 2 * p_t * r_t / max(p_t + r_t, 1e-9)
        flag = yp.mean() * 100
        log(f"  {thr:>8.2f} {p_t:>10.4f} {r_t:>8.4f} {f1_t:>8.4f} {flag:>9.1f}%")
        thresh_rows.append({"threshold": thr, "precision": p_t,
                            "recall": r_t, "f1": f1_t, "flagged_pct": flag})
    log("=" * 60)

    tag_safe = tag.replace(" ", "_")
    with open(os.path.join(output_dir, f"binary_metrics_{tag_safe}.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(output_dir, f"binary_metrics_{tag_safe}.json"), "w") as f:
        json.dump({
            "tag": tag, "accuracy": float(acc),
            "precision_active": float(prec), "recall_active": float(rec),
            "specificity_NA": float(spec), "f1_active": float(f1),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "threshold_analysis": thresh_rows,
        }, f, indent=2)
    print(f"  Binary metrics → {output_dir}/binary_metrics_{tag_safe}.[txt|json]")


def fg_metrics_report(y_true_fg, y_pred_fg, y_prob_fg,
                      fg_classes, output_dir, tag):
    lines = []

    def log(s=""):
        print(s); lines.append(s)

    K       = len(fg_classes)
    top1    = (y_pred_fg == y_true_fg).mean() * 100
    top2    = top_k_accuracy_score(y_true_fg, y_prob_fg, k=min(2, K)) * 100
    bal_acc = balanced_accuracy_score(y_true_fg, y_pred_fg) * 100
    kappa   = cohen_kappa_score(y_true_fg, y_pred_fg) \
              if len(np.unique(y_true_fg)) > 1 else float("nan")
    mcc     = matthews_corrcoef(y_true_fg, y_pred_fg)

    lb  = label_binarize(y_true_fg, classes=list(range(K)))
    aps = [
        average_precision_score(lb[:, c], y_prob_fg[:, c])
        if lb.shape[1] > c and lb[:, c].sum() > 0 else float("nan")
        for c in range(K)
    ]
    mAP = float(np.nanmean(aps))

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true_fg, y_pred_fg, average="macro",    zero_division=0)
    p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(
        y_true_fg, y_pred_fg, average="weighted", zero_division=0)
    _,     _,     f1_mi,  _ = precision_recall_fscore_support(
        y_true_fg, y_pred_fg, average="micro",    zero_division=0)
    prec, rec, f1, sup       = precision_recall_fscore_support(
        y_true_fg, y_pred_fg, labels=list(range(K)), zero_division=0)
    per_acc = [
        (y_pred_fg[y_true_fg == c] == c).mean() * 100
        if (y_true_fg == c).sum() > 0 else 0.0
        for c in range(K)
    ]
    cm = confusion_matrix(y_true_fg, y_pred_fg, labels=list(range(K)))

    log("\n" + "=" * 60)
    log(f"STAGE-2 FINE-GRAINED METRICS  [{tag}]")
    log("=" * 60)
    log(f"\nOVERALL")
    log(f"  Top-1 Accuracy    : {top1:.2f}%")
    log(f"  Top-2 Accuracy    : {top2:.2f}%")
    log(f"  Balanced Accuracy : {bal_acc:.2f}%")
    log(f"  Cohen's Kappa     : {kappa:.4f}")
    log(f"  MCC               : {mcc:.4f}")
    log(f"  mAP               : {mAP:.4f}")
    log(f"  F1 Micro          : {f1_mi:.4f}")
    log(f"  F1 Macro          : {f1_mac:.4f}")
    log(f"  F1 Weighted       : {f1_wt:.4f}")
    log(f"  Precision Macro   : {p_mac:.4f}")
    log(f"  Recall Macro      : {r_mac:.4f}")

    log(f"\nPER-CLASS")
    log(f"  {'Class':<30} {'Acc%':>7} {'Prec':>7} {'Rec':>7} "
        f"{'F1':>7} {'AP':>8} {'N':>6}")
    log(f"  {'-'*70}")
    for i, cls in enumerate(fg_classes):
        ap_s = f"{aps[i]:.4f}" if not np.isnan(aps[i]) else "   N/A"
        log(f"  {cls:<30} {per_acc[i]:>7.2f} {prec[i]:>7.4f} "
            f"{rec[i]:>7.4f} {f1[i]:>7.4f} {ap_s:>8} {sup[i]:>6}")

    log(f"\nCONFUSION MATRIX (Rows=Actual, Cols=Predicted)")
    log(f"  {'':30}" + "".join(f"{c[:8]:>10}" for c in fg_classes))
    for i, cls in enumerate(fg_classes):
        log(f"  {cls:<30}" + "".join(f"{cm[i,j]:>10}" for j in range(K)))

    log(f"\nCLASSIFICATION REPORT")
    log(classification_report(y_true_fg, y_pred_fg,
                              target_names=fg_classes, digits=4, zero_division=0))
    log("=" * 60)

    tag_safe = tag.replace(" ", "_")
    with open(os.path.join(output_dir, f"fg_metrics_{tag_safe}.txt"), "w") as f:
        f.write("\n".join(lines))

    def cv(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    with open(os.path.join(output_dir, f"fg_metrics_{tag_safe}.json"), "w") as f:
        json.dump({
            "tag": tag,
            "top1_acc": float(top1), "top2_acc": float(top2),
            "balanced_acc": float(bal_acc),
            "cohen_kappa": float(kappa) if not np.isnan(kappa) else None,
            "mcc": float(mcc), "mAP": float(mAP),
            "f1_micro": float(f1_mi), "f1_macro": float(f1_mac),
            "f1_weighted": float(f1_wt),
            "precision_macro": float(p_mac), "recall_macro": float(r_mac),
            "per_class": {
                cls: {
                    "accuracy":  float(per_acc[i]),
                    "precision": float(prec[i]),
                    "recall":    float(rec[i]),
                    "f1":        float(f1[i]),
                    "AP":        float(aps[i]) if not np.isnan(aps[i]) else None,
                    "support":   int(sup[i]),
                }
                for i, cls in enumerate(fg_classes)
            },
            "confusion_matrix": cm.tolist(),
        }, f, indent=2, default=cv)

    print(f"  FG metrics → {output_dir}/fg_metrics_{tag_safe}.[txt|json]")


# ============================================================
# 8. INFERENCE
# ============================================================
def run_inference(model, test_samples, fg_label_map, device, output_dir):
    print("\n" + "=" * 60)
    print("INFERENCE — TEST SET")
    print("=" * 60)

    model.eval().to(device)
    id_to_fg   = {v: k for k, v in fg_label_map.items()}
    fg_classes = [id_to_fg[i] for i in range(len(id_to_fg))]
    ds         = TwoStageVideoDataset(test_samples, training=False)
    softmax    = nn.Softmax(dim=1)
    window_rows = []

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            frames, bin_lbl, fg_lbl = ds[i]
            frames = frames.unsqueeze(0).to(device)
            with torch.no_grad():
                bin_logits, fg_logits = model(frames)

            bin_probs = softmax(bin_logits)[0].cpu().numpy()
            fg_probs  = softmax(fg_logits )[0].cpu().numpy()
            bin_pred  = int(np.argmax(bin_probs))
            fg_pred   = int(np.argmax(fg_probs))

            window_rows.append({
                "video_path":    s["video_path"],
                "start_frame":   s["start_frame"],
                "end_frame":     s["end_frame"],
                "true_label":    s["label_str"],
                "true_bin":      s["bin_label"],
                "true_fg":       s["fg_label"],
                "pred_bin":      bin_pred,
                "pred_bin_label": "N/A" if bin_pred == BIN_NA else "active",
                "prob_NA":        round(float(bin_probs[0]), 4),
                "prob_active":    round(float(bin_probs[1]), 4),
                "pred_fg":        fg_pred,
                "pred_fg_label":  id_to_fg[fg_pred],
                **{f"prob_fg_{id_to_fg[j]}": round(float(fg_probs[j]), 4)
                   for j in range(len(fg_classes))},
            })

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(ds)}] {os.path.basename(s['video_path'])} "
                      f"[{s['start_frame']}-{s['end_frame']}] "
                      f"true={s['label_str']}  "
                      f"bin={'NA' if bin_pred==0 else 'act'}({bin_probs[bin_pred]:.2f})  "
                      f"fg={id_to_fg[fg_pred]}({fg_probs[fg_pred]:.2f})")
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            window_rows.append({
                "video_path":    s["video_path"],
                "start_frame":   s["start_frame"],
                "end_frame":     s["end_frame"],
                "true_label":    s["label_str"],
                "true_bin":      s["bin_label"],
                "true_fg":       s["fg_label"],
                "pred_bin":      -1, "pred_bin_label": "ERROR",
                "prob_NA": 0.0, "prob_active": 0.0,
                "pred_fg": -1, "pred_fg_label": "ERROR",
            })

    win_df  = pd.DataFrame(window_rows)
    win_csv = os.path.join(output_dir, "test_predictions_window.csv")
    win_df.to_csv(win_csv, index=False)
    print(f"\nWindow predictions → {win_csv}")

    valid        = win_df[win_df["pred_bin"] >= 0].copy()
    fg_prob_cols = [f"prob_fg_{c}" for c in fg_classes
                    if f"prob_fg_{c}" in valid.columns]

    # ── Stage-1 window-level binary metrics ──────────────────────────
    print("\n--- STAGE 1: BINARY (window level) ---")
    binary_metrics_report(
        y_true_bin=valid["true_bin"].values,
        y_prob_active=valid["prob_active"].values,
        output_dir=output_dir, tag="window_binary")

    # ── Stage-2 window-level fg metrics (non-N/A ground truth only) ──
    fg_win = valid[valid["true_fg"] >= 0].copy()
    if len(fg_win):
        print("\n--- STAGE 2: FINE-GRAINED (window level, non-N/A only) ---")
        fg_probs_mat = fg_win[fg_prob_cols].values if fg_prob_cols \
                       else np.zeros((len(fg_win), len(fg_classes)))
        fg_metrics_report(
            y_true_fg=fg_win["true_fg"].values,
            y_pred_fg=fg_win["pred_fg"].values,
            y_prob_fg=fg_probs_mat,
            fg_classes=fg_classes,
            output_dir=output_dir, tag="window_fg")

    # ── Video-level aggregation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("VIDEO-LEVEL AGGREGATION")
    print("=" * 60)

    video_rows = []
    for vpath, grp in valid.groupby("video_path"):
        avg_prob_active = grp["prob_active"].mean()
        vid_bin_pred    = int(avg_prob_active >= 0.5)
        true_bin_vid    = int(Counter(grp["true_bin"].tolist()).most_common(1)[0][0])

        avg_fg_probs = grp[fg_prob_cols].mean(axis=0).values if fg_prob_cols \
                       else np.zeros(len(fg_classes))
        vid_fg_pred  = int(np.argmax(avg_fg_probs))

        fg_grp      = grp[grp["true_fg"] >= 0]
        true_fg_vid = int(Counter(fg_grp["true_fg"].tolist()).most_common(1)[0][0]) \
                      if len(fg_grp) > 0 else -1

        video_rows.append({
            "video_path":      vpath,
            "n_windows":       len(grp),
            "true_bin":        true_bin_vid,
            "pred_bin":        vid_bin_pred,
            "avg_prob_active": round(float(avg_prob_active), 4),
            "true_fg":         true_fg_vid,
            "pred_fg":         vid_fg_pred,
            "pred_fg_label":   id_to_fg[vid_fg_pred],
            "true_fg_label":   id_to_fg[true_fg_vid] if true_fg_vid >= 0 else NA_LABEL,
            **{pc: round(float(v), 4)
               for pc, v in zip(fg_prob_cols, avg_fg_probs)},
        })

    vid_df  = pd.DataFrame(video_rows)
    vid_csv = os.path.join(output_dir, "test_predictions_video.csv")
    vid_df.to_csv(vid_csv, index=False)
    print(f"Video predictions → {vid_csv}")

    # Binary video-level
    print("\n--- STAGE 1: BINARY (video level) ---")
    binary_metrics_report(
        y_true_bin=vid_df["true_bin"].values,
        y_prob_active=vid_df["avg_prob_active"].values,
        output_dir=output_dir, tag="video_binary")

    # FG video-level
    fg_vid = vid_df[vid_df["true_fg"] >= 0].copy()
    if len(fg_vid) and fg_prob_cols:
        print("\n--- STAGE 2: FINE-GRAINED (video level) ---")
        fg_metrics_report(
            y_true_fg=fg_vid["true_fg"].values,
            y_pred_fg=fg_vid["pred_fg"].values,
            y_prob_fg=fg_vid[fg_prob_cols].values,
            fg_classes=fg_classes,
            output_dir=output_dir, tag="video_fg")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VideoMAE V2 2-stage sliding window fine-tuning")
    parser.add_argument("--task", choices=["loco", "rmm"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    cfg            = TASK_CONFIG[args.task]
    label_col      = cfg["label_col"]
    num_fg_classes = cfg["num_fg_classes"]
    base_dir   = cfg["output_dir"]
    output_dir = os.path.join(base_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    pl.seed_everything(args.seed)


    print(f"\n{'='*60}")
    print(f"TASK       : {args.task.upper()}")
    print(f"LABEL COL  : {label_col}")
    print(f"FG CLASSES : {num_fg_classes}  (+ N/A binary)")
    print(f"WINDOW     : {WINDOW_SEC}s / stride {WINDOW_STRIDE}s")
    print(f"{'='*60}\n")


    dm = TwoStageDataModule(
        label_col=label_col,
        num_fg_classes=num_fg_classes,
        output_dir=output_dir,
    )
    dm.setup()

    model = TwoStageFineTune(
        num_fg_classes=num_fg_classes,
        freeze=True,
        bin_weights=dm.bin_weights,
        fg_weights=dm.fg_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, monitor="val_loss", mode="min", save_top_k=2,
        filename=f"vmae2-{args.task}-twostage-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

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
    best_model = TwoStageFineTune.load_from_checkpoint(
        best, num_fg_classes=num_fg_classes, freeze=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best_model, dm.test_samples, dm.fg_label_map, device, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()