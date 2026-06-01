"""
VideoMAE V2 ViT-Base — Full-Video Sliding Window (2-sec window, 1-sec stride)
Single classification head including N/A as a class.
Unified for Locomotion and Repetitive Motor Movements tasks.

Usage:
    python videomae2_fullvideo_sliding.py --task loco
    python videomae2_fullvideo_sliding.py --task rmm

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
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

# ============================================================
# TASK CONFIG  (N/A added → +1 class vs clip script)
# ============================================================
TASK_CONFIG = {
    "loco": {
        "label_col":   "Locomotion",
        "num_classes": 6,          # original 5 + N/A
        "output_dir":  "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco_fullvideo",
    },
    "rmm": {
        "label_col":   "Repetitive_Motor_Movements",
        "num_classes": 5,          # original 4 + N/A
        "output_dir":  "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm_fullvideo",
    },
}

SPLIT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

# ============================================================
# GLOBAL CONFIG
# ============================================================
BATCH_SIZE    = 8
NUM_WORKERS   = 4
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-4
SEED          = 42

# VideoMAE V2 ViT-B expects 16 frames at 224x224
NUM_FRAMES  = 16
CROP_SIZE   = 224
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)

# Annotation timing / windowing
ANN_FPS        = 15.0
WINDOW_SEC     = 2.0
WINDOW_STRIDE  = 1.0
MIN_WIN_FRAMES = 5

NA_LABEL = "N/A"

VMAE2_CKPT = os.path.expanduser("~/.cache/videomae2/vit_b_k710_dl_from_giant.pth")


# ============================================================
# 1. BBOX LOADING  (full-video h5)
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
    """Majority label in [ann_start, ann_end). Treats empty/nan as N/A."""
    labels = []
    for f in range(ann_start, ann_end):
        lbl = frame_to_label.get(f, NA_LABEL)
        if lbl in ("", "nan", "None"):
            lbl = NA_LABEL
        labels.append(lbl)
    if not labels:
        return NA_LABEL
    return Counter(labels).most_common(1)[0][0]


def build_samples(split_csv, label_col):
    """
    Slide a 2-sec / 1-sec-stride window across the full annotated span.

    VIDEO-LEVEL FILTER: only include videos that have at least one window
    whose majority label is NOT N/A. All-N/A videos are skipped entirely.

    Returns {split: [sample_dict, ...]}
    Uses interpolated_full_h5 column for bbox.
    """
    split_df = pd.read_csv(split_csv)
    for c in ["video_path", "label_path", "interpolated_full_h5", "split"]:
        if c not in split_df.columns:
            raise ValueError(f"Split CSV missing column: '{c}'")

    by_split          = {"train": [], "val": [], "test": []}
    window_ann_frames = int(WINDOW_SEC    * ANN_FPS)   # 30 ann frames per window
    stride_ann_frames = int(WINDOW_STRIDE * ANN_FPS)   # 15 ann frames per stride
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

        # Build windows into temp list first for video-level filter
        video_samples = []
        start = 0
        while start + window_ann_frames <= total_ann_frames + stride_ann_frames:
            end     = min(start + window_ann_frames, total_ann_frames)
            n_valid = end - start
            if n_valid < MIN_WIN_FRAMES:
                start += stride_ann_frames
                continue

            label_str = get_window_label(frame_to_label, start, end)
            video_samples.append({
                "video_path":  vp,
                "h5_path":     hp,
                "start_frame": int(start),
                "end_frame":   int(end - 1),
                "label_str":   label_str,
                "ann_fps":     ANN_FPS,
            })
            start += stride_ann_frames

        # VIDEO-LEVEL FILTER: skip if all windows are N/A
        if not any(s["label_str"] != NA_LABEL for s in video_samples):
            skipped_allna += 1
            print(f"  [skip-allNA] {os.path.basename(vp)}")
            continue

        by_split[sp].extend(video_samples)

    print(f"\nBuild done. skipped_files={skipped_files}  skipped_all_NA={skipped_allna}")
    return by_split


# ============================================================
# 3. DATASET — output shape (C,T,H,W) for VideoMAE V2
# ============================================================
class BBoxCropVideoDataset(Dataset):
    def __init__(self, samples, label_map,
                 num_frames=NUM_FRAMES, crop_size=CROP_SIZE, training=False):
        self.samples    = samples
        self.label_map  = label_map
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
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx):
        s     = self.samples[idx]
        label = self.label_map[s["label_str"]]
        try:
            frames = self._read_segment(s)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as e:
            print(f"  load error {os.path.basename(s['video_path'])} "
                  f"[{s['start_frame']}-{s['end_frame']}]: {e}")
            return (
                torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
                label,
            )


def collate(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


# ============================================================
# 4. DATA MODULE
# ============================================================
class H5BBoxDataModule(pl.LightningDataModule):
    def __init__(self, label_col, output_dir):
        super().__init__()
        self.label_col     = label_col
        self.output_dir    = output_dir
        self.label_map     = None
        self.train_samples = None
        self.val_samples   = None
        self.test_samples  = None
        self.class_weights = None

    def setup(self, stage=None):
        print(f"\nBuilding sliding-window samples (label_col={self.label_col})...")
        by_split = build_samples(SPLIT_CSV, self.label_col)

        n_tr, n_v, n_te = (len(by_split["train"]),
                           len(by_split["val"]),
                           len(by_split["test"]))
        print(f"Windows  train={n_tr}  val={n_v}  test={n_te}")
        if n_tr == 0:
            raise RuntimeError("No training windows built.")

        # Label map — always include N/A
        all_labels = sorted({s["label_str"] for sp in by_split.values() for s in sp})
        if NA_LABEL not in all_labels:
            all_labels = sorted(all_labels + [NA_LABEL])
        self.label_map = {lab: i for i, lab in enumerate(all_labels)}
        print(f"Label map ({len(self.label_map)} classes): {self.label_map}")

        for sp_name, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
            print(f"  {sp_name}: {dict(sorted(dist.items()))}")

        self.train_samples = by_split["train"]
        self.val_samples   = by_split["val"]
        self.test_samples  = by_split["test"]

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "label_mapping.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False)

        # Class weights from TRAIN only
        n_classes = len(self.label_map)
        counts    = np.zeros(n_classes, dtype=np.float64)
        for s in self.train_samples:
            counts[self.label_map[s["label_str"]]] += 1
        counts  = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print("\nClass weights (train):")
        for lab, idx in sorted(self.label_map.items(), key=lambda x: x[1]):
            print(f"  {lab:35s}  count={int(counts[idx]):5d}  weight={weights[idx]:.4f}")

    def train_dataloader(self):
        ds = BBoxCropVideoDataset(self.train_samples, self.label_map, training=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate,
                          pin_memory=True)

    def val_dataloader(self):
        ds = BBoxCropVideoDataset(self.val_samples, self.label_map, training=False)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate,
                          pin_memory=True)


# ============================================================
# 5. VideoMAE V2 ViT-Base MODEL
# ============================================================
def build_videomae2_vitb(num_classes, freeze_all_but_last_block=True):
    try:
        from modeling_finetune import vit_base_patch16_224
    except ImportError as e:
        raise ImportError(
            "Download modeling_finetune.py:\n"
            "  wget -O modeling_finetune.py "
            "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py"
        ) from e

    # Build with K710 classes to load pretrained weights
    model = vit_base_patch16_224(num_classes=710)

    if not os.path.exists(VMAE2_CKPT):
        print(f"Downloading VideoMAE V2 ViT-B K710 checkpoint → {VMAE2_CKPT}")
        os.makedirs(os.path.dirname(VMAE2_CKPT), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/"
            "vit_b_k710_dl_from_giant.pth",
            VMAE2_CKPT,
        )

    ckpt  = torch.load(VMAE2_CKPT, map_location="cpu")
    state = ckpt.get("module", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded VideoMAE V2 ViT-B K710: missing={len(missing)} unexpected={len(unexpected)}")

    # Replace head for our task
    model.head = nn.Linear(768, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    if freeze_all_but_last_block:
        print("Freezing all but blocks.11 + head + fc_norm")
        for name, p in model.named_parameters():
            p.requires_grad = (
                name.startswith("head.")
                or "blocks.11." in name
                or "fc_norm" in name
            )
        tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {tr/1e6:.2f}M / {tot/1e6:.2f}M")

    return model


# ============================================================
# 6. LIGHTNING MODULE
# ============================================================
class VideoMAE2FineTune(pl.LightningModule):
    def __init__(self, num_classes, freeze=True, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_videomae2_vitb(num_classes, freeze_all_but_last_block=freeze)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y   = batch
        logits  = self.model(x)
        loss    = F.cross_entropy(logits, y, weight=self.class_weights)
        acc     = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y   = batch
        logits  = self.model(x)
        loss    = F.cross_entropy(logits, y)   # unweighted for true perf
        acc     = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)
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
def compute_full_metrics(y_true, y_pred, y_prob, all_labels, output_dir, tag):
    """
    Compute and save full classification metrics.
    y_true / y_pred : integer class indices
    y_prob          : (N, K) softmax probabilities
    all_labels      : list of class name strings ordered by index
    """
    lines = []

    def log(s=""):
        print(s); lines.append(s)

    K       = len(all_labels)
    top1    = (y_pred == y_true).mean() * 100
    top2    = top_k_accuracy_score(y_true, y_prob, k=min(2, K)) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    kappa   = cohen_kappa_score(y_true, y_pred) \
              if len(np.unique(y_true)) > 1 else float("nan")
    mcc     = matthews_corrcoef(y_true, y_pred)

    lb  = label_binarize(y_true, classes=list(range(K)))
    aps = []
    for c in range(K):
        if lb.shape[1] > c and lb[:, c].sum() > 0:
            aps.append(average_precision_score(lb[:, c], y_prob[:, c]))
        else:
            aps.append(float("nan"))
    mAP = float(np.nanmean(aps))

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0)
    p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    _,     _,     f1_mi,  _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro",    zero_division=0)
    prec, rec, f1, sup       = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(K)), zero_division=0)
    per_acc = [
        (y_pred[y_true == c] == c).mean() * 100
        if (y_true == c).sum() > 0 else 0.0
        for c in range(K)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))

    log("\n" + "=" * 60)
    log(f"METRICS [{tag}]")
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
    for i, cls in enumerate(all_labels):
        ap_s = f"{aps[i]:.4f}" if not np.isnan(aps[i]) else "   N/A"
        log(f"  {cls:<30} {per_acc[i]:>7.2f} {prec[i]:>7.4f} "
            f"{rec[i]:>7.4f} {f1[i]:>7.4f} {ap_s:>8} {sup[i]:>6}")

    log(f"\nCONFUSION MATRIX (Rows=Actual, Cols=Predicted)")
    log(f"  {'':30}" + "".join(f"{c[:8]:>10}" for c in all_labels))
    for i, cls in enumerate(all_labels):
        log(f"  {cls:<30}" + "".join(f"{cm[i,j]:>10}" for j in range(K)))

    log(f"\nCLASSIFICATION REPORT")
    log(classification_report(y_true, y_pred,
                              target_names=all_labels, digits=4, zero_division=0))
    log("=" * 60)

    tag_safe = tag.replace(" ", "_")
    with open(os.path.join(output_dir, f"metrics_{tag_safe}.txt"), "w") as f:
        f.write("\n".join(lines))

    def cv(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    with open(os.path.join(output_dir, f"metrics_{tag_safe}.json"), "w") as f:
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
                for i, cls in enumerate(all_labels)
            },
            "confusion_matrix": cm.tolist(),
        }, f, indent=2, default=cv)

    print(f"  Metrics saved → {output_dir}/metrics_{tag_safe}.[txt|json]")


# ============================================================
# 8. INFERENCE — window-level + video-level aggregation
# ============================================================
def run_inference(model, test_samples, label_map, device, output_dir):
    print("\n" + "=" * 60)
    print("INFERENCE — TEST SET (window + video level)")
    print("=" * 60)

    model.eval().to(device)
    id_to_label  = {v: k for k, v in label_map.items()}
    all_labels   = [id_to_label[i] for i in range(len(id_to_label))]
    ds           = BBoxCropVideoDataset(test_samples, label_map, training=False)
    softmax      = nn.Softmax(dim=1)
    window_rows  = []

    for i in range(len(ds)):
        s = test_samples[i]
        try:
            x, _ = ds[i]
            x    = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            probs   = softmax(logits)[0].cpu().numpy()
            top     = int(np.argmax(probs))
            window_rows.append({
                "video_path":  s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame":   s["end_frame"],
                "true_label":  s["label_str"],
                "true_idx":    label_map[s["label_str"]],
                "pred_label":  id_to_label[top],
                "pred_idx":    top,
                "confidence":  round(float(probs[top]), 4),
                **{f"prob_{id_to_label[j]}": round(float(probs[j]), 4)
                   for j in range(len(all_labels))},
            })
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(ds)}] {os.path.basename(s['video_path'])} "
                      f"[{s['start_frame']}-{s['end_frame']}] "
                      f"true={s['label_str']} pred={id_to_label[top]} "
                      f"({probs[top]:.2f})")
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            window_rows.append({
                "video_path":  s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame":   s["end_frame"],
                "true_label":  s["label_str"],
                "true_idx":    label_map[s["label_str"]],
                "pred_label":  "ERROR",
                "pred_idx":    -1,
                "confidence":  0.0,
            })

    win_df  = pd.DataFrame(window_rows)
    win_csv = os.path.join(output_dir, "test_predictions_window.csv")
    win_df.to_csv(win_csv, index=False)
    print(f"\nWindow predictions → {win_csv}")

    # ── Window-level metrics ──────────────────────────────────────────
    valid     = win_df[win_df["pred_idx"] >= 0].copy()
    prob_cols = [f"prob_{c}" for c in all_labels if f"prob_{c}" in valid.columns]

    if len(valid):
        y_true  = valid["true_idx"].values
        y_pred  = valid["pred_idx"].values
        y_prob  = valid[prob_cols].values if prob_cols else np.zeros((len(valid), len(all_labels)))
        compute_full_metrics(y_true, y_pred, y_prob, all_labels, output_dir,
                             tag="window_level")

    # ── Video-level aggregation (avg prob → argmax) ───────────────────
    print("\n" + "=" * 60)
    print("VIDEO-LEVEL AGGREGATION")
    print("=" * 60)

    video_rows = []
    for vpath, grp in valid.groupby("video_path"):
        avg_probs  = grp[prob_cols].mean(axis=0).values if prob_cols \
                     else np.zeros(len(all_labels))
        pred_idx   = int(np.argmax(avg_probs))
        pred_label = id_to_label[pred_idx]
        true_label = Counter(grp["true_label"].tolist()).most_common(1)[0][0]
        true_idx   = label_map[true_label]

        video_rows.append({
            "video_path":  vpath,
            "n_windows":   len(grp),
            "true_label":  true_label,
            "true_idx":    true_idx,
            "pred_label":  pred_label,
            "pred_idx":    pred_idx,
            **{pc: round(float(v), 4) for pc, v in zip(prob_cols, avg_probs)},
        })

    vid_df  = pd.DataFrame(video_rows)
    vid_csv = os.path.join(output_dir, "test_predictions_video.csv")
    vid_df.to_csv(vid_csv, index=False)
    print(f"Video predictions → {vid_csv}")

    if len(vid_df):
        y_true_v = vid_df["true_idx"].values
        y_pred_v = vid_df["pred_idx"].values
        y_prob_v = vid_df[prob_cols].values if prob_cols \
                   else np.zeros((len(vid_df), len(all_labels)))
        compute_full_metrics(y_true_v, y_pred_v, y_prob_v, all_labels,
                             output_dir, tag="video_level")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VideoMAE V2 full-video sliding window fine-tuning (single head + N/A)")
    parser.add_argument("--task", choices=["loco", "rmm"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg        = TASK_CONFIG[args.task]
    label_col  = cfg["label_col"]
    base_dir   = cfg["output_dir"]
    output_dir = os.path.join(base_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    pl.seed_everything(args.seed)


    print(f"\n{'='*60}")
    print(f"TASK : {args.task.upper()}")
    print(f"LABEL: {label_col}")
    print(f"MODE : full-video sliding window ({WINDOW_SEC}s / {WINDOW_STRIDE}s stride)")
    print(f"{'='*60}\n")


    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes for '{args.task}', found {n_classes}. "
        f"Update TASK_CONFIG or check your data.")
    print(f"\nNum classes: {n_classes}")

    model = VideoMAE2FineTune(
        num_classes=n_classes,
        freeze=True,
        class_weights=dm.class_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, monitor="val_loss", mode="min", save_top_k=2,
        filename=f"vmae2-{args.task}-fullvideo-{{epoch:02d}}-{{val_loss:.3f}}",
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
    best_model = VideoMAE2FineTune.load_from_checkpoint(
        best, num_classes=n_classes, freeze=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best_model, dm.test_samples, dm.label_map, device, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()