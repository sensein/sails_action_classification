"""
End-to-end VideoMAE V2 ViT-Base fine-tuning pipeline.

Distilled ViT-B checkpoint (K710, 86.6% K400 top-1).
Uses the official modeling_finetune.py from OpenGVLab/VideoMAEv2 as backbone.

Unified for Locomotion and Repetitive Motor Movements tasks.
Usage:
    python videomae2_finetune.py --task loco
    python videomae2_finetune.py --task rmm

Setup (one-time):
    # Clone the VideoMAEv2 repo (we only need one file from it)
    wget -O modeling_finetune.py \\
      "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py"

    # Download the distilled ViT-B K710 checkpoint
    mkdir -p ~/.cache/videomae2
    wget -O ~/.cache/videomae2/vit_b_k710_dl_from_giant.pth \\
      "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth"

    # Install deps (in your pytorchvideo_env)
    pip install timm
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import cv2
import pytorch_lightning as pl
from functools import partial
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# TASK CONFIG
# ============================================================
TASK_CONFIG = {
    "loco": {
        "label_col":   "Locomotion",
        "num_classes": 5,
        "output_dir":  "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco",
    },
    "rmm": {
        "label_col":   "Repetitive_Motor_Movements",
        "num_classes": 4,
        "output_dir":  "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm",
    },
}

SPLIT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

# ============================================================
# GLOBAL CONFIG
# ============================================================
BATCH_SIZE      = 8      # A100-80GB can handle this easily for ViT-B
NUM_WORKERS     = 4
MAX_EPOCHS      = 50
LEARNING_RATE   = 1e-4
SEED            = 42

# VideoMAE V2 ViT-B expects 16 frames at 224x224
NUM_FRAMES      = 16
CROP_SIZE       = 224
MEAN            = (0.485, 0.456, 0.406)
STD             = (0.229, 0.224, 0.225)

# Annotation timing / clipping
ANN_FPS         = 15.0
MIN_FRAMES      = 15
CLIP_FRAMES     = 30

# Checkpoint
VMAE2_CKPT = os.path.expanduser("~/.cache/videomae2/vit_b_k710_dl_from_giant.pth")


# ============================================================
# 1. BBOX LOADING
# ============================================================
def load_bbox_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


# ============================================================
# 2. ACTION RUNS + CLIPPING
# ============================================================
def find_action_runs(ann, label_col):
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()
    runs = []
    i, n = 0, len(df)
    while i < n:
        lab = labels[i].strip()
        if lab in ("N/A", ""):
            i += 1; continue
        j = i
        while (j + 1 < n and labels[j + 1].strip() == lab
               and frames[j + 1] == frames[j] + 1):
            j += 1
        runs.append((frames[i], frames[j], lab))
        i = j + 1
    return runs


def chunk_run(start, end):
    total = end - start + 1
    if total < MIN_FRAMES:
        return []
    if total < CLIP_FRAMES * 2:
        if total < 45:
            return [(start, end)]
        split_pt = start + CLIP_FRAMES
        return [(start, split_pt - 1), (split_pt, end)]
    clips = []
    s = start
    while s <= end:
        e = min(s + CLIP_FRAMES - 1, end)
        if (e - s + 1) >= MIN_FRAMES:
            clips.append((s, e))
        s += CLIP_FRAMES
    return clips


def build_samples(split_csv, label_col):
    split_df = pd.read_csv(split_csv)
    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    for c in required:
        if c not in split_df.columns:
            raise ValueError(f"Split CSV missing column: {c}")

    by_split = {"train": [], "val": [], "test": []}

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
        except Exception as e:
            print(f"  skip ({e}): {lp}"); continue

        if label_col not in ann.columns:
            continue

        runs = find_action_runs(ann, label_col)
        for sf, ef, lab in runs:
            for cs, ce in chunk_run(sf, ef):
                by_split[sp].append({
                    "video_path":  vp,
                    "h5_path":     hp,
                    "start_frame": int(cs),
                    "end_frame":   int(ce),
                    "label_str":   lab,
                    "ann_fps":     ANN_FPS,
                })
    return by_split


# ============================================================
# 3. DATASET — output shape (C,T,H,W) for VideoMAE V2
# ============================================================
class BBoxCropVideoDataset(Dataset):
    def __init__(self, samples, label_map, num_frames=NUM_FRAMES,
                 crop_size=CROP_SIZE, training=False):
        self.samples    = samples
        self.label_map  = label_map
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.training   = training
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _read_segment(self, s):
        cap = cv2.VideoCapture(s["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {s['video_path']}")
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step    = max(1, int(round(vid_fps / s["ann_fps"])))

        bbox_map = load_bbox_map(s["h5_path"])
        if not bbox_map:
            cap.release(); raise ValueError("empty bbox map")

        ann_frames = np.arange(s["start_frame"], s["end_frame"] + 1)
        idxs   = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]

        bbox_keys = np.array(sorted(bbox_map.keys()))
        frames = []
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

            x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(crop)

        cap.release()
        arr = np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.label_map[s["label_str"]]
        try:
            frames = self._read_segment(s)
            if self.training and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])
            return frames, label
        except Exception as e:
            print(f"  load error {os.path.basename(s['video_path'])} "
                  f"[{s['start_frame']}-{s['end_frame']}]: {e}")
            return torch.zeros(3, self.num_frames, self.crop_size, self.crop_size), label


def collate(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


# ============================================================
# 4. DATA MODULE
# ============================================================
class H5BBoxDataModule(pl.LightningDataModule):
    def __init__(self, label_col, output_dir):
        super().__init__()
        self.label_col  = label_col
        self.output_dir = output_dir
        self.label_map  = None
        self.train_samples = None
        self.val_samples   = None
        self.test_samples  = None

    def setup(self, stage=None):
        print(f"Building samples (label_col={self.label_col})...")
        by_split = build_samples(SPLIT_CSV, self.label_col)
        n_train = len(by_split["train"])
        n_val   = len(by_split["val"])
        n_test  = len(by_split["test"])
        print(f"Clips  train={n_train}  val={n_val}  test={n_test}")
        if n_train == 0:
            raise RuntimeError("No training clips built.")

        all_labels = sorted({s["label_str"] for split in by_split.values() for s in split})
        self.label_map = {lab: i for i, lab in enumerate(all_labels)}
        print(f"Label map: {self.label_map}")

        from collections import Counter
        for sp, samps in by_split.items():
            dist = Counter(s["label_str"] for s in samps)
            print(f"  {sp} distribution: {dict(dist)}")

        self.train_samples = by_split["train"]
        self.val_samples   = by_split["val"]
        self.test_samples  = by_split["test"]

        with open(os.path.join(self.output_dir, "label_mapping.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(self.test_samples).to_csv(
            os.path.join(self.output_dir, "test_split.csv"), index=False)

        # Class weights from TRAIN only
        n_classes = len(self.label_map)
        counts = np.zeros(n_classes, dtype=np.float64)
        for s in self.train_samples:
            counts[self.label_map[s["label_str"]]] += 1
        counts  = np.maximum(counts, 1.0)
        weights = counts.sum() / (n_classes * counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights (train):")
        for lab, idx in self.label_map.items():
            print(f"  {lab:30s} count={int(counts[idx]):4d}  weight={weights[idx]:.3f}")

    def train_dataloader(self):
        ds = BBoxCropVideoDataset(self.train_samples, self.label_map, training=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate)

    def val_dataloader(self):
        ds = BBoxCropVideoDataset(self.val_samples, self.label_map, training=False)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate)


# ============================================================
# 5. VideoMAE V2 ViT-Base MODEL
# ============================================================
def build_videomae2_vitb(num_classes, freeze_all_but_last_block=True):
    """
    Loads VideoMAE V2 ViT-Base with K710 distilled checkpoint (86.6% K400).

    Uses the official modeling_finetune.py from OpenGVLab/VideoMAEv2.

    Architecture: ViT-B/16, 16 frames, tubelet_size=2
      patch_size=16, embed_dim=768, depth=12, num_heads=12
    """
    try:
        from sailsprep.action_model_testing.Videomaev2.clip.modeling_finetune import vit_base_patch16_224
    except ImportError as e:
        raise ImportError(
            "Could not import from modeling_finetune.py.\n"
            "Download it:\n"
            "  wget -O modeling_finetune.py "
            "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py\n"
            "Place it in the same directory as this script."
        ) from e

    # Build ViT-B with K710's 710 classes first to load the checkpoint
    model = vit_base_patch16_224(num_classes=710)

    # Load distilled K710 checkpoint
    if not os.path.exists(VMAE2_CKPT):
        print(f"Downloading VideoMAE V2 ViT-B K710 checkpoint -> {VMAE2_CKPT}")
        os.makedirs(os.path.dirname(VMAE2_CKPT), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth",
            VMAE2_CKPT,
        )

    ckpt = torch.load(VMAE2_CKPT, map_location="cpu")
    state = ckpt.get("module", ckpt)
    # Strip 'module.' prefix if DDP wrapped
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded VideoMAE V2 ViT-B K710. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (sample): {missing[:5]}")
    if unexpected:
        print(f"  unexpected keys (sample): {unexpected[:5]}")

    # Replace the classification head for our task
    model.head = nn.Linear(768, num_classes)
    # Re-init head
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    if freeze_all_but_last_block:
        print("Freezing all but last transformer block (blocks.11) + head + fc_norm")
        for name, p in model.named_parameters():
            trainable = False
            if name.startswith("head."):
                trainable = True
            elif "blocks.11." in name:
                trainable = True
            elif "fc_norm" in name:
                trainable = True
            p.requires_grad = trainable

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

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
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        acc  = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ============================================================
# 7. INFERENCE (on test split)
# ============================================================
def run_inference(model, test_samples, label_map, device, output_csv, output_dir):
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
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            probs = softmax(logits)
            top   = int(probs.argmax(1).item())
            conf  = float(probs[0, top].item())
            rows.append({
                "video_path":  s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame":   s["end_frame"],
                "true_label":  s["label_str"],
                "pred_label":  id_to_label[top],
                "confidence":  round(conf, 4),
                "correct":     int(id_to_label[top] == s["label_str"]),
            })
            print(f"[{i+1}/{len(ds)}] {os.path.basename(s['video_path'])} "
                  f"[{s['start_frame']}-{s['end_frame']}] "
                  f"true={s['label_str']} pred={id_to_label[top]} ({conf:.2f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "video_path":  s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame":   s["end_frame"],
                "true_label":  s["label_str"],
                "pred_label":  "ERROR",
                "confidence":  0.0,
                "correct":     0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions -> {output_csv}")

    valid = df[df["pred_label"] != "ERROR"]
    if len(valid):
        acc = valid["correct"].mean()
        print(f"\nAccuracy: {acc:.4f} ({int(valid['correct'].sum())}/{len(valid)})")
        all_labels = sorted(label_map.keys())
        print("\nClassification report:")
        print(classification_report(valid["true_label"], valid["pred_label"],
                                    labels=all_labels, zero_division=0))
        cm = confusion_matrix(valid["true_label"], valid["pred_label"], labels=all_labels)
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        print("Confusion matrix:")
        print(cm_df)

        with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(valid["true_label"], valid["pred_label"],
                                          labels=all_labels, zero_division=0))
            f.write(f"\n{cm_df.to_string()}\n")


# ============================================================
# 8. MAIN
# ============================================================
# In TASK_CONFIG, remove hardcoded output_dir (we'll build it dynamically)
# Change main() to accept --seed:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["loco", "rmm"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    label_col  = cfg["label_col"]
    
    # Seed-specific output dir
    base_dir   = cfg["output_dir"]
    output_dir = os.path.join(base_dir, f"seed_{args.seed}")
    output_csv = os.path.join(output_dir, "test_predictions.csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}\nTASK: {args.task.upper()} | label_col={label_col} | seed={args.seed}\n{'='*60}")

    pl.seed_everything(args.seed)  
    dm = H5BBoxDataModule(label_col=label_col, output_dir=output_dir)
    dm.setup()

    n_classes = len(dm.label_map)
    assert n_classes == cfg["num_classes"], (
        f"Expected {cfg['num_classes']} classes for {args.task}, found {n_classes}")
    print(f"\nNum classes: {n_classes}")

    model = VideoMAE2FineTune(
        num_classes=n_classes,
        freeze=True,
        class_weights=dm.class_weights,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, monitor="val_loss", mode="min", save_top_k=2,
        filename=f"vmae2-{args.task}-{{epoch:02d}}-{{val_loss:.3f}}",
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
        best, num_classes=n_classes, freeze=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best_model, dm.test_samples, dm.label_map, device,
                  output_csv, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()