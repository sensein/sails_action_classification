"""End-to-end V-JEPA 2 fine-tuning on H5-bbox-cropped action segments.

Default mode: frozen encoder + attentive probe.
Use --full_finetune to unfreeze the encoder.
"""

import json
import math
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoVideoProcessor

# ============================================================
# CONFIG
# ============================================================
SPLIT_CSV  = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
LABEL_COL  = "Repetitive_Motor_Movements"
OUTPUT_DIR = "/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vjepa_finetune/rmm/"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "test_predictions.csv")

MODEL_NAME      = "facebook/vjepa2-vitg-fpc64-256"
EMBED_DIM       = 1408
FRAMES_PER_CLIP = 64
TUBELET_SIZE    = 2
CROP_SIZE       = 256

NUM_CLASSES   = 4
BATCH_SIZE    = 4          # ViT-g is heavy; 2 fits comfortably on A100 80GB with frozen encoder
NUM_WORKERS   = 4
MAX_EPOCHS    = 40
LR_HEAD       = 1e-3       # head lr
LR_BACKBONE   = 1e-5       # only used if --full_finetune
TEST_SPLIT    = 0.30
SEED          = 42
ANN_FPS       = 15.0
MIN_RUN       = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# H5 + segment helpers (identical to SlowFast pipeline)
# ============================================================
def load_bbox_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def find_action_runs(ann, label_col, min_frames=MIN_RUN):
    df = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()
    runs = []
    i, n = 0, len(df)
    while i < n:
        lab = labels[i]
        if lab in ("N/A", ""): i += 1; continue
        j = i
        while (j + 1 < n and labels[j + 1] == lab and frames[j + 1] == frames[j] + 1):
            j += 1
        if frames[j] - frames[i] + 1 >= min_frames:
            runs.append((frames[i], frames[j], lab))
        i = j + 1
    return runs


def build_samples(split_csv, label_col=LABEL_COL):
    df = pd.read_csv(split_csv)
    for c in ["video_path", "label_path", "interpolated_anno_h5"]:
        if c not in df.columns:
            raise ValueError(f"Split CSV missing column: {c}")
    samples = []
    for _, row in df.iterrows():
        vp = str(row["video_path"]).strip()
        lp = str(row["label_path"]).strip()
        hp = str(row["interpolated_anno_h5"]).strip()
        if not (os.path.exists(vp) and os.path.exists(lp) and os.path.exists(hp)):
            print(f"  skip missing: {os.path.basename(vp)}"); continue
        try:
            ann = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            ann.columns = ann.columns.str.strip()
        except Exception as e:
            print(f"  csv error: {e}"); continue
        if label_col not in ann.columns:
            print(f"  no '{label_col}': {lp}"); continue
        for sf, ef, lab in find_action_runs(ann, label_col):
            samples.append({"video_path": vp, "h5_path": hp,
                            "start_frame": int(sf), "end_frame": int(ef),
                            "label_str": lab})
    return samples


# ============================================================
# Dataset: read 64 cropped frames per segment, return uint8 RGB
# ============================================================
class VJEPASegmentDataset(Dataset):
    def __init__(self, samples, label_map, num_frames=FRAMES_PER_CLIP,
                 crop_size=CROP_SIZE, training=False):
        self.samples = samples
        self.label_map = label_map
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.training = training

    def __len__(self): return len(self.samples)

    def _read_segment(self, s):
        cap = cv2.VideoCapture(s["video_path"])
        if not cap.isOpened(): raise IOError("cannot open video")
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(vid_fps / ANN_FPS)))

        bbox_map = load_bbox_map(s["h5_path"])
        if not bbox_map:
            cap.release(); raise ValueError("empty bbox map")
        bbox_keys = np.array(sorted(bbox_map.keys()))

        ann_frames = np.arange(s["start_frame"], s["end_frame"] + 1)
        idxs = np.linspace(0, len(ann_frames) - 1, self.num_frames).astype(int)
        chosen = ann_frames[idxs]

        out = np.empty((self.num_frames, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        for k, af in enumerate(chosen):
            vf = int(af * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ret, frame = cap.read()
            if not ret:
                out[k] = 0; continue
            H, W = frame.shape[:2]
            bb = bbox_map[af] if af in bbox_map else bbox_map[int(bbox_keys[np.argmin(np.abs(bbox_keys - af))])]
            x1, y1, x2, y2 = bb
            x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            out[k] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cap.release()
        return out  # (T, H, W, 3) uint8

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.label_map[s["label_str"]]
        try:
            frames = self._read_segment(s)
            if self.training and np.random.rand() < 0.5:
                frames = frames[:, :, ::-1, :].copy()  # horizontal flip
            return frames, label
        except Exception as e:
            print(f"  load err {os.path.basename(s['video_path'])} "
                  f"[{s['start_frame']}-{s['end_frame']}]: {e}")
            return np.zeros((FRAMES_PER_CLIP, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8), label


def make_collate(processor):
    def collate(batch):
        videos, labels = zip(*batch)
        clip_list = [v for v in videos]  # list of (T,H,W,3) uint8
        inputs = processor(clip_list, return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.long)
    return collate


# ============================================================
# Attentive pooling head + classifier
# ============================================================
class AttentivePoolHead(nn.Module):
    """Single learned query cross-attends over token sequence -> (B, D)."""
    def __init__(self, dim=EMBED_DIM, num_heads=8, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(dim)
        self.cls   = nn.Linear(dim, num_classes)

    def forward(self, tokens):  # tokens: (B, N, D)
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        return self.cls(pooled)


# ============================================================
# Lightning module
# ============================================================
class VJEPAFineTune(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, full_finetune=False, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.full_finetune = full_finetune

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, attn_implementation="sdpa",
        )
        if not full_finetune:
            for p in self.encoder.parameters(): p.requires_grad = False
            self.encoder.eval()
            print("Encoder FROZEN — training attentive probe only")
        else:
            print("Encoder UNFROZEN — full fine-tune (heavy memory)")

        self.head = AttentivePoolHead(EMBED_DIM, num_classes=num_classes)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None

    def forward(self, inputs):
        if self.full_finetune:
            out = self.encoder(**inputs)
        else:
            with torch.no_grad():
                out = self.encoder(**inputs)
        tokens = out.last_hidden_state.float()  # (B, N, D)
        return self.head(tokens)

    def _step(self, batch, stage):
        inputs, labels = batch
        # Cast float inputs to match encoder dtype
        inputs = {k: (v.to(self.encoder.dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels,
                               weight=self.class_weights if stage == "train" else None)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log(f"{stage}_acc",  acc,  prog_bar=True, batch_size=labels.size(0))
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def configure_optimizers(self):
        if self.full_finetune:
            params = [
                {"params": self.encoder.parameters(), "lr": LR_BACKBONE},
                {"params": self.head.parameters(),    "lr": LR_HEAD},
            ]
        else:
            params = self.head.parameters()
        opt = torch.optim.AdamW(params, lr=LR_HEAD, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": sch}


# ============================================================
# Data module
# ============================================================
class VJEPADataModule(pl.LightningDataModule):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.label_map = None
        self.class_weights = None

    def setup(self, stage=None):
        print("Building samples...")
        samples = build_samples(SPLIT_CSV)
        print(f"Total segments: {len(samples)}")
        if not samples: raise RuntimeError("No samples")

        labels = sorted({s["label_str"] for s in samples})
        self.label_map = {l: i for i, l in enumerate(labels)}
        print(f"Label map: {self.label_map}")

        from collections import Counter
        for k, v in Counter(s["label_str"] for s in samples).items():
            print(f"  {k}: {v}")

        y = [s["label_str"] for s in samples]
        self.train_s, self.test_s = train_test_split(
            samples, test_size=TEST_SPLIT, random_state=SEED, stratify=y)
        print(f"Train: {len(self.train_s)} | Test: {len(self.test_s)}")

        # class weights
        n = len(self.label_map)
        counts = np.zeros(n)
        for s in self.train_s: counts[self.label_map[s["label_str"]]] += 1
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (n * counts)
        self.class_weights = torch.tensor(w, dtype=torch.float32)
        for lab, idx in self.label_map.items():
            print(f"  {lab:25s} count={int(counts[idx]):4d}  weight={w[idx]:.3f}")

        with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)
        pd.DataFrame(self.test_s).to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

    def train_dataloader(self):
        ds = VJEPASegmentDataset(self.train_s, self.label_map, training=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=make_collate(self.processor))

    def val_dataloader(self):
        ds = VJEPASegmentDataset(self.test_s, self.label_map, training=False)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=make_collate(self.processor))


# ============================================================
# Inference
# ============================================================
def run_inference(model, test_samples, label_map, processor, device):
    print("\n" + "=" * 60); print("INFERENCE"); print("=" * 60)
    model.eval().to(device)
    id_to_label = {v: k for k, v in label_map.items()}
    ds = VJEPASegmentDataset(test_samples, label_map, training=False)
    rows = []
    for i in range(len(ds)):
        s = test_samples[i]
        try:
            frames, label = ds[i]
            inputs = processor([frames], return_tensors="pt")
            inputs = {k: (v.to(device=device, dtype=model.encoder.dtype) if v.is_floating_point() else v.to(device))
                      for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            top = int(probs.argmax(1).item())
            conf = float(probs[0, top].item())
            rows.append({
                "video_path": s["video_path"], "start_frame": s["start_frame"],
                "end_frame": s["end_frame"], "true_label": s["label_str"],
                "pred_label": id_to_label[top], "confidence": round(conf, 4),
                "correct": int(id_to_label[top] == s["label_str"]),
            })
            print(f"[{i+1}/{len(ds)}] true={s['label_str']:15s} pred={id_to_label[top]:15s} ({conf:.2f})")
        except Exception as e:
            print(f"  ERR: {e}")
            rows.append({"video_path": s["video_path"], "start_frame": s["start_frame"],
                         "end_frame": s["end_frame"], "true_label": s["label_str"],
                         "pred_label": "ERROR", "confidence": 0.0, "correct": 0})

    df = pd.DataFrame(rows); df.to_csv(OUTPUT_CSV, index=False)
    valid = df[df["pred_label"] != "ERROR"]
    if len(valid):
        acc = valid["correct"].mean()
        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(valid["true_label"], valid["pred_label"], zero_division=0))
        labs = sorted(valid["true_label"].unique())
        cm = confusion_matrix(valid["true_label"], valid["pred_label"], labels=labs)
        print(pd.DataFrame(cm, index=labs, columns=labs))
        with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(valid["true_label"], valid["pred_label"], zero_division=0))


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full_finetune", action="store_true",
                   help="Unfreeze encoder (heavy; default is attentive probe)")
    args = p.parse_args()

    pl.seed_everything(SEED)
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
    dm = VJEPADataModule(processor); dm.setup()

    n = len(dm.label_map)
    model = VJEPAFineTune(num_classes=n, full_finetune=args.full_finetune,
                          class_weights=dm.class_weights)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR, monitor="val_loss", mode="min", save_top_k=2,
        filename="vjepa-{epoch:02d}-{val_loss:.3f}")
    early_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10, precision="16-mixed",
    )
    trainer.fit(model, dm)

    print(f"\nBest: {ckpt_cb.best_model_path}")
    best = VJEPAFineTune.load_from_checkpoint(
        ckpt_cb.best_model_path, num_classes=n, full_finetune=args.full_finetune)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(best, dm.test_s, dm.label_map, processor, device)
    print("\nDone.")


if __name__ == "__main__":
    main()