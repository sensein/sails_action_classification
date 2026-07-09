"""
VJEPA2 Clip-Level Classification — Head Ablation
=================================================

  linear       : mean-pool tokens -> single Linear layer
  mlp_small    : mean-pool -> LayerNorm -> 512-dim MLP -> classifier
  mlp_large    : mean-pool -> LayerNorm -> 1024 -> 512-dim MLP -> classifier
  attentive    : cross-attention query token -> Linear  (original)
  transformer  : 2-layer Transformer encoder -> CLS token -> Linear

Usage:
    # Single head
    python vjepa_clip_level_ablation.py --label loco --head attentive

    # Run ALL heads in one go
    python vjepa_clip_level_ablation.py --label loco --head all
    python vjepa_clip_level_ablation.py --label rmm  --head all

    # Skip encoder inference if features already cached
    python vjepa_clip_level_ablation.py --label loco --head all --skip_extraction

"""

import argparse
import json
import os
from collections import Counter, defaultdict

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ============================================================
# ARGUMENT PARSING
# ============================================================
HEAD_CHOICES = ["linear", "mlp_small", "mlp_large", "attentive", "transformer", "all"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="VJEPA2 clip-level ablation: loco or rmm, multiple head types"
    )
    parser.add_argument("--label", type=str, choices=["loco", "rmm"], required=True,
                        help="Which label column: loco or rmm")
    parser.add_argument("--head", type=str, choices=HEAD_CHOICES, default="all",
                        help="Classification head to use. 'all' runs every head.")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip encoder feature extraction and load cached .pt file")
    parser.add_argument("--seed", type=int, default=42) 
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================
SPLIT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"

LABEL_CONFIGS = {
    "loco": {
        "label_col":  "Locomotion",
        "output_dir": "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vjepa_new_crop/clip_level_ablation/loco",
    },
    "rmm": {
        "label_col":  "Repetitive_Motor_Movements",
        "output_dir": "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vjepa_new_crop/clip_level_ablation/rmm",
    },
}

HF_MODEL_NAME  = "facebook/vjepa2-vitg-fpc64-256"
EMBED_DIM      = 1408
CROP_SIZE      = 256
MEAN           = (0.45, 0.45, 0.45)
STD            = (0.225, 0.225, 0.225)
ANN_FPS        = 15.0

# Clipping rules (identical to SlowFast)
MIN_FRAMES   = 15
CLIP_FRAMES  = 30

# VJEPA input: uniformly sample this many frames from each clip
VJEPA_FRAMES = 8

BATCH_SIZE    = 16
NUM_WORKERS   = 8
MAX_EPOCHS    = 50
LEARNING_RATE = 1e-3



# ============================================================
# 1. CLIPPING LOGIC  (identical to SlowFast check_clips.py)
# ============================================================
def chunk_run(start, end):
    total = end - start + 1
    if total < MIN_FRAMES:
        return []
    if total < CLIP_FRAMES * 2:
        if total < 45:
            return [(start, end)]
        else:
            split_pt = start + CLIP_FRAMES
            return [(start, split_pt - 1), (split_pt, end)]
    clips, s = [], start
    while s <= end:
        e = min(s + CLIP_FRAMES - 1, end)
        if (e - s + 1) >= MIN_FRAMES:
            clips.append((s, e))
        s += CLIP_FRAMES
    return clips


# ============================================================
# 2. FIND ACTION RUNS  (NA / empty breaks the run)
# ============================================================
def find_action_runs(ann, label_col):
    df     = ann.sort_values("Frame").reset_index(drop=True)
    frames = df["Frame"].astype(int).tolist()
    labels = df[label_col].astype(str).tolist()

    runs, i, n = [], 0, len(df)
    while i < n:
        lab = labels[i].strip()
        if lab in ("N/A", "", "nan"):
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and labels[j + 1].strip() == lab
            and labels[j + 1].strip() not in ("N/A", "", "nan")
            and frames[j + 1] == frames[j] + 1
        ):
            j += 1
        runs.append((frames[i], frames[j], lab))
        i = j + 1
    return runs


# ============================================================
# 3. H5 BBOX LOADING
# ============================================================
def load_bbox_map(h5_path):
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


# ============================================================
# 4. BUILD CLIP SAMPLES
# ============================================================
def build_samples(split_csv, label_col):
    df_csv = pd.read_csv(split_csv)
    required = ["video_path", "label_path", "interpolated_anno_h5", "split"]
    missing  = [c for c in required if c not in df_csv.columns]
    if missing:
        raise ValueError(f"Split CSV missing columns: {missing}")

    split_buckets = defaultdict(list)

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
                    "video_path":  vp,
                    "h5_path":     hp,
                    "start_frame": int(cs),
                    "end_frame":   int(ce),
                    "label_str":   lab,
                    "split":       sp,
                })

    train_s = split_buckets.get("train", [])
    val_s   = split_buckets.get("val",   [])
    test_s  = split_buckets.get("test",  [])

    print(f"  Clips -> train:{len(train_s)} | val:{len(val_s)} | test:{len(test_s)}")
    dist = Counter(s["label_str"] for s in train_s + val_s + test_s)
    print("  Class distribution (all splits):")
    for k, v in sorted(dist.items()):
        print(f"    {k}: {v}")

    return train_s, val_s, test_s


# ============================================================
# 5. DATASET
# ============================================================
class ClipDataset(Dataset):
    def __init__(self, samples, label_map, vjepa_frames=VJEPA_FRAMES,
                 crop_size=CROP_SIZE, training=False):
        self.samples      = samples
        self.label_map    = label_map
        self.vjepa_frames = vjepa_frames
        self.crop_size    = crop_size
        self.training     = training
        self.mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1, 1)

    def __len__(self): return len(self.samples)

    def _load_clip(self, s):
        cap = cv2.VideoCapture(s["video_path"])
        if not cap.isOpened():
            raise IOError(f"cannot open {s['video_path']}")
        vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step      = max(1, int(round(vid_fps / ANN_FPS)))
        total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox_map  = load_bbox_map(s["h5_path"])
        bbox_keys = np.array(sorted(bbox_map.keys()))

        ann_frames = np.arange(s["start_frame"], s["end_frame"] + 1)
        idxs       = np.linspace(0, len(ann_frames) - 1, self.vjepa_frames).astype(int)
        chosen     = ann_frames[idxs]

        frames = []
        for af in chosen:
            vf = max(0, min(int(af * step), total_vid - 1))
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
            crop = cv2.resize(frame[y1:y2, x1:x2], (self.crop_size, self.crop_size))
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        cap.release()

        arr    = np.ascontiguousarray(np.stack(frames), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)   # [C, T, H, W]
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx):
        s     = self.samples[idx]
        label = self.label_map[s["label_str"]]
        try:
            clip = self._load_clip(s)
            if self.training and torch.rand(1).item() < 0.5:
                clip = torch.flip(clip, dims=[3])
            return clip, label
        except Exception as e:
            print(f"  load error [{os.path.basename(s['video_path'])} "
                  f"{s['start_frame']}-{s['end_frame']}]: {e}")
            return torch.zeros(3, self.vjepa_frames, self.crop_size, self.crop_size), label


# ============================================================
# 6. CLASSIFICATION HEADS  (ablation variants)
# ============================================================

class LinearProbe(nn.Module):
    """
    Simplest baseline: mean-pool all patch tokens -> single Linear layer.
    No nonlinearity. Purely tests linear separability of VJEPA features.
    """
    def __init__(self, embed_dim, num_classes, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        return self.head(self.norm(x.mean(dim=1)))


class MLPSmallProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> one hidden layer (512) -> GELU -> Dropout -> Linear.
    Adds nonlinearity over LinearProbe with minimal parameters.
    """
    def __init__(self, embed_dim, num_classes, hidden=512, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))


class MLPLargeProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> 1024 -> GELU -> 512 -> GELU -> Dropout -> Linear.
    Deeper MLP; more capacity to learn non-linear feature combinations.
    """
    def __init__(self, embed_dim, num_classes, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))


class AttentiveProbe(nn.Module):
    """
    Cross-attention: a single learned query attends over all patch tokens.
    Lets the model select the most task-relevant spatial/temporal tokens.
    This is the original head from vjepa_clip_level.py.
    """
    def __init__(self, embed_dim, num_classes, num_heads=8, **kwargs):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm  = nn.LayerNorm(embed_dim)
        self.head  = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.head(self.norm(out).squeeze(1))


class TransformerProbe(nn.Module):
    """
    Prepend a learnable CLS token, run 2-layer Transformer encoder,
    then classify from the CLS output.
    Adds self-attention among all tokens before pooling; most powerful head.
    """
    def __init__(self, embed_dim, num_classes, num_heads=8,
                 num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)           # [B, N+1, D]
        x   = self.transformer(x)
        return self.head(self.norm(x[:, 0]))        # CLS token


# ---- Factory -------------------------------------------------------
def build_probe(head_name, embed_dim, num_classes):
    registry = {
        "linear":      LinearProbe,
        "mlp_small":   MLPSmallProbe,
        "mlp_large":   MLPLargeProbe,
        "attentive":   AttentiveProbe,
        "transformer": TransformerProbe,
    }
    if head_name not in registry:
        raise ValueError(f"Unknown head '{head_name}'. Choose from {list(registry)}")
    probe    = registry[head_name](embed_dim=embed_dim, num_classes=num_classes)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  Head '{head_name}': {n_params:,} parameters")
    return probe


# ============================================================
# 7. FEATURE EXTRACTION  (frozen encoder, run once)
# ============================================================
@torch.no_grad()
def extract_features(encoder, samples, label_map, device, batch_size=4):
    encoder.eval()
    ds     = ClipDataset(samples, label_map, training=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_feats, all_labels = [], []
    for i, (clips, labels) in enumerate(loader):
        if i % 50 == 0:
            print(f"    batch {i}/{len(loader)}")
        # clips: [B, C, T, H, W] -> encoder expects [B, T, C, H, W]
        clips = clips.permute(0, 2, 1, 3, 4).to(device)
        try:
            out = encoder(pixel_values_videos=clips, skip_predictor=True)
            all_feats.append(out.last_hidden_state.cpu().float())
            all_labels.append(labels)
        except Exception as e:
            print(f"    batch {i} error: {e}")
            all_feats.append(torch.zeros(clips.shape[0], 1, EMBED_DIM))
            all_labels.append(labels)

    return torch.cat(all_feats, 0), torch.cat(all_labels, 0)


# ============================================================
# 8. FEATURE DATASET
# ============================================================
class FeatureDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats  = feats
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.feats[i], self.labels[i]


# ============================================================
# 9. TRAIN PROBE
# ============================================================
def train_probe(probe, tr_dl, va_dl, device,
                class_weights=None, max_epochs=MAX_EPOCHS, tag="probe"):
    probe = probe.to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    cw    = class_weights.to(device) if class_weights is not None else None

    best_acc = 0.0; best_state = None; patience = 10; patience_ctr = 0

    for epoch in range(max_epochs):
        probe.train()
        tl = tc = tt = 0
        for feats, labels in tr_dl:
            feats, labels = feats.to(device), labels.to(device)
            logits = probe(feats)
            loss   = F.cross_entropy(logits, labels, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * labels.size(0)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += labels.size(0)
        sch.step()

        probe.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for feats, labels in va_dl:
                feats, labels = feats.to(device), labels.to(device)
                logits = probe(feats)
                vl += F.cross_entropy(logits, labels).item() * labels.size(0)
                vc += (logits.argmax(1) == labels).sum().item()
                vt += labels.size(0)

        val_acc = vc / vt
        print(f"  [{tag}] Ep {epoch+1:3d} | "
              f"train loss={tl/tt:.4f} acc={tc/tt:.4f} | "
              f"val loss={vl/vt:.4f} acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_ctr = 0
            print(f"  [{tag}] -> best val acc: {best_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [{tag}] Early stop at ep {epoch+1}")
                break

    if best_state:
        probe.load_state_dict(best_state)
    print(f"  [{tag}] Final best val acc: {best_acc:.4f}")
    return probe, best_acc


# ============================================================
# 10. INFERENCE
# ============================================================
@torch.no_grad()
def run_inference(probe, te_feats, test_samples, label_map, device,
                  out_dir, head_name):
    probe.eval().to(device)
    id2lab  = {v: k for k, v in label_map.items()}
    softmax = nn.Softmax(dim=1)

    te_labels_enc = torch.tensor(
        [label_map[s["label_str"]] for s in test_samples], dtype=torch.long
    )
    ds     = FeatureDataset(te_feats, te_labels_enc)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 4, shuffle=False)
    rows   = []
    idx    = 0

    for feats, _labels in loader:
        probs = softmax(probe(feats.to(device)))
        for i in range(probs.shape[0]):
            top  = int(probs[i].argmax().item())
            conf = float(probs[i, top].item())
            s    = test_samples[idx]
            rows.append({
                "video_path":  s["video_path"],
                "start_frame": s["start_frame"],
                "end_frame":   s["end_frame"],
                "true_label":  s["label_str"],
                "pred_label":  id2lab[top],
                "confidence":  round(conf, 4),
                "correct":     int(id2lab[top] == s["label_str"]),
            })
            idx += 1

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"test_predictions_{head_name}.csv")
    df.to_csv(csv_path, index=False)

    valid = df[df["pred_label"] != "ERROR"]
    acc   = valid["correct"].mean() if len(valid) else 0.0
    if len(valid):
        print(f"\n[{head_name}] Accuracy: {acc:.4f}  "
              f"({int(valid['correct'].sum())}/{len(valid)})")
        print(classification_report(valid["true_label"], valid["pred_label"],
                                    zero_division=0))
        labs = sorted(valid["true_label"].unique())
        cm   = pd.DataFrame(
            confusion_matrix(valid["true_label"], valid["pred_label"], labels=labs),
            index=labs, columns=labs,
        )
        print("Confusion matrix:\n", cm)
        txt_path = os.path.join(out_dir, f"test_metrics_{head_name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Head: {head_name}\n")
            f.write(f"Model: {HF_MODEL_NAME}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(valid["true_label"], valid["pred_label"],
                                          zero_division=0))
            f.write(f"\nConfusion matrix:\n{cm.to_string()}\n")
    return acc


# ============================================================
# 11. PER-HEAD RUNNER
# ============================================================
def run_one_head(head_name, tr_f, tr_l, va_f, va_l, te_f,
                 train_s, test_s, lmap, cw, device, base_out_dir):
    """Train + evaluate one head variant. Returns (head_name, val_acc, test_acc)."""
    head_dir = os.path.join(base_out_dir, head_name)
    os.makedirs(head_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  HEAD: {head_name.upper()}")
    print("=" * 60)

    probe = build_probe(head_name, EMBED_DIM, len(lmap))
    tr_dl = DataLoader(FeatureDataset(tr_f, tr_l),
                       batch_size=BATCH_SIZE * 4, shuffle=True,  num_workers=0)
    va_dl = DataLoader(FeatureDataset(va_f, va_l),
                       batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0)

    probe, best_val_acc = train_probe(
        probe, tr_dl, va_dl, device, cw, MAX_EPOCHS, tag=head_name
    )
    torch.save(probe.state_dict(), os.path.join(head_dir, "best_probe.pt"))

    test_acc = run_inference(probe, te_f, test_s, lmap, device, head_dir, head_name)
    return head_name, best_val_acc, test_acc


# ============================================================
# 12. MAIN
# ============================================================
def main():
    args      = parse_args()
    cfg       = LABEL_CONFIGS[args.label]
    label_col = cfg["label_col"]
    base_dir = os.path.join(cfg["output_dir"], f"seed_{args.seed}")
    os.makedirs(base_dir, exist_ok=True)

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    heads_to_run = (
        [h for h in HEAD_CHOICES if h != "all"]
        if args.head == "all" else [args.head]
    )

    print("=" * 60)
    print(f"  VJEPA2 Clip-Level Ablation  |  {args.label.upper()}")
    print(f"  Heads to run : {heads_to_run}")
    print(f"  Model        : {HF_MODEL_NAME}")
    print(f"  Output       : {base_dir}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Build samples & label map
    # ----------------------------------------------------------------
    train_s, val_s, test_s = build_samples(SPLIT_CSV, label_col)
    if not (train_s or val_s or test_s):
        raise RuntimeError("No samples found. Check CSV / annotation paths.")

    all_s  = train_s + val_s + test_s
    labels = sorted({s["label_str"] for s in all_s})
    lmap   = {lab: i for i, lab in enumerate(labels)}
    print(f"\nLabel map: {lmap}")
    with open(os.path.join(base_dir, "label_mapping.json"), "w") as f:
        json.dump(lmap, f, indent=2)
    pd.DataFrame(test_s).to_csv(os.path.join(base_dir, "test_split.csv"), index=False)

    # Inverse-frequency class weights from train split
    n  = len(lmap)
    ct = np.zeros(n)
    for s in train_s: ct[lmap[s["label_str"]]] += 1
    ct = np.maximum(ct, 1.0)
    cw = torch.tensor(ct.sum() / (n * ct), dtype=torch.float32)
    print("\nClass weights (train):")
    for lab, idx in lmap.items():
        print(f"  {lab:30s}  count={int(ct[idx]):4d}  weight={cw[idx]:.3f}")

    feat_cache = os.path.join(base_dir, "extracted_features.pt")

    # ----------------------------------------------------------------
    # Feature extraction  (skip if --skip_extraction and cache exists)
    # ----------------------------------------------------------------
    if args.skip_extraction and os.path.exists(feat_cache):
        print(f"\nLoading cached features from {feat_cache}")
        cache      = torch.load(feat_cache, map_location="cpu")
        tr_f, tr_l = cache["train"]
        va_f, va_l = cache["val"]
        te_f, te_l = cache["test"]
        print(f"  train: {tr_f.shape}  val: {va_f.shape}  test: {te_f.shape}")
    else:
        from transformers import AutoModel
        print(f"\nLoading frozen encoder: {HF_MODEL_NAME}")
        encoder = AutoModel.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).eval().to(device)
        for p in encoder.parameters():
            p.requires_grad = False
        print(f"  embed_dim={EMBED_DIM}  |  all params frozen")

        print("\nExtracting train features...")
        tr_f, tr_l = extract_features(encoder, train_s, lmap, device)
        print("Extracting val features...")
        va_f, va_l = extract_features(encoder, val_s,   lmap, device)
        print("Extracting test features...")
        te_f, te_l = extract_features(encoder, test_s,  lmap, device)

        torch.save({"train": (tr_f, tr_l), "val": (va_f, va_l), "test": (te_f, te_l)},
                   feat_cache)
        print(f"Features cached -> {feat_cache}")

        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFeature shapes: train={tuple(tr_f.shape)} "
          f"val={tuple(va_f.shape)} test={tuple(te_f.shape)}")

    # ----------------------------------------------------------------
    # Ablation loop
    # ----------------------------------------------------------------
    summary_rows = []
    for head_name in heads_to_run:
        h_name, val_acc, test_acc = run_one_head(
            head_name,
            tr_f, tr_l, va_f, va_l, te_f,
            train_s, test_s, lmap, cw, device, base_dir,
        )
        summary_rows.append({
            "head":     h_name,
            "val_acc":  round(float(val_acc),  4),
            "test_acc": round(float(test_acc), 4),
            "label":    args.label,
        })

    # ----------------------------------------------------------------
    # Ablation summary table
    # ----------------------------------------------------------------
    if len(summary_rows) > 1:
        summary_df   = pd.DataFrame(summary_rows).sort_values("test_acc", ascending=False)
        summary_path = os.path.join(base_dir, "ablation_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 60)
        print("ABLATION SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved -> {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

