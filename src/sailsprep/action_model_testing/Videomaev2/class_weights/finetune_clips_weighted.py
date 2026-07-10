"""
VideoMAE V2 Fine-tuning for Locomotion Classification
- Class weights (inverse frequency) to fix class imbalance
- Supports all models: vit_s, vit_b, vit_l, vit_h, vit_g
- Detailed evaluation metrics at the end
Set: export MODEL=vit_b  (or vit_s, vit_l, vit_h, vit_g)
Run: python videomae_finetune_weighted.py
"""

import shutil
import os
import sys
import subprocess
import json
import re
import pandas as pd
import torch
import numpy as np
from collections import Counter
from pathlib import Path

# ========================== MODEL SELECTION ==========================
MODEL_KEY = os.environ.get("MODEL", "vit_b")

MODEL_CONFIGS = {
    "vit_s": {
        "model_name":    "vit_small_patch16_224",
        "checkpoint":    "vit_s_k710_dl_from_giant.pth",
        "batch_size":    8,
        "lr":            1e-4,
        "layer_decay":   0.65,
        "drop_path":     0.1,
        "tubelet_size":  2,
    },
    "vit_b": {
        "model_name":    "vit_base_patch16_224",
        "checkpoint":    "vit_b_k710_dl_from_giant.pth",
        "batch_size":    8,
        "lr":            1e-4,
        "layer_decay":   0.65,
        "drop_path":     0.2,
        "tubelet_size":  2,
    },
    "vit_l": {
        "model_name":    "vit_large_patch16_224",
        "checkpoint":    "vit_l_from_hf.pth",
        "batch_size":    4,
        "lr":            1e-4,
        "layer_decay":   0.75,
        "drop_path":     0.2,
        "tubelet_size":  2,
    },
    "vit_h": {
        "model_name":    "vit_huge_patch16_224",
        "checkpoint":    "vit_h_from_hf.pth",
        "batch_size":    2,
        "lr":            1e-4,
        "layer_decay":   0.80,
        "drop_path":     0.3,
        "tubelet_size":  2,
    },
    "vit_g": {
        "model_name":    "vit_giant_patch14_224",
        "checkpoint":    "vit_g_from_hf.pth",
        "batch_size":    2,
        "lr":            1e-4,
        "layer_decay":   0.90,
        "drop_path":     0.3,
        "tubelet_size":  1,   # giant uses tubelet=1
    },
}

if MODEL_KEY not in MODEL_CONFIGS:
    print(f"[ERROR] Unknown MODEL='{MODEL_KEY}'. Choose from: {list(MODEL_CONFIGS.keys())}")
    sys.exit(1)

cfg = MODEL_CONFIGS[MODEL_KEY]
print(f"\n[INFO] Running model: {MODEL_KEY} → {cfg['model_name']}")

# ========================== CONFIG ==========================
BASE     = "/orcd/scratch/orcd/007/aparnabg/all_project_files"
REPO_DIR = f"/orcd/data/satra/002/projects/SAILS/feature_processing/pipeline_outputs/pose_outputs/VideoMAEv2/"
CKPT_DIR = f"{REPO_DIR}/checkpoints"

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{BASE}/action_sota_models/Videomaev2/class_weights/output/locomotion_{MODEL_KEY}_weighted"
)
MASTER_CSV = os.environ.get(
    "MASTER_CSV",
    "/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v2.csv"
)
CHECKPOINT  = os.path.join(CKPT_DIR, cfg["checkpoint"])
CLIP_COL    = "cut_clip_path"
SPLIT_COL   = "split"
CLASSES     = ["Walking", "Cruising", "Crawling", "Vehicle", "Running"]

# ========================== ENV SETUP ==========================
os.environ["SLURM_NTASKS"] = "1"
os.environ["SLURM_PROCID"] = "0"
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    os.environ["RANK"]       = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
env = os.environ.copy()

# ========================== STEP 1: PREPARE DATA ==========================
print("\n" + "=" * 55)
print("  STEP 1: Preparing dataset from master CSV")
print("=" * 55 + "\n")

df = pd.read_csv(MASTER_CSV)
print(f"Loaded {len(df)} rows from {MASTER_CSV}")

df = df.dropna(subset=[CLIP_COL])
df = df[df[CLIP_COL].str.strip() != ""]

df["label_name"] = df[CLIP_COL].apply(lambda p: os.path.basename(os.path.dirname(p)))
class_to_idx     = {cls: idx for idx, cls in enumerate(CLASSES)}
print(f"Class mapping: {class_to_idx}")

df = df[df["label_name"].isin(class_to_idx)]
df["label_idx"] = df["label_name"].map(class_to_idx)

df_train = df[df[SPLIT_COL] == "train"]
df_val   = df[df[SPLIT_COL] == "val"]
df_test  = df[df[SPLIT_COL] == "test"]

print(f"\n{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-" * 54)
for cls in CLASSES:
    n_tr = len(df_train[df_train["label_name"] == cls])
    n_v  = len(df_val  [df_val  ["label_name"] == cls])
    n_te = len(df_test [df_test ["label_name"] == cls])
    print(f"{cls:<12} {n_tr:>8} {n_v:>8} {n_te:>8} {n_tr+n_v+n_te:>8}")
print("-" * 54)
print(f"{'TOTAL':<12} {len(df_train):>8} {len(df_val):>8} {len(df_test):>8} {len(df):>8}")

for _, row in df.iterrows():
    if " " in str(row[CLIP_COL]):
        print(f"[WARNING] Space in path: {row[CLIP_COL]}")

# ========================== STEP 1b: CLASS WEIGHTS ==========================
print("\n" + "=" * 55)
print("  STEP 1b: Computing class weights")
print("=" * 55 + "\n")

train_counts = Counter(df_train["label_idx"].values)
total_train  = len(df_train)
weights      = [total_train / (len(CLASSES) * train_counts.get(i, 1))
                for i in range(len(CLASSES))]
weights_tensor = torch.tensor(weights, dtype=torch.float32)

print("Class weights (inverse frequency):")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<12} count={train_counts.get(i,0):>6}  weight={weights[i]:.4f}")

weights_path = os.path.join(REPO_DIR, "class_weights.pt")
torch.save(weights_tensor, weights_path)
print(f"\n[OK] Saved class weights → {weights_path}")

# ========================== STEP 1c: PATCH run_class_finetuning.py ==========================
print("\n" + "=" * 55)
print("  STEP 1c: Patching run_class_finetuning.py")
print("=" * 55 + "\n")

run_ft_path   = os.path.join(REPO_DIR, "run_class_finetuning.py")
run_ft_backup = run_ft_path + ".bak"
PATCH_MARKER  = "# PATCHED: class weights"

with open(run_ft_path, "r") as f:
    run_ft_code = f.read()

if PATCH_MARKER not in run_ft_code:
    # Backup
    with open(run_ft_backup, "w") as f:
        f.write(run_ft_code)
    print(f"[OK] Backed up original → {run_ft_backup}")

    # 1. Inject weight loader after last top-level import
    weight_loader = f"""
{PATCH_MARKER}
import os as _os
import torch as _torch
_cw_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "class_weights.pt")
_class_weights = _torch.load(_cw_path, weights_only=True) if _os.path.exists(_cw_path) else None
if _class_weights is not None:
    print(f"[INFO] Class weights loaded: {{_class_weights.tolist()}}")
"""
    lines      = run_ft_code.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1
    lines.insert(insert_idx, weight_loader)
    run_ft_code = "\n".join(lines)

    # 2. Patch plain CrossEntropyLoss() → weighted version
    run_ft_code = re.sub(
        r'(criterion\s*=\s*(?:torch\.nn\.|nn\.)CrossEntropyLoss\s*\(\s*)\)',
        r'\1weight=_class_weights.to(device) if _class_weights is not None else None)',
        run_ft_code,
    )

    # 3. If LabelSmoothing / SoftTarget / Mixup criterion exists, add override block
    if "LabelSmoothingCrossEntropy" in run_ft_code or "SoftTargetCrossEntropy" in run_ft_code:
        override = """
    # CLASS WEIGHT OVERRIDE — force weighted CE loss when weights file exists
    if _class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(
            weight=_class_weights.to(device)
        )
        print("[INFO] Overriding loss with weighted CrossEntropyLoss")
"""
        matches = list(re.finditer(
            r'criterion\s*=\s*(?:LabelSmoothingCrossEntropy|SoftTargetCrossEntropy'
            r'|torch\.nn\.CrossEntropyLoss|nn\.CrossEntropyLoss)\s*\([^)]*\)',
            run_ft_code
        ))
        if matches:
            pos = run_ft_code.find("\n", matches[-1].end())
            run_ft_code = run_ft_code[:pos] + "\n" + override + run_ft_code[pos:]

    with open(run_ft_path, "w") as f:
        f.write(run_ft_code)
    print("[OK] Patched run_class_finetuning.py with weighted CrossEntropyLoss")
else:
    print("[SKIP] Already patched")

# ========================== STEP 1d: WRITE CSV FILES ==========================
for path, subset in [
    (os.path.join(REPO_DIR, "train.csv"), df_train),
    (os.path.join(REPO_DIR, "val.csv"),   df_val),
    (os.path.join(REPO_DIR, "test.csv"),  df_test),
]:
    with open(path, "w") as f:
        for _, row in subset.iterrows():
            f.write(f"{row[CLIP_COL]} {row['label_idx']}\n")

print(f"\n[OK] CSVs written: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# ========================== STEP 2: TRAIN ==========================
print("\n" + "=" * 55)
print(f"  STEP 2: Fine-tuning {MODEL_KEY.upper()} with class weights")
print("=" * 55 + "\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_cmd = (
    f"python {REPO_DIR}/run_class_finetuning.py"
    f" --model {cfg['model_name']}"
    f" --data_set Kinetics-400"
    f" --nb_classes {len(CLASSES)}"
    f" --data_path {REPO_DIR}"
    f" --finetune {CHECKPOINT}"
    f" --log_dir {OUTPUT_DIR}"
    f" --output_dir {OUTPUT_DIR}"
    f" --batch_size {cfg['batch_size']}"
    f" --num_sample 2"
    f" --input_size 224"
    f" --short_side_size 224"
    f" --save_ckpt_freq 50"
    f" --num_frames 16"
    f" --sampling_rate 8"           # fixed for 30fps
    f" --opt adamw"
    f" --lr {cfg['lr']}"
    f" --warmup_lr 1e-6"
    f" --min_lr 1e-6"
    f" --layer_decay {cfg['layer_decay']}"
    f" --opt_betas 0.9 0.999"
    f" --weight_decay 0.05"
    f" --epochs 50"
    f" --test_num_segment 5"
    f" --test_num_crop 3"
    f" --warmup_epochs 10"          # more warmup helps imbalanced data
    f" --drop_path {cfg['drop_path']}"
    f" --head_drop_rate 0.0"        # removed — hurts minority classes
    f" --smoothing 0"               # disabled — let weighted CE handle it
    f" --mixup 0"                   # disabled — confuses weighted loss
    f" --cutmix 0"                  # disabled — confuses weighted loss
    f" --num_workers 8"
)

print(f"[CMD] {train_cmd}\n")
result = subprocess.run(train_cmd, shell=True, cwd=REPO_DIR, env=env)
if result.returncode != 0:
    print(f"\n[ERROR] Training failed (exit code {result.returncode})")
    sys.exit(1)
# ========================== CLEANUP: Remove epoch checkpoints ==========================
print("\n[INFO] Cleaning up intermediate checkpoints...")
for ckpt_file in Path(OUTPUT_DIR).glob("checkpoint-[0-9]*.pth"):
    if "best" not in ckpt_file.name:
        size_mb = ckpt_file.stat().st_size / (1024 * 1024)
        print(f"  Removing {ckpt_file.name} ({size_mb:.0f} MB)")
        ckpt_file.unlink()
print("[OK] Only checkpoint-best.pth retained")
# ========================== STEP 3: EVALUATE (VideoMAE built-in) ==========================
print("\n" + "=" * 55)
print("  STEP 3: VideoMAE built-in evaluation")
print("=" * 55 + "\n")

ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint-best.pth")
if not os.path.exists(ckpt_path):
    ckpts     = sorted(Path(OUTPUT_DIR).glob("checkpoint-*.pth"))
    ckpt_path = str(ckpts[-1]) if ckpts else None

if ckpt_path is None:
    print("[ERROR] No checkpoint found.")
    sys.exit(1)

eval_cmd = (
    f"python {REPO_DIR}/run_class_finetuning.py"
    f" --model {cfg['model_name']}"
    f" --data_set Kinetics-400"
    f" --nb_classes {len(CLASSES)}"
    f" --data_path {REPO_DIR}"
    f" --finetune {ckpt_path}"
    f" --log_dir {OUTPUT_DIR}/eval"
    f" --output_dir {OUTPUT_DIR}/eval"
    f" --batch_size {cfg['batch_size']}"
    f" --num_sample 1"
    f" --input_size 224"
    f" --short_side_size 224"
    f" --num_frames 16"
    f" --sampling_rate 8"
    f" --test_num_segment 5"
    f" --test_num_crop 3"
    f" --eval"
    f" --num_workers 8"
)

print(f"[CMD] {eval_cmd}\n")
result = subprocess.run(eval_cmd, shell=True, cwd=REPO_DIR, env=env)
if result.returncode != 0:
    print(f"\n[ERROR] Eval failed (exit code {result.returncode})")
    sys.exit(1)

# ========================== STEP 4: DETAILED METRICS ==========================
print("\n" + "=" * 55)
print(f"  STEP 4: Detailed metrics — {MODEL_KEY.upper()}")
print("=" * 55 + "\n")

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        cohen_kappa_score, matthews_corrcoef,
        balanced_accuracy_score, average_precision_score,
        top_k_accuracy_score, precision_recall_fscore_support,
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    print("[ERROR] pip install scikit-learn")
    sys.exit(1)

try:
    import decord
    decord.bridge.set_bridge("torch")
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[ERROR] pip install decord torchvision")
    sys.exit(1)

# ── Dataset ──────────────────────────────────────────────────
class VideoClipDataset(Dataset):
    def __init__(self, df, num_frames=16, size=224):
        self.df         = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform  = T.Compose([
            T.Resize((size, size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.df)

    def _load(self, path):
        try:
            vr      = decord.VideoReader(path, num_threads=1)
            idx     = np.linspace(0, len(vr)-1, self.num_frames).astype(int)
            frames  = vr.get_batch(idx).float() / 255.0   # (T,H,W,C)
            frames  = frames.permute(0, 3, 1, 2)           # (T,C,H,W)
            frames  = torch.stack([self.transform(f) for f in frames])
            return frames.permute(1, 0, 2, 3)             # (C,T,H,W)
        except Exception as e:
            print(f"[WARN] {path}: {e}")
            return torch.zeros(3, self.num_frames, 224, 224)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self._load(row[CLIP_COL]), int(row["label_idx"])

# ── Load model ────────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
import models.modeling_finetune
from timm.models import create_model as timm_create

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

model = timm_create(
    cfg["model_name"],
    pretrained=False,
    num_classes=len(CLASSES),
    all_frames=16,
    tubelet_size=cfg["tubelet_size"],
    drop_rate=0.0,
    drop_path_rate=0.0,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
)
ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model", ckpt)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
msg        = model.load_state_dict(state_dict, strict=False)
print(f"[INFO] Loaded. Missing={len(msg.missing_keys)} Unexpected={len(msg.unexpected_keys)}")
model      = model.to(device).eval()

# ── Inference on test set ─────────────────────────────────────
loader = DataLoader(VideoClipDataset(df_test), batch_size=8,
                    shuffle=False, num_workers=4, pin_memory=True)

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for i, (videos, labels) in enumerate(loader):
        probs = torch.softmax(model(videos.to(device)), dim=-1)
        all_preds.extend(probs.argmax(-1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        if (i+1) % 10 == 0:
            print(f"  {min((i+1)*8, len(df_test))}/{len(df_test)} clips done")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── Metrics ───────────────────────────────────────────────────
top1    = (all_preds == all_labels).mean() * 100
top2    = top_k_accuracy_score(all_labels, all_probs, k=2) * 100
bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100
kappa   = cohen_kappa_score(all_labels, all_preds)
mcc     = matthews_corrcoef(all_labels, all_preds)

labels_bin   = label_binarize(all_labels, classes=list(range(len(CLASSES))))
ap_per_class = [average_precision_score(labels_bin[:, c], all_probs[:, c])
                for c in range(len(CLASSES))]
mAP = np.mean(ap_per_class)

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(len(CLASSES))), zero_division=0)

p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro",    zero_division=0)
p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
_,     _,     f1_mi,  _ = precision_recall_fscore_support(all_labels, all_preds, average="micro",    zero_division=0)

cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
per_class_acc = [
    (all_preds[all_labels == c] == c).mean() * 100
    if (all_labels == c).sum() > 0 else 0.0
    for c in range(len(CLASSES))
]

# ── Print + save report ───────────────────────────────────────
report_lines = []
def log(line=""):
    print(line)
    report_lines.append(line)

log("\n" + "=" * 60)
log(f"  METRICS REPORT — {MODEL_KEY.upper()} (with class weights)")
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
log(f"  Precision Weighted: {p_wt:.4f}")
log(f"  Recall Macro      : {r_mac:.4f}")
log(f"  Recall Weighted   : {r_wt:.4f}")

log(f"\nPER-CLASS")
log(f"  {'Class':<12} {'Acc%':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7} {'N':>6}")
log(f"  {'-'*56}")
for i, cls in enumerate(CLASSES):
    log(f"  {cls:<12} {per_class_acc[i]:>7.2f} {precision[i]:>7.4f} "
        f"{recall[i]:>7.4f} {f1[i]:>7.4f} {ap_per_class[i]:>7.4f} {support[i]:>6}")

log(f"\nCONFUSION MATRIX (Rows=Actual, Cols=Predicted)")
log(f"  {'':12}" + "".join(f"{c[:7]:>8}" for c in CLASSES))
for i, cls in enumerate(CLASSES):
    log(f"  {cls:<12}" + "".join(f"{cm[i,j]:>8}" for j in range(len(CLASSES))))

log(f"\nCLASSIFICATION REPORT")
log(classification_report(all_labels, all_preds,
                           target_names=CLASSES, digits=4, zero_division=0))
log("=" * 60)

eval_dir = os.path.join(OUTPUT_DIR, "eval")
os.makedirs(eval_dir, exist_ok=True)

with open(os.path.join(eval_dir, f"report_{MODEL_KEY}.txt"), "w") as f:
    f.write("\n".join(report_lines))
print(f"[OK] Report → {eval_dir}/report_{MODEL_KEY}.txt")

with open(os.path.join(eval_dir, f"metrics_{MODEL_KEY}.json"), "w") as f:
    json.dump({
        "model": MODEL_KEY, "checkpoint": ckpt_path,
        "top1_acc": float(top1), "top2_acc": float(top2),
        "balanced_acc": float(bal_acc), "cohen_kappa": float(kappa),
        "mcc": float(mcc), "mAP": float(mAP),
        "f1_micro": float(f1_mi), "f1_macro": float(f1_mac), "f1_weighted": float(f1_wt),
        "precision_macro": float(p_mac), "precision_weighted": float(p_wt),
        "recall_macro": float(r_mac),    "recall_weighted": float(r_wt),
        "per_class": {
            cls: {
                "accuracy": float(per_class_acc[i]), "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]),
                "AP": float(ap_per_class[i]), "support": int(support[i]),
            } for i, cls in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)
print(f"[OK] Metrics JSON → {eval_dir}/metrics_{MODEL_KEY}.json")

print("\n" + "=" * 55)
print(f"  ALL DONE! Model: {MODEL_KEY.upper()}")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Metrics: {eval_dir}/")
print("=" * 55)