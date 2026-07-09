"""
VJEPA2 ViT-G Feature Extraction — RMM (Run Once)
=================================================
Extracts features from ALL RMM clips using frozen facebook/vjepa2-vitg-fpc64-256.
Saves a single .pt file shared by all seed training jobs.

Classes: hands_flapping / jumping / rocking / spinning

Output: /orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/rmm/
        extracted_features.pt   <- shared by all seeds
        dataset_meta.json       <- shared by all seeds
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoVideoProcessor

try:
    from decord import VideoReader, cpu
except ImportError as e:
    raise ImportError("Please install decord: pip install eva-decord") from e

# ============================================================
# CONFIG
# ============================================================
CLIPS_DIR     = "/home/aparnabg/orcd/pool/cut_rmm_clips_with_timepotins_bruke_and_other/"
OUTPUT_BASE   = "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/rmm/"
HF_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-256"
EMBED_DIM     = 1408
NUM_FRAMES    = 64
CROP_SIZE     = 256
BATCH_SIZE    = 2        # small for ViT-G (1B params)
NUM_WORKERS   = 8
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES   = 4        # hands_flapping, jumping, rocking, spinning

os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# 1. BUILD FULL DATASET
# ============================================================
def build_dataset_from_folders(clips_dir):
    data = []
    classes = sorted([d for d in os.listdir(clips_dir)
                      if os.path.isdir(os.path.join(clips_dir, d))])
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_dir = os.path.join(clips_dir, cls)
        clips   = glob.glob(os.path.join(cls_dir, "*.mp4"))
        for clip_path in clips:
            data.append({"video_path": clip_path, "label": cls})
        print(f"  {cls}: {len(clips)} clips")

    df = pd.DataFrame(data)
    print(f"Total clips: {len(df)}")
    return df


# ============================================================
# 2. VIDEO DATASET
# ============================================================
class VJEPA2VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor, num_frames=NUM_FRAMES):
        self.video_paths = video_paths
        self.labels      = labels
        self.processor   = processor
        self.num_frames  = num_frames

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames(self, video_path):
        vr           = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, self.num_frames - total_frames), mode='wrap')
        frames = vr.get_batch(indices).asnumpy()                   # [T, H, W, C]
        return torch.from_numpy(frames).permute(0, 3, 1, 2)        # [T, C, H, W]

    def __getitem__(self, idx):
        try:
            frames       = self._sample_frames(self.video_paths[idx])
            inputs       = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].squeeze(0)
            return pixel_values, self.labels[idx]
        except Exception as e:
            print(f"  Error loading {self.video_paths[idx]}: {e}", flush=True)
            return torch.zeros(self.num_frames, 3, CROP_SIZE, CROP_SIZE), self.labels[idx]


# ============================================================
# 3. EXTRACT FEATURES
# ============================================================
@torch.no_grad()
def extract_all_features(model, processor, video_paths, labels, device):
    model.eval()
    model = model.to(device)

    dataset = VJEPA2VideoDataset(video_paths, labels, processor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_features, all_labels, errors = [], [], []

    for batch_idx, (pixel_values, batch_labels) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}", flush=True)
        try:
            pixel_values = pixel_values.to(device)
            outputs      = model(pixel_values_videos=pixel_values, skip_predictor=True)
            features     = outputs.last_hidden_state       # [B, N_tokens, 1408]
            all_features.append(features.cpu().float())
            all_labels.append(batch_labels)
        except Exception as e:
            print(f"  Error at batch {batch_idx}: {e}", flush=True)
            errors.append(batch_idx)

    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.cat(
        [l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in all_labels], dim=0
    )
    print(f"  Features shape : {all_features.shape}", flush=True)
    print(f"  Labels shape   : {all_labels.shape}", flush=True)
    if errors:
        print(f"  Failed batches : {len(errors)}", flush=True)
    return all_features, all_labels


# ============================================================
# 4. MAIN
# ============================================================
def main():
    feat_path = os.path.join(OUTPUT_BASE, "extracted_features.pt")
    meta_path = os.path.join(OUTPUT_BASE, "dataset_meta.json")

    if os.path.exists(feat_path):
        print(f"Features already exist at {feat_path} — skipping extraction.")
        return

    device = torch.device(DEVICE)
    print(f"Device      : {device}")
    print(f"Model       : {HF_MODEL_NAME}")
    print(f"Output base : {OUTPUT_BASE}")

    # Load encoder
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    encoder   = AutoModel.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder hidden size: {encoder.config.hidden_size}")

    # Build full dataset (no split — seeds handle splitting)
    df        = build_dataset_from_folders(CLIPS_DIR)
    classes   = sorted(df["label"].unique())
    label_map = {cls: i for i, cls in enumerate(classes)}
    df["label_encoded"] = df["label"].map(label_map)
    print(f"Label map: {label_map}")

    # Extract features for ALL clips
    print("\nExtracting features for ALL clips...")
    features, labels = extract_all_features(
        encoder, processor,
        df["video_path"].tolist(),
        df["label_encoded"].tolist(),
        device,
    )

    # Save features
    torch.save({"features": features, "labels": labels}, feat_path)
    print(f"Features saved to: {feat_path}")

    # Save metadata so seed jobs know which index = which clip
    meta = {
        "video_paths"   : df["video_path"].tolist(),
        "labels"        : df["label"].tolist(),
        "label_encoded" : df["label_encoded"].tolist(),
        "label_map"     : label_map,
        "embed_dim"     : EMBED_DIM,
        "model"         : HF_MODEL_NAME,
        "num_frames"    : NUM_FRAMES,
        "num_classes"   : NUM_CLASSES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nRMM feature extraction complete!")


if __name__ == "__main__":
    main()
