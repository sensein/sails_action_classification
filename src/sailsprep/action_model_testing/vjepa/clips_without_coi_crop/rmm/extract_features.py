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

import json
import os
import sys

import torch
from transformers import AutoModel, AutoVideoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from clips_without_coi_crop.common.extraction import build_dataset_from_folders, extract_all_features

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
# MAIN
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
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    encoder   = AutoModel.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=dtype,
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
        num_frames=NUM_FRAMES, crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        flush=True,
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
