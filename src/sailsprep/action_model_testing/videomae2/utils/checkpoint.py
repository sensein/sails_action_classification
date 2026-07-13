import os

import torch
import torch.nn as nn


def load_pretrained_vitb_k710(ckpt_path):
    """
    Instantiates VideoMAE V2 ViT-B/16 with a 710-way (K710) head and loads
    the distilled K710 checkpoint into it, downloading it to ckpt_path if
    not already present.

    Returns (model, missing_keys, unexpected_keys).
    """
    from modeling_finetune import vit_base_patch16_224

    model = vit_base_patch16_224(num_classes=710)

    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/"
            "vit_b_k710_dl_from_giant.pth",
            ckpt_path,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("module", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, missing, unexpected


def build_videomae2_vitb(num_classes, ckpt_path, freeze_all_but_last_block=True):
    """
    Loads VideoMAE V2 ViT-Base with K710 distilled checkpoint (86.6% K400),
    replaces the classification head for num_classes, and optionally
    freezes all but the last transformer block (blocks.11) + head + fc_norm.
    """
    try:
        model, missing, unexpected = load_pretrained_vitb_k710(ckpt_path)
    except ImportError as e:
        raise ImportError(
            "Could not import from modeling_finetune.py.\n"
            "Download it:\n"
            "  wget -O modeling_finetune.py "
            "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py\n"
            "Place it in the same directory as this script."
        ) from e

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
