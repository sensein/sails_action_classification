"""Shared helpers used by the Video Swin training scripts.

These are the functions/classes whose logic was byte-for-byte identical
(modulo docstring wording) across clip_based/video_swin_finetune.py,
sliding_window/video_swin_fullvideo_sliding.py, and
sliding_window/video_swin_binary_sliding.py. Moved here verbatim so all
three import the same implementation instead of duplicating it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import h5py
import torch
import torch.nn as nn


def load_bbox_map(h5_path: str) -> Dict[int, Tuple[int, int, int, int]]:
    """Load per-frame bounding boxes from an HDF5 annotations file.

    Args:
        h5_path: Path to the interpolated annotation HDF5 file.

    Returns:
        Mapping from annotation frame index to ``(x1, y1, x2, y2)`` bbox.
    """
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {
        int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1
    }


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(video, label)`` pairs into batched tensors.

    Args:
        batch: List of ``(video_tensor, class_index)`` tuples.

    Returns:
        Tuple of ``(videos, labels)`` with shapes ``(B, C, T, H, W)``
        and ``(B,)`` respectively.
    """
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)


class VideoSwinClassifier(nn.Module):
    """Classification head wrapper around the Video Swin-B backbone.

    Args:
        backbone: Video Swin-B trunk (``SwinTransformer3D``).
        feat_dim: Channel dimension of the backbone output.
        num_classes: Number of target classes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the backbone and classification head.

        Args:
            x: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Unnormalized logits of shape ``(B, num_classes)``.
        """
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        return self.head(feats)
