"""Shared H5 bbox loading and frame cropping (identical across extractor scripts)."""

from __future__ import annotations

import cv2
import h5py
import numpy as np


# ============================================================
# H5 bbox loading (same convention as SlowFast pipeline)
# ============================================================
def load_bbox_map(h5_path: str) -> dict:
    """Return {ann_frame_idx: (x1,y1,x2,y2)} from interpolated H5."""
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


def crop_frame_with_bbox(frame: np.ndarray, bbox, out_size: int = 224) -> np.ndarray:
    """Crop frame to bbox and resize to (out_size, out_size). frame is HWC uint8 RGB."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, W - 1)); x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1)); y2 = max(y1 + 1, min(y2, H))
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size))
    return crop
