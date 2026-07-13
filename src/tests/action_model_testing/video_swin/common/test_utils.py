"""
Tests for src/sailsprep/action_model_testing/video_swin/common/utils.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from sailsprep.action_model_testing.video_swin.common.utils import (
    load_bbox_map,
    collate_fn,
    VideoSwinClassifier,
)


class TestLoadBboxMap:
    def test_parses_frame_to_bbox(self, tmp_path):
        import h5py

        h5_path = tmp_path / "bboxes.h5"
        dtype = np.dtype([("values_block_1", np.int64, (6,))])
        rows = np.zeros(2, dtype=dtype)
        rows["values_block_1"][0] = [1, 0, 10, 20, 30, 40]
        rows["values_block_1"][1] = [2, 0, 50, 60, 70, 80]

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("bboxes/table", data=rows)

        bbox_map = load_bbox_map(str(h5_path))
        assert bbox_map[1] == (10, 20, 30, 40)
        assert bbox_map[2] == (50, 60, 70, 80)


class TestCollateFn:
    def test_stacks_videos_and_labels(self):
        t = torch.zeros(3, 8, 64, 64)
        batch = [(t, 0), (t, 1)]
        videos, labels = collate_fn(batch)
        assert videos.shape == (2, 3, 8, 64, 64)
        assert labels.tolist() == [0, 1]
        assert labels.dtype == torch.long


class TestVideoSwinClassifier:
    def test_output_shape(self):
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        clf = VideoSwinClassifier(backbone, feat_dim=1024, num_classes=5)
        x = torch.zeros(2, 3, 32, 224, 224)
        logits = clf(x)
        assert logits.shape == (2, 5)
