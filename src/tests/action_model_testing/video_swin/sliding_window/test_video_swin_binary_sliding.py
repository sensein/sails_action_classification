"""
Tests for src/sailsprep/action_model_testing/video_swin/sliding_window/video_swin_binary_sliding.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sailsprep.action_model_testing.video_swin.sliding_window import video_swin_binary_sliding as mod


def _make_mock_cap(num_frames: int = 32, h: int = 240, w: int = 320):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_mock_bbox_map(n: int = 50):
    return {i: (10, 10, 100, 100) for i in range(n)}


class TestGetWindowBinaryLabel:
    def test_majority_non_na(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        assert mod.get_window_binary_label(ftl, 0, 3) == "non-N/A"

    def test_tie_goes_to_non_na(self):
        ftl = {0: "Walking", 1: "N/A"}
        assert mod.get_window_binary_label(ftl, 0, 2) == "non-N/A"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        assert mod.get_window_binary_label(ftl, 0, 2) == "N/A"


class TestBBoxCropVideoDatasetBinary:
    def test_getitem_shape_and_label(self, tmp_path):
        sample = {
            "video_path": str(tmp_path / "fake.mp4"),
            "h5_path": "fake.h5",
            "start_frame": 0,
            "end_frame": 29,
            "label_str": "non-N/A",
            "ann_fps": 15.0,
        }
        label_map = {"N/A": 0, "non-N/A": 1}
        ds = mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            tensor, label = ds[0]

        assert tensor.shape == (3, 8, 224, 224)
        assert label == 1


class TestCollateFn:
    def test_collate_shapes(self):
        t = torch.zeros(3, 8, 64, 64)
        batch = [(t, 0), (t, 1)]
        videos, labels = mod.collate_fn(batch)
        assert videos.shape == (2, 3, 8, 64, 64)
        assert labels.tolist() == [0, 1]


class TestVideoSwinClassifierForward:
    def test_output_shape(self):
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        clf = mod.VideoSwinClassifier(backbone, feat_dim=1024, num_classes=2)
        x = torch.zeros(2, 3, 32, 224, 224)
        logits = clf(x)
        assert logits.shape == (2, 2)
