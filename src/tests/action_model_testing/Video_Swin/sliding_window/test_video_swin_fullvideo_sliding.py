"""
Tests for src/sailsprep/action_model_testing/Video_Swin/sliding_window/video_swin_fullvideo_sliding.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from sailsprep.action_model_testing.Video_Swin.sliding_window import video_swin_fullvideo_sliding as mod


class TestGetWindowLabel:
    def test_majority_non_na(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        assert mod.get_window_label(ftl, 0, 3) == "Walking"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        assert mod.get_window_label(ftl, 0, 2) == "N/A"

    def test_empty_window_returns_na(self):
        assert mod.get_window_label({}, 0, 0) == "N/A"


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
        clf = mod.VideoSwinClassifier(backbone, feat_dim=1024, num_classes=6)
        x = torch.zeros(2, 3, 32, 224, 224)
        logits = clf(x)
        assert logits.shape == (2, 6)
