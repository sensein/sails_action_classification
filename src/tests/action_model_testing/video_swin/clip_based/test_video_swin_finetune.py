"""
Tests for src/sailsprep/action_model_testing/video_swin/clip_based/video_swin_finetune.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from sailsprep.action_model_testing.video_swin.clip_based import video_swin_finetune as mod


def _make_mock_cap(num_frames: int = 32, h: int = 240, w: int = 320):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_mock_bbox_map(n: int = 50):
    return {i: (10, 10, 100, 100) for i in range(n)}


class TestFindActionRuns:
    def test_single_run(self):
        df = pd.DataFrame({
            "Frame": [0, 1, 2],
            "Locomotion": ["Walking", "Walking", "Walking"],
        })
        runs = mod.find_action_runs(df, "Locomotion")
        assert runs == [(0, 2, "Walking")]

    def test_two_runs(self):
        df = pd.DataFrame({
            "Frame": [0, 1, 2, 3],
            "Locomotion": ["Walking", "Walking", "Running", "Running"],
        })
        runs = mod.find_action_runs(df, "Locomotion")
        assert len(runs) == 2
        assert runs[0][2] == "Walking"
        assert runs[1][2] == "Running"

    def test_na_skipped(self):
        df = pd.DataFrame({
            "Frame": [0, 1],
            "Locomotion": ["N/A", "Walking"],
        })
        runs = mod.find_action_runs(df, "Locomotion")
        assert all(r[2] != "N/A" for r in runs)

    def test_non_contiguous_split(self):
        df = pd.DataFrame({
            "Frame": [0, 1, 5, 6],
            "Locomotion": ["Walking"] * 4,
        })
        runs = mod.find_action_runs(df, "Locomotion")
        assert len(runs) == 2


class TestChunkRun:
    def test_too_short_returns_empty(self):
        assert mod.chunk_run(0, 4) == []

    def test_single_chunk_for_short_run(self):
        chunks = mod.chunk_run(0, 20)
        assert len(chunks) == 1
        assert chunks[0] == (0, 20)

    def test_long_run_chunks_evenly(self):
        chunks = mod.chunk_run(0, 89)
        assert len(chunks) == 3


class TestBBoxCropVideoDatasetClip:
    def test_getitem_returns_correct_shapes(self, tmp_path):
        sample = {
            "video_path": str(tmp_path / "fake.mp4"),
            "h5_path": "fake.h5",
            "start_frame": 0,
            "end_frame": 31,
            "label_str": "Walking",
            "ann_fps": 15.0,
        }
        label_map = {"Walking": 0}
        ds = mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        with (
            patch(f"{mod.__name__}.cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            tensor, label = ds[0]

        assert tensor.shape == (3, 8, 224, 224)
        assert label == 0

    def test_getitem_bad_video_returns_zeros(self, tmp_path):
        sample = {
            "video_path": str(tmp_path / "bad.mp4"),
            "h5_path": "fake.h5",
            "start_frame": 0,
            "end_frame": 31,
            "label_str": "Walking",
            "ann_fps": 15.0,
        }
        label_map = {"Walking": 0}
        ds = mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=bad_cap):
            tensor, label = ds[0]

        assert torch.all(tensor == 0)


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
        clf = mod.VideoSwinClassifier(backbone, feat_dim=1024, num_classes=5)
        x = torch.zeros(2, 3, 32, 224, 224)
        logits = clf(x)
        assert logits.shape == (2, 5)


class TestClassWeights:
    """Inverse-frequency weights: rarer class must get higher weight."""

    def test_rare_class_higher_weight(self):
        counts = np.array([90.0, 10.0])
        weights = counts.sum() / (2 * counts)
        assert weights[1] > weights[0]

    def test_equal_counts_equal_weights(self):
        counts = np.array([50.0, 50.0])
        weights = counts.sum() / (2 * counts)
        assert weights[0] == pytest.approx(weights[1])
