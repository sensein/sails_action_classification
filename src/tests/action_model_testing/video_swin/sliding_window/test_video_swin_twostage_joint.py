"""
Tests for src/sailsprep/action_model_testing/video_swin/sliding_window/video_swin_twostage_joint.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sailsprep.action_model_testing.video_swin.sliding_window import video_swin_twostage_joint as mod


def _make_mock_cap(num_frames: int = 32, h: int = 240, w: int = 320):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_mock_bbox_map(n: int = 50):
    return {i: (10, 10, 100, 100) for i in range(n)}


class TestGetWindowLabelsTwoStage:
    def test_majority_action_and_binary(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        action, binary = mod.get_window_labels(ftl, 0, 3)
        assert action == "Walking"
        assert binary == "non-N/A"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        action, binary = mod.get_window_labels(ftl, 0, 2)
        assert action == "N/A"
        assert binary == "N/A"

    def test_empty(self):
        action, binary = mod.get_window_labels({}, 0, 0)
        assert action == "N/A"
        assert binary == "N/A"


class TestTwoStageDataset:
    def test_na_window_action_label_is_minus_one(self, tmp_path):
        sample = {
            "video_path": str(tmp_path / "fake.mp4"),
            "h5_path": "fake.h5",
            "start_frame": 0,
            "end_frame": 29,
            "label_str": "N/A",
            "binary_label": "N/A",
            "ann_fps": 15.0,
        }
        binary_map = {"N/A": 0, "non-N/A": 1}
        action_map = {"Walking": 0, "Running": 1}
        ds = mod.TwoStageVideoDataset([sample], binary_map, action_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            tensor, bin_lbl, act_lbl = ds[0]

        assert bin_lbl == 0
        assert act_lbl == -1  # N/A windows masked from action loss

    def test_non_na_window_has_valid_action_label(self, tmp_path):
        sample = {
            "video_path": str(tmp_path / "fake.mp4"),
            "h5_path": "fake.h5",
            "start_frame": 0,
            "end_frame": 29,
            "label_str": "Walking",
            "binary_label": "non-N/A",
            "ann_fps": 15.0,
        }
        binary_map = {"N/A": 0, "non-N/A": 1}
        action_map = {"Walking": 0, "Running": 1}
        ds = mod.TwoStageVideoDataset([sample], binary_map, action_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            _, bin_lbl, act_lbl = ds[0]

        assert bin_lbl == 1
        assert act_lbl == 0


class TestCollateFn:
    def test_twostage_collate_shapes(self):
        t = torch.zeros(3, 8, 64, 64)
        batch = [(t, 0, 1), (t, 1, -1)]
        videos, bin_labels, act_labels = mod.collate_fn(batch)
        assert videos.shape == (2, 3, 8, 64, 64)
        assert bin_labels.tolist() == [0, 1]
        assert act_labels.tolist() == [1, -1]


class TestVideoSwinTwoStageForward:
    def test_output_shapes(self):
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        clf = mod.VideoSwinTwoStage(backbone, feat_dim=1024, num_action_classes=4)
        x = torch.zeros(2, 3, 32, 224, 224)
        feats, binary_logits, action_logits = clf(x)
        assert feats.shape == (2, 1024)
        assert binary_logits.shape == (2, 2)
        assert action_logits.shape == (2, 4)


class TestTwoStageLoss:
    """Action loss must be zero when all windows are N/A (action_label == -1)."""

    def _make_lightning_mod(self, num_action_classes: int):
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        inner = mod.VideoSwinTwoStage(backbone, feat_dim=1024, num_action_classes=num_action_classes)

        class _Stub(mod.VideoSwinTwoStageModule):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.model = inner
                self.binary_weights = None
                self.action_weights = None

        return _Stub()

    def test_all_na_action_loss_is_zero(self):
        lightning_mod = self._make_lightning_mod(num_action_classes=3)

        binary_logits = torch.randn(2, 2)
        action_logits = torch.randn(2, 3)
        binary_labels = torch.tensor([0, 1])
        action_labels = torch.tensor([-1, -1])  # all N/A

        _, _, act_loss = lightning_mod._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )
        assert act_loss.item() == pytest.approx(0.0)

    def test_mixed_batch_action_loss_nonzero(self):
        lightning_mod = self._make_lightning_mod(num_action_classes=3)

        binary_logits = torch.randn(2, 2)
        action_logits = torch.randn(2, 3)
        binary_labels = torch.tensor([0, 1])
        action_labels = torch.tensor([-1, 2])  # one real label

        _, _, act_loss = lightning_mod._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )
        assert act_loss.item() > 0.0
