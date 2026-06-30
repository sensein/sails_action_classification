"""Tests for Video Swin action-recognition scripts.

Covers:
  - video_swin_finetune          (clip-based)
  - video_swin_binary_sliding    (binary N/A vs non-N/A)
  - video_swin_twostage_joint    (joint two-stage)
  - video_swin_fullvideo_sliding (sliding window, N/A as class)

Run with:
    poetry run pytest src/tests/test_Video_Swin.py -v
"""

from __future__ import annotations

import types
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch



_MODULE_PATHS = {
    "video_swin_finetune":          "sailsprep.action_model_testing.Video_Swin.clip_based.video_swin_finetune",
    "video_swin_binary_sliding":    "sailsprep.action_model_testing.Video_Swin.sliding_window.video_swin_binary_sliding",
    "video_swin_twostage_joint":    "sailsprep.action_model_testing.Video_Swin.sliding_window.video_swin_twostage_joint",
    "video_swin_fullvideo_sliding": "sailsprep.action_model_testing.Video_Swin.sliding_window.video_swin_fullvideo_sliding",
}

def _import(module_name: str):
    import importlib
    full_path = _MODULE_PATHS[module_name]
    return importlib.import_module(full_path)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_label_map() -> dict[str, int]:
    return {"N/A": 0, "Walking": 1, "Running": 2}


@pytest.fixture()
def tiny_binary_map() -> dict[str, int]:
    return {"N/A": 0, "non-N/A": 1}


@pytest.fixture()
def dummy_video_tensor() -> torch.Tensor:
    """Minimal (C, T, H, W) tensor — no real video needed."""
    return torch.zeros(3, 32, 224, 224)


@pytest.fixture()
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    videos = torch.zeros(2, 3, 32, 224, 224)
    labels = torch.tensor([0, 1])
    return videos, labels


# ---------------------------------------------------------------------------
# 1. Utility / data-building helpers
# ---------------------------------------------------------------------------

class TestFindActionRuns:
    """clip-based: find_action_runs"""

    def setup_method(self):
        self.mod = _import("video_swin_finetune")

    def test_single_run(self):
        df = pd.DataFrame({
            "Frame": [0, 1, 2],
            "Locomotion": ["Walking", "Walking", "Walking"],
        })
        runs = self.mod.find_action_runs(df, "Locomotion")
        assert runs == [(0, 2, "Walking")]

    def test_two_runs(self):
        df = pd.DataFrame({
            "Frame": [0, 1, 2, 3],
            "Locomotion": ["Walking", "Walking", "Running", "Running"],
        })
        runs = self.mod.find_action_runs(df, "Locomotion")
        assert len(runs) == 2
        assert runs[0][2] == "Walking"
        assert runs[1][2] == "Running"

    def test_na_skipped(self):
        df = pd.DataFrame({
            "Frame": [0, 1],
            "Locomotion": ["N/A", "Walking"],
        })
        runs = self.mod.find_action_runs(df, "Locomotion")
        assert all(r[2] != "N/A" for r in runs)

    def test_non_contiguous_split(self):
        """A gap in frame numbers must split a run."""
        df = pd.DataFrame({
            "Frame": [0, 1, 5, 6],
            "Locomotion": ["Walking"] * 4,
        })
        runs = self.mod.find_action_runs(df, "Locomotion")
        assert len(runs) == 2


class TestChunkRun:
    """clip-based: chunk_run"""

    def setup_method(self):
        self.mod = _import("video_swin_finetune")

    def test_too_short_returns_empty(self):
        # MIN_FRAMES = 15; a run of 5 frames should produce nothing.
        assert self.mod.chunk_run(0, 4) == []

    def test_single_chunk_for_short_run(self):
        chunks = self.mod.chunk_run(0, 20)
        assert len(chunks) == 1
        assert chunks[0] == (0, 20)

    def test_long_run_chunks_evenly(self):
        # CLIP_FRAMES = 30; a run of 90 frames → 3 chunks.
        chunks = self.mod.chunk_run(0, 89)
        assert len(chunks) == 3


class TestGetWindowLabel:
    """sliding-window: majority-vote label"""

    def setup_method(self):
        self.mod = _import("video_swin_fullvideo_sliding")

    def test_majority_non_na(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        assert self.mod.get_window_label(ftl, 0, 3) == "Walking"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        assert self.mod.get_window_label(ftl, 0, 2) == "N/A"

    def test_empty_window_returns_na(self):
        assert self.mod.get_window_label({}, 0, 0) == "N/A"


class TestGetWindowBinaryLabel:
    """binary sliding: N/A vs non-N/A majority vote"""

    def setup_method(self):
        self.mod = _import("video_swin_binary_sliding")

    def test_majority_non_na(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        assert self.mod.get_window_binary_label(ftl, 0, 3) == "non-N/A"

    def test_tie_goes_to_non_na(self):
        ftl = {0: "Walking", 1: "N/A"}
        # non_na_count == na_count → non-N/A wins (>=)
        assert self.mod.get_window_binary_label(ftl, 0, 2) == "non-N/A"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        assert self.mod.get_window_binary_label(ftl, 0, 2) == "N/A"


class TestGetWindowLabelsTwoStage:
    """two-stage: returns (action_label, binary_label) tuple"""

    def setup_method(self):
        self.mod = _import("video_swin_twostage_joint")

    def test_majority_action_and_binary(self):
        ftl = {0: "Walking", 1: "Walking", 2: "N/A"}
        action, binary = self.mod.get_window_labels(ftl, 0, 3)
        assert action == "Walking"
        assert binary == "non-N/A"

    def test_all_na(self):
        ftl = {0: "N/A", 1: "N/A"}
        action, binary = self.mod.get_window_labels(ftl, 0, 2)
        assert action == "N/A"
        assert binary == "N/A"

    def test_empty(self):
        action, binary = self.mod.get_window_labels({}, 0, 0)
        assert action == "N/A"
        assert binary == "N/A"


# ---------------------------------------------------------------------------
# 2. Dataset __getitem__ — mock video I/O
# ---------------------------------------------------------------------------

def _make_mock_cap(num_frames: int = 32, h: int = 240, w: int = 320):
    """Return a MagicMock that behaves like cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.0
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_mock_bbox_map(n: int = 50) -> dict[int, tuple[int, int, int, int]]:
    return {i: (10, 10, 100, 100) for i in range(n)}


class TestBBoxCropVideoDatasetClip:
    """clip-based dataset smoke test"""

    def setup_method(self):
        self.mod = _import("video_swin_finetune")

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
        ds = self.mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        with (
            patch(f"{self.mod.__name__}.cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(self.mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
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
        ds = self.mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=bad_cap):
            tensor, label = ds[0]

        assert torch.all(tensor == 0)


class TestBBoxCropVideoDatasetBinary:
    """binary sliding-window dataset smoke test"""

    def setup_method(self):
        self.mod = _import("video_swin_binary_sliding")

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
        ds = self.mod.BBoxCropVideoDataset([sample], label_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(self.mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            tensor, label = ds[0]

        assert tensor.shape == (3, 8, 224, 224)
        assert label == 1


class TestTwoStageDataset:
    """two-stage dataset returns (tensor, binary_label, action_label)"""

    def setup_method(self):
        self.mod = _import("video_swin_twostage_joint")

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
        ds = self.mod.TwoStageVideoDataset([sample], binary_map, action_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(self.mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
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
        ds = self.mod.TwoStageVideoDataset([sample], binary_map, action_map, num_frames=8)

        with (
            patch("cv2.VideoCapture", return_value=_make_mock_cap()),
            patch.object(self.mod, "load_bbox_map", return_value=_make_mock_bbox_map()),
        ):
            _, bin_lbl, act_lbl = ds[0]

        assert bin_lbl == 1
        assert act_lbl == 0


# ---------------------------------------------------------------------------
# 3. collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:

    @pytest.mark.parametrize("module_name", [
        "video_swin_finetune",
        "video_swin_binary_sliding",
        "video_swin_fullvideo_sliding",
    ])
    def test_collate_shapes(self, module_name):
        mod = _import(module_name)
        t = torch.zeros(3, 8, 64, 64)
        batch = [(t, 0), (t, 1)]
        videos, labels = mod.collate_fn(batch)
        assert videos.shape == (2, 3, 8, 64, 64)
        assert labels.tolist() == [0, 1]

    def test_twostage_collate_shapes(self):
        mod = _import("video_swin_twostage_joint")
        t = torch.zeros(3, 8, 64, 64)
        batch = [(t, 0, 1), (t, 1, -1)]
        videos, bin_labels, act_labels = mod.collate_fn(batch)
        assert videos.shape == (2, 3, 8, 64, 64)
        assert bin_labels.tolist() == [0, 1]
        assert act_labels.tolist() == [1, -1]


# ---------------------------------------------------------------------------
# 4. Model forward pass (CPU, no real backbone)
# ---------------------------------------------------------------------------

class TestVideoSwinClassifierForward:
    """Smoke-test the classifier head with a stub backbone."""

    @pytest.mark.parametrize("module_name,num_classes", [
        ("video_swin_finetune", 5),
        ("video_swin_fullvideo_sliding", 6),
        ("video_swin_binary_sliding", 2),
    ])
    def test_output_shape(self, module_name, num_classes):
        mod = _import(module_name)
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        clf = mod.VideoSwinClassifier(backbone, feat_dim=1024, num_classes=num_classes)
        x = torch.zeros(2, 3, 32, 224, 224)
        logits = clf(x)
        assert logits.shape == (2, num_classes)


class TestVideoSwinTwoStageForward:
    """Two-headed model returns three tensors."""

    def test_output_shapes(self):
        mod = _import("video_swin_twostage_joint")
        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        clf = mod.VideoSwinTwoStage(backbone, feat_dim=1024, num_action_classes=4)
        x = torch.zeros(2, 3, 32, 224, 224)
        feats, binary_logits, action_logits = clf(x)
        assert feats.shape == (2, 1024)
        assert binary_logits.shape == (2, 2)
        assert action_logits.shape == (2, 4)


# ---------------------------------------------------------------------------
# 5. Loss masking in two-stage Lightning module
# ---------------------------------------------------------------------------

class TestTwoStageLoss:
    """Action loss must be zero when all windows are N/A (action_label == -1)."""

    def _make_lightning_mod(self, mod, num_action_classes: int):
        """Create a minimal VideoSwinTwoStageModule without loading weights."""

        backbone = MagicMock(return_value=torch.zeros(2, 1024, 2, 7, 7))
        inner = mod.VideoSwinTwoStage(backbone, feat_dim=1024, num_action_classes=num_action_classes)

        # Subclass to bypass build_video_swin_twostage in __init__
        class _Stub(mod.VideoSwinTwoStageModule):
            def __init__(self):
                # Call nn.Module.__init__ only, skip LightningModule setup
                torch.nn.Module.__init__(self)
                self.model = inner
                self.binary_weights = None
                self.action_weights = None

        return _Stub()

    def test_all_na_action_loss_is_zero(self):
        mod = _import("video_swin_twostage_joint")
        lightning_mod = self._make_lightning_mod(mod, num_action_classes=3)

        binary_logits = torch.randn(2, 2)
        action_logits = torch.randn(2, 3)
        binary_labels = torch.tensor([0, 1])
        action_labels = torch.tensor([-1, -1])  # all N/A

        _, _, act_loss = lightning_mod._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )
        assert act_loss.item() == pytest.approx(0.0)

    def test_mixed_batch_action_loss_nonzero(self):
        mod = _import("video_swin_twostage_joint")
        lightning_mod = self._make_lightning_mod(mod, num_action_classes=3)

        binary_logits = torch.randn(2, 2)
        action_logits = torch.randn(2, 3)
        binary_labels = torch.tensor([0, 1])
        action_labels = torch.tensor([-1, 2])  # one real label

        _, _, act_loss = lightning_mod._compute_loss(
            binary_logits, action_logits, binary_labels, action_labels
        )
        assert act_loss.item() > 0.0


# ---------------------------------------------------------------------------
# 6. Class-weight computation (DataModule.setup logic)
# ---------------------------------------------------------------------------

class TestClassWeights:
    """Inverse-frequency weights: rarer class must get higher weight."""

    def test_rare_class_higher_weight(self):
        # 90 samples of class 0, 10 of class 1
        counts = np.array([90.0, 10.0])
        weights = counts.sum() / (2 * counts)
        assert weights[1] > weights[0]

    def test_equal_counts_equal_weights(self):
        counts = np.array([50.0, 50.0])
        weights = counts.sum() / (2 * counts)
        assert weights[0] == pytest.approx(weights[1])