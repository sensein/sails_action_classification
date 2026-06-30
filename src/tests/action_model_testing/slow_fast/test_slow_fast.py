"""
Tests for src/sailsprep/action_model_testing/slow_fast/slow_fast.py

Run with:
    poetry run pytest
"""

import pytest
import pandas as pd
import numpy as np
import torch

from sailsprep.action_model_testing.slow_fast.slow_fast import (
    chunk_run,
    find_action_runs,
    BBoxCropVideoDataset,
    slowfast_collate,
    SlowFastFineTune,
)
import torch.nn as nn  # noqa: E402
# ============================================================
# chunk_run
# ============================================================

class TestChunkRun:
    def test_below_min_frames_returns_empty(self):
        # 14 frames -> below MIN_FRAMES (15), skip
        assert chunk_run(0, 13) == []

    def test_exactly_min_frames_returns_single_clip(self):
        # exactly 15 frames -> one clip
        assert chunk_run(0, 14) == [(0, 14)]

    def test_single_clip_range(self):
        # 15-44 frames -> 1 clip, unchanged bounds
        assert chunk_run(10, 40) == [(10, 40)]

    def test_split_into_two_clips_45_frames(self):
        # 45 frames -> 2 clips: [0,29] and [30,44]
        result = chunk_run(0, 44)
        assert len(result) == 2
        assert result[0] == (0, 29)
        assert result[1] == (30, 44)

    def test_split_into_two_clips_59_frames(self):
        # 59 frames -> 2 clips
        result = chunk_run(0, 58)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][1] == 58

    def test_exactly_60_frames_two_clips(self):
        # 60 frames -> 2 full 30-frame chunks
        result = chunk_run(0, 59)
        assert len(result) == 2
        assert result[0] == (0, 29)
        assert result[1] == (30, 59)

    def test_90_frames_three_clips(self):
        result = chunk_run(0, 89)
        assert len(result) == 3
        assert result[0] == (0, 29)
        assert result[1] == (30, 59)
        assert result[2] == (60, 89)

    def test_last_chunk_dropped_if_too_short(self):
        # 60 + 14 = 74 frames -> last chunk is 14 frames, below MIN_FRAMES
        result = chunk_run(0, 73)
        assert len(result) == 2
        assert result[-1] == (30, 59)

    def test_last_chunk_kept_if_ge_min_frames(self):
        # 60 + 15 = 75 frames -> last chunk is 15 frames, kept
        result = chunk_run(0, 74)
        assert len(result) == 3
        assert result[-1] == (60, 74)

    def test_nonzero_start_preserved(self):
        # start offset should propagate through all clips
        result = chunk_run(100, 189)  # 90 frames
        assert result[0][0] == 100
        assert result[-1][1] == 189

    def test_single_frame_returns_empty(self):
        assert chunk_run(5, 5) == []


# ============================================================
# find_action_runs
# ============================================================

class TestFindActionRuns:
    def _make_df(self, frames, labels, col="Locomotion"):
        return pd.DataFrame({"Frame": frames, col: labels})

    def test_single_consecutive_run(self):
        df = self._make_df([0, 1, 2], ["walk", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == [(0, 2, "walk")]

    def test_two_distinct_labels(self):
        df = self._make_df([0, 1, 2, 3], ["walk", "walk", "run", "run"])
        runs = find_action_runs(df, "Locomotion")
        assert len(runs) == 2
        assert runs[0] == (0, 1, "walk")
        assert runs[1] == (2, 3, "run")

    def test_na_label_breaks_run(self):
        df = self._make_df([0, 1, 2, 3, 4], ["walk", "N/A", "walk", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        # frame 0 alone, then frames 2-4
        assert len(runs) == 2
        assert runs[0] == (0, 0, "walk")
        assert runs[1] == (2, 4, "walk")

    def test_empty_string_label_breaks_run(self):
        df = self._make_df([0, 1, 2], ["walk", "", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert len(runs) == 2

    def test_gap_in_frames_breaks_run(self):
        # frames 0,1 then jump to 3 — not consecutive
        df = self._make_df([0, 1, 3, 4], ["walk", "walk", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert len(runs) == 2
        assert runs[0] == (0, 1, "walk")
        assert runs[1] == (3, 4, "walk")

    def test_unsorted_input_is_sorted(self):
        df = self._make_df([2, 0, 1], ["walk", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == [(0, 2, "walk")]

    def test_all_na_returns_empty(self):
        df = self._make_df([0, 1, 2], ["N/A", "N/A", "N/A"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == []

    def test_single_frame_run(self):
        df = self._make_df([5], ["jump"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == [(5, 5, "jump")]


# ============================================================
# slowfast_collate
# ============================================================

class TestSlowfastCollate:
    def _make_batch(self, n=2, slow_t=8, fast_t=32, h=224, w=224):
        """Build a fake batch as the dataset would produce."""
        batch = []
        for i in range(n):
            slow = torch.zeros(3, slow_t, h, w)
            fast = torch.zeros(3, fast_t, h, w)
            batch.append(([slow, fast], i))
        return batch

    def test_output_shapes(self):
        batch = self._make_batch(n=4)
        inputs, labels = slowfast_collate(batch)
        slow, fast = inputs
        assert slow.shape == (4, 3, 8, 224, 224)
        assert fast.shape == (4, 3, 32, 224, 224)
        assert labels.shape == (4,)
        assert labels.dtype == torch.long

    def test_label_values_preserved(self):
        batch = self._make_batch(n=3)
        _, labels = slowfast_collate(batch)
        assert labels.tolist() == [0, 1, 2]


# ============================================================
# BBoxCropVideoDataset._pack_pathway
# ============================================================

class TestPackPathway:
    def test_slow_has_fewer_frames(self):
        label_map = {"walk": 0}
        ds = BBoxCropVideoDataset(samples=[], label_map=label_map,
                                  num_frames=32, alpha=4)
        # 3 channels, 32 frames, 224x224
        frames = torch.rand(3, 32, 224, 224)
        slow, fast = ds._pack_pathway(frames)
        assert fast.shape == frames.shape
        assert slow.shape[1] == 32 // 4   # 8 slow frames

    def test_slow_is_subset_of_fast(self):
        label_map = {"walk": 0}
        ds = BBoxCropVideoDataset(samples=[], label_map=label_map,
                                  num_frames=8, alpha=2)
        frames = torch.arange(24, dtype=torch.float).reshape(3, 8, 1, 1)
        slow, fast = ds._pack_pathway(frames)
        assert slow.shape[1] == 4
        # slow frames should be evenly sampled from fast
        assert fast.shape[1] == 8


# ============================================================
# SlowFastFineTune — head replacement & freeze logic (CPU, no pretrained weights)
# ============================================================

class TestSlowFastFineTune:
    @pytest.fixture
    def model_2class(self, mocker):
        # Patch torch.hub.load so no download happens in CI
        mocker.patch(
            "sailsprep.action_model_testing.slow_fast.slow_fast.torch.hub.load",
            return_value=_make_fake_slowfast(num_classes_original=400),
        )
        return SlowFastFineTune(num_classes=2, freeze_backbone=False)

    def test_head_replaced_correctly(self, model_2class):
        out_features = model_2class.model.blocks[-1].proj.out_features
        assert out_features == 2

    def test_frozen_backbone_only_head_trains(self, mocker):
        mocker.patch(
            "sailsprep.action_model_testing.slow_fast.slow_fast.torch.hub.load",
            return_value=_make_fake_slowfast(num_classes_original=400),
        )
        m = SlowFastFineTune(num_classes=3, freeze_backbone=True)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        # Only head params (blocks.6) should be trainable
        assert all("blocks.6" in n for n in trainable), (
            f"Non-head params are trainable: "
            f"{[n for n in trainable if 'blocks.6' not in n]}"
        )

    def test_class_weights_registered_as_buffer(self, mocker):
        mocker.patch(
            "sailsprep.action_model_testing.slow_fast.slow_fast.torch.hub.load",
            return_value=_make_fake_slowfast(num_classes_original=400),
        )
        weights = torch.tensor([1.0, 2.0, 3.0])
        m = SlowFastFineTune(num_classes=3, freeze_backbone=False,
                             class_weights=weights)
        assert hasattr(m, "class_weights")
        assert torch.allclose(m.class_weights, weights)


# ============================================================
# Helpers
# ============================================================


class _FakeHead(nn.Module):
    """Minimal stand-in for the SlowFast classification head."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.proj = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.proj(x.mean([2, 3, 4]))  # pool spatial+temporal


class _FakeSlowFast(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # SlowFastFineTune accesses self.model.blocks[-1].proj
        self.blocks = nn.ModuleList([
            nn.Identity(),   # blocks[0..5] placeholders
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            _FakeHead(256, num_classes),   # blocks[6] = head (index -1)
        ])

    def forward(self, x):
        return self.blocks[-1].proj(torch.zeros(x[0].shape[0], 256))


def _make_fake_slowfast(num_classes_original=400):
    return _FakeSlowFast(num_classes_original)