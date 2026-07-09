"""
Tests for sailsprep/action_model_testing/vjepa/clips_fixed_length/vjepa_clip_level_ablation.py
No disk I/O beyond tmp_path, no GPU, no HF downloads required.

Run: poetry run pytest
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import torch

from sailsprep.action_model_testing.vjepa.clips_fixed_length.vjepa_clip_level_ablation import (
    EMBED_DIM,
    AttentiveProbe,
    ClipDataset,
    FeatureDataset,
    LinearProbe,
    MLPLargeProbe,
    MLPSmallProbe,
    TransformerProbe,
    build_probe,
    build_samples,
    chunk_run,
    find_action_runs,
    load_bbox_map,
    run_inference,
    train_probe,
)

DEVICE = torch.device("cpu")


# ============================================================
# Helpers
# ============================================================

def _make_h5_bbox(path: str, frames: range) -> None:
    """Write a minimal bboxes/table dataset shaped like the real H5 files.

    load_bbox_map reads frame id from values_block_1[:, 0] and the bbox from
    columns 2..5, so values_block_1 needs at least 6 columns per row.
    """
    dtype = np.dtype([
        ("index", "i8"),
        ("values_block_1", "i8", (6,)),
    ])
    table = np.zeros(len(frames), dtype=dtype)
    for i, f in enumerate(frames):
        table[i] = (f, (f, 0, 10, 20, 110, 220))
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("bboxes/table", data=table)


# ============================================================
# 1. chunk_run
# ============================================================

class TestChunkRun:
    def test_below_min_frames_returns_empty(self):
        assert chunk_run(0, 13) == []

    def test_single_clip_range(self):
        assert chunk_run(10, 40) == [(10, 40)]

    def test_split_into_two_clips_45_frames(self):
        result = chunk_run(0, 44)
        assert len(result) == 2
        assert result[0] == (0, 29)
        assert result[1] == (30, 44)

    def test_60_frames_two_full_chunks(self):
        result = chunk_run(0, 59)
        assert result == [(0, 29), (30, 59)]

    def test_last_chunk_dropped_if_too_short(self):
        result = chunk_run(0, 73)  # 74 frames: 60 + 14 (14 < MIN_FRAMES)
        assert result == [(0, 29), (30, 59)]

    def test_last_chunk_kept_if_ge_min_frames(self):
        result = chunk_run(0, 74)  # 75 frames: 60 + 15
        assert result == [(0, 29), (30, 59), (60, 74)]


# ============================================================
# 2. find_action_runs
# ============================================================

class TestFindActionRuns:
    def _df(self, frames, labels, col="Locomotion"):
        return pd.DataFrame({"Frame": frames, col: labels})

    def test_single_run(self):
        df = self._df([0, 1, 2], ["walk", "walk", "walk"])
        assert find_action_runs(df, "Locomotion") == [(0, 2, "walk")]

    def test_na_breaks_run(self):
        df = self._df([0, 1, 2, 3], ["walk", "N/A", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == [(0, 0, "walk"), (2, 3, "walk")]

    def test_empty_and_nan_string_break_run(self):
        df = self._df([0, 1, 2], ["walk", "", "walk"])
        assert len(find_action_runs(df, "Locomotion")) == 2

    def test_gap_in_frames_breaks_run(self):
        df = self._df([0, 1, 3, 4], ["walk", "walk", "walk", "walk"])
        runs = find_action_runs(df, "Locomotion")
        assert runs == [(0, 1, "walk"), (3, 4, "walk")]

    def test_unsorted_input_is_sorted(self):
        df = self._df([2, 0, 1], ["walk", "walk", "walk"])
        assert find_action_runs(df, "Locomotion") == [(0, 2, "walk")]


# ============================================================
# 3. load_bbox_map
# ============================================================

class TestLoadBboxMap:
    def test_maps_frame_to_bbox(self, tmp_path):
        h5_path = str(tmp_path / "bbox.h5")
        _make_h5_bbox(h5_path, range(3))
        bbox_map = load_bbox_map(h5_path)
        assert set(bbox_map.keys()) == {0, 1, 2}
        assert bbox_map[0] == (10, 20, 110, 220)


# ============================================================
# 4. build_samples
# ============================================================

class TestBuildSamples:
    def test_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "split.csv"
        pd.DataFrame({"video_path": ["a.mp4"]}).to_csv(csv_path, index=False)
        try:
            build_samples(str(csv_path), "Locomotion")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_skips_rows_with_missing_files(self, tmp_path):
        csv_path = tmp_path / "split.csv"
        pd.DataFrame({
            "video_path": ["/nonexistent/video.mp4"],
            "label_path": ["/nonexistent/labels.csv"],
            "interpolated_anno_h5": ["/nonexistent/anno.h5"],
            "split": ["train"],
        }).to_csv(csv_path, index=False)
        train_s, val_s, test_s = build_samples(str(csv_path), "Locomotion")
        assert train_s == [] and val_s == [] and test_s == []

    def test_builds_clips_for_valid_row(self, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        h5_path = tmp_path / "anno.h5"
        _make_h5_bbox(str(h5_path), range(20))
        label_path = tmp_path / "labels.csv"
        pd.DataFrame({
            "Frame": list(range(20)),
            "Locomotion": ["walk"] * 20,
        }).to_csv(label_path, index=False)

        csv_path = tmp_path / "split.csv"
        pd.DataFrame({
            "video_path": [str(video_path)],
            "label_path": [str(label_path)],
            "interpolated_anno_h5": [str(h5_path)],
            "split": ["train"],
        }).to_csv(csv_path, index=False)

        train_s, val_s, test_s = build_samples(str(csv_path), "Locomotion")
        assert len(train_s) == 1
        assert val_s == [] and test_s == []
        assert train_s[0]["label_str"] == "walk"
        assert train_s[0]["start_frame"] == 0
        assert train_s[0]["end_frame"] == 19


# ============================================================
# 5. ClipDataset
# ============================================================

class TestClipDataset:
    def test_load_error_returns_zero_clip(self):
        samples = [{
            "video_path": "/nonexistent/video.mp4",
            "h5_path": "/nonexistent/anno.h5",
            "start_frame": 0,
            "end_frame": 10,
            "label_str": "walk",
        }]
        ds = ClipDataset(samples, {"walk": 0}, vjepa_frames=4, crop_size=16)
        clip, label = ds[0]
        assert clip.shape == (3, 4, 16, 16)
        assert label == 0

    def test_len(self):
        samples = [{"video_path": "a", "h5_path": "b", "start_frame": 0,
                    "end_frame": 1, "label_str": "walk"}] * 5
        ds = ClipDataset(samples, {"walk": 0})
        assert len(ds) == 5


# ============================================================
# 6. Classification heads
# ============================================================

class TestClassificationHeads:
    def test_linear_probe_shape(self):
        probe = LinearProbe(embed_dim=32, num_classes=3)
        x = torch.randn(2, 5, 32)
        assert probe(x).shape == (2, 3)

    def test_mlp_small_probe_shape(self):
        probe = MLPSmallProbe(embed_dim=32, num_classes=3, hidden=16)
        x = torch.randn(2, 5, 32)
        assert probe(x).shape == (2, 3)

    def test_mlp_large_probe_shape(self):
        probe = MLPLargeProbe(embed_dim=32, num_classes=3)
        x = torch.randn(2, 5, 32)
        assert probe(x).shape == (2, 3)

    def test_attentive_probe_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=3, num_heads=4)
        x = torch.randn(2, 5, 32)
        assert probe(x).shape == (2, 3)

    def test_transformer_probe_shape(self):
        probe = TransformerProbe(embed_dim=32, num_classes=3, num_heads=4)
        x = torch.randn(2, 5, 32)
        assert probe(x).shape == (2, 3)

    def test_build_probe_unknown_head_raises(self):
        try:
            build_probe("bogus", 32, 3)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_build_probe_returns_correct_type(self):
        probe = build_probe("linear", 32, 3)
        assert isinstance(probe, LinearProbe)


# ============================================================
# 7. FeatureDataset
# ============================================================

class TestFeatureDataset:
    def test_len_and_getitem(self):
        feats = torch.randn(4, 5, 32)
        labels = torch.tensor([0, 1, 0, 1])
        ds = FeatureDataset(feats, labels)
        assert len(ds) == 4
        f, l = ds[0]
        assert f.shape == (5, 32)
        assert l.item() == 0


# ============================================================
# 8. train_probe / run_inference (tiny end-to-end, monkeypatched epochs)
# ============================================================

class TestTrainAndInfer:
    def _make_loaders(self, n=8, tokens=5, dim=EMBED_DIM, num_classes=2):
        feats = torch.randn(n, tokens, dim)
        labels = torch.randint(0, num_classes, (n,))
        ds = FeatureDataset(feats, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        return loader

    def test_train_probe_runs_one_epoch(self):
        probe = LinearProbe(embed_dim=EMBED_DIM, num_classes=2)
        tr_dl = self._make_loaders()
        va_dl = self._make_loaders()

        trained, best_acc = train_probe(probe, tr_dl, va_dl, DEVICE, max_epochs=1, tag="test")
        assert isinstance(trained, LinearProbe)
        assert 0.0 <= best_acc <= 1.0

    def test_run_inference_writes_predictions(self, tmp_path):
        probe = LinearProbe(embed_dim=EMBED_DIM, num_classes=2)
        n = 6
        te_feats = torch.randn(n, 5, EMBED_DIM)
        label_map = {"walk": 0, "run": 1}
        test_samples = [
            {"video_path": f"v{i}.mp4", "start_frame": 0, "end_frame": 10,
             "label_str": "walk" if i % 2 == 0 else "run"}
            for i in range(n)
        ]
        acc = run_inference(probe, te_feats, test_samples, label_map, DEVICE,
                            str(tmp_path), "linear")
        assert 0.0 <= acc <= 1.0
        assert (tmp_path / "test_predictions_linear.csv").exists()
