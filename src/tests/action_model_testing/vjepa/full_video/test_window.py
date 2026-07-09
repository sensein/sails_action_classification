"""
Tests for sailsprep/action_model_testing/vjepa/full_video/window.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.full_video.window import (
    EMBED_DIM,
    STRIDE_FRAMES,
    WINDOW_FRAMES,
    AttentiveProbe,
    WindowDataset,
    build_windows_from_video,
    encode_labels,
    run_inference,
    train_probe,
)

DEVICE = torch.device("cpu")


def _make_feat_npy(tmp_dir: str, T: int = 90, embed: int = EMBED_DIM) -> str:
    path = os.path.join(tmp_dir, "feat.npy")
    np.save(path, np.random.randn(embed, T).astype(np.float32))
    return path


def _make_label_csv(tmp_dir: str, T: int = 90, column: str = "Locomotion") -> str:
    path = os.path.join(tmp_dir, "labels.csv")
    labels = (["walk"] * (T // 2)) + (["run"] * (T - T // 2))
    pd.DataFrame({column: labels}).to_csv(path, index=False)
    return path


class TestBuildWindowsFromVideo:
    def test_bad_feat_path_returns_empty(self, tmp_path):
        label_path = _make_label_csv(str(tmp_path))
        assert build_windows_from_video("/nonexistent/feat.npy", label_path, "Locomotion") == []

    def test_bad_label_path_returns_empty(self, tmp_path):
        feat_path = _make_feat_npy(str(tmp_path))
        assert build_windows_from_video(feat_path, "/nonexistent/labels.csv", "Locomotion") == []

    def test_missing_column_returns_empty(self, tmp_path):
        feat_path = _make_feat_npy(str(tmp_path), T=60)
        label_path = _make_label_csv(str(tmp_path), T=60, column="Other")
        assert build_windows_from_video(feat_path, label_path, "Locomotion") == []

    def test_correct_number_of_windows(self, tmp_path):
        T = 90
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        expected = len(range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES))
        assert len(windows) == expected

    def test_window_shape_and_majority_label(self, tmp_path):
        T = 90
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        feat, label = windows[0]
        assert feat.shape == (WINDOW_FRAMES, EMBED_DIM)
        assert label == "walk"  # first window is entirely within the "walk" half

    def test_short_annotation_padded_with_none(self, tmp_path):
        T = 60
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = os.path.join(str(tmp_path), "labels.csv")
        pd.DataFrame({"Locomotion": ["walk"] * 10}).to_csv(label_path, index=False)
        windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        # frames beyond the annotated 10 should be padded with "None"
        assert windows[-1][1] == "None"


class TestEncodeLabels:
    def test_assigns_sorted_int_codes(self):
        windows = [
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "run"),
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "walk"),
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "None"),
        ]
        encoded, label_map = encode_labels(windows)
        assert label_map == {"None": 0, "run": 1, "walk": 2}
        assert [lbl for _, lbl in encoded] == [1, 2, 0]


class TestWindowDataset:
    def _raw(self, n=6):
        return [(torch.randn(WINDOW_FRAMES, EMBED_DIM), i % 2) for i in range(n)]

    def test_len_and_getitem(self):
        ds = WindowDataset(self._raw(6))
        assert len(ds) == 6
        feat, label = ds[0]
        assert feat.shape == (WINDOW_FRAMES, EMBED_DIM)
        assert label.dtype == torch.long


class TestAttentiveProbe:
    def test_forward_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=3, num_heads=4)
        x = torch.randn(4, WINDOW_FRAMES, 32)
        assert probe(x).shape == (4, 3)


class TestTrainProbe:
    def _loader(self, n=8, num_classes=2, batch_size=4):
        feats = torch.randn(n, WINDOW_FRAMES, EMBED_DIM)
        labels = torch.randint(0, num_classes, (n,))
        ds = WindowDataset(list(zip(feats, labels.tolist())))
        return DataLoader(ds, batch_size=batch_size)

    def test_runs_one_epoch(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.full_video.window as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        trained = train_probe(probe, self._loader(), self._loader(), DEVICE, str(tmp_path))
        assert isinstance(trained, AttentiveProbe)
        assert (tmp_path / "training_log.csv").exists()


class TestRunInference:
    def test_writes_predictions_and_metrics(self, tmp_path):
        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        label_map = {"walk": 0, "run": 1}
        test_windows_raw = [
            (torch.randn(WINDOW_FRAMES, EMBED_DIM), "walk" if i % 2 == 0 else "run")
            for i in range(6)
        ]
        results_df = run_inference(probe, test_windows_raw, label_map, DEVICE, str(tmp_path))
        assert len(results_df) == 6
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "test_metrics.txt").exists()

    def test_unknown_label_falls_back_to_none(self, tmp_path):
        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        label_map = {"walk": 0, "None": 1}
        test_windows_raw = [(torch.randn(WINDOW_FRAMES, EMBED_DIM), "unseen_label")]
        results_df = run_inference(probe, test_windows_raw, label_map, DEVICE, str(tmp_path))
        assert results_df.iloc[0]["true_label"] == "None"
