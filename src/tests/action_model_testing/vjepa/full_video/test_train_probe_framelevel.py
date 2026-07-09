"""
Tests for sailsprep/action_model_testing/vjepa/full_video/train_probe_framelevel.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.full_video.train_probe_framelevel import (
    CONTEXT,
    EMBED_DIM,
    WINDOW_SIZE,
    AttentiveProbe,
    FrameDataset,
    build_frames_from_video,
    run_inference_per_video,
    train_probe,
)

DEVICE = torch.device("cpu")


def _make_feat_npy(tmp_dir: str, T: int = 30, embed: int = EMBED_DIM) -> str:
    path = os.path.join(tmp_dir, "feat.npy")
    np.save(path, np.random.randn(embed, T).astype(np.float32))
    return path


def _make_label_csv(tmp_dir: str, T: int = 30, column: str = "Locomotion") -> str:
    path = os.path.join(tmp_dir, "labels.csv")
    labels = (["walk"] * (T // 2)) + (["run"] * (T - T // 2))
    pd.DataFrame({column: labels}).to_csv(path, index=False)
    return path


class TestBuildFramesFromVideo:
    def test_bad_feat_path_returns_empty(self, tmp_path):
        label_path = _make_label_csv(str(tmp_path))
        assert build_frames_from_video("/nonexistent/feat.npy", label_path, "Locomotion") == []

    def test_missing_column_returns_empty(self, tmp_path):
        feat_path = _make_feat_npy(str(tmp_path))
        label_path = _make_label_csv(str(tmp_path), column="Other")
        assert build_frames_from_video(feat_path, label_path, "Locomotion") == []

    def test_one_frame_per_timestep(self, tmp_path):
        T = 30
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        frames = build_frames_from_video(feat_path, label_path, "Locomotion")
        assert len(frames) == T

    def test_window_shape_is_context_padded(self, tmp_path):
        T = 30
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        frames = build_frames_from_video(feat_path, label_path, "Locomotion")
        window, label, t = frames[0]
        assert window.shape == (WINDOW_SIZE, EMBED_DIM)
        assert t == 0
        assert label == "walk"

    def test_edge_padding_repeats_boundary_frame(self, tmp_path):
        T = 30
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        raw_feat = np.load(feat_path).T  # (T, D)
        frames = build_frames_from_video(feat_path, label_path, "Locomotion")
        window0, _, _ = frames[0]
        # First CONTEXT rows of window for t=0 should all equal frame 0
        # (edge padding repeats the boundary).
        for i in range(CONTEXT):
            assert torch.allclose(window0[i], torch.tensor(raw_feat[0]))


class TestFrameDataset:
    def test_len_and_encoding(self):
        frames_raw = [
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "walk", 0),
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "unseen", 1),
        ]
        label_map = {"walk": 0, "None": 1}
        ds = FrameDataset(frames_raw, label_map)
        assert len(ds) == 2
        window, enc, lbl, t = ds[1]
        assert enc.item() == 1  # falls back to "None"
        assert lbl == "unseen"
        assert t == 1


class TestAttentiveProbe:
    def test_forward_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=3, num_heads=4)
        x = torch.randn(4, WINDOW_SIZE, 32)
        assert probe(x).shape == (4, 3)


class TestTrainProbe:
    def _loader(self, n=8, num_classes=2, batch_size=4):
        frames_raw = [
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "walk" if i % 2 == 0 else "run", i)
            for i in range(n)
        ]
        label_map = {"walk": 0, "run": 1, "None": 2}
        ds = FrameDataset(frames_raw, label_map)
        return DataLoader(ds, batch_size=batch_size)

    def test_runs_one_epoch(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.full_video.train_probe_framelevel as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        trained = train_probe(probe, self._loader(), self._loader(), DEVICE, str(tmp_path))
        assert isinstance(trained, AttentiveProbe)
        assert (tmp_path / "training_log.csv").exists()


class TestRunInferencePerVideo:
    def test_writes_per_video_csv_and_metrics(self, tmp_path):
        T = 20
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        splits_csv = tmp_path / "splits.csv"
        pd.DataFrame({
            "split": ["test"],
            "vjpe_features_full_video_vit_h_features": [feat_path],
            "label_path": [label_path],
        }).to_csv(splits_csv, index=False)

        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=3, num_heads=8)
        label_map = {"walk": 0, "run": 1, "None": 2}

        run_inference_per_video(
            probe, str(splits_csv), "Locomotion", label_map, DEVICE, str(tmp_path)
        )

        pred_dir = tmp_path / "per_video_predictions"
        assert pred_dir.exists()
        assert len(list(pred_dir.glob("*_predictions.csv"))) == 1
        assert (tmp_path / "test_metrics.txt").exists()
