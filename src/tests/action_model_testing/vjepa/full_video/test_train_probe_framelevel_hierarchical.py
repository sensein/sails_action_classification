"""
Tests for sailsprep/action_model_testing/vjepa/full_video/train_probe_framelevel_hierarchical.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.full_video.train_probe_framelevel_hierarchical import (
    EMBED_DIM,
    WINDOW_SIZE,
    FrameDataset,
    HierarchicalProbe,
    build_frames_from_video,
    build_label_maps,
    hierarchical_loss,
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
    def test_one_frame_per_timestep(self, tmp_path):
        T = 20
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        frames = build_frames_from_video(feat_path, label_path, "Locomotion")
        assert len(frames) == T
        window, label, t = frames[0]
        assert window.shape == (WINDOW_SIZE, EMBED_DIM)


class TestBuildLabelMaps:
    def test_stage_maps(self):
        train_raw = [
            (torch.zeros(WINDOW_SIZE, EMBED_DIM), "walk", 0),
            (torch.zeros(WINDOW_SIZE, EMBED_DIM), "run", 1),
            (torch.zeros(WINDOW_SIZE, EMBED_DIM), "None", 2),
        ]
        stage1_map, stage2_map = build_label_maps(train_raw)
        assert stage1_map == {"None": 0, "run": 1, "walk": 1}
        assert stage2_map == {"run": 0, "walk": 1}


class TestFrameDataset:
    def test_getitem_encodes_both_stages(self):
        stage1_map = {"None": 0, "walk": 1}
        stage2_map = {"walk": 0}
        frames_raw = [
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "None", 0),
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "walk", 1),
        ]
        ds = FrameDataset(frames_raw, stage1_map, stage2_map)
        assert len(ds) == 2
        window, s1, s2, lbl, t = ds[0]
        assert s1.item() == 0 and s2.item() == -1
        window, s1, s2, lbl, t = ds[1]
        assert s1.item() == 1 and s2.item() == 0


class TestHierarchicalProbe:
    def test_forward_shapes(self):
        probe = HierarchicalProbe(embed_dim=32, num_stage2_classes=3, num_heads=4)
        l1, l2 = probe(torch.randn(4, WINDOW_SIZE, 32))
        assert l1.shape == (4, 2)
        assert l2.shape == (4, 3)


class TestHierarchicalLoss:
    def test_no_nonnone_samples_gives_zero_stage2_loss(self):
        logits1 = torch.randn(3, 2)
        logits2 = torch.randn(3, 2)
        s1 = torch.zeros(3, dtype=torch.long)
        s2 = torch.full((3,), -1, dtype=torch.long)
        total, l1, l2 = hierarchical_loss(logits1, logits2, s1, s2)
        assert l2.item() == 0.0


class TestTrainProbe:
    def _loader(self, n=8, batch_size=4):
        stage1_map = {"None": 0, "walk": 1}
        stage2_map = {"walk": 0}
        frames_raw = [
            (torch.randn(WINDOW_SIZE, EMBED_DIM), "walk" if i % 2 == 0 else "None", i)
            for i in range(n)
        ]
        ds = FrameDataset(frames_raw, stage1_map, stage2_map)
        return DataLoader(ds, batch_size=batch_size)

    def test_runs_one_epoch(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.full_video.train_probe_framelevel_hierarchical as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=1, num_heads=8)
        trained = train_probe(probe, self._loader(), self._loader(), DEVICE, str(tmp_path))
        assert isinstance(trained, HierarchicalProbe)
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

        stage1_map = {"None": 0, "walk": 1, "run": 1}
        stage2_map = {"walk": 0, "run": 1}
        probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=2, num_heads=8)

        run_inference_per_video(
            probe, str(splits_csv), "Locomotion", stage1_map, stage2_map, DEVICE, str(tmp_path)
        )

        pred_dir = tmp_path / "per_video_predictions"
        assert pred_dir.exists()
        assert len(list(pred_dir.glob("*_predictions.csv"))) == 1
        assert (tmp_path / "test_metrics.txt").exists()
