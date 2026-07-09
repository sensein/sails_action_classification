"""
Tests for sailsprep/action_model_testing/vjepa/full_video/two_stage.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.full_video.two_stage import (
    EMBED_DIM,
    STRIDE_FRAMES,
    WINDOW_FRAMES,
    AttentiveBackbone,
    HierarchicalProbe,
    WindowDataset,
    build_label_maps,
    build_windows_from_video,
    hierarchical_loss,
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

    def test_correct_number_of_windows(self, tmp_path):
        T = 90
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        expected = len(range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES))
        assert len(windows) == expected


class TestBuildLabelMaps:
    def test_stage1_binary_stage2_nonnone_only(self):
        train_raw = [
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "walk"),
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "run"),
            (torch.zeros(WINDOW_FRAMES, EMBED_DIM), "None"),
        ]
        stage1_map, stage2_map = build_label_maps(train_raw)
        assert stage1_map == {"None": 0, "run": 1, "walk": 1}
        assert stage2_map == {"run": 0, "walk": 1}

    def test_none_added_if_absent(self):
        train_raw = [(torch.zeros(WINDOW_FRAMES, EMBED_DIM), "walk")]
        stage1_map, _ = build_label_maps(train_raw)
        assert stage1_map["None"] == 0


class TestWindowDataset:
    def test_none_windows_get_stage2_minus_one(self):
        stage1_map = {"None": 0, "walk": 1}
        stage2_map = {"walk": 0}
        raw = [(torch.randn(WINDOW_FRAMES, EMBED_DIM), "None"),
               (torch.randn(WINDOW_FRAMES, EMBED_DIM), "walk")]
        ds = WindowDataset(raw, stage1_map, stage2_map)
        assert len(ds) == 2
        feat, s1, s2, lbl = ds[0]
        assert s1.item() == 0 and s2.item() == -1 and lbl == "None"
        feat, s1, s2, lbl = ds[1]
        assert s1.item() == 1 and s2.item() == 0 and lbl == "walk"


class TestModel:
    def test_backbone_output_shape(self):
        backbone = AttentiveBackbone(embed_dim=32, num_heads=4)
        out = backbone(torch.randn(3, WINDOW_FRAMES, 32))
        assert out.shape == (3, 32)

    def test_hierarchical_probe_output_shapes(self):
        probe = HierarchicalProbe(embed_dim=32, num_stage2_classes=4, num_heads=4)
        logits1, logits2 = probe(torch.randn(3, WINDOW_FRAMES, 32))
        assert logits1.shape == (3, 2)
        assert logits2.shape == (3, 4)


class TestHierarchicalLoss:
    def test_all_none_batch_stage2_loss_is_zero(self):
        logits1 = torch.randn(4, 2)
        logits2 = torch.randn(4, 3)
        s1 = torch.zeros(4, dtype=torch.long)
        s2 = torch.full((4,), -1, dtype=torch.long)
        total, l1, l2 = hierarchical_loss(logits1, logits2, s1, s2)
        assert l2.item() == 0.0
        assert torch.isclose(total, l1)

    def test_mixed_batch_computes_both_losses(self):
        logits1 = torch.randn(4, 2)
        logits2 = torch.randn(4, 3)
        s1 = torch.tensor([0, 1, 1, 0])
        s2 = torch.tensor([-1, 0, 2, -1])
        total, l1, l2 = hierarchical_loss(logits1, logits2, s1, s2)
        assert l2.item() > 0.0
        assert total.item() > 0.0


class TestTrainProbe:
    def _loader(self, n=8, num_stage2=2, batch_size=4):
        stage1_map = {"None": 0, "a": 1, "b": 1}
        stage2_map = {"a": 0, "b": 1}
        raw = [
            (torch.randn(WINDOW_FRAMES, EMBED_DIM), "a" if i % 2 == 0 else "None")
            for i in range(n)
        ]
        ds = WindowDataset(raw, stage1_map, stage2_map)
        return DataLoader(ds, batch_size=batch_size)

    def test_runs_one_epoch(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.full_video.two_stage as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=2, num_heads=8)
        trained = train_probe(probe, self._loader(), self._loader(), DEVICE, str(tmp_path))
        assert isinstance(trained, HierarchicalProbe)
        assert (tmp_path / "training_log.csv").exists()


class TestRunInference:
    def test_writes_predictions_and_metrics(self, tmp_path):
        probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=2, num_heads=8)
        stage1_map = {"None": 0, "walk": 1, "run": 1}
        stage2_map = {"walk": 0, "run": 1}
        test_windows_raw = [
            (torch.randn(WINDOW_FRAMES, EMBED_DIM), "walk" if i % 2 == 0 else "None")
            for i in range(6)
        ]
        results_df = run_inference(
            probe, test_windows_raw, stage1_map, stage2_map, DEVICE, str(tmp_path)
        )
        assert len(results_df) == 6
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "test_metrics.txt").exists()
