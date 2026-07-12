"""
Tests for sailsprep/action_model_testing/vjepa/clips_without_coi_crop/train_probe.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.locomotion.train_probe import (
    EMBED_DIM,
    AttentiveProbe,
    FeatureDataset,
    run_inference,
    train_probe,
)

DEVICE = torch.device("cpu")


class TestAttentiveProbe:
    def test_forward_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=4, num_heads=4)
        x = torch.randn(3, 6, 32)
        out = probe(x)
        assert out.shape == (3, 4)

    def test_batch_size_1(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=2, num_heads=4)
        x = torch.randn(1, 6, 32)
        assert probe(x).shape == (1, 2)


class TestFeatureDataset:
    def test_len_and_getitem(self):
        feats = torch.randn(5, 6, 32)
        labels = torch.tensor([0, 1, 0, 1, 0])
        ds = FeatureDataset(feats, labels)
        assert len(ds) == 5
        f, l = ds[2]
        assert f.shape == (6, 32)
        assert l.item() == 0


class TestTrainProbe:
    def _make_loader(self, n=8, tokens=6, dim=EMBED_DIM, num_classes=2, batch_size=4):
        feats = torch.randn(n, tokens, dim)
        labels = torch.randint(0, num_classes, (n,))
        return DataLoader(FeatureDataset(feats, labels), batch_size=batch_size)

    def test_runs_one_epoch_and_saves_log(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.clips_without_coi_crop.locomotion.train_probe as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        tr_dl = self._make_loader()
        va_dl = self._make_loader()

        trained = train_probe(probe, tr_dl, va_dl, DEVICE, str(tmp_path))
        assert isinstance(trained, AttentiveProbe)
        assert (tmp_path / "training_log.csv").exists()


class TestRunInference:
    def test_writes_predictions_and_metrics(self, tmp_path):
        n = 6
        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        test_features = torch.randn(n, 6, EMBED_DIM)
        test_labels_enc = [i % 2 for i in range(n)]
        original_labels = ["walk" if i % 2 == 0 else "run" for i in range(n)]
        video_paths = [f"v{i}.mp4" for i in range(n)]
        label_map = {"walk": 0, "run": 1}

        results_df = run_inference(
            probe, test_features, test_labels_enc, original_labels,
            video_paths, label_map, DEVICE, str(tmp_path),
        )
        assert len(results_df) == n
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "test_metrics.txt").exists()
