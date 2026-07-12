"""
Tests for sailsprep/action_model_testing/vjepa/clips_without_coi_crop/train_probe_ablation.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.locomotion.train_probe_ablation import (
    EMBED_DIM,
    HEAD_CHOICES,
    AttentiveProbe,
    FeatureDataset,
    LinearProbe,
    MLPLargeProbe,
    MLPSmallProbe,
    TransformerProbe,
    build_probe,
    run_inference,
    train_probe,
)

DEVICE = torch.device("cpu")


class TestHeadChoices:
    def test_all_five_heads_present(self):
        assert set(HEAD_CHOICES) == {
            "linear", "mlp_small", "mlp_large", "attentive", "transformer",
        }


class TestClassificationHeads:
    def test_linear_probe_shape(self):
        probe = LinearProbe(embed_dim=32, num_classes=3)
        assert probe(torch.randn(2, 5, 32)).shape == (2, 3)

    def test_mlp_small_probe_shape(self):
        probe = MLPSmallProbe(embed_dim=32, num_classes=3, hidden=16)
        assert probe(torch.randn(2, 5, 32)).shape == (2, 3)

    def test_mlp_large_probe_shape(self):
        probe = MLPLargeProbe(embed_dim=32, num_classes=3)
        assert probe(torch.randn(2, 5, 32)).shape == (2, 3)

    def test_attentive_probe_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=3, num_heads=4)
        assert probe(torch.randn(2, 5, 32)).shape == (2, 3)

    def test_transformer_probe_shape(self):
        probe = TransformerProbe(embed_dim=32, num_classes=3, num_heads=4)
        assert probe(torch.randn(2, 5, 32)).shape == (2, 3)


class TestBuildProbe:
    def test_builds_each_registered_head(self):
        for head_name, cls in [
            ("linear", LinearProbe), ("mlp_small", MLPSmallProbe),
            ("mlp_large", MLPLargeProbe), ("attentive", AttentiveProbe),
            ("transformer", TransformerProbe),
        ]:
            probe = build_probe(head_name, embed_dim=32, num_classes=3)
            assert isinstance(probe, cls)


class TestFeatureDataset:
    def test_len_and_getitem(self):
        feats = torch.randn(4, 5, 32)
        labels = torch.tensor([0, 1, 0, 1])
        ds = FeatureDataset(feats, labels)
        assert len(ds) == 4
        f, l = ds[0]
        assert f.shape == (5, 32)


class TestTrainProbe:
    def _make_loader(self, n=8, tokens=5, dim=EMBED_DIM, num_classes=2, batch_size=4):
        feats = torch.randn(n, tokens, dim)
        labels = torch.randint(0, num_classes, (n,))
        return DataLoader(FeatureDataset(feats, labels), batch_size=batch_size)

    def test_runs_one_epoch_and_saves_log(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.clips_without_coi_crop.locomotion.train_probe_ablation as mod
        monkeypatch.setattr(mod, "MAX_EPOCHS", 1)
        monkeypatch.setattr(mod, "PATIENCE", 1)

        probe = LinearProbe(embed_dim=EMBED_DIM, num_classes=2)
        tr_dl = self._make_loader()
        va_dl = self._make_loader()

        trained, best_acc = train_probe(probe, tr_dl, va_dl, DEVICE, str(tmp_path), "linear")
        assert isinstance(trained, LinearProbe)
        assert 0.0 <= best_acc <= 1.0
        assert (tmp_path / "training_log.csv").exists()


class TestRunInference:
    def test_writes_predictions_and_metrics(self, tmp_path):
        n = 6
        probe = LinearProbe(embed_dim=EMBED_DIM, num_classes=2)
        test_features = torch.randn(n, 5, EMBED_DIM)
        test_labels_enc = [i % 2 for i in range(n)]
        original_labels = ["walk" if i % 2 == 0 else "run" for i in range(n)]
        video_paths = [f"v{i}.mp4" for i in range(n)]
        label_map = {"walk": 0, "run": 1}

        acc = run_inference(
            probe, test_features, test_labels_enc, original_labels,
            video_paths, label_map, DEVICE, str(tmp_path), "linear",
        )
        assert 0.0 <= acc <= 1.0
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "test_metrics.txt").exists()
