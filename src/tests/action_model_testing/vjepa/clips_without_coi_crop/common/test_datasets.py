"""
Tests for src/sailsprep/action_model_testing/vjepa/clips_without_coi_crop/common/datasets.py
"""
import torch

from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.common.datasets import FeatureDataset


class TestFeatureDataset:
    def test_len(self):
        ds = FeatureDataset(torch.zeros(5, 1408), torch.zeros(5, dtype=torch.long))
        assert len(ds) == 5

    def test_getitem_returns_feature_and_label(self):
        features = torch.arange(6).reshape(2, 3).float()
        labels = torch.tensor([0, 1])
        ds = FeatureDataset(features, labels)
        feat, label = ds[1]
        assert torch.equal(feat, features[1])
        assert label == 1
