"""
Tests for src/sailsprep/action_model_testing/vjepa/common/probes.py
"""
import torch

from sailsprep.action_model_testing.vjepa.common.probes import (
    LinearProbe,
    MLPLargeProbe,
    MLPSmallProbe,
)


class TestLinearProbe:
    def test_output_shape(self):
        probe = LinearProbe(embed_dim=32, num_classes=5)
        x = torch.randn(2, 10, 32)
        out = probe(x)
        assert out.shape == (2, 5)


class TestMLPSmallProbe:
    def test_output_shape(self):
        probe = MLPSmallProbe(embed_dim=32, num_classes=4, hidden=16)
        x = torch.randn(3, 6, 32)
        out = probe(x)
        assert out.shape == (3, 4)


class TestMLPLargeProbe:
    def test_output_shape(self):
        probe = MLPLargeProbe(embed_dim=32, num_classes=6)
        x = torch.randn(2, 6, 32)
        out = probe(x)
        assert out.shape == (2, 6)
