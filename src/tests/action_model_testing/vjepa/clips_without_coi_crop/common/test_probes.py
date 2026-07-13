"""
Tests for src/sailsprep/action_model_testing/vjepa/clips_without_coi_crop/common/probes.py
"""
import torch

from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.common.probes import AttentiveProbe


class TestAttentiveProbe:
    def test_output_shape(self):
        probe = AttentiveProbe(embed_dim=32, num_classes=5, num_heads=4, num_queries=1)
        x = torch.randn(2, 10, 32)
        out = probe(x)
        assert out.shape == (2, 5)

    def test_multi_query_output_shape(self):
        probe = AttentiveProbe(embed_dim=16, num_classes=3, num_heads=2, num_queries=4)
        x = torch.randn(3, 8, 16)
        out = probe(x)
        assert out.shape == (3, 3)
