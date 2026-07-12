"""
Tests for src/sailsprep/action_model_testing/slow_fast/experiments/common/pack_pathway.py
"""
import torch

from sailsprep.action_model_testing.slow_fast.experiments.common.pack_pathway import PackPathway


class TestPackPathway:
    def test_fast_pathway_unchanged(self):
        pp = PackPathway(alpha=4)
        frames = torch.rand(3, 32, 224, 224)
        slow, fast = pp(frames)
        assert torch.equal(fast, frames)

    def test_slow_pathway_subsampled(self):
        pp = PackPathway(alpha=4)
        frames = torch.rand(3, 32, 224, 224)
        slow, fast = pp(frames)
        assert slow.shape[1] == 32 // 4

    def test_default_alpha(self):
        pp = PackPathway()
        assert pp.alpha == 4
