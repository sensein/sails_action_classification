"""Shared SlowFast pathway-packing transform."""

import torch


class PackPathway(torch.nn.Module):
    """Splits a uniformly-sampled clip into SlowFast's (slow, fast) pathway pair."""

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        return [slow_pathway, fast_pathway]
