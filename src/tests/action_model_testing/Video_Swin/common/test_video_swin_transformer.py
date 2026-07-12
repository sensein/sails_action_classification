"""
Tests for src/sailsprep/action_model_testing/Video_Swin/common/video_swin_transformer.py
"""
from __future__ import annotations

import torch

from sailsprep.action_model_testing.Video_Swin.common.video_swin_transformer import (
    window_partition,
    window_reverse,
    get_window_size,
    PatchEmbed3D,
    SwinTransformer3D,
)


class TestWindowPartitionReverse:
    def test_round_trip(self):
        B, D, H, W, C = 1, 2, 4, 4, 3
        window_size = (2, 2, 2)
        x = torch.arange(B * D * H * W * C, dtype=torch.float32).reshape(B, D, H, W, C)

        windows = window_partition(x, window_size)
        assert windows.shape == (4, 8, C)  # (B*num_windows, prod(window_size), C)

        restored = window_reverse(windows, window_size, B, D, H, W)
        assert torch.equal(restored, x)


class TestGetWindowSize:
    def test_shrinks_when_input_smaller_than_window(self):
        result = get_window_size((2, 3, 3), window_size=(2, 7, 7))
        assert result == (2, 3, 3)

    def test_keeps_window_when_input_larger(self):
        result = get_window_size((8, 16, 16), window_size=(2, 7, 7))
        assert result == (2, 7, 7)

    def test_with_shift_size(self):
        win, shift = get_window_size((2, 3, 3), window_size=(2, 7, 7), shift_size=(1, 3, 3))
        assert win == (2, 3, 3)
        assert shift == (0, 0, 0)


class TestPatchEmbed3D:
    def test_output_shape(self):
        pe = PatchEmbed3D(patch_size=(2, 4, 4), in_chans=3, embed_dim=32)
        x = torch.zeros(1, 3, 8, 32, 32)
        out = pe(x)
        # D/2=4, H/4=8, W/4=8
        assert out.shape == (1, 32, 4, 8, 8)

    def test_pads_non_divisible_input(self):
        pe = PatchEmbed3D(patch_size=(2, 4, 4), in_chans=3, embed_dim=16)
        x = torch.zeros(1, 3, 5, 10, 10)
        out = pe(x)
        assert out.shape[0] == 1 and out.shape[1] == 16


class TestSwinTransformer3DForward:
    def test_tiny_model_forward_shape(self):
        model = SwinTransformer3D(
            patch_size=(2, 4, 4),
            embed_dim=24,
            depths=[1, 1],
            num_heads=[2, 2],
            window_size=(2, 3, 3),
            drop_path_rate=0.0,
            patch_norm=True,
        )
        model.eval()
        x = torch.zeros(1, 3, 4, 12, 12)
        with torch.no_grad():
            out = model(x)
        assert out.ndim == 5
        assert out.shape[0] == 1
