"""
Tests for src/sailsprep/action_model_testing/slow_fast/experiments/common/video_dataset.py
"""
from unittest.mock import MagicMock, patch

import pytest
import torch

from sailsprep.action_model_testing.slow_fast.experiments.common.video_dataset import VideoDataset


def _fake_encoded_video(video_tensor):
    ev = MagicMock()
    ev.get_clip.return_value = {"video": video_tensor}
    return ev


class TestVideoDataset:
    def test_getitem_success(self):
        ds = VideoDataset(
            video_paths=["a.mp4", "b.mp4"],
            labels=[0, 1],
            clip_duration=1.0,
            num_frames=8,
            crop_size=224,
            alpha=4,
        )
        video_tensor = torch.zeros(3, 8, 224, 224)
        with patch(
            "sailsprep.action_model_testing.slow_fast.experiments.common.video_dataset.EncodedVideo.from_path",
            return_value=_fake_encoded_video(video_tensor),
        ):
            out, label = ds[0]
        assert torch.equal(out, video_tensor)
        assert label == 0

    def test_getitem_falls_back_to_next_video_on_decode_failure(self):
        ds = VideoDataset(
            video_paths=["bad.mp4", "good.mp4"],
            labels=[0, 1],
            clip_duration=1.0,
            num_frames=8,
            crop_size=224,
            alpha=4,
        )
        good_tensor = torch.ones(3, 8, 224, 224)

        def _from_path(path, **kwargs):
            if path == "bad.mp4":
                raise RuntimeError("decode failed")
            return _fake_encoded_video(good_tensor)

        with patch(
            "sailsprep.action_model_testing.slow_fast.experiments.common.video_dataset.EncodedVideo.from_path",
            side_effect=_from_path,
        ):
            out, label = ds[0]
        assert torch.equal(out, good_tensor)
        assert label == 1

    def test_getitem_dummy_tensor_when_all_fail_slowfast(self):
        ds = VideoDataset(
            video_paths=["bad1.mp4", "bad2.mp4"],
            labels=[0, 1],
            clip_duration=1.0,
            num_frames=8,
            crop_size=224,
            alpha=4,
            model_name="slowfast_r50",
        )
        with patch(
            "sailsprep.action_model_testing.slow_fast.experiments.common.video_dataset.EncodedVideo.from_path",
            side_effect=RuntimeError("decode failed"),
        ):
            out, label = ds[0]
        assert isinstance(out, list)
        assert out[0].shape == (3, 2, 224, 224)  # num_frames // alpha
        assert out[1].shape == (3, 8, 224, 224)

    def test_getitem_dummy_tensor_when_all_fail_slow_only(self):
        ds = VideoDataset(
            video_paths=["bad1.mp4", "bad2.mp4"],
            labels=[0, 1],
            clip_duration=1.0,
            num_frames=8,
            crop_size=224,
            alpha=4,
            model_name="slow_r50",
        )
        with patch(
            "sailsprep.action_model_testing.slow_fast.experiments.common.video_dataset.EncodedVideo.from_path",
            side_effect=RuntimeError("decode failed"),
        ):
            out, label = ds[0]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 8, 224, 224)

    def test_len(self):
        ds = VideoDataset(
            video_paths=["a.mp4", "b.mp4", "c.mp4"],
            labels=[0, 1, 2],
            clip_duration=1.0,
            num_frames=8,
            crop_size=224,
            alpha=4,
        )
        assert len(ds) == 3
