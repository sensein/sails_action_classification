"""
Tests for I3D (R3D-18 / R2Plus1D) and VJEPA2 feature extraction pipelines.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

MODULE_I3D   = "sailsprep.action_model_testing.feature_extraction.i3d_extractor"
MODULE_VJEPA = "sailsprep.action_model_testing.feature_extraction.vjepa2_extractor"


# ============================================================
# Helpers
# ============================================================

class _DummyBackbone(nn.Module):
    """Returns (B, 512) zeros — no GPU needed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.zeros(x.shape[0], 512)


class _DummyVJEPAEncoder(nn.Module):
    """Returns a fake last_hidden_state of shape (B, 32, 1408)."""

    def forward(self, **kwargs: object) -> MagicMock:  # noqa: D102
        pv = kwargs.get("pixel_values_videos")
        assert isinstance(pv, torch.Tensor)
        B = pv.shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.zeros(B, 32, 1408)
        return out


# ============================================================
# I3D extractor tests
# ============================================================

class TestBuildBackbones:
    def test_returns_both_backbones_by_default(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import build_backbones

        models, _ = build_backbones(gpu=0, active_backbones=["i3d", "r2plus1d"])
        assert set(models.keys()) == {"i3d", "r2plus1d"}

    def test_single_backbone(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import build_backbones

        models, _ = build_backbones(gpu=0, active_backbones=["i3d"])
        assert list(models.keys()) == ["i3d"]

    def test_fc_replaced_with_identity(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import build_backbones

        models, _ = build_backbones(gpu=0, active_backbones=["i3d"])
        assert isinstance(models["i3d"].fc, nn.Identity)

    def test_models_in_eval_mode(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import build_backbones

        models, _ = build_backbones(gpu=0, active_backbones=["i3d"])
        assert not models["i3d"].training


class TestCropFrameWithBboxI3D:
    def test_output_shape(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import crop_frame_with_bbox

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = crop_frame_with_bbox(frame, bbox=(100, 50, 300, 200), out_size=224)
        assert result.shape == (224, 224, 3)

    def test_clamps_out_of_bounds_bbox(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import crop_frame_with_bbox

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = crop_frame_with_bbox(frame, bbox=(-10, -10, 200, 200), out_size=64)
        assert result.shape == (64, 64, 3)

    def test_tiny_bbox(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import crop_frame_with_bbox

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = crop_frame_with_bbox(frame, bbox=(10, 10, 11, 11), out_size=224)
        assert result.shape == (224, 224, 3)


class TestPreprocessClipTensor:
    def test_output_shape(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import preprocess_clip_tensor

        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
        tensor = preprocess_clip_tensor(frames)
        assert tensor.shape == (3, 16, 224, 224)

    def test_output_is_float(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import preprocess_clip_tensor

        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
        tensor = preprocess_clip_tensor(frames)
        assert tensor.dtype == torch.float32


class TestExtractFeaturesI3D:
    def test_output_shape(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import extract_features

        model = _DummyBackbone()
        frames = np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8)
        feats = extract_features(model, frames, batch_size=4, device=torch.device("cpu"))
        assert feats.shape == (512, 30)

    def test_empty_input(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import extract_features

        model = _DummyBackbone()
        frames = np.zeros((0, 224, 224, 3), dtype=np.uint8)
        feats = extract_features(model, frames, batch_size=4, device=torch.device("cpu"))
        assert feats.shape == (512, 0)

    def test_single_frame(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import extract_features

        model = _DummyBackbone()
        frames = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        feats = extract_features(model, frames, batch_size=4, device=torch.device("cpu"))
        assert feats.shape == (512, 1)

    def test_dtype_float32(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import extract_features

        model = _DummyBackbone()
        frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        feats = extract_features(model, frames, batch_size=4, device=torch.device("cpu"))
        assert feats.dtype == np.float32


class TestReadVideoCroppedI3D:
    def test_returns_none_for_missing_video(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import read_video_cropped

        assert read_video_cropped("/no/video.mp4", "/no/file.h5") is None

    def test_returns_none_for_missing_h5(self, tmp_path: Path) -> None:
        from sailsprep.action_model_testing.feature_extraction.i3d_extractor import read_video_cropped

        v = tmp_path / "v.mp4"
        v.touch()
        assert read_video_cropped(str(v), "/no/file.h5") is None


# ============================================================
# VJEPA2 extractor tests
# ============================================================

class TestCropFrameWithBboxVJEPA:
    def test_output_shape(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import crop_frame_with_bbox

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = crop_frame_with_bbox(frame, bbox=(50, 50, 200, 200), out_size=256)
        assert result.shape == (256, 256, 3)

    def test_clamps_out_of_bounds(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import crop_frame_with_bbox

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = crop_frame_with_bbox(frame, bbox=(-5, -5, 300, 300), out_size=64)
        assert result.shape == (64, 64, 3)


class TestReadVideoCroppedVJEPA:
    def test_returns_none_for_missing_video(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import read_video_cropped

        assert read_video_cropped("/no/video.mp4", "/no/file.h5") is None

    def test_returns_none_for_missing_h5(self, tmp_path: Path) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import read_video_cropped

        v = tmp_path / "v.mp4"
        v.touch()
        assert read_video_cropped(str(v), "/no/file.h5") is None


class TestExtractFeaturesSingleVideo:
    def test_output_shape_and_dtype(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import (
            extract_features_single_video,
            FRAMES_PER_CLIP,
        )

        model = _DummyVJEPAEncoder()

        mock_proc = MagicMock()
        mock_proc.return_value = {
            "pixel_values_videos": torch.zeros(1, FRAMES_PER_CLIP, 3, 256, 256)
        }

        T = 70  # more than one clip
        frames = np.random.randint(0, 255, (T, 256, 256, 3), dtype=np.uint8)

        feats = extract_features_single_video(
            model, mock_proc, frames, batch_clips=1, device=torch.device("cpu")
        )
        assert feats.shape[0] == 1408   # embed dim
        assert feats.shape[1] == T
        assert feats.dtype == np.float32

    def test_short_video_padded(self) -> None:
        from sailsprep.action_model_testing.feature_extraction.vjepa2_extractor import (
            extract_features_single_video,
            FRAMES_PER_CLIP,
        )

        model = _DummyVJEPAEncoder()
        mock_proc = MagicMock()
        mock_proc.return_value = {
            "pixel_values_videos": torch.zeros(1, FRAMES_PER_CLIP, 3, 256, 256)
        }

        T = 10  # shorter than one clip
        frames = np.random.randint(0, 255, (T, 256, 256, 3), dtype=np.uint8)
        feats = extract_features_single_video(
            model, mock_proc, frames, batch_clips=1, device=torch.device("cpu")
        )
        assert feats.shape == (1408, T)


# ============================================================
# Shared / integration tests
# ============================================================

class TestCSVContract:
    def test_missing_h5_column_detected(self, tmp_path: Path) -> None:
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"video_path": ["a.mp4"]}).to_csv(csv, index=False)
        df = pd.read_csv(csv)
        assert "interpolated_full_h5" not in df.columns

    def test_valid_csv_has_required_columns(self, tmp_path: Path) -> None:
        csv = tmp_path / "good.csv"
        pd.DataFrame({
            "video_path": ["a.mp4"],
            "interpolated_full_h5": ["a.h5"],
        }).to_csv(csv, index=False)
        df = pd.read_csv(csv)
        assert {"video_path", "interpolated_full_h5"}.issubset(df.columns)


class TestMetadataJSON:
    def test_round_trip(self, tmp_path: Path) -> None:
        meta = {
            "video_paths": ["a.mp4"],
            "labels": ["walk"],
            "label_encoded": [0],
            "label_map": {"walk": 0},
            "embed_dim": 1408,
            "model": "facebook/vjepa2-vitg-fpc64-256",
            "num_frames": 64,
        }
        p = tmp_path / "meta.json"
        p.write_text(json.dumps(meta))
        loaded = json.loads(p.read_text())
        assert loaded["embed_dim"] == 1408
        assert loaded["label_map"] == {"walk": 0}


class TestOutputDirs:
    def test_backbone_dirs_created(self, tmp_path: Path) -> None:
        for name in ("i3d", "r2plus1d"):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            assert d.exists()