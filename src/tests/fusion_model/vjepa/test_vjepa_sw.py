"""
Tests for vjepa_sw.py — no disk I/O, no GPU required.
Run: poetry run pytest src/tests/tests_vjepa_sw.py
"""

from __future__ import annotations

import tempfile
import os
import numpy as np
import pandas as pd
import pytest
import torch

from sailsprep.fusion_model.vjepa.vjepa_sw import (
    EMBED_DIM,
    WINDOW_FRAMES,
    STRIDE_FRAMES,
    CLASS_WEIGHT_STRENGTH,
    ImprovedAttentiveProbe,
    SinusoidalPositionalEncoding,
    WindowDataset,
    build_windows_from_video,
    compute_class_weights,
    infer_one_video,
    load_video_data,
)

DEVICE = torch.device("cpu")

# ============================================================
# Helpers
# ============================================================

def _make_feat_npy(tmp_dir: str, T: int = 60, embed: int = EMBED_DIM) -> str:
    """Save a random (embed, T) .npy feature file and return path."""
    path = os.path.join(tmp_dir, "feat.npy")
    arr = np.random.randn(embed, T).astype(np.float32)
    np.save(path, arr)
    return path


def _make_label_csv(tmp_dir: str, T: int = 60, column: str = "Locomotion") -> str:
    """Save a label CSV with T rows and return path."""
    path = os.path.join(tmp_dir, "labels.csv")
    labels = (["walk"] * (T // 2)) + (["run"] * (T - T // 2))
    pd.DataFrame({column: labels}).to_csv(path, index=False)
    return path


def _make_probe(num_classes: int = 3) -> ImprovedAttentiveProbe:
    return ImprovedAttentiveProbe(embed_dim=EMBED_DIM, num_classes=num_classes)


# ============================================================
# 1. SinusoidalPositionalEncoding
# ============================================================

class TestSinusoidalPositionalEncoding:
    def test_output_shape(self) -> None:
        pe = SinusoidalPositionalEncoding(embed_dim=16, max_len=64)
        x = torch.zeros(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)

    def test_adds_positional_signal(self) -> None:
        pe = SinusoidalPositionalEncoding(embed_dim=16, max_len=64)
        x = torch.zeros(1, 5, 16)
        out = pe(x)
        # output should differ across time steps
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_no_learnable_params(self) -> None:
        pe = SinusoidalPositionalEncoding(embed_dim=16, max_len=64)
        assert sum(p.numel() for p in pe.parameters()) == 0


# ============================================================
# 2. ImprovedAttentiveProbe
# ============================================================

class TestImprovedAttentiveProbe:
    def test_forward_shape(self) -> None:
        probe = _make_probe(num_classes=4)
        x = torch.randn(8, WINDOW_FRAMES, EMBED_DIM)
        out = probe(x)
        assert out.shape == (8, 4)

    def test_batch_size_1(self) -> None:
        probe = _make_probe(num_classes=2)
        x = torch.randn(1, WINDOW_FRAMES, EMBED_DIM)
        out = probe(x)
        assert out.shape == (1, 2)

    def test_output_is_tensor(self) -> None:
        probe = _make_probe()
        x = torch.randn(4, WINDOW_FRAMES, EMBED_DIM)
        out = probe(x)
        assert isinstance(out, torch.Tensor)

    def test_eval_deterministic(self) -> None:
        probe = _make_probe()
        probe.eval()
        x = torch.randn(2, WINDOW_FRAMES, EMBED_DIM)
        with torch.no_grad():
            o1 = probe(x)
            o2 = probe(x)
        assert torch.allclose(o1, o2)


# ============================================================
# 3. WindowDataset
# ============================================================

class TestWindowDataset:
    def _make_raw(self, n: int = 10) -> list[tuple[torch.Tensor, str]]:
        return [
            (torch.randn(WINDOW_FRAMES, EMBED_DIM), "walk" if i % 2 == 0 else "None")
            for i in range(n)
        ]

    def test_len(self) -> None:
        label_map = {"walk": 0, "None": 1}
        ds = WindowDataset(self._make_raw(7), label_map)
        assert len(ds) == 7

    def test_getitem_shapes(self) -> None:
        label_map = {"walk": 0, "None": 1}
        ds = WindowDataset(self._make_raw(4), label_map)
        feat, lbl = ds[0]
        assert feat.shape == (WINDOW_FRAMES, EMBED_DIM)
        assert lbl.dtype == torch.long

    def test_unknown_label_falls_back_to_none(self) -> None:
        label_map = {"walk": 0, "None": 1}
        raw = [(torch.randn(WINDOW_FRAMES, EMBED_DIM), "unknown")]
        ds = WindowDataset(raw, label_map)
        _, lbl = ds[0]
        assert lbl.item() == 1  # "None" fallback


# ============================================================
# 4. build_windows_from_video
# ============================================================

class TestBuildWindowsFromVideo:
    def test_returns_empty_on_bad_feat_path(self, tmp_path: pytest.TempPathFactory) -> None:
        with tempfile.TemporaryDirectory() as td:
            label_path = _make_label_csv(td)
            result = build_windows_from_video("/nonexistent/feat.npy", label_path, "Locomotion")
        assert result == []

    def test_returns_empty_on_bad_label_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            feat_path = _make_feat_npy(td, T=60)
            result = build_windows_from_video(feat_path, "/nonexistent/labels.csv", "Locomotion")
        assert result == []

    def test_returns_empty_on_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            feat_path = _make_feat_npy(td, T=60)
            label_path = _make_label_csv(td, column="OtherCol")
            result = build_windows_from_video(feat_path, label_path, "Locomotion")
        assert result == []

    def test_correct_number_of_windows(self) -> None:
        T = 60
        with tempfile.TemporaryDirectory() as td:
            feat_path  = _make_feat_npy(td, T=T)
            label_path = _make_label_csv(td, T=T)
            windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        expected = len(range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES))
        assert len(windows) == expected

    def test_window_tensor_shape(self) -> None:
        T = 60
        with tempfile.TemporaryDirectory() as td:
            feat_path  = _make_feat_npy(td, T=T)
            label_path = _make_label_csv(td, T=T)
            windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        feat, label = windows[0]
        assert feat.shape == (WINDOW_FRAMES, EMBED_DIM)
        assert isinstance(label, str)

    def test_short_video_no_windows(self) -> None:
        """Video shorter than WINDOW_FRAMES → empty list."""
        T = WINDOW_FRAMES - 1
        with tempfile.TemporaryDirectory() as td:
            feat_path  = _make_feat_npy(td, T=T)
            label_path = _make_label_csv(td, T=T)
            windows = build_windows_from_video(feat_path, label_path, "Locomotion")
        assert windows == []


# ============================================================
# 5. compute_class_weights
# ============================================================

class TestComputeClassWeights:
    def test_shape(self) -> None:
        w = compute_class_weights({0: 100, 1: 50, 2: 10}, num_classes=3)
        assert w.shape == (3,)

    def test_mean_is_one(self) -> None:
        w = compute_class_weights({0: 100, 1: 50, 2: 10}, num_classes=3)
        assert abs(w.mean().item() - 1.0) < 1e-5

    def test_uniform_when_strength_zero(self) -> None:
        w = compute_class_weights({0: 100, 1: 50, 2: 10}, num_classes=3, strength=0.0)
        assert torch.allclose(w, torch.ones(3))

    def test_rare_class_gets_higher_weight(self) -> None:
        w = compute_class_weights({0: 1000, 1: 10}, num_classes=2)
        # class 1 (rare) should weigh more
        assert w[1] > w[0]


# ============================================================
# 6. load_video_data
# ============================================================

class TestLoadVideoData:
    def test_returns_none_on_bad_feat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            label_path = _make_label_csv(td)
            feat, labels = load_video_data("/bad/path.npy", label_path, "Locomotion")
        assert feat is None and labels is None

    def test_returns_none_on_bad_label(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            feat_path = _make_feat_npy(td)
            feat, labels = load_video_data(feat_path, "/bad/labels.csv", "Locomotion")
        assert feat is None and labels is None

    def test_returns_none_on_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            feat_path  = _make_feat_npy(td, T=30)
            label_path = _make_label_csv(td, T=30, column="Other")
            feat, labels = load_video_data(feat_path, label_path, "Locomotion")
        assert feat is None and labels is None

    def test_correct_shapes(self) -> None:
        T = 40
        with tempfile.TemporaryDirectory() as td:
            feat_path  = _make_feat_npy(td, T=T)
            label_path = _make_label_csv(td, T=T)
            feat, labels = load_video_data(feat_path, label_path, "Locomotion")
        assert feat is not None and labels is not None
        assert feat.shape == (T, EMBED_DIM)
        assert len(labels) == T


# ============================================================
# 7. infer_one_video
# ============================================================

class TestInferOneVideo:
    def _setup(self, T: int = 60, num_classes: int = 3) -> tuple:
        label_map = {f"cls{i}": i for i in range(num_classes)}
        probe = _make_probe(num_classes=num_classes)
        probe.eval()
        feat = np.random.randn(T, EMBED_DIM).astype(np.float32)
        return probe, feat, label_map

    def test_output_lengths(self) -> None:
        T = 60
        probe, feat, label_map = self._setup(T=T)
        preds, confs = infer_one_video(probe, feat, label_map, DEVICE)
        assert len(preds) == T
        assert len(confs) == T

    def test_predictions_are_valid_labels(self) -> None:
        probe, feat, label_map = self._setup(T=60)
        preds, _ = infer_one_video(probe, feat, label_map, DEVICE)
        valid = set(label_map.keys()) | {"None"}
        assert all(p in valid for p in preds)

    def test_confidences_in_range(self) -> None:
        probe, feat, label_map = self._setup(T=60)
        _, confs = infer_one_video(probe, feat, label_map, DEVICE)
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_short_video_returns_none(self) -> None:
        """Video too short for any window → all 'None'."""
        T = WINDOW_FRAMES - 1
        probe, _, label_map = self._setup()
        feat = np.random.randn(T, EMBED_DIM).astype(np.float32)
        preds, confs = infer_one_video(probe, feat, label_map, DEVICE)
        assert len(preds) == T
        assert all(p == "None" for p in preds)
        assert all(c == 0.0 for c in confs)