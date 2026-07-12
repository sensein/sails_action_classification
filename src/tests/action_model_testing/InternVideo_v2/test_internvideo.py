"""Tests for InternVideo2 fine-tuning pipeline.

These tests are designed to run in CI without a GPU, real video files,
or the 12 GB InternVideo2 checkpoint.  Heavy external dependencies
(transformers, pytorch_lightning, cv2, h5py) are mocked where needed.

Run with::

    pytest src/tests/test_internvideo.py -v
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stubs for heavy optional imports so the module can be imported
# without a GPU or the full conda environment.
# ---------------------------------------------------------------------------

def _make_pl_stub() -> types.ModuleType:
    """Return a minimal pytorch_lightning stub."""
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = object  # type: ignore[attr-defined]
    pl.LightningDataModule = object  # type: ignore[attr-defined]

    class _Trainer:
        def __init__(self, **kw: Any) -> None:
            self.is_global_zero = True
            self.callback_metrics: dict = {}

        def fit(self, *a: Any, **kw: Any) -> None:
            pass

    class _MCkpt:
        best_model_path = ""
        def __init__(self, **kw: Any) -> None:
            pass

    class _ES:
        def __init__(self, **kw: Any) -> None:
            pass

    pl.Trainer = _Trainer  # type: ignore[attr-defined]
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.ModelCheckpoint = _MCkpt  # type: ignore[attr-defined]
    callbacks.EarlyStopping = _ES  # type: ignore[attr-defined]
    pl.callbacks = callbacks  # type: ignore[attr-defined]
    pl.seed_everything = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    return pl


def _make_cv2_stub() -> types.ModuleType:
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    cv2.CAP_PROP_POS_FRAMES = 1  # type: ignore[attr-defined]
    cv2.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]

    class _Cap:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return 30.0

        def set(self, prop: int, val: float) -> None:
            pass

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    cv2.VideoCapture = _Cap  # type: ignore[attr-defined]
    cv2.resize = lambda img, sz: np.zeros((*sz[::-1], 3), dtype=np.uint8)  # type: ignore[attr-defined]
    cv2.cvtColor = lambda img, code: img  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2
    return cv2


def _make_h5py_stub() -> types.ModuleType:
    h5py = types.ModuleType("h5py")

    class _File:
        def __init__(self, path: str, mode: str = "r") -> None:
            pass

        def __enter__(self) -> _File:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def __getitem__(self, key: str) -> Any:
            # Mimic the real HDF5 structure:
            # f["bboxes/table"][()] returns a structured array
            # that has a field "values_block_1"
            dt = np.dtype([
                ("index",         np.int32),
                ("values_block_0",np.int32),
                ("values_block_1", np.dtype([
                    ("f0", np.int32),  # frame idx
                    ("f1", np.int32),  # unused
                    ("f2", np.int32),  # x1
                    ("f3", np.int32),  # y1
                    ("f4", np.int32),  # x2
                    ("f5", np.int32),  # y2
                ]))
            ])
            # just return the table object which supports ["values_block_1"]
            return _Table()

    class _Table:
        """Mimics f["bboxes/table"]."""
        def __call__(self) -> _Vb1Wrapper:
            return _Vb1Wrapper()

        def __getitem__(self, key: str) -> _Vb1Wrapper:
            return _Vb1Wrapper()

    class _Vb1Wrapper:
        """Mimics table[()], supports table["values_block_1"]."""
        def __getitem__(self, key: str) -> np.ndarray:
            # Each row: (frame_idx, unused, x1, y1, x2, y2)
            dt = np.dtype([
                ("f0", np.int32),
                ("f1", np.int32),
                ("f2", np.int32),
                ("f3", np.int32),
                ("f4", np.int32),
                ("f5", np.int32),
            ])
            return np.array([(0, 0, 10, 20, 110, 120)], dtype=dt)

    h5py.File = _File  # type: ignore[attr-defined]
    sys.modules["h5py"] = h5py
    return h5py


def _make_sklearn_stub() -> None:
    sklearn = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")
    metrics.classification_report = lambda *a, **kw: "mocked report"  # type: ignore[attr-defined]
    metrics.confusion_matrix = lambda *a, **kw: np.zeros((2, 2), dtype=int)  # type: ignore[attr-defined]
    sklearn.metrics = metrics  # type: ignore[attr-defined]
    if _real_sklearn is not None:
        sys.modules["sklearn"] = _real_sklearn
    else:
        sys.modules.pop("sklearn", None)

    if _real_sklearn_metrics is not None:
        sys.modules["sklearn.metrics"] = _real_sklearn_metrics
    else:
        sys.modules.pop("sklearn.metrics", None)


# Install stubs before the module under test is imported. These stub
# modules are only good enough for loading InternVideo.py itself — leaving
# them in sys.modules afterward would break every other action_model_testing
# suite that needs the REAL pytorch_lightning/cv2/h5py (this file is
# collected first, alphabetically, in a full-suite run). So snapshot
# whatever was really there beforehand and restore it right after import.
_real_sklearn = sys.modules.get("sklearn")
_real_sklearn_metrics = sys.modules.get("sklearn.metrics")
_real_pl = sys.modules.get("pytorch_lightning")
_real_pl_callbacks = sys.modules.get("pytorch_lightning.callbacks")
_real_cv2 = sys.modules.get("cv2")
_real_h5py = sys.modules.get("h5py")

_make_pl_stub()
_make_cv2_stub()
_make_h5py_stub()
_make_sklearn_stub()


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_SRC_ROOT))

_mod_path = _SRC_ROOT / "sailsprep" / "action_model_testing" / "InternVideo_v2" / "InternVideo.py"
if not _mod_path.exists():
    raise FileNotFoundError(
        f"InternVideo.py not found at expected path: {_mod_path}\n"
        f"Resolved from __file__: {Path(__file__).resolve()}"
    )

_spec = importlib.util.spec_from_file_location("iv2", _mod_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
iv2 = _mod

# Restore real modules (or remove the stub) now that InternVideo.py is loaded,
# so later-collected test suites in the same session get the real packages.
for _name, _real in (
    ("pytorch_lightning", _real_pl),
    ("pytorch_lightning.callbacks", _real_pl_callbacks),
    ("cv2", _real_cv2),
    ("h5py", _real_h5py),
):
    if _real is not None:
        sys.modules[_name] = _real
    else:
        sys.modules.pop(_name, None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotation_df(
    label_col: str = "Locomotion",
    n: int = 60,
    label: str = "Walk",
) -> pd.DataFrame:
    return pd.DataFrame({
        "Frame": list(range(n)),
        label_col: [label] * n,
    })


def _make_split_csv(tmp_path: Path, label_col: str = "Locomotion") -> Path:
    """Write a minimal split CSV with valid-looking but fake file paths."""
    video = tmp_path / "video.mp4"
    label = tmp_path / "label.csv"
    h5 = tmp_path / "bbox.h5"
    # Create empty placeholder files so os.path.exists passes.
    video.touch()
    h5.touch()
    ann_df = _make_annotation_df(label_col)
    ann_df.to_csv(label, index=False)

    split_df = pd.DataFrame({
        "video_path": [str(video)],
        "label_path": [str(label)],
        "interpolated_anno_h5": [str(h5)],
        "split": ["train"],
    })
    csv_path = tmp_path / "split.csv"
    split_df.to_csv(csv_path, index=False)
    return csv_path


# ===========================================================================
# Unit tests
# ===========================================================================


class TestLoadBboxMap:
    """load_bbox_map returns a dict mapping frame index to (x1,y1,x2,y2)."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        # h5py stub always returns a structured row for any path.
        result = iv2.load_bbox_map(str(tmp_path / "dummy.h5"))
        assert isinstance(result, dict)

    def test_keys_are_ints(self, tmp_path: Path) -> None:
        result = iv2.load_bbox_map(str(tmp_path / "dummy.h5"))
        for k in result:
            assert isinstance(k, int)

    def test_values_are_4_tuples(self, tmp_path: Path) -> None:
        result = iv2.load_bbox_map(str(tmp_path / "dummy.h5"))
        for v in result.values():
            assert len(v) == 4


class TestFindActionRuns:
    """find_action_runs groups contiguous frames with the same label."""

    def test_single_run(self) -> None:
        df = _make_annotation_df(n=5, label="Walk")
        runs = iv2.find_action_runs(df, "Locomotion")
        assert len(runs) == 1
        start, end, lab = runs[0]
        assert lab == "Walk"
        assert start == 0
        assert end == 4

    def test_two_runs(self) -> None:
        df = pd.DataFrame({
            "Frame": list(range(10)),
            "Locomotion": ["Walk"] * 5 + ["Run"] * 5,
        })
        runs = iv2.find_action_runs(df, "Locomotion")
        assert len(runs) == 2
        assert runs[0][2] == "Walk"
        assert runs[1][2] == "Run"

    def test_na_labels_skipped(self) -> None:
        df = pd.DataFrame({
            "Frame": list(range(6)),
            "Locomotion": ["N/A", "N/A", "Walk", "Walk", "Walk", "Walk"],
        })
        runs = iv2.find_action_runs(df, "Locomotion")
        assert len(runs) == 1
        assert runs[0][2] == "Walk"

    def test_empty_label_skipped(self) -> None:
        df = pd.DataFrame({
            "Frame": [0, 1],
            "Locomotion": ["", "Walk"],
        })
        runs = iv2.find_action_runs(df, "Locomotion")
        assert len(runs) == 1

    def test_non_contiguous_creates_separate_runs(self) -> None:
        df = pd.DataFrame({
            "Frame": [0, 1, 5, 6],  # gap between 1 and 5
            "Locomotion": ["Walk"] * 4,
        })
        runs = iv2.find_action_runs(df, "Locomotion")
        assert len(runs) == 2

    def test_rmm_column(self) -> None:
        df = pd.DataFrame({
            "Frame": list(range(5)),
            "Repetitive_Motor_Movements": ["Hand"] * 5,
        })
        runs = iv2.find_action_runs(df, "Repetitive_Motor_Movements")
        assert len(runs) == 1


class TestChunkRun:
    """chunk_run produces non-overlapping clips of the expected length."""

    def test_too_short_returns_empty(self) -> None:
        # MIN_FRAMES = 15; anything shorter → []
        result = iv2.chunk_run(0, iv2.MIN_FRAMES - 2)
        assert result == []

    def test_single_clip_when_short_enough(self) -> None:
        # 20 frames → one clip
        result = iv2.chunk_run(0, 19)
        assert len(result) == 1
        assert result[0] == (0, 19)

    def test_split_around_45(self) -> None:
        # total = 46 → two clips
        result = iv2.chunk_run(0, 45)
        assert len(result) == 2

    def test_multiple_clips_long_run(self) -> None:
        # 120 frames → 4 × CLIP_FRAMES(30) clips
        result = iv2.chunk_run(0, 119)
        assert len(result) == 4
        for s, e in result:
            assert (e - s + 1) >= iv2.MIN_FRAMES

    def test_clips_cover_full_range(self) -> None:
        result = iv2.chunk_run(10, 69)  # 60 frames
        assert result[0][0] == 10
        assert result[-1][1] == 69

    def test_each_clip_at_least_min_frames(self) -> None:
        for total in [15, 30, 45, 60, 90, 120]:
            result = iv2.chunk_run(0, total - 1)
            for s, e in result:
                assert (e - s + 1) >= iv2.MIN_FRAMES


class TestBuildSamples:
    """build_samples reads the CSV and produces split dicts."""

    def test_returns_three_splits(self, tmp_path: Path) -> None:
        csv = _make_split_csv(tmp_path)
        with patch.object(iv2, "SPLIT_CSV", str(csv)):
            result = iv2.build_samples(str(csv), "Locomotion")
        assert set(result.keys()) == {"train", "val", "test"}

    def test_train_has_samples(self, tmp_path: Path) -> None:
        csv = _make_split_csv(tmp_path)
        with patch.object(iv2, "SPLIT_CSV", str(csv)):
            result = iv2.build_samples(str(csv), "Locomotion")
        assert len(result["train"]) > 0

    def test_sample_has_required_keys(self, tmp_path: Path) -> None:
        csv = _make_split_csv(tmp_path)
        with patch.object(iv2, "SPLIT_CSV", str(csv)):
            result = iv2.build_samples(str(csv), "Locomotion")
        sample = result["train"][0]
        for key in ("video_path", "h5_path", "start_frame", "end_frame",
                    "label_str", "ann_fps"):
            assert key in sample, f"Missing key: {key}"

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing column"):
            iv2.build_samples(str(bad_csv), "Locomotion")

    def test_unknown_split_is_ignored(self, tmp_path: Path) -> None:
        csv = _make_split_csv(tmp_path)
        df = pd.read_csv(csv)
        df["split"] = "unknown"
        df.to_csv(csv, index=False)
        with patch.object(iv2, "SPLIT_CSV", str(csv)):
            result = iv2.build_samples(str(csv), "Locomotion")
        assert all(len(v) == 0 for v in result.values())


class TestBBoxCropVideoDataset:
    """BBoxCropVideoDataset: shape, dtype, and error-recovery."""

    def _make_ds(self, training: bool = False) -> iv2.BBoxCropVideoDataset:
        samples = [{
            "video_path": "/fake/video.mp4",
            "h5_path": "/fake/bbox.h5",
            "start_frame": 0,
            "end_frame": 29,
            "label_str": "Walk",
            "ann_fps": 15.0,
        }]
        label_map = {"Walk": 0}
        return iv2.BBoxCropVideoDataset(
            samples, label_map, num_frames=4, crop_size=64, training=training
        )

    def test_len(self) -> None:
        ds = self._make_ds()
        assert len(ds) == 1

    def test_output_shape(self) -> None:
        ds = self._make_ds()
        frames, label = ds[0]
        assert frames.shape == (3, 4, 64, 64)  # (C, T, H, W)
        assert isinstance(label, int)

    def test_output_dtype_float32(self) -> None:
        ds = self._make_ds()
        frames, _ = ds[0]
        assert frames.dtype == torch.float32

    def test_label_correct(self) -> None:
        ds = self._make_ds()
        _, label = ds[0]
        assert label == 0

    def test_training_flag_does_not_crash(self) -> None:
        ds = self._make_ds(training=True)
        frames, label = ds[0]
        assert frames.shape == (3, 4, 64, 64)

    def test_bad_video_returns_zeros(self) -> None:
        """If the video cannot be decoded, __getitem__ returns a zero tensor."""
        import cv2 as _cv2

        class _BadCap:
            def isOpened(self) -> bool:
                return False
            def get(self, *a: Any) -> float:
                return 0.0
            def set(self, *a: Any) -> None:
                pass
            def read(self) -> tuple[bool, None]:
                return False, None
            def release(self) -> None:
                pass

        with patch.object(_cv2, "VideoCapture", return_value=_BadCap()):
            ds = self._make_ds()
            frames, label = ds[0]
        assert frames.shape == (3, 4, 64, 64)
        assert torch.all(frames == 0)


class TestCollateFn:
    """collate_fn produces correctly shaped batched tensors."""

    def test_batch_shape(self) -> None:
        B, C, T, H, W = 3, 3, 4, 64, 64
        batch = [(torch.zeros(C, T, H, W), i) for i in range(B)]
        videos, labels = iv2.collate_fn(batch)
        assert videos.shape == (B, C, T, H, W)
        assert labels.shape == (B,)
        assert labels.dtype == torch.long

    def test_label_values(self) -> None:
        batch = [(torch.zeros(3, 4, 32, 32), lbl) for lbl in [0, 1, 2]]
        _, labels = iv2.collate_fn(batch)
        assert labels.tolist() == [0, 1, 2]


class TestInternVideo2Classifier:
    """InternVideo2Classifier: forward pass shape and output dtype."""

    def _make_clf(self, feat_dim: int = 16, num_classes: int = 5) -> iv2.InternVideo2Classifier:
        backbone = nn.Linear(feat_dim, feat_dim, bias=False)
        # Make backbone return a 2-D tensor (B, feat_dim).
        return iv2.InternVideo2Classifier(backbone, feat_dim, num_classes)

    def test_forward_shape(self) -> None:
        clf = self._make_clf()
        # Backbone is a Linear; we pass (B, feat_dim) directly via a wrapper.
        # Override forward to bypass the video tensor handling.
        B = 4
        feat = torch.randn(B, 16)
        logits = clf.head(clf.norm(clf.backbone(feat)))
        assert logits.shape == (B, 5)

    def test_tuple_output_handled(self) -> None:
        """Backbone returning a tuple is unwrapped correctly."""
        class _TupleBackbone(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return (x, x)

        clf = iv2.InternVideo2Classifier(_TupleBackbone(), feat_dim=8, num_classes=3)
        x = torch.randn(2, 8)
        out = clf.backbone(x)
        # Simulate the classifier's tuple-unwrapping logic.
        if isinstance(out, (tuple, list)):
            out = out[0]
        assert out.shape == (2, 8)

    def test_sequence_output_cls_token(self) -> None:
        """3-D backbone output (B, N, D) → CLS token at position 0."""
        class _SeqBackbone(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B = x.shape[0]
                return torch.randn(B, 10, 16)  # (B, N, D)

        clf = iv2.InternVideo2Classifier(_SeqBackbone(), feat_dim=16, num_classes=4)
        x = torch.randn(2, 3, 4, 32, 32)
        with patch.object(clf.backbone, "forward",
                          return_value=torch.randn(2, 10, 16)):
            out = clf.backbone(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feat = out[:, 0] if out.dim() == 3 else out
        assert feat.shape == (2, 16)


class TestTaskConfig:
    """TASK_CONFIG contains all expected keys and valid values."""

    @pytest.mark.parametrize("task", ["loco", "rmm"])
    def test_task_keys(self, task: str) -> None:
        cfg = iv2.TASK_CONFIG[task]
        assert "label_col" in cfg
        assert "num_classes" in cfg
        assert "output_dir" in cfg

    def test_loco_num_classes(self) -> None:
        assert iv2.TASK_CONFIG["loco"]["num_classes"] == 5

    def test_rmm_num_classes(self) -> None:
        assert iv2.TASK_CONFIG["rmm"]["num_classes"] == 4

    def test_loco_label_col(self) -> None:
        assert iv2.TASK_CONFIG["loco"]["label_col"] == "Locomotion"

    def test_rmm_label_col(self) -> None:
        assert iv2.TASK_CONFIG["rmm"]["label_col"] == "Repetitive_Motor_Movements"


class TestHyperparameters:
    """Smoke-test that global constants are sane."""

    def test_batch_size_positive(self) -> None:
        assert iv2.BATCH_SIZE > 0

    def test_num_frames_positive(self) -> None:
        assert iv2.NUM_FRAMES > 0

    def test_crop_size_positive(self) -> None:
        assert iv2.CROP_SIZE > 0

    def test_mean_std_length(self) -> None:
        assert len(iv2.MEAN) == 3
        assert len(iv2.STD) == 3

    def test_learning_rate_in_range(self) -> None:
        assert 0 < iv2.LEARNING_RATE < 1

    def test_min_frames_leq_clip_frames(self) -> None:
        assert iv2.MIN_FRAMES <= iv2.CLIP_FRAMES


class TestRunInference:
    """run_inference writes a predictions CSV and handles errors."""

    def _minimal_setup(
        self, tmp_path: Path
    ) -> tuple[nn.Module, list[dict], dict[str, int]]:
        class _FakeModel(nn.Module):
            def eval(self) -> _FakeModel:
                return self

            def to(self, device: Any) -> _FakeModel:
                return self

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.9, 0.1]])

        model = _FakeModel()
        samples = [{
            "video_path": "/fake/v.mp4",
            "h5_path": "/fake/b.h5",
            "start_frame": 0,
            "end_frame": 29,
            "label_str": "Walk",
            "ann_fps": 15.0,
        }]
        label_map = {"Walk": 0, "Run": 1}
        return model, samples, label_map

    def test_csv_created(self, tmp_path: Path) -> None:
        model, samples, label_map = self._minimal_setup(tmp_path)
        out_csv = str(tmp_path / "preds.csv")

        with patch.object(iv2.BBoxCropVideoDataset, "__getitem__",
                          return_value=(torch.zeros(3, 4, 64, 64), 0)):
            iv2.run_inference(
                model, samples, label_map,
                torch.device("cpu"), out_csv, str(tmp_path)
            )

        assert Path(out_csv).exists()
        df = pd.read_csv(out_csv)
        assert "pred_label" in df.columns
        assert "true_label" in df.columns

    def test_metrics_file_created(self, tmp_path: Path) -> None:
        model, samples, label_map = self._minimal_setup(tmp_path)
        out_csv = str(tmp_path / "preds.csv")

        with patch.object(iv2.BBoxCropVideoDataset, "__getitem__",
                          return_value=(torch.zeros(3, 4, 64, 64), 0)):
            iv2.run_inference(
                model, samples, label_map,
                torch.device("cpu"), out_csv, str(tmp_path)
            )

        assert (tmp_path / "test_metrics.txt").exists()