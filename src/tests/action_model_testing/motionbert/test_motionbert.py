"""
Tests for sailsprep.action_model_testing.motionbert.motionbert
"""

from __future__ import annotations

import os
import sys
import types
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stub heavy optional deps before motionbert is imported
# ---------------------------------------------------------------------------

# ultralytics  ----------------------------------------------------------------
_ul = types.ModuleType("ultralytics")


class _FakeYOLO:
    def __init__(self, *a: Any, **kw: Any) -> None:
        pass


_ul.YOLO = _FakeYOLO  # type: ignore[attr-defined]
sys.modules.setdefault("ultralytics", _ul)

# lib / lib.model.DSTformer  --------------------------------------------------
# These are only imported inside functions (lift_2d_to_3d,
# build_action_recognition_model) so we pre-populate sys.modules so that the
# per-test patches land correctly too.

class _StubDSTformer(nn.Module):
    """Tiny stand-in for MotionBERT's DSTformer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, C = x.shape
        # Return same (B, T, J*C) layout the real model produces
        return x.reshape(B, T, J * C)


_lib = types.ModuleType("lib")
_lib_model = types.ModuleType("lib.model")
_lib_model_dst = types.ModuleType("lib.model.DSTformer")
_lib_model_dst.DSTformer = _StubDSTformer  # type: ignore[attr-defined]
sys.modules.setdefault("lib", _lib)
sys.modules.setdefault("lib.model", _lib_model)
sys.modules.setdefault("lib.model.DSTformer", _lib_model_dst)

# Now import the module under test  ------------------------------------------
import sailsprep.action_model_testing.motionbert.motionbert as mb  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vid(video_id: str = "walk_clip01", cls: str = "walk") -> dict[str, Any]:
    return {
        "video_id": video_id,
        "path": f"/fake/{video_id}.mp4",
        "filename": f"{video_id}.mp4",
        "label": mb.CLASS_TO_IDX[cls],
        "class_name": cls,
    }


def _write_npy(directory: str, video_id: str, T: int) -> None:
    arr = np.random.rand(T, mb.NUM_KEYPOINTS, 3).astype(np.float32)
    np.save(os.path.join(directory, f"{video_id}.npy"), arr)


def _build_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "splits.csv"
    df.to_csv(p, index=False)
    return p


def _csv_row(csv_cls: str, split: str, stem: str, tmp_path: Path) -> dict[str, Any]:
    return {
        "cut_clip_path": str(tmp_path / csv_cls / f"{stem}.mp4"),
        "split": split,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_five_action_classes(self) -> None:
        assert len(mb.ACTION_CLASSES) == 5

    def test_action_class_names(self) -> None:
        assert set(mb.ACTION_CLASSES) == {"walk", "cruise", "crawl", "vehicle", "run"}

    def test_class_to_idx_covers_all_classes(self) -> None:
        assert set(mb.CLASS_TO_IDX.keys()) == set(mb.ACTION_CLASSES)

    def test_idx_to_class_is_inverse_of_class_to_idx(self) -> None:
        for cls, idx in mb.CLASS_TO_IDX.items():
            assert mb.IDX_TO_CLASS[idx] == cls

    def test_csv_class_to_internal_values_are_valid(self) -> None:
        for v in mb.CSV_CLASS_TO_INTERNAL.values():
            assert v in mb.ACTION_CLASSES

    def test_max_frames(self) -> None:
        assert mb.MAX_FRAMES == 243

    def test_num_keypoints(self) -> None:
        assert mb.NUM_KEYPOINTS == 17

    def test_video_extensions_have_dots(self) -> None:
        for ext in mb.VIDEO_EXTENSIONS:
            assert ext.startswith(".")


# ---------------------------------------------------------------------------
# setup_output_dirs
# ---------------------------------------------------------------------------

class TestSetupOutputDirs:
    def test_returns_six_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mb, "OUTPUT_ROOT", str(tmp_path / "pose"))
        monkeypatch.setattr(mb, "ACTION_OUTPUT_ROOT", str(tmp_path / "action"))
        dirs = mb.setup_output_dirs()
        assert set(dirs) == {"pose_2d", "pose_3d", "predictions", "checkpoints", "logs", "metadata"}

    def test_all_directories_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mb, "OUTPUT_ROOT", str(tmp_path / "pose"))
        monkeypatch.setattr(mb, "ACTION_OUTPUT_ROOT", str(tmp_path / "action"))
        dirs = mb.setup_output_dirs()
        for d in dirs.values():
            assert os.path.isdir(d)

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mb, "OUTPUT_ROOT", str(tmp_path / "pose"))
        monkeypatch.setattr(mb, "ACTION_OUTPUT_ROOT", str(tmp_path / "action"))
        mb.setup_output_dirs()
        mb.setup_output_dirs()  # must not raise

    def test_pose_dirs_under_output_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = str(tmp_path / "pose")
        monkeypatch.setattr(mb, "OUTPUT_ROOT", root)
        monkeypatch.setattr(mb, "ACTION_OUTPUT_ROOT", str(tmp_path / "action"))
        dirs = mb.setup_output_dirs()
        assert dirs["pose_2d"].startswith(root)
        assert dirs["pose_3d"].startswith(root)


# ---------------------------------------------------------------------------
# load_splits_from_csv
# ---------------------------------------------------------------------------

class TestLoadSplitsFromCsv:
    def _default_csv(self, tmp_path: Path) -> Path:
        rows = [
            _csv_row("Walking",  "train", "c1", tmp_path),
            _csv_row("Walking",  "train", "c2", tmp_path),
            _csv_row("Cruising", "val",   "c3", tmp_path),
            _csv_row("Crawling", "test",  "c4", tmp_path),
            _csv_row("Running",  "train", "c5", tmp_path),
            _csv_row("Vehicle",  "val",   "c6", tmp_path),
        ]
        return _build_csv(tmp_path, rows)

    def test_returns_train_val_test_keys(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_split_counts(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        assert len(splits["train"]) == 3
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 1

    def test_vid_info_required_keys(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        for vids in splits.values():
            for v in vids:
                for k in ("path", "label", "class_name", "filename", "video_id"):
                    assert k in v

    def test_class_names_are_internal(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        for vids in splits.values():
            for v in vids:
                assert v["class_name"] in mb.ACTION_CLASSES

    def test_labels_match_class_to_idx(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        for vids in splits.values():
            for v in vids:
                assert v["label"] == mb.CLASS_TO_IDX[v["class_name"]]

    def test_no_overlap_between_splits(self, tmp_path: Path) -> None:
        csv = self._default_csv(tmp_path)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        tr = {v["video_id"] for v in splits["train"]}
        va = {v["video_id"] for v in splits["val"]}
        te = {v["video_id"] for v in splits["test"]}
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)

    def test_unknown_csv_class_dropped(self, tmp_path: Path) -> None:
        rows = [
            _csv_row("Walking", "train", "c1", tmp_path),
            _csv_row("Skateboarding", "train", "c9", tmp_path),  # unknown
        ]
        csv = _build_csv(tmp_path, rows)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        assert len(splits["train"]) == 1

    def test_unknown_split_value_dropped(self, tmp_path: Path) -> None:
        rows = [
            _csv_row("Walking", "train",   "c1", tmp_path),
            _csv_row("Running", "holdout", "c9", tmp_path),  # unknown split
        ]
        csv = _build_csv(tmp_path, rows)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        total = sum(len(v) for v in splits.values())
        assert total == 1

    def test_video_id_format(self, tmp_path: Path) -> None:
        rows = [_csv_row("Walking", "train", "myclip", tmp_path)]
        csv = _build_csv(tmp_path, rows)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        assert splits["train"][0]["video_id"] == "walk_myclip"

    def test_all_csv_classes_map_correctly(self, tmp_path: Path) -> None:
        mapping = {
            "Walking": "walk", "Cruising": "cruise", "Crawling": "crawl",
            "Vehicle": "vehicle", "Running": "run",
        }
        rows = [_csv_row(csv_cls, "train", f"c{i}", tmp_path)
                for i, csv_cls in enumerate(mapping)]
        csv = _build_csv(tmp_path, rows)
        splits = mb.load_splits_from_csv(str(csv), "cut_clip_path", "split")
        found = {v["class_name"] for v in splits["train"]}
        assert found == set(mapping.values())


# ---------------------------------------------------------------------------
# get_all_videos
# ---------------------------------------------------------------------------

class TestGetAllVideos:
    def test_deduplicates(self) -> None:
        v1 = _vid("walk_a", "walk")
        v2 = _vid("cruise_b", "cruise")
        splits: dict[str, list[dict[str, Any]]] = {
            "train": [v1], "val": [v2], "test": [v1],
        }
        assert len(mb.get_all_videos(splits)) == 2

    def test_preserves_train_val_test_order(self) -> None:
        v1 = _vid("walk_a",   "walk")
        v2 = _vid("cruise_b", "cruise")
        v3 = _vid("crawl_c",  "crawl")
        splits: dict[str, list[dict[str, Any]]] = {
            "train": [v1], "val": [v2], "test": [v3],
        }
        ids = [v["video_id"] for v in mb.get_all_videos(splits)]
        assert ids == ["walk_a", "cruise_b", "crawl_c"]

    def test_empty_splits(self) -> None:
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        assert mb.get_all_videos(splits) == []

    def test_returns_all_when_no_duplicates(self) -> None:
        vids = [_vid(f"walk_{i}", "walk") for i in range(6)]
        splits: dict[str, list[dict[str, Any]]] = {
            "train": vids[:3], "val": vids[3:5], "test": vids[5:],
        }
        assert len(mb.get_all_videos(splits)) == 6


# ---------------------------------------------------------------------------
# SkeletonActionDataset
# ---------------------------------------------------------------------------

class TestSkeletonActionDataset:
    def test_len(self, tmp_path: Path) -> None:
        samples = [_vid(f"walk_{i}") for i in range(4)]
        for s in samples:
            _write_npy(str(tmp_path), s["video_id"], 60)
        ds = mb.SkeletonActionDataset(samples, str(tmp_path))
        assert len(ds) == 4

    def test_short_clip_padded_to_max_frames(self, tmp_path: Path) -> None:
        v = _vid("walk_short")
        _write_npy(str(tmp_path), v["video_id"], 50)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        kps, _ = ds[0]
        assert kps.shape == (mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)

    def test_long_clip_subsampled_to_max_frames(self, tmp_path: Path) -> None:
        v = _vid("walk_long")
        _write_npy(str(tmp_path), v["video_id"], 400)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        kps, _ = ds[0]
        assert kps.shape == (mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)

    def test_exact_length_clip(self, tmp_path: Path) -> None:
        v = _vid("walk_exact")
        _write_npy(str(tmp_path), v["video_id"], mb.MAX_FRAMES)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        kps, _ = ds[0]
        assert kps.shape == (mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)

    def test_correct_label_returned(self, tmp_path: Path) -> None:
        v = _vid("cruise_x", "cruise")
        _write_npy(str(tmp_path), v["video_id"], 80)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        _, label = ds[0]
        assert label.item() == mb.CLASS_TO_IDX["cruise"]

    def test_returns_float32_tensor(self, tmp_path: Path) -> None:
        v = _vid("walk_dtype")
        _write_npy(str(tmp_path), v["video_id"], 60)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        kps, _ = ds[0]
        assert kps.dtype == torch.float32

    def test_returns_int64_label(self, tmp_path: Path) -> None:
        v = _vid("walk_lbltype")
        _write_npy(str(tmp_path), v["video_id"], 60)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        _, label = ds[0]
        assert label.dtype == torch.int64

    def test_root_joint_xy_is_zero(self, tmp_path: Path) -> None:
        """Root subtraction must zero out joint-0 xy."""
        v = _vid("walk_root")
        arr = np.ones((60, mb.NUM_KEYPOINTS, 3), dtype=np.float32) * 5.0
        np.save(os.path.join(str(tmp_path), f"{v['video_id']}.npy"), arr)
        ds = mb.SkeletonActionDataset([v], str(tmp_path))
        kps, _ = ds[0]
        assert torch.allclose(kps[:, 0, :2], torch.zeros(mb.MAX_FRAMES, 2), atol=1e-5)

    def test_dataloader_batch_shape(self, tmp_path: Path) -> None:
        from torch.utils.data import DataLoader
        samples = [_vid(f"walk_b{i}") for i in range(3)]
        for s in samples:
            _write_npy(str(tmp_path), s["video_id"], 70)
        ds = mb.SkeletonActionDataset(samples, str(tmp_path))
        loader = DataLoader(ds, batch_size=3)
        kps_batch, lbl_batch = next(iter(loader))
        assert kps_batch.shape == (3, mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)
        assert lbl_batch.shape == (3,)

    def test_custom_max_frames(self, tmp_path: Path) -> None:
        v = _vid("walk_custom")
        _write_npy(str(tmp_path), v["video_id"], 60)
        ds = mb.SkeletonActionDataset([v], str(tmp_path), max_frames=100)
        kps, _ = ds[0]
        assert kps.shape == (100, mb.NUM_KEYPOINTS, 3)


# ---------------------------------------------------------------------------
# build_action_recognition_model
# ---------------------------------------------------------------------------

class TestBuildActionRecognitionModel:
    """DSTformer stub is already in sys.modules from module-level setup."""

    def test_returns_nn_module(self) -> None:
        model = mb.build_action_recognition_model("/fake", 5, "cpu")
        assert isinstance(model, nn.Module)

    def test_output_shape_five_classes(self) -> None:
        model = mb.build_action_recognition_model("/fake", 5, "cpu")
        model.eval()
        x = torch.randn(2, mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_output_shape_three_classes(self) -> None:
        model = mb.build_action_recognition_model("/fake", 3, "cpu")
        model.eval()
        x = torch.randn(1, mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_no_checkpoint_does_not_raise(self) -> None:
        mb.build_action_recognition_model("/fake", 5, "cpu", ckpt_path=None)

    def test_nonexistent_checkpoint_does_not_raise(self) -> None:
        mb.build_action_recognition_model(
            "/fake", 5, "cpu", ckpt_path="/does/not/exist.bin"
        )

    def test_batch_size_one(self) -> None:
        model = mb.build_action_recognition_model("/fake", 5, "cpu")
        model.eval()
        x = torch.randn(1, mb.MAX_FRAMES, mb.NUM_KEYPOINTS, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# run_action_inference — early exit (no checkpoint)
# ---------------------------------------------------------------------------

class TestRunActionInferenceEarlyExit:
    def test_returns_none_when_checkpoint_missing(self, tmp_path: Path) -> None:
        dirs = {
            "checkpoints": str(tmp_path / "ckpts"),
            "predictions": str(tmp_path / "preds"),
        }
        os.makedirs(dirs["checkpoints"])
        os.makedirs(dirs["predictions"])
        splits: dict[str, list[dict[str, Any]]] = {
            "train": [], "val": [], "test": [_vid("walk_t1")],
        }
        result = mb.run_action_inference(splits, str(tmp_path / "poses"), dirs, device="cpu")
        assert result is None

    def test_returns_none_with_empty_test_split_and_no_checkpoint(
        self, tmp_path: Path
    ) -> None:
        dirs = {
            "checkpoints": str(tmp_path / "ckpts"),
            "predictions": str(tmp_path / "preds"),
        }
        os.makedirs(dirs["checkpoints"])
        os.makedirs(dirs["predictions"])
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        result = mb.run_action_inference(splits, str(tmp_path / "poses"), dirs, device="cpu")
        assert result is None


# ---------------------------------------------------------------------------
# finetune_action_recognition — early exit (no pose files)
# ---------------------------------------------------------------------------

class TestFinetuneEarlyExit:
    def _dirs(self, tmp_path: Path) -> dict[str, str]:
        dirs = {
            "checkpoints": str(tmp_path / "ckpts"),
            "logs":        str(tmp_path / "logs"),
            "predictions": str(tmp_path / "preds"),
        }
        for d in dirs.values():
            os.makedirs(d)
        return dirs

    def test_returns_none_when_no_npy_files(self, tmp_path: Path) -> None:
        splits: dict[str, list[dict[str, Any]]] = {
            "train": [_vid("walk_a")],
            "val":   [],
            "test":  [],
        }
        result = mb.finetune_action_recognition(
            splits, str(tmp_path / "empty_poses"), self._dirs(tmp_path), device="cpu"
        )
        assert result is None

    def test_returns_none_when_all_splits_empty(self, tmp_path: Path) -> None:
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        result = mb.finetune_action_recognition(
            splits, str(tmp_path / "empty_poses"), self._dirs(tmp_path), device="cpu"
        )
        assert result is None