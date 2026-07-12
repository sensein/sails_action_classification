"""
Tests for sailsprep/action_model_testing/vjepa/clips_without_coi_crop/locomotion/extract_features.py
No disk I/O beyond tmp_path, no GPU, no HF downloads, no video codec required.

Run: poetry run pytest
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types

import numpy as np
import torch

# The module hard-imports `decord` at module load time. decord ships only as
# manylinux wheels, so stub it out here for platforms/CI runners where it
# isn't installed; the tests below never touch VideoReader/cpu directly.
# A bare types.ModuleType leaves __spec__ as None, which later crashes
# importlib.util.find_spec("decord") (e.g. transformers' optional-dep probing)
# for any other test collected in this same pytest process, so give it a spec.
if "decord" not in sys.modules:
    _decord_stub = types.ModuleType("decord")
    _decord_stub.VideoReader = object
    _decord_stub.cpu = lambda *args, **kwargs: None
    _decord_stub.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
    sys.modules["decord"] = _decord_stub

# The module also calls os.makedirs(OUTPUT_BASE, ...) at import time against
# a hardcoded cluster path (/orcd/...) that doesn't exist/isn't writable in
# CI. Swallow that failure for the duration of the import only.
_real_makedirs = os.makedirs
os.makedirs = lambda *a, **k: None
try:
    from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.locomotion.extract_features import (
        CROP_SIZE,
        NUM_FRAMES,
        build_dataset_from_folders,
    )
    # VJEPA2VideoDataset now lives in the shared common/extraction.py module
    # (extract_features.py only re-imports build_dataset_from_folders/extract_all_features).
    from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.common.extraction import (
        VJEPA2VideoDataset,
    )
finally:
    os.makedirs = _real_makedirs

DEVICE = torch.device("cpu")


# ============================================================
# 1. build_dataset_from_folders
# ============================================================

class TestBuildDatasetFromFolders:
    def test_collects_mp4_clips_per_class_folder(self, tmp_path):
        for cls, n in [("walk", 2), ("run", 3)]:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(n):
                (cls_dir / f"clip{i}.mp4").write_bytes(b"fake")

        df = build_dataset_from_folders(str(tmp_path))
        assert len(df) == 5
        assert set(df["label"]) == {"walk", "run"}
        assert (df["label"] == "walk").sum() == 2
        assert (df["label"] == "run").sum() == 3

    def test_ignores_non_mp4_files(self, tmp_path):
        cls_dir = tmp_path / "walk"
        cls_dir.mkdir()
        (cls_dir / "clip.mp4").write_bytes(b"fake")
        (cls_dir / "notes.txt").write_bytes(b"fake")

        df = build_dataset_from_folders(str(tmp_path))
        assert len(df) == 1

    def test_empty_dir_returns_empty_dataframe(self, tmp_path):
        df = build_dataset_from_folders(str(tmp_path))
        assert len(df) == 0


# ============================================================
# 2. VJEPA2VideoDataset — error path (no real decoding)
# ============================================================

class TestVJEPA2VideoDataset:
    def test_len(self):
        ds = VJEPA2VideoDataset(["a.mp4", "b.mp4"], [0, 1], processor=None)
        assert len(ds) == 2

    def test_getitem_falls_back_to_dummy_on_load_error(self):
        ds = VJEPA2VideoDataset(["/nonexistent.mp4"], [0], processor=None)
        pixel_values, label = ds[0]
        assert pixel_values.shape == (NUM_FRAMES, 3, CROP_SIZE, CROP_SIZE)
        assert label == 0
