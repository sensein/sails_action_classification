"""
Tests for src/sailsprep/action_model_testing/vjepa/clips_without_coi_crop/common/extraction.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from sailsprep.action_model_testing.vjepa.clips_without_coi_crop.common.extraction import (
    VJEPA2VideoDataset,
    build_dataset_from_folders,
    extract_all_features,
)


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

    def test_empty_dir_returns_empty_dataframe(self, tmp_path):
        df = build_dataset_from_folders(str(tmp_path))
        assert len(df) == 0


class TestVJEPA2VideoDataset:
    def test_len(self):
        ds = VJEPA2VideoDataset(["a.mp4", "b.mp4"], [0, 1], processor=None)
        assert len(ds) == 2

    def test_getitem_falls_back_to_dummy_on_load_error(self):
        ds = VJEPA2VideoDataset(["/nonexistent.mp4"], [0], processor=None, num_frames=8, crop_size=32)
        pixel_values, label = ds[0]
        assert pixel_values.shape == (8, 3, 32, 32)
        assert label == 0


class TestExtractAllFeatures:
    def test_extracts_and_concatenates_batches(self):
        video_paths = ["/nonexistent1.mp4", "/nonexistent2.mp4"]
        labels = [0, 1]

        fake_model = MagicMock()
        fake_model.eval.return_value = None
        fake_model.to.return_value = fake_model
        fake_model.return_value = MagicMock(last_hidden_state=torch.zeros(1, 4, 8))

        features, out_labels = extract_all_features(
            fake_model, processor=None,
            video_paths=video_paths, labels=labels,
            device=torch.device("cpu"),
            num_frames=4, crop_size=16, batch_size=1, num_workers=0,
        )
        assert features.shape == (2, 4, 8)
        assert out_labels.tolist() == [0, 1]
