"""
Tests for src/sailsprep/action_model_testing/slow_fast/experiments/common/data.py
"""
import pandas as pd
import pytest
import torch

from sailsprep.action_model_testing.slow_fast.experiments.common.data import (
    load_splits_from_csv,
    slow_collate,
    slowfast_collate,
)


class TestLoadSplitsFromCsv:
    def _write_csv(self, tmp_path):
        rows = [
            {"clip": "/data/Walking/clip1.mp4", "split": "train"},
            {"clip": "/data/Cruising/clip2.mp4", "split": "val"},
            {"clip": "/data/Running/clip3.mp4", "split": "test"},
            {"clip": "/data/UnknownClass/clip4.mp4", "split": "train"},
            {"clip": "", "split": "train"},
        ]
        p = tmp_path / "master.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_splits_by_split_column(self, tmp_path):
        p = self._write_csv(tmp_path)
        df_train, df_val, df_test = load_splits_from_csv(str(p), "clip", "split")
        assert len(df_train) == 1
        assert len(df_val) == 1
        assert len(df_test) == 1

    def test_unknown_class_dropped(self, tmp_path):
        p = self._write_csv(tmp_path)
        df_train, df_val, df_test = load_splits_from_csv(str(p), "clip", "split")
        assert "UnknownClass" not in df_train["csv_label"].tolist()

    def test_class_name_mapped_to_internal(self, tmp_path):
        p = self._write_csv(tmp_path)
        df_train, df_val, df_test = load_splits_from_csv(str(p), "clip", "split")
        assert df_train.iloc[0]["class_name"] == "walk"

    def test_no_overlap_between_splits(self, tmp_path):
        p = self._write_csv(tmp_path)
        df_train, df_val, df_test = load_splits_from_csv(str(p), "clip", "split")
        train_paths = set(df_train["video_path"])
        val_paths = set(df_val["video_path"])
        test_paths = set(df_test["video_path"])
        assert not (train_paths & val_paths)
        assert not (train_paths & test_paths)


class TestSlowfastCollate:
    def test_output_shapes(self):
        batch = []
        for i in range(3):
            slow = torch.zeros(3, 8, 224, 224)
            fast = torch.zeros(3, 32, 224, 224)
            batch.append(([slow, fast], i))

        (slow, fast), labels = slowfast_collate(batch)
        assert slow.shape == (3, 3, 8, 224, 224)
        assert fast.shape == (3, 3, 32, 224, 224)
        assert labels.tolist() == [0, 1, 2]
        assert labels.dtype == torch.long


class TestSlowCollate:
    def test_output_shapes(self):
        batch = [(torch.zeros(3, 8, 224, 224), i) for i in range(2)]
        videos, labels = slow_collate(batch)
        assert videos.shape == (2, 3, 8, 224, 224)
        assert labels.tolist() == [0, 1]
