"""
Tests for the dlc2action locomotion classification pipeline.

Run with:
    pytest test_pipeline.py -v
or with coverage:
    pytest test_pipeline.py -v --cov=pipeline --cov-report=term-missing
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from sailsprep.action_model_testing.dlc_action.run import (
    NUM_KEYPOINTS,
    COORDS_PER_KPT,
    FEATURE_DIM,
    analyze_labels,
    convert_labels_to_segments,
    debug_feature_shapes,
    load_pose_from_json,
    match_files,
    prepare_data,
    split_from_csv,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tmp(tmp_path):
    """Alias for pytest's tmp_path for shorter usage."""
    return tmp_path


@pytest.fixture()
def sample_pose_json(tmp):
    """Write a minimal pose JSON with 3 frames and NUM_KEYPOINTS keypoints."""
    frames = {}
    for frame_idx in range(1, 4):          # frames "1", "2", "3"
        kps = {}
        for kp_idx in range(NUM_KEYPOINTS):
            kps[f"kp_{kp_idx:03d}"] = {
                "x": float(frame_idx + kp_idx),
                "y": float(frame_idx * 2 + kp_idx),
                "confidence": 0.9,
            }
        frames[str(frame_idx)] = kps

    path = tmp / "pose.json"
    path.write_text(json.dumps({"frames": frames}))
    return str(path)


@pytest.fixture()
def sample_label_csv(tmp):
    """Three-row label CSV with a Locomotion column."""
    df = pd.DataFrame({
        "frame": [0, 1, 2],
        "Locomotion": ["Walking", "Walking", "Running"],
    })
    path = tmp / "labels.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def sample_mapping_csv(tmp, sample_pose_json, sample_label_csv):
    """Mapping CSV consumed by match_files() and split_from_csv()."""
    path = tmp / "mapping.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "hrnet_full_path", "label_path", "split"],
        )
        writer.writeheader()
        writer.writerow({
            "video_path":      str(tmp / "clip_001.mp4"),
            "hrnet_full_path": sample_pose_json,
            "label_path":      sample_label_csv,
            "split":           "train",
        })
    return str(path)


# ===========================================================================
# match_files
# ===========================================================================

class TestMatchFiles:
    def test_returns_expected_keys(self, sample_mapping_csv):
        result = match_files(sample_mapping_csv)
        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert {"name", "video", "label", "pose"} == set(item.keys())

    def test_name_is_video_stem(self, sample_mapping_csv):
        result = match_files(sample_mapping_csv)
        assert result[0]["name"] == "clip_001"

    def test_missing_required_columns_raises(self, tmp):
        bad_csv = tmp / "bad.csv"
        bad_csv.write_text("video_path\n/foo/bar.mp4\n")
        with pytest.raises(ValueError, match="CSV must contain columns"):
            match_files(str(bad_csv))


# ===========================================================================
# analyze_labels
# ===========================================================================

class TestAnalyzeLabels:
    def test_returns_sorted_unique_classes(self, sample_mapping_csv, sample_label_csv):
        matched = match_files(sample_mapping_csv)
        classes = analyze_labels(matched, "Locomotion")
        assert classes == sorted({"Walking", "Running"})

    def test_missing_column_returns_empty(self, sample_mapping_csv):
        matched = match_files(sample_mapping_csv)
        classes = analyze_labels(matched, "NonExistentColumn")
        assert classes == []


# ===========================================================================
# load_pose_from_json
# ===========================================================================

class TestLoadPoseFromJson:
    def test_shape(self, sample_pose_json):
        pose = load_pose_from_json(sample_pose_json)
        assert pose.shape == (3, NUM_KEYPOINTS, COORDS_PER_KPT)

    def test_dtype_float32(self, sample_pose_json):
        pose = load_pose_from_json(sample_pose_json)
        assert pose.dtype == np.float32

    def test_values_written_correctly(self, sample_pose_json):
        pose = load_pose_from_json(sample_pose_json)
        # Frame 1 (index 0), keypoint 0: x = 1+0 = 1, y = 2+0 = 2, conf = 0.9
        assert pose[0, 0, 0] == pytest.approx(1.0)
        assert pose[0, 0, 1] == pytest.approx(2.0)
        assert pose[0, 0, 2] == pytest.approx(0.9)

    def test_frames_sorted_numerically(self, tmp):
        """Frames "1","2","10" must be ordered 1 < 2 < 10, not lexicographically."""
        frames = {}
        for i in [1, 2, 10]:
            kps = {
                f"kp_{k:03d}": {"x": float(i), "y": 0.0, "confidence": 1.0}
                for k in range(NUM_KEYPOINTS)
            }
            frames[str(i)] = kps

        path = tmp / "sorted_pose.json"
        path.write_text(json.dumps({"frames": frames}))

        pose = load_pose_from_json(str(path))
        # index 2 must be frame "10", so x should be 10, not 2
        assert pose[2, 0, 0] == pytest.approx(10.0)

    def test_unknown_keypoint_name_is_skipped(self, tmp):
        """Keypoints with non-integer suffixes should not crash."""
        frames = {
            "1": {
                "kp_000": {"x": 1.0, "y": 2.0, "confidence": 0.8},
                "kp_bad": {"x": 9.0, "y": 9.0, "confidence": 0.1},
            }
        }
        path = tmp / "bad_kp.json"
        path.write_text(json.dumps({"frames": frames}))

        pose = load_pose_from_json(str(path))
        assert pose.shape == (1, NUM_KEYPOINTS, COORDS_PER_KPT)


# ===========================================================================
# convert_labels_to_segments
# ===========================================================================

class TestConvertLabelsToSegments:
    def test_basic_segmentation(self, sample_label_csv):
        seg = convert_labels_to_segments(sample_label_csv, "Locomotion")
        # frames 0-1 = Walking, frame 2 = Running
        assert len(seg) == 2
        assert seg.iloc[0]["behavior"] == "Walking"
        assert seg.iloc[0]["start"] == 0
        assert seg.iloc[0]["end"] == 1
        assert seg.iloc[1]["behavior"] == "Running"
        assert seg.iloc[1]["start"] == 2
        assert seg.iloc[1]["end"] == 2

    def test_max_frames_truncates(self, sample_label_csv):
        seg = convert_labels_to_segments(sample_label_csv, "Locomotion", max_frames=2)
        # Only frames 0-1 (Walking), Running is cut off
        assert all(seg["behavior"] == "Walking")

    def test_missing_column_returns_empty(self, sample_label_csv):
        seg = convert_labels_to_segments(sample_label_csv, "NonExistent")
        assert seg.empty

    def test_nan_filled_with_unlabeled(self, tmp):
        df = pd.DataFrame({"Locomotion": ["Walking", None, "Running"]})
        path = tmp / "nan_labels.csv"
        df.to_csv(path, index=False)
        seg = convert_labels_to_segments(str(path), "Locomotion")
        behaviors = seg["behavior"].tolist()
        assert "unlabeled" in behaviors

    def test_output_dtypes(self, sample_label_csv):
        seg = convert_labels_to_segments(sample_label_csv, "Locomotion")
        assert seg["start"].dtype == np.int64
        assert seg["end"].dtype == np.int64
        assert seg["behavior"].dtype == object  # str


# ===========================================================================
# prepare_data
# ===========================================================================

class TestPrepareData:
    def test_creates_pose_and_label_files(self, tmp, sample_mapping_csv):
        matched = match_files(sample_mapping_csv)
        prepared = prepare_data(matched, "Locomotion", str(tmp / "output"))

        pose_dir  = tmp / "output" / "pose_data"
        label_dir = tmp / "output" / "labels"

        assert len(prepared) == 1
        assert (pose_dir  / "clip_001_features.pt").exists()
        assert (label_dir / "clip_001.csv").exists()

    def test_feature_tensor_shape(self, tmp, sample_mapping_csv):
        matched = match_files(sample_mapping_csv)
        prepare_data(matched, "Locomotion", str(tmp / "output"))

        pt = torch.load(
            tmp / "output" / "pose_data" / "clip_001_features.pt",
            weights_only=True,
        )
        tensor = pt["ind0"]
        assert tensor.ndim == 2
        assert tensor.shape[1] == FEATURE_DIM

    def test_bad_file_skipped_gracefully(self, tmp):
        matched = [
            {
                "name": "bad_clip",
                "video": "/nonexistent/video.mp4",
                "label": "/nonexistent/labels.csv",
                "pose":  "/nonexistent/pose.json",
            }
        ]
        prepared = prepare_data(matched, "Locomotion", str(tmp / "output"))
        assert prepared == []


# ===========================================================================
# debug_feature_shapes
# ===========================================================================

class TestDebugFeatureShapes:
    def test_passes_with_correct_shape(self, tmp):
        pose_dir = tmp / "pose_data"
        pose_dir.mkdir()
        tensor = torch.zeros(10, FEATURE_DIM)
        torch.save({"ind0": tensor}, pose_dir / "clip_features.pt")
        debug_feature_shapes(str(pose_dir))  # Should not raise

    def test_raises_on_wrong_dim(self, tmp):
        pose_dir = tmp / "pose_data"
        pose_dir.mkdir()
        bad_tensor = torch.zeros(10, FEATURE_DIM + 1)
        torch.save({"ind0": bad_tensor}, pose_dir / "bad_features.pt")
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            debug_feature_shapes(str(pose_dir))


# ===========================================================================
# split_from_csv
# ===========================================================================

class TestSplitFromCsv:
    def _write_mapping(self, tmp, rows):
        path = tmp / "split_mapping.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["video_path", "hrnet_full_path", "label_path", "split"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def test_correct_split_assignment(self, tmp):
        rows = [
            {"video_path": "/a/clip_001.mp4", "hrnet_full_path": "", "label_path": "", "split": "train"},
            {"video_path": "/a/clip_002.mp4", "hrnet_full_path": "", "label_path": "", "split": "val"},
            {"video_path": "/a/clip_003.mp4", "hrnet_full_path": "", "label_path": "", "split": "test"},
        ]
        csv_path = self._write_mapping(tmp, rows)
        names = ["clip_001", "clip_002", "clip_003"]
        split = split_from_csv(csv_path, names)
        assert split["train"] == ["clip_001"]
        assert split["val"]   == ["clip_002"]
        assert split["test"]  == ["clip_003"]

    def test_unknown_split_defaults_to_train(self, tmp, capsys):
        rows = [
            {"video_path": "/a/clip_001.mp4", "hrnet_full_path": "", "label_path": "", "split": "weirdvalue"},
        ]
        csv_path = self._write_mapping(tmp, rows)
        split = split_from_csv(csv_path, ["clip_001"])
        assert "clip_001" in split["train"]

    def test_missing_split_column_raises(self, tmp):
        path = tmp / "no_split.csv"
        path.write_text("video_path,hrnet_full_path,label_path\n/a/b.mp4,,\n")
        with pytest.raises(ValueError, match="'split' column"):
            split_from_csv(str(path), ["b"])

    def test_video_not_in_csv_defaults_to_train(self, tmp):
        rows = [
            {"video_path": "/a/clip_001.mp4", "hrnet_full_path": "", "label_path": "", "split": "train"},
        ]
        csv_path = self._write_mapping(tmp, rows)
        split = split_from_csv(csv_path, ["clip_001", "mystery_clip"])
        assert "mystery_clip" in split["train"]


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestIntegration:
    """
    Light end-to-end test that wires match → analyze → prepare → split.
    Does NOT call Project (requires a full dlc2action installation + GPU).
    """

    def test_full_preprocessing_pipeline(self, tmp, sample_mapping_csv):
        matched = match_files(sample_mapping_csv)
        assert matched

        action_classes = analyze_labels(matched, "Locomotion")
        assert action_classes

        out_dir = str(tmp / "processed")
        prepared = prepare_data(matched, "Locomotion", out_dir)
        assert prepared

        pose_dir = os.path.join(out_dir, "pose_data")
        debug_feature_shapes(pose_dir)  # raises if shapes are wrong

        split = split_from_csv(sample_mapping_csv, prepared)
        assert "train" in split
        total = sum(len(v) for v in split.values())
        assert total == len(prepared)