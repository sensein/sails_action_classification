"""
Tests for src/sailsprep/action_model_testing/OpenTAD/prepare_data/prepare_data.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "OpenTAD" / "prepare_data" / "prepare_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("opentad_prepare_data", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pd_mod():
    return _load_module()


class TestParseLabelCsvMulticlass:
    def _write_csv(self, tmp_path, rows):
        df = pd.DataFrame(rows)
        p = tmp_path / "label.csv"
        df.to_csv(p, index=False)
        return p

    def test_missing_file_returns_empty(self, pd_mod, tmp_path):
        segments, total_frames = pd_mod.parse_label_csv_multiclass(
            str(tmp_path / "nope.csv"), ann_fps=30.0,
            column_name="Locomotion", class_to_id=pd_mod.LOCOMOTION_CLASS_TO_ID,
        )
        assert segments == [] and total_frames == 0

    def test_single_action_segment(self, pd_mod, tmp_path):
        rows = [{"Frame": i, "Locomotion": ("Walking" if 2 <= i <= 5 else None)} for i in range(10)]
        p = self._write_csv(tmp_path, rows)
        segments, total_frames = pd_mod.parse_label_csv_multiclass(
            str(p), ann_fps=1.0, column_name="Locomotion",
            class_to_id=pd_mod.LOCOMOTION_CLASS_TO_ID,
        )
        assert total_frames == 10
        assert len(segments) == 1
        assert segments[0]["label"] == "Walking"
        assert segments[0]["segment"] == [2.0, 6.0]

    def test_missing_target_column_returns_empty(self, pd_mod, tmp_path):
        p = self._write_csv(tmp_path, [{"Frame": 0, "Other": "x"}])
        segments, total_frames = pd_mod.parse_label_csv_multiclass(
            str(p), ann_fps=1.0, column_name="Locomotion",
            class_to_id=pd_mod.LOCOMOTION_CLASS_TO_ID,
        )
        assert segments == [] and total_frames == 0


class TestPoseJsonToNpy:
    def test_missing_file_returns_none(self, pd_mod, tmp_path):
        assert pd_mod.pose_json_to_npy(str(tmp_path / "nope.json")) is None

    def test_empty_frames_returns_none(self, pd_mod, tmp_path):
        p = tmp_path / "pose.json"
        p.write_text(json.dumps({"frames": {}}))
        assert pd_mod.pose_json_to_npy(str(p)) is None

    def test_builds_feature_array(self, pd_mod, tmp_path):
        kp = {"x": 1.0, "y": 2.0, "confidence": 0.9}
        frames = {"0": {"Nose": kp}, "2": {"Nose": kp}}
        p = tmp_path / "pose.json"
        p.write_text(json.dumps({"frames": frames}))
        feats = pd_mod.pose_json_to_npy(str(p))
        assert feats.shape == (3, pd_mod.POSE_DIM)


class TestBuildAnnotationJson:
    def test_basic_split_assignment(self, pd_mod):
        split_df = pd.DataFrame([
            {"video_path": "vidA.mp4", "label_path": "", "split": "train"},
            {"video_path": "vidB.mp4", "label_path": "", "split": "val"},
            {"video_path": "vidC.mp4", "label_path": "", "split": "test"},
        ])
        result = pd_mod.build_annotation_json(
            split_df, ann_fps=30.0, video_feature_lengths={}, task_name="locomotion",
        )
        db = result["database"]
        assert db["vidA"]["subset"] == "training"
        assert db["vidB"]["subset"] == "validation"
        assert db["vidC"]["subset"] == "test"

    def test_uses_video_feature_lengths(self, pd_mod):
        split_df = pd.DataFrame([{"video_path": "vidA.mp4", "label_path": "", "split": "train"}])
        result = pd_mod.build_annotation_json(
            split_df, ann_fps=10.0, video_feature_lengths={"vidA": 100}, task_name="rmm",
        )
        assert result["database"]["vidA"]["frame"] == 100
        assert result["database"]["vidA"]["duration"] == pytest.approx(10.0)
