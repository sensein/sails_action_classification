"""
Tests for src/sailsprep/action_model_testing/vlm_models/common/shared_utils.py
"""
from __future__ import annotations

import pytest

from common.shared_utils import (
    LABEL_FPS,
    compute_top2_from_votes,
    frame_labels_to_clip_labels,
    load_frame_labels,
)


class TestFrameLabelsToClips:
    def test_two_windows(self):
        labels = ["Walking"] * 30 + ["Crawling"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert len(clips) == 2
        assert clips[0]["label_full"] == "Walking"
        assert clips[1]["label_full"] == "Crawling"

    def test_short_window_skipped(self):
        labels = ["Walking"] * 5  # < 50% of 30 frames
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert len(clips) == 0

    def test_unknown_maps_to_no_locomotion(self):
        labels = ["unknown_action"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_binary"] == "No_Locomotion"

    def test_binary_active(self):
        labels = ["Running"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_binary"] == "Locomotion"

    def test_binary_inactive(self):
        labels = ["No_Locomotion"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_binary"] == "No_Locomotion"

    def test_rmm_task_active(self):
        labels = ["Jumping"] * 30
        clips = frame_labels_to_clip_labels(labels, task="rmm")
        assert clips[0]["label_full"] == "Jumping"
        assert clips[0]["label_binary"] == "RMM"

    def test_rmm_inactive(self):
        labels = ["No_RMM"] * 30
        clips = frame_labels_to_clip_labels(labels, task="rmm")
        assert clips[0]["label_binary"] == "No_RMM"

    def test_clip_timing_correct(self):
        labels = ["Walking"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["start_sec"] == pytest.approx(0.0)
        assert clips[0]["end_sec"] == pytest.approx(30 / LABEL_FPS)

    def test_majority_determines_label(self):
        labels = ["Walking"] * 20 + ["Running"] * 10
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_full"] == "Walking"


class TestTop2Accuracy:
    def test_top1_correct(self):
        assert compute_top2_from_votes([["Walking", "Walking"]], ["Walking"]) == 1.0

    def test_top2_second_place(self):
        preds = [["Running", "Running", "Walking"]]
        assert compute_top2_from_votes(preds, ["Walking"]) == 1.0

    def test_top2_miss(self):
        preds = [["Crawling", "Crawling", "Running"]]
        assert compute_top2_from_votes(preds, ["Walking"]) == 0.0

    def test_empty_preds_skip(self):
        assert compute_top2_from_votes([[]], ["Walking"]) == 0.0

    def test_multiple_clips(self):
        preds = [["Walking", "Walking"], ["Running", "Running"]]
        y_true = ["Walking", "Crawling"]
        assert compute_top2_from_votes(preds, y_true) == pytest.approx(0.5)


class TestLoadFrameLabels:
    def test_valid_loco_labels(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,locomotion\n0,Walking\n1,Crawling\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["Walking", "Crawling"]

    def test_unknown_maps_to_no_locomotion(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,locomotion\n0,FlyingThroughAir\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["No_Locomotion"]

    def test_rmm_labels(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,repetitive_motor\n0,Jumping\n1,Rocking\n")
        labels = load_frame_labels(str(csv), task="rmm")
        assert labels == ["Jumping", "Rocking"]

    def test_bom_csv_handled(self, tmp_path):
        """CSV with BOM (utf-8-sig) should load cleanly."""
        csv = tmp_path / "labels.csv"
        csv.write_bytes(b"\xef\xbb\xbfframe,locomotion\n0,Running\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["Running"]
