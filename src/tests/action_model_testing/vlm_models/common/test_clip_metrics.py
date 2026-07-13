"""
Tests for src/sailsprep/action_model_testing/vlm_models/common/clip_metrics.py
"""
from __future__ import annotations

import json

import pytest

from common.clip_metrics import compute_metrics, compute_top2_accuracy, extract_label_from_path


class TestExtractLabelFromPath:
    def test_finds_class_folder(self):
        assert extract_label_from_path("/data/Walking/clip1.mp4", ["Walking", "Running"]) == "Walking"

    def test_returns_none_if_not_found(self):
        assert extract_label_from_path("/data/Unknown/clip1.mp4", ["Walking", "Running"]) is None


class TestComputeMetrics:
    def test_perfect_predictions(self, tmp_path):
        y_true = ["Walking", "Running", "Walking", "Running"]
        y_pred = ["Walking", "Running", "Walking", "Running"]
        metrics = compute_metrics(y_true, y_pred, ["Walking", "Running"], str(tmp_path))
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert (tmp_path / "evaluation_metrics.json").exists()
        assert (tmp_path / "confusion_matrix.csv").exists()

    def test_metrics_json_is_valid(self, tmp_path):
        y_true = ["Walking", "Running", "Walking"]
        y_pred = ["Walking", "Walking", "Walking"]
        compute_metrics(y_true, y_pred, ["Walking", "Running"], str(tmp_path))
        data = json.loads((tmp_path / "evaluation_metrics.json").read_text())
        assert "accuracy" in data
        assert "confusion_matrix" in data


class TestComputeTop2Accuracy:
    def test_top1_hit(self):
        preds = [["Walking", "Walking"]]
        assert compute_top2_accuracy(preds, ["Walking"]) == 1.0

    def test_top2_hit(self):
        preds = [["Running", "Running", "Walking"]]
        assert compute_top2_accuracy(preds, ["Walking"]) == 1.0

    def test_miss(self):
        preds = [["Crawling", "Crawling"]]
        assert compute_top2_accuracy(preds, ["Walking"]) == 0.0

    def test_empty_preds_skipped(self):
        assert compute_top2_accuracy([[]], ["Walking"]) == 0.0
