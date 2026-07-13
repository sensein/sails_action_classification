"""
Tests for src/sailsprep/action_model_testing/vlm_models/window_classification/window_classifier_qwen.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image
import numpy as np

from window_classification.window_classifier_qwen import QwenClassifier
from common.shared_utils import TASK_CONFIG


def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


class TestQwenWindowParser:
    """Parse methods of QwenClassifier -- no GPU."""

    @pytest.fixture()
    def clf(self):
        obj = object.__new__(QwenClassifier)
        cfg = TASK_CONFIG["loco"]
        obj.task = "loco"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.random_frames = False
        obj.seed = 42
        return obj

    @pytest.fixture()
    def clf_rmm(self):
        obj = object.__new__(QwenClassifier)
        cfg = TASK_CONFIG["rmm"]
        obj.task = "rmm"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.random_frames = False
        obj.seed = 42
        return obj

    def test_multiclass_action_tag(self, clf):
        assert clf._parse_multiclass("ACTION: Running") == "Running"

    def test_multiclass_no_loco_label(self, clf):
        assert clf._parse_multiclass("ACTION: No_Locomotion") == "No_Locomotion"

    def test_multiclass_space_variant(self, clf):
        assert clf._parse_multiclass("ACTION: No Locomotion") == "No_Locomotion"

    def test_multiclass_none_garbage(self, clf):
        assert clf._parse_multiclass("I don't know") is None

    def test_binary_yes(self, clf):
        assert clf._parse_binary("ANSWER: YES") is True

    def test_binary_no(self, clf):
        assert clf._parse_binary("ANSWER: NO") is False

    def test_binary_garbage(self, clf):
        assert clf._parse_binary("probably") is None

    def test_finegrained_walking(self, clf):
        assert clf._parse_finegrained("ACTION: Walking") == "Walking"

    def test_finegrained_flap_rmm(self, clf_rmm):
        assert clf_rmm._parse_finegrained("child is flapping hands") == "Hands_flapping"

    def test_classify_multiclass_empty_frames(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "No_Locomotion"

    def test_classify_multiclass_majority(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ACTION: Walking",
            "ACTION: Walking",
            "ACTION: Running",
        ])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "Walking"
        assert conf == pytest.approx(2 / 3)

    def test_classify_binary_empty_frames(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"

    def test_classify_binary_majority_yes(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ANSWER: YES",
            "ANSWER: YES",
            "ANSWER: NO",
        ])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "Locomotion"
        assert conf == pytest.approx(2 / 3)

    def test_classify_binary_majority_no(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ANSWER: NO",
            "ANSWER: NO",
            "ANSWER: YES",
        ])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"
        # confidence is always the fraction of positive (YES) votes, not the winning label's share
        assert conf == pytest.approx(1 / 3)
