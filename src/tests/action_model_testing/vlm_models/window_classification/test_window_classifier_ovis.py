"""
Tests for src/sailsprep/action_model_testing/vlm_models/window_classification/window_classifier_ovis.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image
import numpy as np

from window_classification.window_classifier_ovis import OvisClassifier
from common.shared_utils import TASK_CONFIG


def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


class TestOvisWindowParser:
    """Parse methods of OvisClassifier -- no GPU."""

    @pytest.fixture()
    def clf(self):
        obj = object.__new__(OvisClassifier)
        cfg = TASK_CONFIG["loco"]
        obj.task = "loco"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.max_partition = 9
        obj.random_frames = False
        obj.seed = 42
        return obj

    @pytest.fixture()
    def clf_rmm(self):
        obj = object.__new__(OvisClassifier)
        cfg = TASK_CONFIG["rmm"]
        obj.task = "rmm"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.max_partition = 9
        obj.random_frames = False
        obj.seed = 42
        return obj

    def test_multiclass_action_tag(self, clf):
        assert clf._parse_multiclass("ACTION: Walking") == "Walking"

    def test_multiclass_no_locomotion(self, clf):
        assert clf._parse_multiclass("ACTION: No_Locomotion") == "No_Locomotion"

    def test_multiclass_space_variant(self, clf):
        result = clf._parse_multiclass("ACTION: No Locomotion")
        assert result == "No_Locomotion"

    def test_multiclass_none_on_garbage(self, clf):
        assert clf._parse_multiclass("I am unsure") is None

    def test_multiclass_empty_string(self, clf):
        assert clf._parse_multiclass("") is None

    def test_binary_yes(self, clf):
        assert clf._parse_binary("ANSWER: YES") is True

    def test_binary_no(self, clf):
        assert clf._parse_binary("ANSWER: NO") is False

    def test_binary_bare_yes(self, clf):
        assert clf._parse_binary("YES") is True

    def test_binary_bare_no(self, clf):
        assert clf._parse_binary("NO") is False

    def test_binary_garbage(self, clf):
        assert clf._parse_binary("maybe") is None

    def test_binary_empty(self, clf):
        assert clf._parse_binary("") is None

    def test_finegrained_action_tag(self, clf):
        assert clf._parse_finegrained("ACTION: Cruising") == "Cruising"

    def test_finegrained_flap_alias_rmm(self, clf_rmm):
        assert clf_rmm._parse_finegrained("ACTION: flapping") == "Hands_flapping"

    def test_finegrained_none_on_garbage(self, clf):
        assert clf._parse_finegrained("hmm") is None

    def test_classify_multiclass_empty_frames_returns_no_label(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "No_Locomotion"
        assert conf == 0.0

    def test_classify_binary_empty_frames_returns_no_label(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"
        assert votes == []

    def test_classify_binary_yes_vote(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()])
        clf._make_grid = MagicMock(return_value=dummy_image())
        clf._call = MagicMock(return_value="ANSWER: YES")
        label, conf, _ = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "Locomotion"

    def test_classify_binary_no_vote(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()])
        clf._make_grid = MagicMock(return_value=dummy_image())
        clf._call = MagicMock(return_value="ANSWER: NO")
        label, conf, _ = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"
