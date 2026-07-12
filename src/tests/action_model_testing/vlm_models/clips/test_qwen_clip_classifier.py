"""
Tests for src/sailsprep/action_model_testing/vlm_models/clips/qwen_clip_classifier.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from clips.qwen_clip_classifier import ClipActionClassifier


def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


class TestQwenClipClassifier:
    @pytest.fixture()
    def clf(self):
        obj = object.__new__(ClipActionClassifier)
        obj.task = "loco"
        obj.class_names = ["Crawling", "Cruising", "Walking", "Running", "Vehicle"]
        obj.random_frames = False
        obj.seed = 42
        obj.num_sample_frames = 8
        return obj

    def test_parse_exact(self, clf):
        assert clf._parse_action("ACTION: Running") == "Running"

    def test_parse_fallback_keyword(self, clf):
        assert clf._parse_action("child is cruising along wall") == "Cruising"

    def test_parse_none(self, clf):
        assert clf._parse_action("unknown action") is None

    def test_majority_vote(self, clf):
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 4)
        clf.classify_frame = MagicMock(side_effect=[
            ("Crawling", "ACTION: Crawling"),
            ("Crawling", "ACTION: Crawling"),
            ("Crawling", "ACTION: Crawling"),
            ("Walking", "ACTION: Walking"),
        ])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred == "Crawling"
        assert conf == pytest.approx(3 / 4)

    def test_empty_video(self, clf):
        clf._sample_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None
